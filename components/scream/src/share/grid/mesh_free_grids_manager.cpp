#include "share/grid/mesh_free_grids_manager.hpp"

#include "share/io/scorpio_input.hpp"
#include "share/grid/point_grid.hpp"
#include "share/grid/se_grid.hpp"
#include "share/grid/remap/do_nothing_remapper.hpp"

#include "ekat/std_meta/ekat_std_utils.hpp"

#include <memory>
#include <numeric>

namespace scream {

MeshFreeGridsManager::
MeshFreeGridsManager (const ekat::Comm& comm, const ekat::ParameterList& p)
 : m_params (p)
 , m_comm   (comm)
{
}

MeshFreeGridsManager::remapper_ptr_type
MeshFreeGridsManager::
do_create_remapper (const grid_ptr_type from_grid,
                    const grid_ptr_type to_grid) const
{
  return std::make_shared<DoNothingRemapper>(from_grid,to_grid);
}

void MeshFreeGridsManager::
build_grids ()
{
  int num_vertical_levels = -1;
  int num_global_cols = -1;
  int num_local_elems = -1;
  int num_gp = -1;

  if (m_params.get<bool>("load_data_from_ic_file",true)) {
    // For AD runs, we expect to read dimensions from IC file

    const auto& ic_file = m_params.get<std::string>("Initial Conditions Filename");

    // Register the IC file...
    scorpio::register_file(ic_file,scorpio::FileMode::Read);

    // ... read dims ...
    num_vertical_levels  = scorpio::get_dimlen_c2f(ic_file.c_str(),"lev");
    if (scorpio::has_dim_c2f(ic_file.c_str(),"ncol")) {
      num_global_cols = scorpio::get_dimlen_c2f(ic_file.c_str(),"ncol");
    }
    if (scorpio::has_dim_c2f(ic_file.c_str(),"elem")) {
      int num_global_elems = scorpio::get_dimlen_c2f(ic_file.c_str(),"elem");
      num_local_elems = num_global_elems / m_comm.size();
      auto rem = num_global_elems % m_comm.size();
      if (m_comm.rank()<rem) {
        ++num_local_elems;
      }
    }
    if (scorpio::has_dim_c2f(ic_file.c_str(),"gp")) {
      num_gp = scorpio::get_dimlen_c2f(ic_file.c_str(),"gp");
    }

    // ... close the IC fil
    scorpio::eam_pio_closefile(ic_file);
  } else {
    // For unit testing, passing dims via parameter list is easier
    num_vertical_levels = m_params.get<int>("number_of_vertical_levels",-1);
    num_global_cols     = m_params.get<int>("number_of_global_columns");
    num_local_elems     = m_params.get<int>("number_of_local_elements",-1);
    num_gp              = m_params.get<int>("number_of_gauss_points",-1);
  }

  const bool build_se = num_local_elems>0 && num_gp>=2; 
  const bool build_pt = num_global_cols>0;

  EKAT_REQUIRE_MSG (build_se || build_pt,
      "Error! At least one grid must be built by MeshFreeGridsManager.\n");

  if (build_se) {
    // Build a set of completely disconnected spectral elements.

    // Set up the degrees of freedom.
    SEGrid::dofs_list_type dofs("", num_local_elems*num_gp*num_gp);
    SEGrid::lid_to_idx_map_type dofs_map("", num_local_elems*num_gp*num_gp, 3);

    auto host_dofs = Kokkos::create_mirror_view(dofs);
    auto host_dofs_map = Kokkos::create_mirror_view(dofs_map);

    // Count unique local dofs. On all elems except the very last one (on rank N),
    // we have num_gp*(num_gp-1) unique dofs;
    int num_local_dofs = num_local_elems*num_gp*num_gp;
    int offset = num_local_dofs*m_comm.rank();

    for (int ie = 0; ie < num_local_elems; ++ie) {
      for (int igp = 0; igp < num_gp; ++igp) {
        for (int jgp = 0; jgp < num_gp; ++jgp) {
          int idof = ie*num_gp*num_gp + igp*num_gp + jgp;
          int gid = offset + idof;
          host_dofs(idof) = gid;
          host_dofs_map(idof, 0) = ie;
          host_dofs_map(idof, 1) = igp;
          host_dofs_map(idof, 2) = jgp;
        }
      }
    }

    // Move the data to the device and set the DOFs.
    Kokkos::deep_copy(dofs, host_dofs);
    Kokkos::deep_copy(dofs_map, host_dofs_map);

    // Create the grid, and set the dofs
    std::shared_ptr<SEGrid> se_grid;
    se_grid = std::make_shared<SEGrid>("SE Grid",num_local_elems,num_gp,num_vertical_levels,m_comm);
    se_grid->setSelfPointer(se_grid);

    se_grid->set_dofs(dofs);
    se_grid->set_lid_to_idx_map(dofs_map);

    add_grid(se_grid);
  }
  if (build_pt) {
    auto pt_grid = create_point_grid("Point Grid",num_global_cols,num_vertical_levels,m_comm);
    add_grid(pt_grid);
    this->alias_grid("Point Grid", "Physics");
  }

  // Now that grids are built, if we have an IC file, look for hybrid coordinates,
  // and set them in all the grids
  if (m_params.isParameter("Initial Conditions Filename")) {
    const auto& ic_file = m_params.get<std::string>("Initial Conditions Filename");

    using vos_t = std::vector<std::string>;
    using KT = KokkosTypes<DefaultDevice>;
    using view_1d_dev  = typename KT::template view_ND<Real,1>;
    using view_1d_host = typename view_1d_dev::HostMirror;
    using namespace ShortFieldTagsNames;

    vos_t field_names;
    std::map<std::string,view_1d_dev> dev_views;
    std::map<std::string,view_1d_host> host_views;
    std::map<std::string,FieldLayout> layouts;

    // Register the IC file...
    scorpio::register_file(ic_file,scorpio::FileMode::Read);

    // ... look for vars ...
    for (const std::string& n : {"hyam","hybm"}) {
      if (scorpio::has_var_c2f(ic_file.c_str(),n.c_str())) {
        field_names.push_back(n);
        dev_views[n] = view_1d_dev(n,num_vertical_levels);
        host_views[n] = Kokkos::create_mirror_view(dev_views.at(n));
        layouts.emplace(n,FieldLayout({LEV},{num_vertical_levels}));
      }
    }
    for (const std::string& n : {"hyai","hybi"}) {
      if (scorpio::has_var_c2f(ic_file.c_str(),n.c_str())) {
        field_names.push_back(n);
        dev_views[n] = view_1d_dev(n,num_vertical_levels+1);
        host_views[n] = Kokkos::create_mirror_view(dev_views.at(n));
        layouts.emplace(n,FieldLayout({ILEV},{num_vertical_levels+1}));
      }
    }

    // ... close IC file
    scorpio::eam_pio_closefile(ic_file);

    ekat::ParameterList vcoord_reader_pl;
    vcoord_reader_pl.set("Filename",ic_file);
    vcoord_reader_pl.set<vos_t>("Field Names",field_names);
    AtmosphereInput vcoord_reader(m_comm,vcoord_reader_pl);

    // Need a grid for the IO. Pick either one
    std::vector<std::shared_ptr<const AbstractGrid>> grids;
    if (build_pt) {
      grids.push_back(get_grid("Point Grid"));
    }
    if (build_se) {
      grids.push_back(get_grid("SE Grid"));
    }

    // Load hybrid coordinates from IC file
    vcoord_reader.init(grids[0],host_views,layouts);
    vcoord_reader.read_variables();
    vcoord_reader.finalize();

    for (const auto& n : field_names) {
      Kokkos::deep_copy(dev_views.at(n),host_views.at(n));
      for (auto g : grids) {
        auto g_nc = get_grid_nonconst(g->name());
        g_nc->set_geometry_data("n",dev_views.at(n));
      }
    }
  }
}

std::shared_ptr<GridsManager>
create_mesh_free_grids_manager (const ekat::Comm& comm, const int num_local_elems,
                                const int num_gp, const int num_vertical_levels,
                                const int num_global_cols)
{
  ekat::ParameterList gm_params;
  gm_params.set("number_of_global_columns",num_global_cols);
  gm_params.set("number_of_local_elements",num_local_elems);
  gm_params.set("number_of_gauss_points",num_gp);
  gm_params.set("number_of_vertical_levels",num_vertical_levels);
  auto gm = create_mesh_free_grids_manager(comm,gm_params);
  return gm;
}

} // namespace scream
