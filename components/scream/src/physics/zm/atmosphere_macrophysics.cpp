#include "ekat/scream_assert.hpp"
#include "physics/zm/scream_zm_interface.hpp"
#include "physics/zm/atmosphere_macrophysics.hpp"
#include <iostream>
namespace scream

{


const Int& lchnk = 0;
const Int& ncol = 0;
Real* t;
Real* qh;
Real* prec;
Real* jctop; 
Real* jcbot; 
Real* pblh;
Real *zm; 
Real* geos; 
Real* zi;
			
Real* qtnd; 
Real* heat; 
Real* pap; 
Real* paph; 
Real* dpp; 
const Real &delt = 0;

Real* mcon;
Real* cme;
Real* cape;
Real* tpert;
Real* dlf;
Real* plfx;
Real* zdu;
Real* rprd; 
Real* mu;
Real* md; 
Real* du;
Real* eu; 
Real* ed;
Real* dp; 
Real* dsubcld; 
Real* jt; 
Real* maxg; 
Real* ideep;
const Real& lengath = 0; 
Real* ql; 
Real* rliq; 
Real* landfrac; 
Real* hu_nm1;
Real* cnv_nm1; 
Real* tm1; 
Real* qm1; 
Real** t_star; 
Real** q_star;
Real *dcape; 
Real* q; 
Real** tend_s; 
Real** tend_q; 
Real** cld; 
Real* snow; 
Real* ntprprd; 
Real* ntsnprd; 
Real** flxprec; 
Real** flxsnow;
const Real& ztodt = 0; 
Real* pguall; 
Real* pgdall; 
Real* icwu; 
const Real& ncnst = 0; 
const Real& limcnv_in = 0;
const bool& no_deep_pbl_in = true;

Real*** fracis;


ZMMacrophysics::ZMMacrophysics (const Comm& comm,const ParameterList& /* params */)
  : m_zm_comm (comm)
{
}
void ZMMacrophysics::set_grids(const std::shared_ptr<const GridsManager> grids_manager)
{


  using namespace units;
  auto Q = kg/kg;
  Q.set_string("kg/kg");
  
  constexpr int NVL = 72;  /* TODO THIS NEEDS TO BE CHANGED TO A CONFIGURABLE */
  constexpr int QSZ =  35;  /* TODO THIS NEEDS TO BE CHANGED TO A CONFIGURABLE */
  auto grid = grids_manager->get_grid("Physics");
  const int num_dofs = grid->get_num_local_dofs();
  const int nc = num_dofs;


  auto VL = FieldTag::VerticalLevel;
  auto CO = FieldTag::Column;
  auto VR = FieldTag::Variable;
  
  FieldLayout scalar3d_layout { {CO,VL}, {nc,NVL} }; // Note that C++ and Fortran read array dimensions in reverse
  FieldLayout vector3d_layout { {CO,VR,VL}, {nc,QSZ,NVL} };
  FieldLayout q_forcing_layout  { {CO,VR,VL}, {nc,QSZ,NVL} };
  auto nondim = m/m;

//  m_required_fields.emplace("t",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("t",  vector3d_layout_mid, Q, grid->name()); // in/out??
// 
//
  m_required_fields.emplace("t",  vector3d_layout, Q, grid->name()); // in/out??
  m_computed_fields.emplace("t",  vector3d_layout, Q, grid->name()); // in/out??


//  m_required_fields.emplace("qh",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("prec",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("jctop",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("jcbot",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("pblh",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("zm",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("geos",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("zi",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("gtnd",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("heat",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("pap",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("paph",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("dpp",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("delt",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("mcon",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("cme",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("cape",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("tpert",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("dlf",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("pflx",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("zdu",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("rprd",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("mu",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("md",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("du",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("eu",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("ed",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("dp",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("dsubcld",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("jt",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("maxg",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("ideep",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("lengath",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("ql",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("rliq",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("landfrac",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("hu_nm1",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("cnv_nm1",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("tm1",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("qm1",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("t_star",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("q_star",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("dcape",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("q",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("tend_s",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("tend_q",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("cld",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("snow",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("ntprprd",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("ntsnprd",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("flxprec",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("flxsnow",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("ztodt",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("pguall",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("pgdall",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("icwu",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("ncnst",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  m_required_fields.emplace("fracis",  vector3d_layout_mid, Q, grid->name()); // in/out??
//  
}

// =========================================================================================
void ZMMacrophysics::initialize (const util::TimeStamp& t0)
{
//  std::vector<std::string> zm_inputs = {"t, qh, prec, jctop, jcbot, pblh, zm, geos, zi, qtnd"
//					"heat, pap, paph, dpp, delt, mcon, cme, cape, tpert"
//					"dlf, pflx, zdu, rprd, mu, md, du, eu, ed, dp, dsubcld"
//					"jt, maxg, ideep, lengath, ql, rliq, landfrac, hu_nm1"
//					"cnv_nm1, tm1, qm1, t_star, q_star, dcape, q, tend_s"
//					"tend_q, cld, snow, ntprprd, ntsnprd, flxprec, flxsnow"
//					"ztodt, pguall, pgdall, icwu, ncnst, fracis"};
  std::vector<std::string> zm_inputs = {"t"};
  auto q_dev = m_zm_fields_out.at("t").get_view();
  zm_init_f90 (limcnv_in, no_deep_pbl_in);
  auto q_host = Kokkos::create_mirror_view(q_dev);
  Kokkos::deep_copy(q_host,q_dev);
  auto q_ptr = q_host.data();

}



// =========================================================================================
void ZMMacrophysics::run (const Real dt)
{
   
   zm_main_f90(lchnk, ncol, t, qh, prec, jctop, jcbot, pblh, zm, geos, zi, qtnd, 
		heat, pap, paph, dpp, delt,
		mcon, cme, cape, tpert, dlf, plfx,
		zdu, rprd, mu, md, du, eu, ed, dp, dsubcld, jt, maxg, ideep,
		lengath, ql, rliq, landfrac, hu_nm1, cnv_nm1,
		tm1, qm1, t_star, q_star, dcape, q, tend_s, tend_q, cld,
		snow, ntprprd, ntsnprd, flxprec, flxsnow,
		ztodt, pguall, pgdall, icwu, ncnst, fracis); 
   std :: cout << "In ZMMAcrophysics::run\n"<< std::endl;
}
// =========================================================================================
void ZMMacrophysics::finalize()
{
  zm_finalize_f90 ();
}
// =========================================================================================
void ZMMacrophysics::register_fields (FieldRepository<Real, device_type>& field_repo) const {
     for (auto& fid : m_required_fields) {
     field_repo.register_field(fid);
   }
   for (auto& fid : m_computed_fields) {
     field_repo.register_field(fid);
   }
 }

void ZMMacrophysics::set_required_field_impl (const Field<const Real, device_type>& f) {
  // Store a copy of the field. We need this in order to do some tracking checks
  // at the beginning of the run call. Other than that, there would be really
  // no need to store a scream field here; we could simply set the view ptr
  // in the Homme's view, and be done with it.
  const auto& name = f.get_header().get_identifier().name();
  m_zm_fields_in.emplace(name,f);
  m_zm_host_views_in[name] = Kokkos::create_mirror_view(f.get_view());
  m_zm_raw_ptrs_in[name] = m_zm_host_views_in[name].data();
//
//  // Add myself as customer to the field
//  add_me_as_customer(f);
}

void ZMMacrophysics::set_computed_field_impl (const Field<      Real, device_type>& f) {
  // Store a copy of the field. We need this in order to do some tracking updates
  // at the end of the run call. Other than that, there would be really
  // no need to store a scream field here; we could simply set the view ptr
  // in the Homme's view, and be done with it.
  const auto& name = f.get_header().get_identifier().name();
  m_zm_fields_out.emplace(name,f);
  m_zm_host_views_out[name] = Kokkos::create_mirror_view(f.get_view());
  m_raw_ptrs_out[name] = m_zm_host_views_out[name].data();
//
//  // Add myself as provider for the field
  add_me_as_provider(f);
}
} // namespace scream