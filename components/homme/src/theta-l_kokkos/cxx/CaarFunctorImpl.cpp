#include "CaarFunctorImpl.hpp"

namespace Homme {

template <bool HYDROSTATIC>
void CaarFunctorImpl::blockOps1()
{
  auto buffers_dp_tens = viewAsReal(m_buffers.dp_tens);
  auto buffers_div_vdp = viewAsReal(m_buffers.div_vdp);
  auto buffers_exner = viewAsReal(m_buffers.exner);
  auto buffers_phi = viewAsReal(m_buffers.phi);
  auto buffers_pnh = viewAsReal(m_buffers.pnh);
  auto buffers_vdp = viewAsReal(m_buffers.vdp);

  const Real data_eta_ave_w = m_data.eta_ave_w;
  const int data_n0 = m_data.n0;

  auto derived_vn0 = viewAsReal(m_derived.m_vn0);

  const SphereGlobal sg(m_sphere_ops);

  auto state_dp3d = viewAsReal(m_state.m_dp3d);
  auto state_phinh_i = viewAsReal(m_state.m_phinh_i);
  auto state_v = viewAsReal(m_state.m_v);
  auto state_vtheta_dp= viewAsReal(m_state.m_vtheta_dp);

  Kokkos::parallel_for(
    "caar_compute blockOps1",
    SphereBlockOps::policy(m_num_elems, 2),
    KOKKOS_LAMBDA(const Team &team) {

      SphereBlockOps b(sg, team);
      if (b.skip()) return;

      if (!HYDROSTATIC) {

        const Real dphi = state_phinh_i(b.e,data_n0,b.x,b.y,b.z+1) - state_phinh_i(b.e,data_n0,b.x,b.y,b.z);
        const Real vtheta_dp = state_vtheta_dp(b.e,data_n0,b.x,b.y,b.z);
        if ((vtheta_dp < 0) || (dphi > 0)) abort();
        EquationOfState::compute_pnh_and_exner(vtheta_dp, dphi, buffers_pnh(b.e,b.x,b.y,b.z), buffers_exner(b.e,b.x,b.y,b.z));

        buffers_phi(b.e,b.x,b.y,b.z) = 0.5 * (state_phinh_i(b.e,data_n0,b.x,b.y,b.z) + state_phinh_i(b.e,data_n0,b.x,b.y,b.z+1));
      }

      const Real v0 = state_v(b.e,data_n0,0,b.x,b.y,b.z) * state_dp3d(b.e,data_n0,b.x,b.y,b.z);
      const Real v1 = state_v(b.e,data_n0,1,b.x,b.y,b.z) * state_dp3d(b.e,data_n0,b.x,b.y,b.z);
      buffers_vdp(b.e,0,b.x,b.y,b.z) = v0;
      buffers_vdp(b.e,1,b.x,b.y,b.z) = v1;
      derived_vn0(b.e,0,b.x,b.y,b.z) += data_eta_ave_w * v0;
      derived_vn0(b.e,1,b.x,b.y,b.z) += data_eta_ave_w * v1;

      SphereBlockScratch ttmp0(b);
      SphereBlockScratch ttmp1(b);
      b.divInit(ttmp0, ttmp1, v0, v1);

      b.barrier();

      const Real dvdp = b.div(ttmp0, ttmp1);
      buffers_div_vdp(b.e,b.x,b.y,b.z) = dvdp;
      buffers_dp_tens(b.e,b.x,b.y,b.z) = dvdp;
    });
}

void CaarFunctorImpl::scanOps1()
{
  auto buffers_div_vdp = viewAsReal(m_buffers.div_vdp);
  auto buffers_dp_i = viewAsReal(m_buffers.dp_i);
  auto buffers_w_tens = viewAsReal(m_buffers.w_tens);

  const int data_n0 = m_data.n0;
  const Real pi_i00 = m_hvcoord.ps0 * m_hvcoord.hybrid_ai0;
  auto state_dp3d = viewAsReal(m_state.m_dp3d);

  Kokkos::parallel_for(
    "caar_compute scanOps1",
    SphereScanOps::policy(m_num_elems),
    KOKKOS_LAMBDA(const Team &team) {
      const SphereScanOps s(team);
      s.scan(buffers_dp_i, state_dp3d, data_n0, pi_i00);
      s.scan(buffers_w_tens, buffers_div_vdp, 0);
    });
}

template <bool HYDROSTATIC>
void CaarFunctorImpl::blockOps2()
{
  auto buffers_dp_i = viewAsReal(m_buffers.dp_i);
  auto buffers_exner = viewAsReal(m_buffers.exner);
  auto buffers_omega_p = viewAsReal(m_buffers.omega_p);
  auto buffers_pnh = viewAsReal(m_buffers.pnh);
  auto buffers_w_tens = viewAsReal(m_buffers.w_tens);

  const Real data_eta_ave_w = m_data.eta_ave_w;
  const int data_n0 = m_data.n0;

  auto derived_omega_p = viewAsReal(m_derived.m_omega_p);

  const SphereGlobal sg(m_sphere_ops);

  auto state_v = viewAsReal(m_state.m_v);
  auto state_vtheta_dp= viewAsReal(m_state.m_vtheta_dp);

  Kokkos::parallel_for(
    "caar_compute blockOps2",
    SphereBlockOps::policy(m_num_elems, 1),
    KOKKOS_LAMBDA(const Team &team) {

      SphereBlockOps b(sg, team);
      if (b.skip()) return;

      const Real pi = 0.5 * (buffers_dp_i(b.e,b.x,b.y,b.z) + buffers_dp_i(b.e,b.x,b.y,b.z+1));
      const SphereBlockScratch tmp0(b, pi);

      if (HYDROSTATIC) {
        Real exner = pi;
        EquationOfState::pressure_to_exner(exner);
        buffers_exner(b.e,b.x,b.y,b.z) = exner;
        buffers_pnh(b.e,b.x,b.y,b.z) = EquationOfState::compute_dphi(state_vtheta_dp(b.e,data_n0,b.x,b.y,b.z), exner, pi);
      }

      b.barrier();

      Real grad0, grad1;
      b.grad(grad0, grad1, tmp0);

      Real omega = -0.5 * (buffers_w_tens(b.e,b.x,b.y,b.z) + buffers_w_tens(b.e,b.x,b.y,b.z+1));
      omega += state_v(b.e,data_n0,0,b.x,b.y,b.z) * grad0 + state_v(b.e,data_n0,1,b.x,b.y,b.z) * grad1;
      buffers_omega_p(b.e,b.x,b.y,b.z) = omega;

      derived_omega_p(b.e,b.x,b.y,b.z) += data_eta_ave_w * omega;
    });
}

void CaarFunctorImpl::scanOps2()
{
  auto buffers_phi = viewAsReal(m_buffers.phi);
  auto buffers_pnh = viewAsReal(m_buffers.pnh);

  const int data_n0 = m_data.n0;

  auto &geometry_phis = m_geometry.m_phis;

  auto state_phinh_i = viewAsReal(m_state.m_phinh_i);

  Kokkos::parallel_for(
    "caar_compute scanOps2",
    SphereScanOps::policy(m_num_elems),
    KOKKOS_LAMBDA(const Team &team) {
      const SphereScanOps s(team);
      s.nacs(state_phinh_i, data_n0, buffers_pnh, geometry_phis);
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(s.t, NUM_PHYSICAL_LEV),
        [&](const int z) {
          buffers_phi(s.e,s.x,s.y,z) = 0.5 * (state_phinh_i(s.e,data_n0,s.x,s.y,z) + state_phinh_i(s.e,data_n0,s.x,s.y,z+1));
        });
    });
}

template <bool HYDROSTATIC>
void CaarFunctorImpl::colN()
{
  auto buffers_dp_tens = viewAsReal(m_buffers.dp_tens);
  auto buffers_phi_tens = viewAsReal(m_buffers.phi_tens);
  auto buffers_theta_tens = viewAsReal(m_buffers.theta_tens);
  auto buffers_v_tens = viewAsReal(m_buffers.v_tens);
  auto buffers_w_tens = viewAsReal(m_buffers.w_tens);

  const Real data_dt = m_data.dt;
  const int data_nm1 = m_data.nm1;
  const int data_np1 = m_data.np1;
  const Real data_scale3 = m_data.scale3;

  auto &geometry_spheremp = m_geometry.m_spheremp;

  const Real scale1_dt = m_data.scale1 * m_data.dt;

  auto state_dp3d = viewAsReal(m_state.m_dp3d);
  auto state_phinh_i = viewAsReal(m_state.m_phinh_i);
  auto state_v = viewAsReal(m_state.m_v);
  auto state_vtheta_dp = viewAsReal(m_state.m_vtheta_dp);
  auto state_w_i = viewAsReal(m_state.m_w_i);

  Kokkos::parallel_for(
    "caar_compute colN",
    SphereCol::policy(m_num_elems, NUM_PHYSICAL_LEV),
    KOKKOS_LAMBDA(const Team &team) {

      const SphereCol c(team);

      const Real spheremp = geometry_spheremp(c.e,c.x,c.y);
      const Real scale1_dt_spheremp = scale1_dt * spheremp;
      const Real scale3_spheremp = data_scale3 * spheremp;

      Real dp_tens = buffers_dp_tens(c.e,c.x,c.y,c.z);
      dp_tens *= scale1_dt_spheremp;
      Real dp_np1 = scale3_spheremp * state_dp3d(c.e,data_nm1,c.x,c.y,c.z);
      dp_np1 -= dp_tens;
      state_dp3d(c.e,data_np1,c.x,c.y,c.z) = dp_np1;

      Real theta_tens = buffers_theta_tens(c.e,c.x,c.y,c.z);
      theta_tens *= -scale1_dt_spheremp;
      Real vtheta_np1 = state_vtheta_dp(c.e,data_nm1,c.x,c.y,c.z);
      vtheta_np1 *= scale3_spheremp;
      vtheta_np1 += theta_tens;
      state_vtheta_dp(c.e,data_np1,c.x,c.y,c.z) = vtheta_np1;

      Real u_tens = buffers_v_tens(c.e,0,c.x,c.y,c.z);
      u_tens *= -scale1_dt_spheremp;
      Real u_np1 = state_v(c.e,data_nm1,0,c.x,c.y,c.z);
      u_np1 *= scale3_spheremp;
      u_np1 += u_tens;
      state_v(c.e,data_np1,0,c.x,c.y,c.z) = u_np1;

      Real v_tens = buffers_v_tens(c.e,1,c.x,c.y,c.z);
      v_tens *= -scale1_dt_spheremp;
      Real v_np1 = state_v(c.e,data_nm1,1,c.x,c.y,c.z);
      v_np1 *= scale3_spheremp;
      v_np1 += v_tens;
      state_v(c.e,data_np1,1,c.x,c.y,c.z) = v_np1;

      if (!HYDROSTATIC) {

        const Real dt_spheremp = data_dt * spheremp;

        Real phi_tens = buffers_phi_tens(c.e,c.x,c.y,c.z);
        phi_tens *= dt_spheremp;
        Real phi_np1 = state_phinh_i(c.e,data_nm1,c.x,c.y,c.z);
        phi_np1 *= scale3_spheremp;
        phi_np1 += phi_tens;
        state_phinh_i(c.e,data_np1,c.x,c.y,c.z) = phi_np1;

        Real w_tens = buffers_w_tens(c.e,c.x,c.y,c.z);
        w_tens *= dt_spheremp;
        Real w_np1 = state_w_i(c.e,data_nm1,c.x,c.y,c.z);
        w_np1 *= scale3_spheremp;
        w_np1 += w_tens;
        state_w_i(c.e,data_np1,c.x,c.y,c.z) = w_np1;

        if (c.z == NUM_PHYSICAL_LEV-1) {
          Real w_tens = buffers_w_tens(c.e,c.x,c.y,NUM_PHYSICAL_LEV);
          w_tens *= dt_spheremp;
          buffers_w_tens(c.e,c.x,c.y,NUM_PHYSICAL_LEV) = w_tens;
          Real w_np1 = state_w_i(c.e,data_nm1,c.x,c.y,NUM_PHYSICAL_LEV);
          w_np1 *= scale3_spheremp;
          w_np1 += w_tens;
          state_w_i(c.e,data_np1,c.x,c.y,NUM_PHYSICAL_LEV) = w_np1;
        }
      }
    });
}

void CaarFunctorImpl::caar_compute() 
{

  if (m_theta_hydrostatic_mode) blockOps1<true>();
  else blockOps1<false>();

  scanOps1();

  if (m_theta_hydrostatic_mode) blockOps2<true>();
  else blockOps2<false>();

  if (m_theta_hydrostatic_mode) scanOps2();

  const SphereGlobal sg(m_sphere_ops);

  auto buffers_dp_tens = viewAsReal(m_buffers.dp_tens);
  auto buffers_div_vdp = viewAsReal(m_buffers.div_vdp);
  auto buffers_vdp = viewAsReal(m_buffers.vdp);

  const Real data_eta_ave_w = m_data.eta_ave_w;
  const int data_n0 = m_data.n0;

  auto state_dp3d = viewAsReal(m_state.m_dp3d);
  auto state_v = viewAsReal(m_state.m_v);

  auto buffers_dp_i = viewAsReal(m_buffers.dp_i);
  auto buffers_omega_i = viewAsReal(m_buffers.w_tens);

  const Real pi_i00 = m_hvcoord.ps0 * m_hvcoord.hybrid_ai0;

  auto buffers_exner = viewAsReal(m_buffers.exner);
  auto buffers_omega_p = viewAsReal(m_buffers.omega_p);
  auto buffers_phi = viewAsReal(m_buffers.phi);
  auto buffers_pnh = viewAsReal(m_buffers.pnh);

  auto derived_omega_p = viewAsReal(m_derived.m_omega_p);

  auto state_phinh_i = viewAsReal(m_state.m_phinh_i);
  auto state_vtheta_dp = viewAsReal(m_state.m_vtheta_dp);

  const bool theta_hydrostatic_mode = m_theta_hydrostatic_mode;

  auto buffers_dpnh_dp_i = viewAsReal(m_buffers.dpnh_dp_i);
  auto buffers_v_i = viewAsReal(m_buffers.v_i);
  auto buffers_vtheta_i = viewAsReal(m_buffers.vtheta_i);

  const int rsplit = m_rsplit;

  if ((rsplit == 0) || !theta_hydrostatic_mode) {

    Kokkos::parallel_for(
      "caar compute_interface_quantities",
      SphereColOps::policy(m_num_elems, NUM_INTERFACE_LEV),
      KOKKOS_LAMBDA(const Team &team) {

        const SphereColOps c(sg, team);

        const Real dm = (c.z == 0) ? 0 : state_dp3d(c.e,data_n0,c.x,c.y,c.z-1);
        const Real dz = (c.z == NUM_PHYSICAL_LEV) ? 0 : state_dp3d(c.e,data_n0,c.x,c.y,c.z);
        const Real dp_i = (c.z == 0) ? dz : (c.z == NUM_PHYSICAL_LEV) ? dm : 0.5 * (dz + dm);
        buffers_dp_i(c.e,c.x,c.y,c.z) = dp_i;

        if (!theta_hydrostatic_mode) {

          const Real v0m = (c.z == 0) ? 0 : state_v(c.e,data_n0,0,c.x,c.y,c.z-1);
          const Real v0z = (c.z == NUM_PHYSICAL_LEV) ? 0 : state_v(c.e,data_n0,0,c.x,c.y,c.z);
          const Real v_i0 = (c.z == 0) ? v0z : (c.z == NUM_PHYSICAL_LEV) ? v0m : (dz * v0z + dm * v0m) / (dm + dz);
          buffers_v_i(c.e,0,c.x,c.y,c.z) = v_i0;

          const Real v1m = (c.z == 0) ? 0 : state_v(c.e,data_n0,1,c.x,c.y,c.z-1);
          const Real v1z = (c.z == NUM_PHYSICAL_LEV) ? 0 : state_v(c.e,data_n0,1,c.x,c.y,c.z);
          const Real v_i1 = (c.z == 0) ? v1z : (c.z == NUM_PHYSICAL_LEV) ? v1m :(dz * v1z + dm * v1m) / (dm + dz);
          buffers_v_i(c.e,1,c.x,c.y,c.z) = v_i1;

          const Real pm = (c.z == 0) ? pi_i00 : buffers_pnh(c.e,c.x,c.y,c.z-1);
          const Real pz = (c.z == NUM_PHYSICAL_LEV) ? pm + 0.5 * dm : buffers_pnh(c.e,c.x,c.y,c.z);
          buffers_dpnh_dp_i(c.e,c.x,c.y,c.z) = 2.0 * (pz - pm) / (dm + dz);
        }

        if (rsplit == 0) {

          Real vtheta_i = 0;
          if ((c.z > 0) && (c.z < NUM_PHYSICAL_LEV)) {
            const Real dphi = buffers_phi(c.e,c.x,c.y,c.z) - buffers_phi(c.e,c.x,c.y,c.z-1);
            const Real dexner = buffers_exner(c.e,c.x,c.y,c.z) - buffers_exner(c.e,c.x,c.y,c.z-1);
            vtheta_i = dphi / dexner;
          }
          vtheta_i /= -PhysicalConstants::cp;
          if (!theta_hydrostatic_mode) vtheta_i *= buffers_dpnh_dp_i(c.e,c.x,c.y,c.z);
          buffers_vtheta_i(c.e,c.x,c.y,c.z) = vtheta_i;
        }
      });
  }

  auto buffers_theta_tens = viewAsReal(m_buffers.theta_tens);
  auto buffers_v_tens = viewAsReal(m_buffers.v_tens);
  auto &buffers_w_tens = buffers_omega_i; // reused
  auto state_w_i = viewAsReal(m_state.m_w_i);

  if (rsplit == 0) {

    auto buffers_eta_dot_dpdn = viewAsReal(m_buffers.eta_dot_dpdn);
    auto &hvcoord_hybrid_bi = m_hvcoord.hybrid_bi;

    Kokkos::parallel_for(
      "caar compute_eta_dot_dpn",
      SphereScanOps::policy(m_num_elems),
      KOKKOS_LAMBDA(const Team &team) {

        const SphereScanOps s(team);
        s.scan(buffers_eta_dot_dpdn, buffers_div_vdp, 0);

        const Real last = buffers_eta_dot_dpdn(s.e,s.x,s.y,NUM_PHYSICAL_LEV);

        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(s.t, 1, NUM_PHYSICAL_LEV),
          [&](const int z) {
            Real eta_dot_dpdn = -buffers_eta_dot_dpdn(s.e,s.x,s.y,z);
            eta_dot_dpdn += hvcoord_hybrid_bi(z) * last;
            buffers_eta_dot_dpdn(s.e,s.x,s.y,z) = eta_dot_dpdn;
          });

        Kokkos::single(
          Kokkos::PerThread(team),
          [&]() {
            buffers_eta_dot_dpdn(s.e,s.x,s.y,0) = buffers_eta_dot_dpdn(s.e,s.x,s.y,NUM_PHYSICAL_LEV) = 0;
          });
      });

    auto derived_eta_dot_dpdn = viewAsReal(m_derived.m_eta_dot_dpdn);

    Kokkos::parallel_for(
      "caar compute_v_vadv compute_vtheta_vadv",
      SphereCol::policy(m_num_elems, NUM_PHYSICAL_LEV),
      KOKKOS_LAMBDA(const Team &team) {
        const SphereCol c(team);

        // compute_v_vadv

        const Real uz = state_v(c.e,data_n0,0,c.x,c.y,c.z);
        const Real vz = state_v(c.e,data_n0,1,c.x,c.y,c.z);
        const Real dp = state_dp3d(c.e,data_n0,c.x,c.y,c.z);

        const Real etap = buffers_eta_dot_dpdn(c.e,c.x,c.y,c.z+1);
        const Real etaz = buffers_eta_dot_dpdn(c.e,c.x,c.y,c.z);

        // compute_dp_and_theta_tens
        buffers_dp_tens(c.e,c.x,c.y,c.z) += etap - etaz;

        // compute_accumulated_quantities
        derived_eta_dot_dpdn(c.e,c.x,c.y,c.z) += data_eta_ave_w * etaz;

        Real u = 0;
        Real v = 0;
        if (c.z < NUM_PHYSICAL_LEV-1) {
          const Real facp = 0.5 * etap / dp;
          u = facp * (state_v(c.e,data_n0,0,c.x,c.y,c.z+1) - uz);
          v = facp * (state_v(c.e,data_n0,1,c.x,c.y,c.z+1) - vz);
        }
        if (c.z > 0) {
          const Real facm = 0.5 * etaz / dp;
          u += facm * (uz - state_v(c.e,data_n0,0,c.x,c.y,c.z-1));
          v += facm * (vz - state_v(c.e,data_n0,1,c.x,c.y,c.z-1));
        }
        buffers_v_tens(c.e,0,c.x,c.y,c.z) = u;
        buffers_v_tens(c.e,1,c.x,c.y,c.z) = v;

        // compute_vtheta_vadv

        const Real thetap = etap * buffers_vtheta_i(c.e,c.x,c.y,c.z+1);
        const Real thetaz = etaz * buffers_vtheta_i(c.e,c.x,c.y,c.z);
        buffers_theta_tens(c.e,c.x,c.y,c.z) = thetap - thetaz;
      });


    if (!theta_hydrostatic_mode) {

      auto buffers_temp = viewAsReal(m_buffers.temp);

      Kokkos::parallel_for(
        "caar compute_w_vadv num_physical_lev",
        SphereCol::policy(m_num_elems, NUM_PHYSICAL_LEV),
        KOKKOS_LAMBDA(const Team &team) {
          const SphereCol c(team);

          const Real dw = state_w_i(c.e,data_n0,c.x,c.y,c.z+1) - state_w_i(c.e,data_n0,c.x,c.y,c.z);
          const Real eta = 0.5 * (buffers_eta_dot_dpdn(c.e,c.x,c.y,c.z) + buffers_eta_dot_dpdn(c.e,c.x,c.y,c.z+1));
          buffers_temp(c.e,c.x,c.y,c.z) = dw * eta;
        });

      auto buffers_phi_tens = viewAsReal(m_buffers.phi_tens);

      Kokkos::parallel_for(
        "caar compute_w_vadv num_interface_lev compute_phi_vadv",
        SphereCol::policy(m_num_elems, NUM_INTERFACE_LEV),
        KOKKOS_LAMBDA(const Team &team) {
          const SphereCol c(team);

          // compute_w_vadv

          const Real tempm = (c.z == 0) ? 0 : buffers_temp(c.e,c.x,c.y,c.z-1);
          const Real tempz = (c.z == NUM_PHYSICAL_LEV) ? 0 : buffers_temp(c.e,c.x,c.y,c.z);
          const Real dw = (c.z == 0) ? tempz : (c.z == NUM_PHYSICAL_LEV) ? tempm : 0.5 * (tempz + tempm);
          buffers_w_tens(c.e,c.x,c.y,c.z) = dw / buffers_dp_i(c.e,c.x,c.y,c.z);

          // compute_phi_vadv

          const Real phim = (c.z == 0) ? 0 : buffers_phi(c.e,c.x,c.y,c.z-1);
          const Real phiz = (c.z == NUM_PHYSICAL_LEV) ? 0 : buffers_phi(c.e,c.x,c.y,c.z);
          const Real phi_vadv = ((c.z == 0) || (c.z == NUM_PHYSICAL_LEV)) ? 0 : (phiz - phim) * buffers_eta_dot_dpdn(c.e,c.x,c.y,c.z) / buffers_dp_i(c.e,c.x,c.y,c.z);
          buffers_phi_tens(c.e,c.x,c.y,c.z) = phi_vadv;
        });
    }
  }

  auto buffers_grad_phinh_i = viewAsReal(m_buffers.grad_phinh_i);
  auto buffers_grad_w_i = viewAsReal(m_buffers.grad_w_i);
  auto buffers_phi_tens = viewAsReal(m_buffers.phi_tens);

  if (theta_hydrostatic_mode) {

    Kokkos::parallel_for(
      "caar compute_w_and_phi_tens hydrostatic",
      SphereColOps::policy(m_num_elems, NUM_INTERFACE_LEV),
      KOKKOS_LAMBDA(const Team &team) {
        const SphereColOps c(sg, team);
        c.grad(buffers_grad_phinh_i, state_phinh_i, data_n0);
      });

  } else {

    auto &geometry_gradphis = m_geometry.m_gradphis;

    auto hvcoord_hybrid_bi_packed = viewAsReal(m_hvcoord.hybrid_bi_packed);

    const Real dscale = m_data.scale1 - m_data.scale2;
    const Real gscale1 = m_data.scale1 * PhysicalConstants::g;
    const Real gscale2 = m_data.scale2 * PhysicalConstants::g;
    const Real ndata_scale1 = -m_data.scale1;

    Kokkos::parallel_for(
      "caar compute_w_and_phi_tens nonhydrostatic",
      SphereColOps::policy(m_num_elems, NUM_INTERFACE_LEV),
      KOKKOS_LAMBDA(const Team &team) {

        const SphereColOps c(sg, team);

        c.grad(buffers_grad_phinh_i, state_phinh_i, data_n0);
        c.grad(buffers_grad_w_i, state_w_i, data_n0);
        const Real gscale = (c.z == NUM_PHYSICAL_LEV) ? gscale1 : gscale2;

        Real w_tens = (rsplit) ? 0 : buffers_w_tens(c.e,c.x,c.y,c.z);
        w_tens += buffers_v_i(c.e,0,c.x,c.y,c.z) * buffers_grad_w_i(c.e,0,c.x,c.y,c.z) + buffers_v_i(c.e,1,c.x,c.y,c.z) * buffers_grad_w_i(c.e,1,c.x,c.y,c.z);
        w_tens *= ndata_scale1;
        w_tens += (buffers_dpnh_dp_i(c.e,c.x,c.y,c.z)-Real(1)) * gscale;
        buffers_w_tens(c.e,c.x,c.y,c.z) = w_tens;

        Real phi_tens = (rsplit) ? 0 : buffers_phi_tens(c.e,c.x,c.y,c.z);
        phi_tens += buffers_v_i(c.e,0,c.x,c.y,c.z) * buffers_grad_phinh_i(c.e,0,c.x,c.y,c.z) + buffers_v_i(c.e,1,c.x,c.y,c.z) * buffers_grad_phinh_i(c.e,1,c.x,c.y,c.z);
        phi_tens *= ndata_scale1;
        phi_tens += state_w_i(c.e,data_n0,c.x,c.y,c.z) * gscale;

        if (dscale) phi_tens += dscale * (buffers_v_i(c.e,0,c.x,c.y,c.z) * geometry_gradphis(c.e,0,c.x,c.y) + buffers_v_i(c.e,1,c.x,c.y,c.z) * geometry_gradphis(c.e,1,c.x,c.y)) * hvcoord_hybrid_bi_packed(c.z);

        buffers_phi_tens(c.e,c.x,c.y,c.z) = phi_tens;
      });
  }

  if (m_theta_advection_form == AdvectionForm::Conservative) {

    Kokkos::parallel_for(
      "caar compute_dp_and_theta_tens conservative",
      SphereBlockOps::policy(m_num_elems, 2),
      KOKKOS_LAMBDA(const Team &team) {

        SphereBlockOps b(sg, team); 
        if (b.skip()) return;

        const Real v0 = state_v(b.e,data_n0,0,b.x,b.y,b.z) * state_vtheta_dp(b.e,data_n0,b.x,b.y,b.z);
        const Real v1 = state_v(b.e,data_n0,1,b.x,b.y,b.z) * state_vtheta_dp(b.e,data_n0,b.x,b.y,b.z);

        SphereBlockScratch ttmp0(b);
        SphereBlockScratch ttmp1(b);
        b.divInit(ttmp0, ttmp1, v0, v1);

        b.barrier();

        const Real div = b.div(ttmp0, ttmp1);
        Real theta_tens = (rsplit) ? 0 : buffers_theta_tens(b.e,b.x,b.y,b.z);
        theta_tens += div;
        buffers_theta_tens(b.e,b.x,b.y,b.z) = theta_tens;

      });

  } else { // AdvectionForm::NonConservative

    Kokkos::parallel_for(
      "caar compute_dp_and_theta_tens nonconservative",
      SphereBlockOps::policy(m_num_elems, 1),
      KOKKOS_LAMBDA(const Team &team) {

        SphereBlockOps b(sg, team); 
        if (b.skip()) return;

        const Real vtheta = state_vtheta_dp(b.e,data_n0,b.x,b.y,b.z) / state_dp3d(b.e,data_n0,b.x,b.y,b.z);
        SphereBlockScratch ttmp0(b, vtheta);

        b.barrier();

        Real grad0, grad1;
        b.grad(grad0, grad1, ttmp0);
        Real theta_tens = buffers_div_vdp(b.e,b.x,b.y,b.z) * vtheta;
        theta_tens += grad0 * buffers_vdp(b.e,0,b.x,b.y,b.z);
        theta_tens += grad1 * buffers_vdp(b.e,1,b.x,b.y,b.z);

        if (rsplit) buffers_theta_tens(b.e,b.x,b.y,b.z) = theta_tens;
        else buffers_theta_tens(b.e,b.x,b.y,b.z) += theta_tens;
      });
  }

  auto &geometry_fcor = m_geometry.m_fcor;
  const bool pgrad_correction = m_pgrad_correction;

  Kokkos::parallel_for(
    "caar compute_v_tens",
    SphereBlockOps::policy(m_num_elems, 6),
    KOKKOS_LAMBDA(const Team &team) {

      SphereBlockOps b(sg, team);
      if (b.skip()) return;

      const Real w2 = (theta_hydrostatic_mode) ? 0 : 0.25 * (state_w_i(b.e,data_n0,b.x,b.y,b.z) * state_w_i(b.e,data_n0,b.x,b.y,b.z) + state_w_i(b.e,data_n0,b.x,b.y,b.z+1) * state_w_i(b.e,data_n0,b.x,b.y,b.z+1));
      const SphereBlockScratch ttmp0(b, w2);

      const Real exneriz = buffers_exner(b.e,b.x,b.y,b.z);
      const SphereBlockScratch ttmp1(b, exneriz);

      const Real log_exneriz = (pgrad_correction) ? log(exneriz) : 0;
      const SphereBlockScratch ttmp2(b, log_exneriz);

      const Real v0 = state_v(b.e,data_n0,0,b.x,b.y,b.z);
      const Real v1 = state_v(b.e,data_n0,1,b.x,b.y,b.z);

      SphereBlockScratch ttmp3(b);
      SphereBlockScratch ttmp4(b);
      b.vortInit(ttmp3, ttmp4, v0, v1);

      const SphereBlockScratch ttmp5(b, 0.5 * (v0 * v0 + v1 * v1));

      b.barrier();

      Real grad_v0, grad_v1;
      b.grad(grad_v0, grad_v1, ttmp5);

      Real u_tens = (rsplit) ? 0 : buffers_v_tens(b.e,0,b.x,b.y,b.z);
      Real v_tens = (rsplit) ? 0 : buffers_v_tens(b.e,1,b.x,b.y,b.z);
      u_tens += grad_v0;
      v_tens += grad_v1;

      const Real cp_vtheta = PhysicalConstants::cp * (state_vtheta_dp(b.e,data_n0,b.x,b.y,b.z) / state_dp3d(b.e,data_n0,b.x,b.y,b.z));

      Real grad_exner0, grad_exner1;
      b.grad(grad_exner0, grad_exner1, ttmp1);

      u_tens += cp_vtheta * grad_exner0;
      v_tens += cp_vtheta * grad_exner1;

      Real mgrad_x, mgrad_y;
      if (theta_hydrostatic_mode) {

        mgrad_x = 0.5 * (buffers_grad_phinh_i(b.e,0,b.x,b.y,b.z) + buffers_grad_phinh_i(b.e,0,b.x,b.y,b.z+1));
        mgrad_y = 0.5 * (buffers_grad_phinh_i(b.e,1,b.x,b.y,b.z) + buffers_grad_phinh_i(b.e,1,b.x,b.y,b.z+1));

      } else {

        mgrad_x = 0.5 * (buffers_grad_phinh_i(b.e,0,b.x,b.y,b.z) * buffers_dpnh_dp_i(b.e,b.x,b.y,b.z) + buffers_grad_phinh_i(b.e,0,b.x,b.y,b.z+1) * buffers_dpnh_dp_i(b.e,b.x,b.y,b.z+1));
        mgrad_y = 0.5 * (buffers_grad_phinh_i(b.e,1,b.x,b.y,b.z) * buffers_dpnh_dp_i(b.e,b.x,b.y,b.z) + buffers_grad_phinh_i(b.e,1,b.x,b.y,b.z+1) * buffers_dpnh_dp_i(b.e,b.x,b.y,b.z+1));

      }

      if (pgrad_correction) {

        Real grad_lexner0, grad_lexner1;
        b.grad(grad_lexner0, grad_lexner1, ttmp2);

        namespace PC = PhysicalConstants;
        constexpr Real cpt0 = PC::cp * (PC::Tref - PC::Tref_lapse_rate * PC::Tref * PC::cp / PC::g);
        mgrad_x += cpt0 * (grad_lexner0 - grad_exner0 / exneriz);
        mgrad_y += cpt0 * (grad_lexner1 - grad_exner1 / exneriz);
      }

      Real wvor_x = 0;
      Real wvor_y = 0;
      if (!theta_hydrostatic_mode) {
        b.grad(wvor_x, wvor_y, ttmp0);
        wvor_x -= 0.5 * (buffers_grad_w_i(b.e,0,b.x,b.y,b.z) * state_w_i(b.e,data_n0,b.x,b.y,b.z) + buffers_grad_w_i(b.e,0,b.x,b.y,b.z+1) * state_w_i(b.e,data_n0,b.x,b.y,b.z+1));
        wvor_y -= 0.5 * (buffers_grad_w_i(b.e,1,b.x,b.y,b.z) * state_w_i(b.e,data_n0,b.x,b.y,b.z) + buffers_grad_w_i(b.e,1,b.x,b.y,b.z+1) * state_w_i(b.e,data_n0,b.x,b.y,b.z+1));
      }

      u_tens += mgrad_x + wvor_x;
      v_tens += mgrad_y + wvor_y;

      const Real vort = b.vort(ttmp3, ttmp4) + geometry_fcor(b.e,b.x,b.y);
      u_tens -= v1 * vort;
      v_tens += v0 * vort;

      buffers_v_tens(b.e,0,b.x,b.y,b.z) = u_tens;
      buffers_v_tens(b.e,1,b.x,b.y,b.z) = v_tens;
    });

  if (m_theta_hydrostatic_mode) colN<true>();
  else colN<false>();
}

}
