#ifndef SCREAM_COLUMN_OPS_HPP
#define SCREAM_COLUMN_OPS_HPP

#include <type_traits>
#include "ekat/ekat_pack.hpp"
#include "ekat/ekat_pack_math.hpp"
#include "share/util/scream_combine_ops.hpp"
#include "share/scream_types.hpp"

#include "ekat/kokkos/ekat_kokkos_types.hpp"
#include "ekat/ekat_pack_utils.hpp"
#include "ekat/util//ekat_arch.hpp"
#include "ekat/ekat_pack.hpp"


namespace scream {

/*
 *  ColumnOps: a series of utility kernels that operate on a single column
 *
 *  This class is responsible of implementing common kernels used in
 *  scream to compute quantities at level midpoints and level interfaces.
 *  For instance, compute interface quantities from midpoints
 *  ones, or integrate over a column, or compute increments of interface
 *  quantities (which will be defined at midpoints).
 *  The kernels are meant to be launched from within a parallel region, with
 *  team policy. More precisely, they are meant to be called from the outer most
 *  parallel region. In other words, you should *not* be inside a TeamThreadRange
 *  parallel loop when calling these kernels, since these kernels will attempt
 *  to create such loops. Furthermore, these kernels *assume* that the team policy
 *  vector length (on CUDA) is 1. We have no way of checking this (the vector length
 *  is stored in the policy, but not in the team member), so you must make
 *  sure that this is the case.
 *
 *  In the compute_* methods, InputProvider can either be a functor (e.g., a lambda)
 *  or a 1d view. The only requirement is that operator()(int) is defined,
 *  and returns a Pack<ScalarType,PackSize>.
 *  For instance, one could use a lambda to compute the midpoint average of
 *  the product of two interface quantities, like this:
 *
 *    using col_ops = ColumnOps<DefaultDevice,Real,N>;
 *    using pack_type = typename col_ops::pack_type;
 *
 *    auto prod = [&](const int k)->pack_type { return x(k)*y(k); }
 *    col_ops::compute_midpoint_values(team,nlevs,prod,output);
 *
 *  Note: all methods have a different impl, depending on whether PackSize>1.
 *        The if statement is evaluated at compile-time, so there is no run-time
 *        penalization. The only requirement is that both branches must compile.
 *
 *  RECALL: k=0 is the model top, while k=num_mid_levels+1 is the surface!
 */

template<typename DeviceType, typename ScalarType, int PackSize>
class ColumnOps {
public:
  // Expose template params
  using device_type = DeviceType;
  using scalar_type = ScalarType;
  using pack_type   = ekat::Pack<scalar_type,PackSize>;

  // Pack info
  enum : int {
    pack_size = PackSize,
  };
  using pack_info = ekat::PackInfo<pack_size>;

  // Kokkos types
  using exec_space = typename device_type::execution_space;
  using KT = ekat::KokkosTypes<device_type>;

  using TeamMember = typename KT::MemberType;

  template<typename T>
  using view_1d = typename KT::template view_1d<T>;

  KOKKOS_INLINE_FUNCTION
  static constexpr scalar_type one  () { return scalar_type(1); }
  KOKKOS_INLINE_FUNCTION
  static constexpr scalar_type zero () { return scalar_type(0); }

  // All functions have an 'input' provider template parameter. This can
  // either be a lambda (to allow input calculation on the fly) or a 1d view.
  // By default, it is a view.
  using DefaultProvider = view_1d<const pack_type>;

  // Runs the input lambda with a TeamThreadRange parallel for over [0,count) range
  template<typename Lambda>
  KOKKOS_INLINE_FUNCTION
  static void team_parallel_for (const TeamMember& team,
                                 const int count,
                                 const Lambda& f)
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count),f);
  }

  // Runs the input lambda with a TeamThreadRange parallel for over [start,end) range
  template<typename Lambda>
  KOKKOS_INLINE_FUNCTION
  static void team_parallel_for (const TeamMember& team,
                                 const int start, const int end,
                                 const Lambda& f)
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start,end),f);
  }

  // Runs the input lambda with a TeamThreadRange parallel scan over [0,count) range
  template<typename Lambda>
  KOKKOS_INLINE_FUNCTION
  static void team_parallel_scan (const TeamMember& team,
                                  const int count,
                                  const Lambda& f)
  {
    auto is_pow_of_2 = [](const int n)->bool {
      // This seems funky, but write down a pow of 2 and a non-pow of 2 in binary (both positive),
      // and you'll see why it works
      return n>0 && (n & (n-1))==0;
    };
    EKAT_KERNEL_REQUIRE_MSG (!ekat::OnGpu<typename device_type::execution_space>::value ||
                             is_pow_of_2(team.team_size()),
      "Error! Team-level parallel_scan on CUDA only works for team sizes that are power of 2.\n"
      "       You could try to reduce the team size to the previous pow of 2.\n");
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team,count),f);
  }
  template<typename Lambda>
  KOKKOS_INLINE_FUNCTION
  static void team_parallel_scan (const TeamMember& team,
                                  const int start, const int end,
                                  const Lambda& f)
  {
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team,start,end),f);
  }

  // Runs the input lambda only for one team thread
  template<typename Lambda>
  KOKKOS_INLINE_FUNCTION
  static void team_single (const TeamMember& team,
                           const Lambda& f)
  {
    Kokkos::single(Kokkos::PerTeam(team),f);
  }

  template<typename InputProvider>
  KOKKOS_INLINE_FUNCTION
  static void debug_checks (const int num_levels, const view_1d<pack_type>& x) {

    // Mini function to check that InputProvider supports op()(int)->pack_type,
    // and that the number of levels is compatible with pack_type and x's size.

    const auto npacks_min = pack_info::num_packs(num_levels);
    EKAT_KERNEL_ASSERT_MSG (num_levels>=0 && x.extent_int(0)>=npacks_min,
        "Error! Number of levels out of bounds.\n");

    using ret_type = decltype(std::declval<InputProvider>()(0));
    using raw_ret_type = typename std::remove_const<typename std::remove_reference<ret_type>::type>::type;

    static_assert(std::is_same<raw_ret_type,pack_type>::value,
      "Error! InputProvider should expose op()(int), returning a pack type.\n");
  }

  // Safety checks
  static_assert(!ekat::OnGpu<exec_space>::value || pack_size==1,
                "Error! ColumnOps impl would be buggy on gpu if VECTOR_SIZE>1.\n");

  // Compute X at level midpoints, given X at level interfaces
  template<CombineMode CM = CombineMode::Replace,
           typename InputProvider = DefaultProvider>
  KOKKOS_INLINE_FUNCTION
  static void
  compute_midpoint_values (const TeamMember& team,
                           const int num_mid_levels,
                           const InputProvider& x_i,
                           const view_1d<pack_type>& x_m,
                           const scalar_type alpha = one(),
                           const scalar_type beta = zero())
  {
    // Sanity checks
    debug_checks<InputProvider>(num_mid_levels,x_m);

    // For GPU (or any build with pack size 1), things are simpler
    if (pack_size==1) {
      team_parallel_for(team,num_mid_levels,
                        [&](const int& k) {
        auto tmp = ( x_i(k) + x_i(k+1) ) / 2.0;
        combine<CM>(tmp, x_m(k), alpha, beta);
      });
    } else {

      const auto NUM_MID_PACKS     = pack_info::num_packs(num_mid_levels);
      const auto NUM_INT_PACKS     = pack_info::num_packs(num_mid_levels+1);

      // It is easier to read if we check whether #int_packs==#mid_packs.
      // This lambda can be used in both cases to process packs that have a next pack
      auto shift_and_avg = [&](const int k){
        // Shift's first arg is the scalar to put in the "empty" spot at the end
        pack_type tmp = ekat::shift_left(x_i(k+1)[0], x_i(k));
        tmp += x_i(k);
        tmp /= 2.0;
        combine<CM>(tmp, x_m(k), alpha, beta);
      };
      if (NUM_MID_PACKS==NUM_INT_PACKS) {
        const auto LAST_PACK =  NUM_MID_PACKS-1;

        // Use SIMD operations only on NUM_MID_PACKS-1, since mid pack
        // does not have a 'next' one
        team_parallel_for(team,NUM_MID_PACKS-1,shift_and_avg);

        team_single (team, [&]() {
          // Last level pack treated separately, since int pack k+1 does not exist.
          // Shift's first arg is the scalar to put in the "empty" spot at the end.
          // In this case, we don't need it, so just uze zero.
          const auto& xi_last = x_i(LAST_PACK);
          pack_type tmp = ekat::shift_left(zero(), xi_last);

          tmp += xi_last;
          tmp /= 2.0;
          combine<CM>(tmp, x_m(LAST_PACK), alpha, beta);
        });
      } else {
        // We can use SIMD operations on all NUM_MID_PACKS mid packs,
        // since x_i(k+1) is *always* fine
        team_parallel_for(team,NUM_MID_PACKS,shift_and_avg);
      }
    }
  }

  // Compute X at level interfaces, given X at level midpoints and top or bot bc.
  // Note: with proper bc, then x_int(x_mid(x_int))==x_int.
  template<bool FixTop, typename InputProvider = DefaultProvider>
  KOKKOS_INLINE_FUNCTION
  static void
  compute_interface_values (const TeamMember& team,
                            const int num_mid_levels,
                            const InputProvider& x_m,
                            const scalar_type& bc,
                            const view_1d<pack_type>& x_i)
  {
    compute_interface_values_impl<pack_size,FixTop>(team,num_mid_levels,x_m,bc,x_i);
  }

  // Given X at level interfaces, compute dX at level midpoints.
  template<CombineMode CM = CombineMode::Replace,
           typename InputProvider = DefaultProvider>
  KOKKOS_INLINE_FUNCTION
  static void
  compute_midpoint_delta (const TeamMember& team,
                          const int num_mid_levels,
                          const InputProvider& x_i,
                          const view_1d<pack_type>& dx_m,
                          const scalar_type alpha = one(),
                          const scalar_type beta = zero())
  {
    // Sanity checks
    debug_checks<InputProvider>(num_mid_levels,dx_m);

    // For GPU (or any build with pack size 1), things are simpler
    if (pack_size==1) {
      team_parallel_for(team,num_mid_levels,
                        [&](const int& k) {
        auto tmp = x_i(k+1)-x_i(k);
        combine<CM>(tmp,dx_m(k),alpha,beta);
      });
    } else {
      const auto NUM_MID_PACKS     = pack_info::num_packs(num_mid_levels);
      const auto NUM_INT_PACKS     = pack_info::num_packs(num_mid_levels+1);

      // It is easier to read if we check whether #int_packs==#mid_packs.
      // This lambda can be used in both cases to process packs that have a next pack
      auto shift_and_subtract = [&](const int k){
        auto tmp = ekat::shift_left(x_i(k+1)[0], x_i(k));
        combine<CM>(tmp - x_i(k),dx_m(k),alpha,beta);
      };
      if (NUM_MID_PACKS==NUM_INT_PACKS) {
        const auto LAST_PACK = NUM_MID_PACKS - 1;

        // Use SIMD operations only on NUM_MID_PACKS-1, since mid pack
        // does not have a 'next' one
        team_parallel_for(team,NUM_MID_PACKS-1,shift_and_subtract);

        // Last pack does not have a next one, so needs to be treated separately and serially.
        team_single(team, [&](){
          // Shift's first arg is the scalar to put in the "empty" spot at the end.
          // In this case, we don't need it, so just uze zero.
          const auto& xi_last = x_i(LAST_PACK);
          auto tmp = ekat::shift_left(zero(),xi_last);
          combine<CM>(tmp - xi_last,dx_m(LAST_PACK),alpha,beta);
        });
      } else {
        // We can use SIMD operations on all NUM_MID_PACKS mid packs,
        // since x_i(k+1) is *always* fine
        team_parallel_for(team,NUM_MID_PACKS,shift_and_subtract);
      }
    }
  }

  // Scan sum of a quantity defined at midpoints, to retrieve its integral at interfaces.
  // This function is the logical inverse of the one above
  // Notes:
  //  - FromTop: true means we scan over [0,num_mid_levels), while false is the opposite.
  //  - InputProvider: must provide an input al all mid levels
  //  - s0: used as bc value at k=0 (Forward) or k=num_mid_levels (Backward)
  template<bool FromTop, typename InputProvider>
  KOKKOS_INLINE_FUNCTION
  static void
  column_scan (const TeamMember& team,
               const int num_mid_levels,
               const InputProvider& dx_m,
               const view_1d<pack_type>& x_i,
               const scalar_type& s0 = zero())
  {
    // Sanity checks
    debug_checks<InputProvider>(num_mid_levels+1,x_i);

    // Scan's impl is quite lengthy, so split impl into two fcns, depending on pack size.
    column_scan_impl<pack_size,FromTop>(team,num_mid_levels,dx_m,x_i,s0);
  }

protected:

  template<int PackLength, bool FromTop,typename InputProvider>
  KOKKOS_INLINE_FUNCTION
  static typename std::enable_if<PackLength==1>::type
  column_scan_impl (const TeamMember& team,
                    const int num_mid_levels,
                    const InputProvider& dx_m,
                    const view_1d<pack_type>& x_i,
                    const scalar_type& s0 = zero())
  {
    // If statement is evaluated at compile time, and compiled away.
    if (FromTop) {
      team_single(team,[&](){
        x_i(0)[0] = s0;
      });
      // No need for a barrier here

      team_parallel_scan(team,num_mid_levels,
                         [&](const int k, scalar_type& accumulator, const bool last) {
        accumulator += dx_m(k)[0];
        if (last) {
          x_i(k+1)[0] = s0 + accumulator;
        }
      });
    } else {
      team_single(team,[&](){
        x_i(num_mid_levels)[0] = s0;
      });
      // No need for a barrier here

      team_parallel_scan(team,num_mid_levels,
                         [&](const int k, scalar_type& accumulator, const bool last) {
        const auto k_bwd = num_mid_levels - k - 1;
        accumulator += dx_m(k_bwd)[0];
        if (last) {
          x_i(k_bwd)[0] = s0 + accumulator;
        }
      });
    }
  }

  template<int PackLength,bool FromTop,typename InputProvider>
  KOKKOS_INLINE_FUNCTION
  static typename std::enable_if<(PackLength>1)>::type
  column_scan_impl (const TeamMember& team,
                    const int num_mid_levels,
                    const InputProvider& dx_m,
                    const view_1d<pack_type>& x_i,
                    const scalar_type& s0 = zero())
  {
    // If statement is evaluated at compile time, and compiled away.
    const int NUM_MID_PACKS = pack_info::num_packs(num_mid_levels);
    const int NUM_INT_PACKS = pack_info::num_packs(num_mid_levels+1);
    const int LAST_INT_PACK = NUM_INT_PACKS - 1;

    if (FromTop) {
      // Strategy:
      //  1. Do a packed reduction of x_m, to get x_i(k) = dx_m(0)+...+dx_m(k-1)
      //  2. Let s = s0 + reduce(x_i(k)). The rhs is the sum of all dx_m
      //     on all "physical" levels on all previous packs (plus bc).
      //  3. Do the scan over the current pack: x_i(k)[i] = s + dx_m(k)[0,...,i-1]

      // It is easier to read if we check whether #int_packs==#mid_packs.
      // This lambda can be used so long as there is a 'next' pack x_i(k+1);
      auto packed_scan_from_top = [&](const int& k, pack_type& accumulator, const bool last) {
        accumulator += dx_m(k);
        if (last) {
          x_i(k+1) = accumulator;
        }
      };

      if (NUM_MID_PACKS==NUM_INT_PACKS) {
        // Compute sum of previous packs (hence, stop at second-to-last pack)
        team_parallel_scan(team,NUM_MID_PACKS-1,packed_scan_from_top);
        team.team_barrier();

        // On each pack, reduce the result of the previous step, add the bc s0,
        // then do the scan sum within the current pack.
        team_parallel_for(team,NUM_INT_PACKS,
                          [&](const int k) {
          scalar_type s = s0;
          // If k==0, x_i(k) does not contain any scan sum (and may contain garbage), so ignore it
          if (k!=0) {
            ekat::reduce_sum<false>(x_i(k),s);
          }
          x_i(k)[0] = s;

          const auto this_pack_end = pack_info::vec_end(num_mid_levels,k);
          for (int i=1; i<this_pack_end; ++i) {
            x_i(k)[i] = x_i(k)[i-1] + dx_m(k)[i-1];
          }
        });
      } else {
        // Compute sum of previous packs
        team_parallel_scan(team,NUM_MID_PACKS,packed_scan_from_top);
        team.team_barrier();

        // On each pack, reduce the result of the previous step, add the bc s0,
        // then do the scan sum within the current pack.
        team_parallel_for(team,NUM_INT_PACKS,
                          [&](const int k) {
          scalar_type s = s0;
          // If k==0, x_i(k) does not contain any scan sum (and may contain garbage), so ignore it
          if (k!=0) {
            ekat::reduce_sum<false>(x_i(k),s);
          }
          x_i(k)[0] = s;

          // Note: for the last interface, this_pack_end==1, so we will *not* access
          //       dx_m(LAST_INT_PACK) (which would be OOB).
          const auto this_pack_end = pack_info::vec_end(num_mid_levels+1,k);
          for (int i=1; i<this_pack_end; ++i) {
            x_i(k)[i] = x_i(k)[i-1] + dx_m(k)[i-1];
          }
        });
      }
    } else {
      // Strategy:
      //  1. Do a packed reduction of x_m, to get x_i(k) = dx_m(k)+...+dx_m(num_mid_levs-1)
      //  2. Let s = s0 + reduce(x_i(k)). The rhs is the sum of all dx_m
      //     on all "physical" levels on all subsequent packs (plus bc).
      //  3. Do the scan over the current pack: x_i(k)[i] = s + dx_m(k)[0,...,i-1]

      // It is easier to read if we check whether #int_packs==#mid_packs.
      if (NUM_MID_PACKS==NUM_INT_PACKS) {
        // The easiest thing to do is to do the scan sum in the last pack,
        // where dx_m contains junk, then call this routine again, but
        // for num_mid_levels=(NUM_MID_PACKS-1)*PackLength.
        team_single(team,[&]() {
          auto& xi_last = x_i(NUM_INT_PACKS-1);
          const auto& dxm_last = dx_m(NUM_MID_PACKS-1);
          auto LAST_INT_VEC_END = pack_info::last_vec_end(num_mid_levels+1);
          xi_last[LAST_INT_VEC_END-1] = s0;
          for (int i=LAST_INT_VEC_END-2; i>=0; --i) {
            xi_last[i] = xi_last[i+1] + dxm_last[i];
          }
        });
        team.team_barrier();
        column_scan_impl<PackLength,FromTop>(team,(NUM_MID_PACKS-1)*PackLength,dx_m,x_i,x_i(NUM_INT_PACKS-1)[0]);
      } else {
        // In this case, all packs of dx_m are full of meaningful values.
        auto packed_scan_from_bot = [&](const int& k, pack_type& accumulator, const bool last) {
          const auto k_bwd = NUM_MID_PACKS - k - 1;
          accumulator += dx_m(k_bwd);
          if (last) {
            x_i(k_bwd-1) = accumulator;
          }
        };
        team_parallel_scan(team,NUM_MID_PACKS-1,packed_scan_from_bot);

        // Need to wait for the packed scan to be done before we move fwd
        team.team_barrier();

        // Now let s = s0 + reduce_sum(x_i(k)). The second term is the sum of all dx_m
        // on all "physical" levels on all packs above current one.
        // Then, do the scan over the current pack: x_i(k)[i] = s + dx_m(k)[i,...,pack_end]

        // The last int pack only needs s0, and the second to last had no "following"
        // midpoints pack, and we didn't write anything in it during the scan sum,
        // so fill it with 0's
        team_single(team,[&]() {
          x_i(LAST_INT_PACK)[0] = s0;
          x_i(LAST_INT_PACK-1) = 0;
        });

        team_parallel_for(team,NUM_MID_PACKS,
                          [&](const int k) {
          const auto k_bwd = NUM_MID_PACKS - k - 1;

          scalar_type s = s0;
          if (k_bwd<NUM_MID_PACKS) {
            ekat::reduce_sum(x_i(k_bwd),s);
          }

          auto& xi_kbwd = x_i(k_bwd);
          const auto& dxm_kbwd = dx_m(k_bwd);
          xi_kbwd[PackLength-1] = s + dxm_kbwd[PackLength-1];
          for (int i=PackLength-2; i>=0; --i) {
            xi_kbwd[i] = xi_kbwd[i+1]+dxm_kbwd[i];
          }
        });
      }
    }
  }

  template<int PackLength, bool FixTop,
           typename InputProvider = DefaultProvider>
  KOKKOS_INLINE_FUNCTION
  static typename std::enable_if<PackLength==1>::type
  compute_interface_values_impl (const TeamMember& team,
                                 const int num_mid_levels,
                                 const InputProvider& x_m,
                                 const scalar_type& bc,
                                 const view_1d<pack_type>& x_i)
  {
    // Sanity checks
    debug_checks<InputProvider>(num_mid_levels+1,x_i);

    // Helper function that returns (-1)^k
    auto m1_pow_k = [](const int k) -> scalar_type {
      return 1 - 2*(k%2);
    };

    // The expression of x_i is (N=num_mid_levels)
    //   x_i(k+1) = (-1)^k [ -x_i(0) + 2\Sum_{n=0}^k (-1)^n x_m(n) ]
    //   x_i(k)   = (-1)^k [ (-1)^N x_i(N) + 2\Sum_{n=k}^{N-1} (-1)^n x_m(n) ]
    // for the cases where we fix top and bot respectively. In both cases,
    // we do a scan sum of (-1)^n x_m(n), using the alt_sign impl of column_scan.
    auto scan_input = [&](const int k) -> pack_type {
      // The first term is -1 for k odd and 1 for k even.
      return 2*m1_pow_k(k) * x_m(k);
    };

    column_scan_impl<pack_size,FixTop>(team,num_mid_levels,scan_input,x_i,0);

    if (FixTop) {

      // Need to add -x_i(0), adn multiply everything by (-1)^k
      team.team_barrier();
      team_parallel_for(team,num_mid_levels+1,
                        [&](const int k) {
        x_i(k) -= bc;
        x_i(k) *= m1_pow_k(k+1);
      });
    } else {
      // Need to add (-1)^N x_i(N), adn multiply everything by (-1)^k
      team.team_barrier();
      team_parallel_for(team,num_mid_levels+1,
                        [&](const int k) {
        x_i(k) += m1_pow_k(num_mid_levels) * bc;
        x_i(k) *= m1_pow_k(k);
      });
    }
  }

  template<int PackLength, bool FixTop,
           typename InputProvider = DefaultProvider>
  KOKKOS_INLINE_FUNCTION
  static typename std::enable_if<(PackLength>1)>::type
  compute_interface_values_impl (const TeamMember& team,
                                 const int num_mid_levels,
                                 const InputProvider& x_m,
                                 const scalar_type& bc,
                                 const view_1d<pack_type>& x_i)
  {
    // Sanity checks
    debug_checks<InputProvider>(num_mid_levels+1,x_i);

    auto m1_pow_k = [](const int k)->int {
      return 1 - 2*(k%2);
    };
    // Store a pack of (-1)^k
    pack_type sign = 0;
    vector_simd
    for (int i=0; i<pack_size; ++i) {
      sign[i] = m1_pow_k(i);
    }

    // Scanned quantity is (-1)^n x_m(n)
    auto lambda = [&](const int k)->pack_type {
      return sign*x_m(k);
    };

    // Do a scan sum with 0 bc.
    // Note: the 2nd template arge tells column_scan_impl to perform the final
    // reduction on a single pack using 'interleaved_reduce_sum'
    column_scan_impl<pack_size,FixTop>(team,num_mid_levels,lambda,x_i);
    team.team_barrier();

    const auto NUM_INT_PACKS = pack_info::num_packs(num_mid_levels+1);
    if (FixTop) {
      // Final formula:
      //   x_i(k+1) = (-1)^k [ -x_i(0) + 2\Sum_{n=0}^k (-1)^n x_m(n) ]
      // At this stage, x_i(k) contains the part within the \Sum

      team_parallel_for(team,NUM_INT_PACKS,
                        [&](const int k) {
        x_i(k) = sign*(bc - 2.0*x_i(k));
      });
    } else {
      // Final formula:
      //   x_i(k) = (-1)^k [ (-1)^N x_i(N) + 2\Sum_{n=k}^{N-1} (-1)^n x_m(n) ]
      // At this stage, x_i(k) contains the part within the \Sum

      team_parallel_for(team,NUM_INT_PACKS,
                        [&](const int k) {
        x_i(k) = sign*(bc*m1_pow_k(num_mid_levels) + 2.0*x_i(k));
      });
    }
  }

};

} // namespace Homme

#endif // SCREAM_COLUMN_OPS_HPP