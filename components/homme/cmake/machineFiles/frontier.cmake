SET (HOMME_MACHINE "frontier" CACHE STRING "")

SET (HOMMEXX_CUDA_MAX_WARP_PER_TEAM "16" CACHE STRING  "")

SET (NETCDF_DIR $ENV{CRAY_NETCDF_HDF5PARALLEL_PREFIX} CACHE FILEPATH "")
SET (HDF5_DIR $ENV{CRAY_HDF5_PARALLEL_PREFIX} CACHE FILEPATH "")
SET (CPRNC_DIR /ccs/proj/cli115/software/frontier/cprnc CACHE FILEPATH "")

SET(BUILD_HOMME_WITHOUT_PIOLIBRARY TRUE CACHE BOOL "")

SET(HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

SET(WITH_PNETCDF FALSE CACHE FILEPATH "")

SET(USE_QUEUING FALSE CACHE BOOL "")

SET(BUILD_HOMME_PREQX_KOKKOS TRUE CACHE BOOL "")
SET(BUILD_HOMME_THETA_KOKKOS TRUE CACHE BOOL "")

SET (HOMMEXX_BFB_TESTING FALSE CACHE BOOL "")

SET(USE_TRILINOS OFF CACHE BOOL "")

SET(HIP_BUILD TRUE CACHE BOOL "")

SET(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
SET(Kokkos_ENABLE_DEBUG OFF CACHE BOOL "")
SET(Kokkos_ARCH_VEGA90A ON CACHE BOOL "")
SET(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
SET(Kokkos_ENABLE_HIP ON CACHE BOOL "")
SET(Kokkos_ENABLE_EXPLICIT_INSTANTIATION OFF CACHE BOOL "")

SET(CMAKE_C_COMPILER "cc" CACHE STRING "")
SET(CMAKE_Fortran_COMPILER "ftn" CACHE STRING "")
#SET(CMAKE_Fortran_FLAGS "--gcc-toolchain=$ENV{MEMBERWORK}/cli115/workaround" CACHE STRING "")
SET(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "")

SET(Extrae_LIBRARY "-I$ENV{CRAY_MPICH_DIR}/include -L$ENV{CRAY_MPICH_DIR}/lib -lmpi $ENV{PE_MPICH_GTL_DIR_amd_gfx90a} $ENV{PE_MPICH_GTL_LIBS_amd_gfx90a}" CACHE STRING "")

#SET(ADD_Fortran_FLAGS "-G2 --gcc-toolchain=$ENV{MEMBERWORK}/cli115/workaround ${Extrae_LIBRARY}" CACHE STRING "")
SET(ADD_C_FLAGS "-g -O ${Extrae_LIBRARY}" CACHE STRING "")
SET(ADD_CXX_FLAGS "-g -std=c++14 -O3 -munsafe-fp-atomics --offload-arch=gfx90a -fno-gpu-rdc -Wno-unused-command-line-argument -Wno-unsupported-floating-point-opt -Wno-#pragma-messages ${Extrae_LIBRARY}" CACHE STRING "")
SET(ADD_LINKER_FLAGS "${Extrae_LIBRARY}" CACHE STRING "")


set (ENABLE_OPENMP OFF CACHE BOOL "")
set (ENABLE_COLUMN_OPENMP OFF CACHE BOOL "")
set (ENABLE_HORIZ_OPENMP OFF CACHE BOOL "")

set (HOMME_TESTING_PROFILE "short" CACHE STRING "")
SET (BUILD_HOMME_THETA_KOKKOS TRUE CACHE BOOL "")

set (USE_NUM_PROCS 8 CACHE STRING "")

SET (USE_MPI_OPTIONS "-c7 --gpu-bind=closest --gpus-per-task=1" CACHE FILEPATH "")