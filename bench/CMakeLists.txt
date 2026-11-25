# Benchmark targets are opportunistic: the CPU comparison needs Fortran,
# LAPACK and OpenMP; the head-to-head needs MKL plus cuBLAS/cuSPARSE.  Each is
# built only when its dependencies are available, with no user-facing options.

find_path(MKL_INCLUDE_DIR mkl/mkl.h)
find_library(MKL_RT_LIBRARY mkl_rt)
if (MKL_INCLUDE_DIR AND MKL_RT_LIBRARY AND TARGET CUDA::cublas AND TARGET CUDA::cusparse)
    add_executable(spike_duel duel.cu)
    target_link_libraries(spike_duel PRIVATE gspike::gspike CUDA::cublas CUDA::cusparse ${MKL_RT_LIBRARY})
    target_include_directories(spike_duel PRIVATE ${MKL_INCLUDE_DIR})
    target_compile_options(spike_duel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    set_target_properties(spike_duel PROPERTIES
            LINKER_LANGUAGE CXX
            CXX_STANDARD 20
            CUDA_STANDARD 20
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
else ()
    message(STATUS "spike_duel disabled: MKL, cuBLAS or cuSPARSE not found")
endif ()

add_executable(spike_bench bench.cu)
target_link_libraries(spike_bench PRIVATE gspike::gspike)
target_compile_options(spike_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
set_target_properties(spike_bench PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 20
        CUDA_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include(CheckLanguage)
check_language(Fortran)
if (CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    find_package(LAPACK QUIET)
    find_package(OpenMP QUIET COMPONENTS Fortran)
endif ()

if (CMAKE_Fortran_COMPILER AND LAPACK_FOUND AND OpenMP_Fortran_FOUND)
    # Original Fortran sources, double-precision real path only.  This library
    # is linked into spike_bench alone; the CUDA library never depends on it.
    add_library(gspike_ref STATIC
            ${PROJECT_SOURCE_DIR}/ref/affinity.c
            ${PROJECT_SOURCE_DIR}/ref/f90_functions_wrapper.f90
            ${PROJECT_SOURCE_DIR}/ref/spike_smp_utilities.f90
            ${PROJECT_SOURCE_DIR}/ref/dzlbprim.f90
            ${PROJECT_SOURCE_DIR}/ref/dpivul.f
            ${PROJECT_SOURCE_DIR}/ref/dspike_smp.f90
            ${PROJECT_SOURCE_DIR}/ref/spike_ref.f90
    )
    target_include_directories(gspike_ref PRIVATE ${PROJECT_SOURCE_DIR}/ref)
    if (CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
        target_compile_options(gspike_ref PRIVATE
                $<$<COMPILE_LANGUAGE:Fortran>:-cpp>
                $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>
                $<$<COMPILE_LANGUAGE:Fortran>:-fallow-argument-mismatch>
        )
    endif ()
    target_link_libraries(gspike_ref PUBLIC
            ${LAPACK_LIBRARIES}
            OpenMP::OpenMP_Fortran
            ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
    )

    target_link_libraries(spike_bench PRIVATE gspike_ref)
    target_include_directories(spike_bench PRIVATE ${PROJECT_SOURCE_DIR}/ref)
    target_compile_definitions(spike_bench PRIVATE GSPIKE_HAVE_REF=1)
else ()
    message(STATUS "CPU reference disabled: Fortran, LAPACK or OpenMP not available")
endif ()
