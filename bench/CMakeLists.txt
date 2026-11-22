option(GSPIKE_BUILD_REF "Build the original Fortran CPU solver for benchmark comparison" ON)

add_executable(spike_bench bench.cu)
target_link_libraries(spike_bench PRIVATE gspike::gspike)
set_target_properties(spike_bench PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

if (GSPIKE_BUILD_REF)
    include(CheckLanguage)
    check_language(Fortran)
endif ()

if (GSPIKE_BUILD_REF AND CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    find_package(LAPACK REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS Fortran)

    # Original Fortran sources, double-precision real path only.  This library is
    # linked into spike_bench alone; the CUDA library never depends on it.
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
    target_compile_options(gspike_ref PRIVATE
            -cpp -ffree-line-length-none -fallow-argument-mismatch
    )
    target_link_libraries(gspike_ref PUBLIC
            ${LAPACK_LIBRARIES}
            OpenMP::OpenMP_Fortran
            ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
    )

    target_link_libraries(spike_bench PRIVATE gspike_ref)
    target_include_directories(spike_bench PRIVATE ${PROJECT_SOURCE_DIR}/ref)
    target_compile_definitions(spike_bench PRIVATE GSPIKE_HAVE_REF=1)
elseif (GSPIKE_BUILD_REF)
    message(WARNING "No Fortran compiler; spike_bench will not include the CPU reference")
endif ()
