cmake_minimum_required(VERSION 3.24)

# project() initializes CMAKE_CUDA_ARCHITECTURES to nvcc's default, so the
# standalone default has to be set before it.  A parent project's choice is
# never touched (it may target a completely different GPU).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif ()

project(gspike VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Standalone builds get the benchmark and tests by default.  When gspike is
# added as a subdirectory of another project, only the library is built and
# none of these choices leak into the parent project.
if(PROJECT_IS_TOP_LEVEL)
    option(GSPIKE_BENCH "Build the benchmarks (with optional comparisons)" ON)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif ()
endif ()

if(NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit REQUIRED)
endif ()

add_library(gspike src/spike.cu src/spike_half.cu src/mixed.cu src/mixed_half.cu)
add_library(gspike::gspike ALIAS gspike)

set_target_properties(gspike PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
)
target_include_directories(gspike PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gspike PUBLIC CUDA::cudart)
target_compile_options(gspike PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

if(PROJECT_IS_TOP_LEVEL AND GSPIKE_BENCH)
    add_subdirectory(bench)
endif ()

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif ()
endif ()
