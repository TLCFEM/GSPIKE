cmake_minimum_required(VERSION 3.20)
project(gspike VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

find_package(CUDAToolkit REQUIRED)

# Width of gspike::index_t, used for every size, stride and offset.
set(GSPIKE_INDEX_TYPE "int32" CACHE STRING "Index type used by gspike: int32 or int64")
set_property(CACHE GSPIKE_INDEX_TYPE PROPERTY STRINGS int32 int64)
if (GSPIKE_INDEX_TYPE STREQUAL "int64")
    set(GSPIKE_INDEX_BITS 64)
    set(GSPIKE_INDEX_DEFINE GSPIKE_INDEX_64)
elseif (GSPIKE_INDEX_TYPE STREQUAL "int32")
    set(GSPIKE_INDEX_BITS 32)
    set(GSPIKE_INDEX_DEFINE GSPIKE_INDEX_32)
else ()
    message(FATAL_ERROR "GSPIKE_INDEX_TYPE must be 'int32' or 'int64' (got '${GSPIKE_INDEX_TYPE}')")
endif ()
message(STATUS "gspike: ${GSPIKE_INDEX_BITS}-bit indexing (${GSPIKE_INDEX_TYPE})")

option(GSPIKE_LEGACY_SOLVE "Use the original one-thread-per-RHS band solve (A/B comparisons)" OFF)

add_library(gspike src/spike.cu)
add_library(gspike::gspike ALIAS gspike)
target_include_directories(gspike PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gspike PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)
# PUBLIC so that every consumer sees the same gspike::index_t as the library.
target_compile_definitions(gspike PUBLIC ${GSPIKE_INDEX_DEFINE})
if (GSPIKE_LEGACY_SOLVE)
    target_compile_definitions(gspike PRIVATE GSPIKE_LEGACY_SOLVE=1)
endif ()
set_target_properties(gspike PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)
if (BUILD_TESTING)
    add_executable(spike_tests tests/test_spike.cu)
    target_link_libraries(spike_tests PRIVATE gspike)
    add_test(NAME spike_tests COMMAND spike_tests)
    # A second, differently seeded fuzzy run catches generator-specific blind
    # spots without making the default run non-deterministic.
    add_test(NAME spike_tests_fuzz COMMAND spike_tests --seed 123456789 --fuzzy-iters 256)

    add_executable(spike_bench bench/bench.cu)
    target_link_libraries(spike_bench PRIVATE gspike)
endif ()
