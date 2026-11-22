cmake_minimum_required(VERSION 3.20)
project(gspike VERSION 0.1.0 LANGUAGES C CXX CUDA)

option(GSPIKE_BENCH "Build the benchmark" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# Host-side AddressSanitizer for Debug builds.  ASan does not see device
# allocations; use compute-sanitizer for device-side errors.
add_compile_options(
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fno-omit-frame-pointer>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fsanitize=address,-fno-omit-frame-pointer>
)
add_link_options(
  $<$<CONFIG:Debug>:-fsanitize=address>
)

find_package(CUDAToolkit REQUIRED)

add_library(gspike src/spike.cu src/mixed.cu)
add_library(gspike::gspike ALIAS gspike)
target_include_directories(gspike PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gspike PUBLIC CUDA::cudart)
set_target_properties(gspike PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(GSPIKE_BENCH)
  add_subdirectory(bench)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(spike_tests tests/test_spike.cu)
  target_link_libraries(spike_tests PRIVATE gspike::gspike)
  set_target_properties(spike_tests PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME spike_tests COMMAND spike_tests)
  # CUDA's UVA allocator needs the shadow gap that ASan protects by default.
  set_tests_properties(spike_tests PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0")
endif()
