add_executable(spike_tests test_spike.cu)
target_link_libraries(spike_tests PRIVATE gspike::gspike)
target_compile_options(spike_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
set_target_properties(spike_tests PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)
add_test(NAME spike_tests COMMAND spike_tests)
