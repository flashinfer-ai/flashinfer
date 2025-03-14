# ------------------------------ Dependencies ---------------------------------#

FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG c03033b50e46748207b27685b1cdfcbe4a2fec59)
FetchContent_MakeAvailable(nvbench)

# -----------------------------------------------------------------------------#

set(ALL_BENCHMARK_TARGETS "")

function(configure_benchmark)
  set(oneValueArgs BENCHMARK_TARGET_NAME)
  set(multiValueArgs BENCHMARK_SOURCES EXTRA_LINK_LIBS EXTRA_COMPILE_FLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")

  message(STATUS "Configure ${arg_BENCHMARK_TARGET_NAME} benchmark")

  add_executable(${arg_BENCHMARK_TARGET_NAME} EXCLUDE_FROM_ALL
                 ${arg_BENCHMARK_SOURCES})
  target_include_directories(
    ${arg_BENCHMARK_TARGET_NAME} PRIVATE ${FLASHINFER_INCLUDE_DIR}
                                         ${FLASHINFER_UTILS_INCLUDE_DIR})
  # TODO: Why is this needed at all? The use of dispatch_inc has to be
  # understood. For now we need it as otherwise the generated sources do not
  # compile.
  add_dependencies(${arg_BENCHMARK_TARGET_NAME} dispatch_inc)
  target_link_libraries(${arg_BENCHMARK_TARGET_NAME} PRIVATE nvbench::main)
  foreach(extra_link_lib IN LISTS arg_EXTRA_LINK_LIBS)
    target_link_libraries(${arg_BENCHMARK_TARGET_NAME}
                          PRIVATE ${extra_link_lib})
  endforeach(extra_link_lib)

  # Add the target to the ALL_BENCHMARK_TARGETS list so that it gets built when
  # the build_benchmarks target is invoked.
  list(APPEND ALL_BENCHMARK_TARGETS "${arg_BENCHMARK_TARGET_NAME}")
  set(ALL_BENCHMARK_TARGETS
      "${ALL_BENCHMARK_TARGETS}"
      PARENT_SCOPE)
  message(STATUS "Configured ${arg_BENCHMARK_TARGET_NAME} benchmark")

endfunction(configure_benchmark)

# cmake-format: off
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_single_decode
  BENCHMARK_SOURCES "bench_single_decode.cu"
  EXTRA_LINK_LIBS "decode_kernels;prefill_kernels"
)
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_batch_decode
  BENCHMARK_SOURCES "bench_batch_decode.cu"
  EXTRA_LINK_LIBS "decode_kernels;prefill_kernels"
)
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_batch_decode_mla
  BENCHMARK_SOURCES "bench_batch_decode_mla.cu"
  EXTRA_LINK_LIBS "decode_kernels"
)
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_single_prefill
  BENCHMARK_SOURCES "bench_single_prefill.cu"
  EXTRA_LINK_LIBS "prefill_kernels"
)
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_batch_prefill
  BENCHMARK_SOURCES "bench_batch_prefill.cu"
  EXTRA_LINK_LIBS "prefill_kernels"
)
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_cascade
  BENCHMARK_SOURCES "bench_cascade.cu"
  EXTRA_LINK_LIBS "decode_kernels;prefill_kernels"
)
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_sampling
  BENCHMARK_SOURCES "bench_sampling.cu"
)
configure_benchmark(
  BENCHMARK_TARGET_NAME bench_norm
  BENCHMARK_SOURCES "bench_norm.cu"
)
# cmake-format: on

# A top-level target to build all benchmarks
add_custom_target(build_benchmarks)
# Make the top-level benchmark target depend on all benchmarks
add_dependencies(build_benchmarks ${ALL_BENCHMARK_TARGETS})

# Create a custom target to run all benchmarks
add_custom_target(run_all_benchmarks COMMAND ${CMAKE_COMMAND} -E echo
                                             "Running all benchmarks...")

# Add a custom command for each benchmark executable to run it
foreach(benchmark IN LISTS ALL_BENCHMARK_TARGETS)
  add_custom_command(
    TARGET run_all_benchmarks
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Running ${benchmark}..."
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${benchmark})
endforeach()

# Ensure the run_all_benchmarks target depends on all benchmark executables
add_dependencies(run_all_benchmarks ${ALL_BENCHMARK_TARGETS})
