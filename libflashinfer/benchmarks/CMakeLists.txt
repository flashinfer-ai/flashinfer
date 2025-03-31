# cmake-format: off
# Set global variable for benchmarking
set(ALL_BENCHMARK_TARGETS "")

# Include centralized config utilities
include(ConfigureTargets)

# === Core Benchmarks Configuration ===
configure_flashinfer_target(
  TARGET_NAME bench_single_decode
  SOURCES "bench_single_decode.cu"
  LINK_LIBS "decode_kernels;prefill_kernels"
  IS_BENCHMARK ON
)

configure_flashinfer_target(
  TARGET_NAME bench_batch_decode
  SOURCES "bench_batch_decode.cu"
  LINK_LIBS "decode_kernels;prefill_kernels"
  IS_BENCHMARK ON
)

configure_flashinfer_target(
  TARGET_NAME bench_batch_decode_mla
  SOURCES "bench_batch_decode_mla.cu"
  LINK_LIBS "decode_kernels"
  IS_BENCHMARK ON
)

configure_flashinfer_target(
  TARGET_NAME bench_single_prefill
  SOURCES "bench_single_prefill.cu"
  LINK_LIBS "prefill_kernels"
  IS_BENCHMARK ON
)

configure_flashinfer_target(
  TARGET_NAME bench_batch_prefill
  SOURCES "bench_batch_prefill.cu"
  LINK_LIBS "prefill_kernels"
  IS_BENCHMARK ON
)

configure_flashinfer_target(
  TARGET_NAME bench_cascade
  SOURCES "bench_cascade.cu"
  LINK_LIBS "decode_kernels;prefill_kernels"
  IS_BENCHMARK ON
)

configure_flashinfer_target(
  TARGET_NAME bench_sampling
  SOURCES "bench_sampling.cu"
  IS_BENCHMARK ON
)

configure_flashinfer_target(
  TARGET_NAME bench_norm
  SOURCES "bench_norm.cu"
  IS_BENCHMARK ON
)

# === FP8 Benchmark Configuration ===
if(FLASHINFER_FP8_BENCHMARKS)
  # Set path to FP8 utilities
  set(FP8_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fp8)

  # Define FP8-specific CUDA compile flags - one per line for readability
  set(FP8_CUDA_FLAGS_LIST
      --expt-extended-lambda
      --use_fast_math
      --generate-code=arch=compute_90a,code=sm_90a
  )

  # Combine the list into a generator expression for CUDA language only
  string(JOIN ";" FP8_CUDA_FLAGS_JOINED "${FP8_CUDA_FLAGS_LIST}")
  set(FP8_CUDA_FLAGS "$<$<COMPILE_LANGUAGE:CUDA>:${FP8_CUDA_FLAGS_JOINED}>")

  # Define FP8-specific include directories
  set(FP8_INCLUDE_DIRS
      "${TORCH_INCLUDE_DIRS}"
      "${CUDA_INCLUDE_DIRS}"
      "${Python3_INCLUDE_DIRS}"
      "${FA3_INCLUDE_DIR}"
      "${CUTLASS_INCLUDE_DIRS}"
      "${FP8_UTILS_DIR}"
  )

  # Add the FP8 benchmark with improved readability
  configure_flashinfer_target(
    TARGET_NAME bench_single_prefill_fp8_sm90
    SOURCES "fp8/bench_single_prefill_sm90.cu"
    LINK_LIBS "FA3_LIB;${TORCH_LIBRARIES}"
    COMPILE_FLAGS "${FP8_CUDA_FLAGS}"
    INCLUDE_DIRS "${FP8_INCLUDE_DIRS}"
    IS_BENCHMARK ON
  )
endif()

# === Benchmark Build and Run Targets ===

# Create a target to build all benchmarks
add_custom_target(build_benchmarks)
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
# cmake-format: on
