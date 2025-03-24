# FP8 benchmark configuration - include utils directory for Flash Attention 3
# utilities
set(FP8_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/fp8)

configure_benchmark(
  BENCHMARK_TARGET_NAME
  bench_single_prefill_fp8_sm90
  BENCHMARK_SOURCES
  "bench_single_prefill_sm90.cu"
  EXTRA_LINK_LIBS
  "FA3_LIB;${TORCH_LIBRARIES}"
  EXTRA_COMPILE_FLAGS
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--use_fast_math;--generate-code=arch=compute_90a,code=sm_90a>"
  EXTRA_INCLUDE_DIRS
  "${TORCH_INCLUDE_DIRS};${CUDA_INCLUDE_DIRS};${FA3_INCLUDE_DIR};${CUTLASS_INCLUDE_DIR};${FP8_UTILS_DIR};${Python3_INCLUDE_DIRS}"
)
