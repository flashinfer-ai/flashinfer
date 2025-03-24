# FP8 test configuration
set(FP8_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Add the test with the flashinfer_ops.cu file combined
configure_unit_test(
  TEST_TARGET_NAME
  test_single_prefill_fa3_sm90
  TEST_SOURCES
  "test_single_prefill_fa3_sm90.cu"
  "flashinfer_ops.cu"
  EXTRA_LINK_LIBS
  "${TORCH_LIBRARIES};FA3_LIB"
  EXTRA_COMPILE_FLAGS
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--use_fast_math;--generate-code=arch=compute_90a,code=sm_90a>"
  EXTRA_INCLUDE_DIRS
  "${TORCH_INCLUDE_DIRS};${CUDA_INCLUDE_DIRS};${Python3_INCLUDE_DIRS};${FA3_INCLUDE_DIR};${CUTLASS_INCLUDE_DIR};${FP8_UTILS_DIR}"
  IS_GTEST
  OFF)
