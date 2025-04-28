# Include the custom torch extension helper module
include(ConfigureTorchExtension)

# Source files for PyTorch extensions
file(
  GLOB
  KERNEL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/bmm_fp8.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/cascade.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/norm.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/page.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/quantization.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/rope.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/sampling.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/renorm.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/activation.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_decode.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_prefill.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/single_decode.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/single_prefill.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_ops.cu",
  "${CMAKE_CURRENT_SOURCE_DIR}/custom_all_reduce.cu")

# SM90-specific sources
file(
  GLOB
  KERNEL_SM90_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/single_prefill_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_prefill_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_ops_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_f16_f16_sm90.cu",
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_bf16_bf16_sm90.cu",
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e4m3_f16_sm90.cu",
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e5m2_f16_sm90.cu",
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e4m3_bf16_sm90.cu",
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e5m2_bf16_sm90.cu")

# Find generated source files
file(GLOB DECODE_SOURCES "${FLASHINFER_GENERATED_SOURCE_DIR}/*decode_head*.cu")
file(GLOB PREFILL_SOURCES
     "${FLASHINFER_GENERATED_SOURCE_DIR}/*prefill_head*.cu")
list(FILTER PREFILL_SOURCES EXCLUDE REGEX ".*_sm90.*")
file(GLOB PREFILL_SM90_SOURCES
     "${FLASHINFER_GENERATED_SOURCE_DIR}/*prefill_head*_sm90.cu")

# Common include directories
set(COMMON_INCLUDE_DIRS
    ${FLASHINFER_INCLUDE_DIR} ${CUTLASS_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
    ${FLASHINFER_GENERATED_SOURCE_DIR} ${FLASHINFER_GENERATED_SOURCE_DIR_ROOT})

# Common compile definitions
set(COMMON_COMPILE_DEFS
    $<$<BOOL:${FLASHINFER_ENABLE_FP8_E4M3}>:FLASHINFER_ENABLE_FP8_E4M3>
    $<$<BOOL:${FLASHINFER_ENABLE_FP8_E5M2}>:FLASHINFER_ENABLE_FP8_E5M2>
    $<$<BOOL:${FLASHINFER_ENABLE_BF16}>:FLASHINFER_ENABLE_BF16>
    $<$<BOOL:${FLASHINFER_ENABLE_F16}>:FLASHINFER_ENABLE_F16>)

# Common libraries
set(COMMON_LIBS cublas cublasLt)

# CXX11 ABI handling
if(FLASHINFER_USE_CXX11_ABI)
  set(CXX11_ABI_DEF _GLIBCXX_USE_CXX11_ABI=1)
else()
  set(CXX11_ABI_DEF "")
endif()

# cmake-format: off
add_cuda_torch_extension(
  EXT_NAME flashinfer_kernels
  SOURCES
    ${KERNEL_SOURCES}
  INCLUDE_DIRS
    ${COMMON_INCLUDE_DIRS}
  LINK_LIBS
    ${COMMON_LIBS}
    $<$<TARGET_EXISTS:flashinfer::decode_kernels>:flashinfer::decode_kernels>
    $<$<TARGET_EXISTS:flashinfer::prefill_kernels>:flashinfer::prefill_kernels>
  COMPILE_FLAGS
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>
    $<$<COMPILE_LANGUAGE:CUDA>:${TORCH_EXT_NVCC_FLAGS}>
    $<$<BOOL:${CXX11_ABI_DEF}>:-D${CXX11_ABI_DEF}>
  PY_LIMITED_API ${FLASHINFER_PY_LIMITED_API}
  MIN_PYTHON_ABI "${FLASHINFER_MIN_PYTHON_ABI}"
)
# cmake-format: on

# Set custom RPATH for the extension
set_target_properties(
  flashinfer_kernels
  PROPERTIES INSTALL_RPATH
             "${FLASHINFER_RPATH_BASE};${FLASHINFER_RPATH_BASE}/../lib")

# Install the extension to flashinfer directory
install(TARGETS flashinfer_kernels DESTINATION flashinfer)

# Build SM90 extension conditionally
if(FLASHINFER_ENABLE_SM90 AND KERNEL_SM90_SOURCES)
  # Add SM90-specific compiler flags
  set(SM90A_FLAGS -gencode arch=compute_90a,code=sm_90a)
  # cmake-format: off
  add_cuda_torch_extension(
    EXT_NAME flashinfer_kernels_sm90
    SOURCES
      ${KERNEL_SM90_SOURCES}
    INCLUDE_DIRS
      ${COMMON_INCLUDE_DIRS}
    LINK_LIBS
      ${COMMON_LIBS}
      $<$<TARGET_EXISTS:flashinfer::prefill_kernels_sm90>:flashinfer::prefill_kernels_sm90>
    COMPILE_FLAGS
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>
      $<$<COMPILE_LANGUAGE:CUDA>:${TORCH_EXT_NVCC_FLAGS}
      ${SM90A_FLAGS}>
      $<$<BOOL:${CXX11_ABI_DEF}>:-D${CXX11_ABI_DEF}>
    PY_LIMITED_API ${FLASHINFER_PY_LIMITED_API}
    MIN_PYTHON_ABI "${FLASHINFER_MIN_PYTHON_ABI}"
  )
  # cmake-format: on
  # Set custom RPATH for the SM90 extension
  set_target_properties(
    flashinfer_kernels_sm90
    PROPERTIES INSTALL_RPATH
               "${FLASHINFER_RPATH_BASE};${FLASHINFER_RPATH_BASE}/../lib")

  # Install the SM90 extension to flashinfer directory
  install(TARGETS flashinfer_kernels_sm90 DESTINATION flashinfer)
endif()

# Configure aot_config.py
configure_file("${CMAKE_SOURCE_DIR}/templates/aot_config.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/flashinfer/jit/aot_config.py" @ONLY)

# Install aot config
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/flashinfer/jit/aot_config.py"
        DESTINATION "flashinfer/jit")

message(STATUS "FlashInfer AOT PyTorch extensions configured")
