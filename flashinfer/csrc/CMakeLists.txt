# Include the custom torch extension helper module
include(ConfigureTorchExtension)

if(FLASHINFER_ENABLE_CUDA)
  message(STATUS "Building FlashInfer Torch CUDA Extnesion")
elseif(FLASHINFER_ENABLE_HIP)
  message(STATUS "Building FlashInfer with Torch ROCm Extension")
endif()

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
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_ops.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/custom_all_reduce.cu")

file(
  GLOB
  KERNEL_HIP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/activation_hip.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_decode_hip.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/cascade_hip.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_ops_hip.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/norm_hip.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/page_hip.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/single_decode_hip.cu")

# SM90-specific sources
file(
  GLOB
  KERNEL_SM90_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/single_prefill_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_prefill_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_ops_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_f16_f16_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_bf16_bf16_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e4m3_f16_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e5m2_f16_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e4m3_bf16_sm90.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_e5m2_bf16_sm90.cu")

# Common include directories
set(COMMON_CUDA_INCLUDE_DIRS
    ${FLASHINFER_INCLUDE_DIR} ${CUTLASS_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
    ${FLASHINFER_GENERATED_SOURCE_DIR} ${FLASHINFER_GENERATED_SOURCE_DIR_ROOT})

set(COMMON_HIP_INCLUDE_DIRS
    ${FLASHINFER_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS}
    ${FLASHINFER_GENERATED_SOURCE_DIR} ${FLASHINFER_GENERATED_SOURCE_DIR_ROOT})
# Common compile definitions
set(COMMON_COMPILE_DEFS
    $<$<BOOL:${FLASHINFER_ENABLE_FP8_E4M3}>:FLASHINFER_ENABLE_FP8_E4M3>
    $<$<BOOL:${FLASHINFER_ENABLE_FP8_E5M2}>:FLASHINFER_ENABLE_FP8_E5M2>
    $<$<BOOL:${FLASHINFER_ENABLE_BF16}>:FLASHINFER_ENABLE_BF16>
    $<$<BOOL:${FLASHINFER_ENABLE_F16}>:FLASHINFER_ENABLE_F16>)

# Common libraries
set(COMMON_LIBS cublas cublasLt)

# cmake-format: off
if(FLASHINFER_ENABLE_HIP)
    # Build an extension only with the decode kernels
    add_hip_torch_extension(
      EXT_NAME flashinfer_hip_kernels
      SOURCES
        ${KERNEL_HIP_SOURCES}
      INCLUDE_DIRS
        ${COMMON_HIP_INCLUDE_DIRS}
      LINK_LIBS
        $<$<TARGET_EXISTS:flashinfer::decode_kernels>:flashinfer::decode_kernels>
      COMPILE_FLAGS
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>
      PY_LIMITED_API ${FLASHINFER_PY_LIMITED_API}
      MIN_PYTHON_ABI "${FLASHINFER_MIN_PYTHON_ABI}"
    )
    target_compile_definitions(flashinfer_hip_kernels
                             PRIVATE FLASHINFER_ENABLE_HIP)
    set_target_properties(
      flashinfer_hip_kernels
      PROPERTIES INSTALL_RPATH
                 "${FLASHINFER_RPATH_BASE};${FLASHINFER_RPATH_BASE}/../lib"
                 HIP_SOURCES_PROPERTY_FORMAT 1
                HIP_SEPARABLE_COMPILATION ON
                LINKER_LANGUAGE HIP)
    # Install the extension to flashinfer directory
    install(TARGETS flashinfer_hip_kernels DESTINATION flashinfer)

elseif(FLASHINFER_ENABLE_CUDA)
  # Build an extension only with the decode kernels
  add_cuda_torch_extension(
    EXT_NAME flashinfer_kernels
    SOURCES
      ${KERNEL_SOURCES}
    INCLUDE_DIRS
      ${COMMON_CUDA_INCLUDE_DIRS}
    LINK_LIBS
      ${COMMON_LIBS}
      $<$<TARGET_EXISTS:flashinfer::decode_kernels>:flashinfer::decode_kernels>
      $<$<TARGET_EXISTS:flashinfer::prefill_kernels>:flashinfer::prefill_kernels>
    COMPILE_FLAGS
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>
    PY_LIMITED_API ${FLASHINFER_PY_LIMITED_API}
    MIN_PYTHON_ABI "${FLASHINFER_MIN_PYTHON_ABI}"
  )
  target_compile_definitions(flashinfer_kernels PRIVATE FLASHINFER_ENABLE_CUDA)
  set_target_properties(
  flashinfer_kernels
  PROPERTIES INSTALL_RPATH
             "${FLASHINFER_RPATH_BASE};${FLASHINFER_RPATH_BASE}/../lib")
  # Install the extension to flashinfer directory
  install(TARGETS flashinfer_kernels DESTINATION flashinfer)
endif()
# cmake-format: on

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
      ${SM90A_FLAGS}>
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

message(STATUS "FlashInfer AOT PyTorch extensions configured")
