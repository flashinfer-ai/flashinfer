# cmake/utils/ConfigureKernelGeneration.cmake Module for kernel source
# generation configuration and execution

# Function to configure and run kernel source generation
function(flashinfer_configure_kernel_generation)

  # Setup generated directories
  set(GENERATED_SOURCE_DIR
      "${CMAKE_BINARY_DIR}/libflashinfer/src/generated"
      CACHE INTERNAL "")
  set(GENERATED_SOURCE_DIR_ROOT
      "${CMAKE_BINARY_DIR}/libflashinfer/src"
      CACHE INTERNAL "")
  file(MAKE_DIRECTORY ${GENERATED_SOURCE_DIR})

  # Define the stamp file for tracking regeneration
  set(GENERATION_STAMP "${GENERATED_SOURCE_DIR}/generation.stamp")

  # Find all generator scripts to track for changes
  file(GLOB GENERATOR_SCRIPTS "${PROJECT_SOURCE_DIR}/aot_build_utils/*.py")

  # Check if we need to regenerate based on script changes
  set(NEED_SOURCE_REGENERATION FALSE)
  if(NOT EXISTS "${GENERATION_STAMP}")
    message(STATUS "Generation needed: Stamp file not found")
    set(NEED_SOURCE_REGENERATION TRUE)
  elseif(NOT EXISTS "${GENERATED_SOURCE_DIR}/dispatch.inc")
    message(STATUS "Generation needed: Generated files not found")
    set(NEED_SOURCE_REGENERATION TRUE)
  else()
    foreach(script ${GENERATOR_SCRIPTS})
      if("${script}" IS_NEWER_THAN "${GENERATION_STAMP}")
        message(STATUS "Generation needed: ${script} changed")
        set(NEED_SOURCE_REGENERATION TRUE)
        break()
      endif()
    endforeach()
  endif()

  # cmake-format: off
  # Command to generate kernel sources
  set(AOT_GENERATE_COMMAND
      ${Python3_EXECUTABLE} -m aot_build_utils.generate
      --path ${GENERATED_SOURCE_DIR}
      --head_dims ${FLASHINFER_GEN_HEAD_DIMS}
      --pos_encoding_modes ${FLASHINFER_GEN_POS_ENCODING_MODES}
      --use_fp16_qk_reductions ${FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS}
      --mask_modes ${FLASHINFER_GEN_MASK_MODES}
      --enable_f16 ${FLASHINFER_ENABLE_F16}
      --enable_bf16 ${FLASHINFER_ENABLE_BF16}
      --enable_fp8_e4m3 ${FLASHINFER_ENABLE_FP8_E4M3}
      --enable_fp8_e5m2 ${FLASHINFER_ENABLE_FP8_E5M2}
      --generate_hip ${FLASHINFER_ENABLE_HIP})

  # Command to generate dispatch include file
  set(AOT_GENERATE_DISPATCH_INC_COMMAND
      ${Python3_EXECUTABLE} -m aot_build_utils.generate_dispatch_inc
      --path "${GENERATED_SOURCE_DIR}/dispatch.inc"
      --head_dims ${FLASHINFER_GEN_HEAD_DIMS}
      --head_dims_sm90 ${HEAD_DIMS_SM90}
      --pos_encoding_modes ${FLASHINFER_GEN_POS_ENCODING_MODES}
      --use_fp16_qk_reductions ${FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS}
      --mask_modes ${FLASHINFER_GEN_MASK_MODES})

  set(AOT_GENERATE_SM90_COMMAND COMMAND
      ${Python3_EXECUTABLE} -m aot_build_utils.generate_sm90
      --path ${GENERATED_SOURCE_DIR}
      --head_dims ${HEAD_DIMS_SM90}
      --pos_encoding_modes ${FLASHINFER_GEN_POS_ENCODING_MODES}
      --use_fp16_qk_reductions ${FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS}
      --mask_modes ${FLASHINFER_GEN_MASK_MODES}
      --enable_f16 ${FLASHINFER_ENABLE_F16}
      --enable_bf16 ${FLASHINFER_ENABLE_BF16}
  )
  # cmake-format: on

  # Only regenerate at configure time if needed
  if(NEED_SOURCE_REGENERATION)
    message(STATUS "Generating kernel sources (configure time)")
    execute_process(
      COMMAND ${AOT_GENERATE_COMMAND}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE PREBUILT_URI
      RESULT_VARIABLE GEN_RESULT)
    if(NOT GEN_RESULT EQUAL 0)
      message(FATAL_ERROR "Kernel generation failed with error ${GEN_RESULT}")
    endif()

    set_property(GLOBAL PROPERTY FLASHINFER_PREBUILT_URIS "${PREBUILT_URI}")
    set(FLASHINFER_PREBUILT_URIS_CACHE
        "${PREBUILT_URI}"
        CACHE STRING
              "Cached value of FLASHINFER_PREBUILT_URIS_CACHE global property"
              FORCE)

    execute_process(
      COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      RESULT_VARIABLE DISPATCH_RESULT)
    if(NOT DISPATCH_RESULT EQUAL 0)
      message(
        FATAL_ERROR
          "Dispatch header generation failed with error ${DISPATCH_RESULT}")
    endif()

    if(FLASHINFER_ENABLE_SM90)
      message(STATUS "Generating SM90 kernel sources (configure time)")
      execute_process(
        COMMAND ${AOT_GENERATE_SM90_COMMAND}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE GEN_RESULT)
      if(NOT GEN_RESULT EQUAL 0)
        message(FATAL_ERROR "Kernel generation failed with error ${GEN_RESULT}")
      endif()
    endif()

    # Create a stamp file to track when we last generated
    execute_process(COMMAND ${CMAKE_COMMAND} -E touch "${GENERATION_STAMP}")
  endif()

  # Now we can safely glob the files (they exist at configure time)
  file(GLOB_RECURSE DECODE_KERNELS_SRCS
       ${GENERATED_SOURCE_DIR}/*decode_head*.cu)
  file(GLOB_RECURSE PREFILL_KERNELS_SRCS
       ${GENERATED_SOURCE_DIR}/*prefill_head*.cu)
  file(GLOB_RECURSE PREFILL_KERNELS_SM90_SRCS
       ${GENERATED_SOURCE_DIR}/*prefill_head*_sm90.cu)
  set(DISPATCH_INC_FILE "${GENERATED_SOURCE_DIR}/dispatch.inc")

  # Create a custom target for manual kernel regeneration
  add_custom_target(
    generate_kernels
    COMMAND ${AOT_GENERATE_COMMAND}
    COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
    COMMAND ${CMAKE_COMMAND} -E touch "${GENERATION_STAMP}"
    # After generation, force CMake to reconfigure to pick up new files
    COMMAND ${CMAKE_COMMAND} -E echo "Generated files - reconfiguration needed"
    COMMAND ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Manually generating kernel sources and reconfiguring"
    VERBATIM)

  add_custom_target(dispatch_inc DEPENDS ${DISPATCH_INC_FILE})

  # Export variables to parent scope
  set(DECODE_KERNELS_SRCS
      ${DECODE_KERNELS_SRCS}
      PARENT_SCOPE)
  set(PREFILL_KERNELS_SRCS
      ${PREFILL_KERNELS_SRCS}
      PARENT_SCOPE)
  set(DISPATCH_INC_FILE
      ${DISPATCH_INC_FILE}
      PARENT_SCOPE)
  set(PREFILL_KERNELS_SM90_SRCS
      ${PREFILL_KERNELS_SM90_SRCS}
      PARENT_SCOPE)
endfunction()
