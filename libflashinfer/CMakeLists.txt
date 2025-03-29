# Set global paths and directories
set(FLASHINFER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FLASHINFER_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(CUTLASS_INCLUDE_DIRS ${NvidiaCutlass_DIR}/include
                         ${NvidiaCutlass_DIR}/tools/util/include)

# ==== RPATH CONFIGURATION FOR C++ LIBRARY ====
# Use standard GNU installation directories
include(GNUInstallDirs)

# Calculate relative path from bin to lib directory
file(RELATIVE_PATH _rel_path "${CMAKE_INSTALL_FULL_BINDIR}"
     "${CMAKE_INSTALL_FULL_LIBDIR}")

# Set RPATH for core library targets
set(CMAKE_INSTALL_RPATH "${FLASHINFER_RPATH_BASE}/${_rel_path}")
# Add the absolute library path as a fallback for third-party applications
list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")

# Apply data type definitions based on enabled options
if(FLASHINFER_ENABLE_FP8_E4M3)
  message(STATUS "Compile fp8_e4m3 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8_E4M3)
endif()

if(FLASHINFER_ENABLE_FP8_E5M2)
  message(STATUS "Compile fp8_e5m2 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8_E5M2)
endif()

if(FLASHINFER_ENABLE_BF16)
  message(STATUS "Compile bf16 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_BF16)
endif()

# FP16 QK Reductions support
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  add_definitions(-DFP16_QK_REDUCTION_SUPPORTED)
endif()

# generate kernel inst
set(HEAD_DIMS ${FLASHINFER_GEN_HEAD_DIMS})
set(POS_ENCODING_MODES ${FLASHINFER_GEN_POS_ENCODING_MODES})
set(MASK_MODES ${FLASHINFER_GEN_MASK_MODES})
set(USE_FP16_QK_REDUCTIONS ${FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS})

# log options
message(STATUS "FLASHINFER_HEAD_DIMS=${HEAD_DIMS}")
message(STATUS "FLASHINFER_POS_ENCODING_MODES=${POS_ENCODING_MODES}")
message(
  STATUS "FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
message(STATUS "FLASHINFER_MASK_MODES=${MASK_MODES}")

# ----------------------- SM90 head dims computation ------------------------#
# Include the logic to caclulate the head dims for SM90
include(CalculateSM90HeadDims)
set(HEAD_DIMS_SM90 "")
flashinfer_compute_sm90_head_dims(RESULT HEAD_DIMS_SM90)
# Log SM90_ALLOWED_HEAD_DIMS and HEAD_DIMS_SM90
message(STATUS "SM90_ALLOWED_HEAD_DIMS=${FLASHINFER_SM90_ALLOWED_HEAD_DIMS}")
message(STATUS "HEAD_DIMS_SM90=${HEAD_DIMS_SM90}")
# ---------------------------------------------------------------------------#

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
    --head_dims ${HEAD_DIMS}
    --pos_encoding_modes ${POS_ENCODING_MODES}
    --use_fp16_qk_reductions ${USE_FP16_QK_REDUCTIONS}
    --mask_modes ${MASK_MODES}
    --enable_f16 ${FLASHINFER_ENABLE_F16}
    --enable_bf16 ${FLASHINFER_ENABLE_BF16}
    --enable_fp8_e4m3 ${FLASHINFER_ENABLE_FP8_E4M3}
    --enable_fp8_e5m2 ${FLASHINFER_ENABLE_FP8_E5M2})

# Command to generate dispatch include file
set(AOT_GENERATE_DISPATCH_INC_COMMAND
    ${Python3_EXECUTABLE} -m aot_build_utils.generate_dispatch_inc
    --path "${GENERATED_SOURCE_DIR}/dispatch.inc"
    --head_dims ${HEAD_DIMS}
    --head_dims_sm90 ${HEAD_DIMS_SM90}
    --pos_encoding_modes ${POS_ENCODING_MODES}
    --use_fp16_qk_reductions ${USE_FP16_QK_REDUCTIONS}
    --mask_modes ${MASK_MODES})
# cmake-format: on

# Only regenerate at configure time if needed
if(NEED_SOURCE_REGENERATION)
  message(STATUS "Generating kernel sources (configure time)")
  execute_process(
    COMMAND ${AOT_GENERATE_COMMAND}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE GEN_RESULT)
  if(NOT GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Kernel generation failed with error ${GEN_RESULT}")
  endif()

  execute_process(
    COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE DISPATCH_RESULT)
  if(NOT DISPATCH_RESULT EQUAL 0)
    message(
      FATAL_ERROR
        "Dispatch header generation failed with error ${DISPATCH_RESULT}")
  endif()

  # Create a stamp file to track when we last generated
  execute_process(COMMAND ${CMAKE_COMMAND} -E touch "${GENERATION_STAMP}")
endif()

# Now we can safely glob the files (they exist at configure time)
file(GLOB_RECURSE DECODE_KERNELS_SRCS ${GENERATED_SOURCE_DIR}/*decode_head*.cu)
file(GLOB_RECURSE PREFILL_KERNELS_SRCS
     ${GENERATED_SOURCE_DIR}/*prefill_head*.cu)
file(GLOB_RECURSE DISPATCH_INC_FILE ${GENERATED_SOURCE_DIR}/dispatch.inc)

# A custom target for manual kernel regeneration
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

# Build decode_kernels and prefill_kernels if FLASHINFER_BUILD_KERNELS is ON or
# if tests or benchmarks are enabled
if(FLASHINFER_BUILD_KERNELS
   OR FLASHINFER_UNITTESTS
   OR FLASHINFER_CXX_BENCHMARKS)
  add_library(decode_kernels STATIC ${DECODE_KERNELS_SRCS})
  add_library(flashinfer::decode_kernels ALIAS decode_kernels)
  target_include_directories(decode_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
  if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
    target_link_libraries(decode_kernels PRIVATE Boost::math)
  endif()

  add_library(prefill_kernels STATIC ${PREFILL_KERNELS_SRCS})
  add_library(flashinfer::prefill_kernels ALIAS prefill_kernels)
  target_include_directories(prefill_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
  if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
    target_link_libraries(prefill_kernels PRIVATE Boost::math)
  endif()

  # Add targets to export set
  install(
    TARGETS decode_kernels
    EXPORT libflashinferTargets
    COMPONENT Kernels)
  install(
    TARGETS prefill_kernels
    EXPORT libflashinferTargets
    COMPONENT Kernels)

  # Export targets from the build tree
  export(
    EXPORT libflashinferTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/libflashinferTargets.cmake"
    NAMESPACE flashinfer::)

  # Install the export set for use with the install-tree
  install(
    EXPORT libflashinferTargets
    FILE libflashinferTargets.cmake
    NAMESPACE flashinfer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer
    COMPONENT Kernels)
endif()

# Compile Flash Attention 3 libraries if FP8 unit tests are enabled
if(FLASHINFER_FP8_TESTS OR FLASHINFER_FP8_BENCHMARKS)
  add_library(FA3_LIB STATIC ${FA3_IMPL_FILES})
  target_compile_options(
    FA3_LIB
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-extended-lambda --use_fast_math
            --compiler-options -fPIC
            --generate-code=arch=compute_90a,code=sm_90a>)
  target_include_directories(FA3_LIB PRIVATE ${CUTLASS_INCLUDE_DIRS}
                                             ${FA3_INCLUDE_DIR})
endif()

# Install all headers (always installed) with Headers component except the
# distributed directory as it is installed separately
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Headers
  FILES_MATCHING
  REGEX "\\.(cuh|h|hpp)$"
  PATTERN "distributed" EXCLUDE)

# Install Boost::math headers with Headers component
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  install(
    DIRECTORY ${boost_math_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
    COMPONENT Headers
    FILES_MATCHING
    REGEX "\\.(h|hpp)$")
endif()

# Install libraries if kernels are built
if(FLASHINFER_BUILD_KERNELS)
  # Install libraries
  install(
    TARGETS decode_kernels prefill_kernels
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Kernels)
endif()

# Install distributed headers if enabled
if(FLASHINFER_DISTRIBUTED)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/flashinfer/distributed/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flashinfer/distributed
    COMPONENT Distributed
    FILES_MATCHING
    REGEX "\\.(cuh|h|hpp)$")

  # Install mscclpp headers with Distributed component
  install(
    DIRECTORY ${mscclpp_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Distributed
    FILES_MATCHING
    REGEX "\\.(h|hpp|cuh)$")

  # Export the target files
  export(
    EXPORT flashinferDistTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/flashinferDistTargets.cmake"
    NAMESPACE flashinfer::)

  # Install the export set with proper component
  install(
    EXPORT flashinferDistTargets
    FILE flashinferDistTargets.cmake
    NAMESPACE flashinfer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer
    COMPONENT Distributed)
endif()

# Add test and benchmark subdirectories if enabled
if(FLASHINFER_UNITTESTS)
  add_subdirectory(tests)
endif()

if(FLASHINFER_CXX_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
