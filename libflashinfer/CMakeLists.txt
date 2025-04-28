# Set global paths and directories
set(FLASHINFER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FLASHINFER_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)

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
# ---------------------- Kernels and dispatch.inc generation ----------------#
# The function does the following: a) Generates the kernels and dispatch.inc
# files b) Sets the generate_kernels and generate_dispatch_inc targets c) Sets
# the DECODE_KERNELS_SRCS, PREFILL_KERNELS_SRCS, and DISPATCH_INC_FILE
# variables.
include(ConfigureKernelGeneration)
flashinfer_configure_kernel_generation()
# ---------------------------------------------------------------------------#

# Set the install path for the libflashinfer headers based on whether the build
# is for a Python Wheels package or for standalone libflashinfer.

if(FLASHINFER_BUILD_WHEELS)
  # When building wheels, install headers inside the Python package
  set(_FLASHINFER_INCLUDE_INSTALL_DIR
      "${SKBUILD_PLATLIB_DIR}/flashinfer/include")
else()
  # For standalone installation, use standard paths
  set(_FLASHINFER_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

# ---------------------- Generate configure.h -------------------------------#
# cmake-format: off
include(${CMAKE_SOURCE_DIR}/cmake/utils/GenerateConfigHeader.cmake)
flashinfer_generate_config_header(
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  INSTALL_DIR ${_FLASHINFER_INCLUDE_INSTALL_DIR}/flashinfer
  COMPONENT Headers
  EXCLUDE_PATTERNS
    ".*_DIR$"
    ".*_PATH$"
    ".*_FOUND$"
    ".*_FILE$"
    ".*_VERSION$"
)
# cmake-format: on
# ---------------------------------------------------------------------------#

# Build decode_kernels and prefill_kernels if needed
if(FLASHINFER_BUILD_KERNELS
   OR FLASHINFER_UNITTESTS
   OR FLASHINFER_CXX_BENCHMARKS)

  set(FLASHINFER_KERNEL_TARGETS decode_kernels prefill_kernels)

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

  if(FLASHINFER_ENABLE_SM90)
    list(APPEND FLASHINFER_KERNEL_TARGETS prefill_kernels_sm90)
    add_library(prefill_kernels_sm90 STATIC ${PREFILL_KERNELS_SM90_SRCS})
    add_library(flashinfer::prefill_kernels_sm90 ALIAS prefill_kernels_sm90)
    target_include_directories(prefill_kernels_sm90
                               PRIVATE ${FLASHINFER_INCLUDE_DIR})
    target_compile_options(
      prefill_kernels_sm90
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
              --generate-code=arch=compute_90a,code=sm_90a>)
    if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
      target_link_libraries(prefill_kernels_sm90 PRIVATE Boost::math)
    endif()
  endif()

  # Only for non-wheel builds: Install libraries and export targets
  if(NOT FLASHINFER_BUILD_WHEELS)
    install(
      TARGETS ${FLASHINFER_KERNEL_TARGETS}
      EXPORT libflashinferTargets
      COMPONENT Kernels
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      INCLUDES
      DESTINATION ${_FLASHINFER_INCLUDE_INSTALL_DIR})

    # Export targets from the build tree
    export(
      EXPORT libflashinferTargets
      FILE "${CMAKE_CURRENT_BINARY_DIR}/libflashinferTargets.cmake"
      NAMESPACE flashinfer::)

    # Install the export set
    install(
      EXPORT libflashinferTargets
      FILE libflashinferTargets.cmake
      NAMESPACE flashinfer::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer
      COMPONENT Kernels)
  endif()
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
  DESTINATION ${_FLASHINFER_INCLUDE_INSTALL_DIR}
  COMPONENT Headers
  FILES_MATCHING
  REGEX "\\.(cuh|h|hpp)$"
  PATTERN "distributed" EXCLUDE)

# Install Boost::math headers with Headers component
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  install(
    DIRECTORY ${boost_math_SOURCE_DIR}/include/
    DESTINATION ${_FLASHINFER_INCLUDE_INSTALL_DIR}/
    COMPONENT Headers
    FILES_MATCHING
    REGEX "\\.(h|hpp)$")
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
    DESTINATION ${_FLASHINFER_INCLUDE_INSTALL_DIR}
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
