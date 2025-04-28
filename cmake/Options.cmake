# cmake-format: off
# NOTE:
# a) Do not modify this file to change option values. Options should be
#    configured using either a config.cmake file (refer the default file
#    inside the cmake folder), or by setting the required -DFLASHINFER_XXX
#    option through command-line.
#
# b) This file should only contain option definitions and should not contain
#    any other CMake commands.
#
# c) All new options should be defined here with a default value and a short
#    description.
#
# d) Add new options under the appropriate section.

# === BUILD COMPONENT OPTIONS ===
# Core FlashInfer kernel libraries (C++ libraries used by all components)
flashinfer_option(FLASHINFER_BUILD_KERNELS
  "Build and install kernel libraries (required for AOT PyTorch extensions)" OFF)

# PyTorch CUDA extensions for Python bindings
# These provide high-performance operations through precompiled kernels
flashinfer_option(FLASHINFER_AOT_TORCH_EXTS_CUDA
  "Build ahead-of-time compiled PyTorch CUDA extensions (requires FLASHINFER_BUILD_KERNELS)" OFF)

flashinfer_option(FLASHINFER_TVM_BINDING "Build TVM binding support" OFF)
flashinfer_option(FLASHINFER_DISTRIBUTED "Build distributed support" OFF)
flashinfer_option(FLASHINFER_BUILD_WHEELS "Build distributed support" ON)

# === DATA TYPE OPTIONS ===
flashinfer_option(FLASHINFER_ENABLE_FP8 "Enable FP8 data type support" ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E4M3 "Enable FP8 E4M3 format specifically" ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E5M2 "Enable FP8 E5M2 format specifically" ON)
flashinfer_option(FLASHINFER_ENABLE_F16 "Enable F16 data type support" ON)
flashinfer_option(FLASHINFER_ENABLE_BF16 "Enable BF16 data type support" ON)

# === CODE GENERATION OPTIONS ===
flashinfer_option(FLASHINFER_GEN_HEAD_DIMS "Head dimensions to enable" 128 256)
flashinfer_option(FLASHINFER_GEN_POS_ENCODING_MODES "Position encoding modes to enable" 0)
flashinfer_option(FLASHINFER_GEN_MASK_MODES "Mask modes to enable" 0 1 2)
flashinfer_option(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS "Use FP16 for QK reductions" OFF)
flashinfer_option(FLASHINFER_SM90_ALLOWED_HEAD_DIMS "64,64" "128,128" "256,256" "192,128")

# === BUILD TYPE OPTIONS ===
flashinfer_option(FLASHINFER_UNITTESTS "Build unit tests" OFF)
flashinfer_option(FLASHINFER_CXX_BENCHMARKS "Build benchmarks" OFF)
flashinfer_option(FLASHINFER_DIST_UNITTESTS "Build distributed unit tests" OFF)

# === VERSION OPTIONS ===
# Custom version suffix for builds
flashinfer_option(FLASHINFER_VERSION_SUFFIX "Custom version suffix for builds" "")

# === FEATURE-SPECIFIC TESTS/BENCHMARKS ===
flashinfer_option(FLASHINFER_FP8_TESTS "Build FP8 tests" OFF)
flashinfer_option(FLASHINFER_FP8_BENCHMARKS "Build FP8 benchmarks" OFF)

# === ARCHITECTURE OPTIONS ===
flashinfer_option(FLASHINFER_CUDA_ARCHITECTURES "CUDA architectures to compile for" "")
flashinfer_option(FLASHINFER_MIN_CUDA_ARCH "Minimum CUDA architecture required (SM_XX)" 75)

# === PATH OPTIONS ===
flashinfer_option(FLASHINFER_CUTLASS_DIR "Path to CUTLASS installation" "")
flashinfer_option(FLASHINFER_TVM_SOURCE_DIR "Path to TVM source directory" "")

# === COMPILER OPTIONS ===
# Control the C++ ABI for PyTorch compatibility
flashinfer_option(FLASHINFER_USE_CXX11_ABI "Use the C++11 ABI for PyTorch compatibility" OFF)

# === PYTHON OPTIONS ===
flashinfer_option(FLASHINFER_PY_LIMITED_API "Use Python's limited API for better version compatibility" ON)
flashinfer_option(FLASHINFER_MIN_PYTHON_ABI "Minimum Python ABI version for limited API compatibility" "3.9")

# === AUTO-DERIVED OPTIONS ===
# Handle CUDA architectures
if(FLASHINFER_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES set to ${FLASHINFER_CUDA_ARCHITECTURES}.")
else()
  # No user-provided architectures, try to detect the CUDA archs based on where
  # the project is being built
  set(detected_archs "")
  detect_cuda_architectures(detected_archs)
  if(detected_archs)
    set(FLASHINFER_CUDA_ARCHITECTURES ${detected_archs} CACHE STRING
        "CUDA architectures" FORCE)
    message(STATUS "Setting FLASHINFER_CUDA_ARCHITECTURES to detected values: ${FLASHINFER_CUDA_ARCHITECTURES}")
  else()
    # No architectures detected, use safe defaults
    set(FLASHINFER_CUDA_ARCHITECTURES "75;80;86" CACHE STRING
        "CUDA architectures to compile for" FORCE)
    message(STATUS "No architectures detected, using defaults: ${FLASHINFER_CUDA_ARCHITECTURES}")
  endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES ${FLASHINFER_CUDA_ARCHITECTURES})

# Derive SM90 support automatically from CUDA architectures
set(FLASHINFER_ENABLE_SM90 OFF CACHE INTERNAL "SM90 architecture support enabled")

if(FLASHINFER_CUDA_ARCHITECTURES)
  string(REGEX MATCH "(^|;)90($|;|a)" FOUND_SM90 "${FLASHINFER_CUDA_ARCHITECTURES}")
  if(FOUND_SM90)
    set(FLASHINFER_ENABLE_SM90 ON CACHE INTERNAL "SM90 architecture support enabled")
    message(STATUS "Enabling SM90-specific optimizations based on CUDA architecture selection")
  endif()
endif()

# Handle automatic enabling of dependent features
if(FLASHINFER_FP8_TESTS)
  set(FLASHINFER_UNITTESTS ON CACHE BOOL "Tests enabled for FP8" FORCE)
endif()

if(FLASHINFER_FP8_BENCHMARKS)
  set(FLASHINFER_CXX_BENCHMARKS ON CACHE BOOL "Benchmarks enabled for FP8" FORCE)
endif()

if(FLASHINFER_DIST_UNITTESTS)
  set(FLASHINFER_UNITTESTS ON CACHE BOOL "Tests enabled for distributed" FORCE)
endif()

if(FLASHINFER_TVM_BINDING AND NOT FLASHINFER_BUILD_KERNELS)
  message(FATAL_ERROR "TVM binding requires FLASHINFER_BUILD_KERNELS to be ON")
endif()

if(FLASHINFER_ENABLE_FP8)
  # Enable both FP8 formats when FP8 is enabled
  set(FLASHINFER_ENABLE_FP8_E4M3 ON CACHE BOOL "Enable FP8 E4M3 format" FORCE)
  set(FLASHINFER_ENABLE_FP8_E5M2 ON CACHE BOOL "Enable FP8 E5M2 format" FORCE)
endif()

# Ensure FP8 is enabled for FP8 tests/benchmarks
if(FLASHINFER_FP8_TESTS OR FLASHINFER_FP8_BENCHMARKS)
  set(FLASHINFER_ENABLE_FP8 ON CACHE BOOL "FP8 enabled for tests/benchmarks" FORCE)
  set(FLASHINFER_ENABLE_FP8_E4M3 ON CACHE BOOL "FP8_E4M3 enabled for tests/benchmarks" FORCE)
endif()

# PyTorch extensions require kernels to be built
if(FLASHINFER_AOT_TORCH_EXTS_CUDA AND NOT FLASHINFER_BUILD_KERNELS)
  message(STATUS "FLASHINFER_AOT_TORCH_EXTS_CUDA requires FLASHINFER_BUILD_KERNELS, enabling it")
  set(FLASHINFER_BUILD_KERNELS ON CACHE BOOL "Build kernels (required by PyTorch extensions)" FORCE)
endif()
# cmake-format: on
