# === Required Dependencies for Core Functionality ===
if(FLASHINFER_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(Thrust REQUIRED)
endif()

# === HIP Dependencies ===
if(FLASHINFER_ENABLE_HIP)
  # Check for HIP
  include(ConfigureRocmPath)
  find_package(HIP REQUIRED)
  message(STATUS "Found HIP: ${HIP_VERSION}")
endif()

find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(
    FATAL_ERROR
      "Python3 not found it is required to generate the kernel sources.")
endif()

# === Test Dependencies ===
if(FLASHINFER_UNITTESTS)
  include(FetchContent)

  # Google Test for unit testing
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5 # release-1.16.0
    FIND_PACKAGE_ARGS NAMES GTest)
  FetchContent_MakeAvailable(googletest)
endif()

# === Benchmark Dependencies ===
if(FLASHINFER_CXX_BENCHMARKS)
  include(FetchContent)

  # NVBench for GPU benchmarking
  FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG c03033b50e46748207b27685b1cdfcbe4a2fec59)
  FetchContent_MakeAvailable(nvbench)
endif()

# === Boost Dependency for FP16 QK Reductions ===
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  include(FetchContent)
  set(BOOST_ENABLE_CMAKE ON)
  FetchContent_Declare(boost_math
                       GIT_REPOSITORY https://github.com/boostorg/math.git)
  FetchContent_MakeAvailable(boost_math)

  set(USE_FP16_QK_REDUCTIONS "true")
  message(STATUS "USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
else()
  set(USE_FP16_QK_REDUCTIONS "false")
  message(STATUS "USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
endif()

# === Distributed component dependencies ===
if(FLASHINFER_DISTRIBUTED OR FLASHINFER_DIST_UNITTESTS)
  include(FetchContent)
  FetchContent_Declare(
    mscclpp
    GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
    GIT_TAG 11e62024d3eb190e005b4689f8c8443d91a6c82e)
  FetchContent_MakeAvailable(mscclpp)

  # Create alias for distributed component
  if(NOT TARGET flashinfer::mscclpp)
    add_library(flashinfer::mscclpp ALIAS mscclpp)
  endif()

  # Fetch spdlog for distributed tests (header-only usage)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG f355b3d58f7067eee1706ff3c801c2361011f3d5 # release-1.15.1
    FIND_PACKAGE_ARGS NAMES spdlog)

  # Use Populate instead of MakeAvailable since we only need the headers
  FetchContent_Populate(spdlog)

  # Set the include directory for later use
  set(SPDLOG_INCLUDE_DIR "${spdlog_SOURCE_DIR}/include")
  message(STATUS "Using spdlog from ${SPDLOG_INCLUDE_DIR}")

  find_package(MPI REQUIRED)
endif()

# === FP8 Dependencies ===
if(FLASHINFER_FP8_TESTS OR FLASHINFER_FP8_BENCHMARKS)
  # Verify CUDA architecture is SM90 or higher
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "90"
     AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "90a")
    message(
      FATAL_ERROR "FP8 tests/benchmarks require SM90 or higher architecture")
  endif()

  # Find PyTorch which is required for FP8 features
  find_package(Torch REQUIRED)
  if(NOT Torch_FOUND)
    message(
      FATAL_ERROR "PyTorch is required for FP8 tests/benchmarks but not found")
  endif()
  message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")

  # Fetch Flash Attention repository with specific commit
  include(FetchContent)
  FetchContent_Declare(
    flash_attention
    GIT_REPOSITORY https://github.com/Dao-AILab/flash-attention.git
    GIT_TAG 29ef580560761838c0e9e82bc0e98d04ba75f949)
  FetchContent_Populate(flash_attention)

  # Set Flash Attention 3 include directory
  set(FA3_INCLUDE_DIR "${flash_attention_SOURCE_DIR}/csrc/flash_attn/hopper")
  message(STATUS "Flash Attention 3 source directory: ${FA3_INCLUDE_DIR}")

  # Compile Flash Attention 3 kernel library
  file(GLOB FA3_IMPL_FILES "${FA3_INCLUDE_DIR}/flash_fwd_*.cu")
endif()

# === TVM Binding dependencies ===
if(FLASHINFER_TVM_BINDING)
  # Resolve TVM source directory
  if(NOT FLASHINFER_TVM_SOURCE_DIR STREQUAL "")
    set(TVM_SOURCE_DIR_SET ${FLASHINFER_TVM_SOURCE_DIR})
  elseif(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR_SET $ENV{TVM_SOURCE_DIR})
  elseif(DEFINED ENV{TVM_HOME})
    set(TVM_SOURCE_DIR_SET $ENV{TVM_HOME})
  else()
    message(
      FATAL_ERROR
        "TVM source directory not found. Set FLASHINFER_TVM_SOURCE_DIR.")
  endif()
endif()

# === Path definitions ===
# Define all include paths centrally - don't use global include_directories

# FlashInfer internal paths
set(FLASHINFER_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/libflashinfer/include"
    CACHE INTERNAL "FlashInfer include directory")

set(FLASHINFER_UTILS_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/libflashinfer/utils"
    CACHE INTERNAL "FlashInfer utilities include directory")

# Generated code paths
set(FLASHINFER_GENERATED_SOURCE_DIR
    "${CMAKE_BINARY_DIR}/libflashinfer/src/generated"
    CACHE INTERNAL "FlashInfer generated source directory")

set(FLASHINFER_GENERATED_SOURCE_DIR_ROOT
    "${CMAKE_BINARY_DIR}/libflashinfer/src"
    CACHE INTERNAL "FlashInfer generated source root directory")

# === CUTLASS Configuration ===
if(FLASHINFER_ENABLE_CUDA)
  if(FLASHINFER_CUTLASS_DIR)
    if(IS_ABSOLUTE ${FLASHINFER_CUTLASS_DIR})
      set(CUTLASS_DIR ${FLASHINFER_CUTLASS_DIR})
    else()
      set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/${FLASHINFER_CUTLASS_DIR}")
    endif()

    list(APPEND CMAKE_PREFIX_PATH ${CUTLASS_DIR})
    set(CUTLASS_INCLUDE_DIRS
        "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include"
        CACHE INTERNAL "CUTLASS include directories")

    message(STATUS "CUTLASS include directories: ${CUTLASS_INCLUDE_DIRS}")
  else()
    message(
      FATAL_ERROR "FLASHINFER_CUTLASS_DIR must be set to the path of CUTLASS")
  endif()
endif()

# === Python dependencies for PyTorch extensions ===
if(FLASHINFER_AOT_TORCH_EXTS)
  find_package(
    Python
    COMPONENTS Interpreter Development.Module
    REQUIRED)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c"
            "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PREFIX})

  if(FLASHINFER_ENABLE_CUDA)
    find_package(CUDA)
  endif()

  # Find PyTorch
  find_package(Torch REQUIRED)

  # Report found versions
  message(STATUS "Found Python: ${Python_VERSION}")
  message(STATUS "Found PyTorch: ${TORCH_VERSION}")

  # pybind11 for core module
  if(NOT TARGET pybind11::module)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.11.1)
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()
