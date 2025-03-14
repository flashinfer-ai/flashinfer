# === Required Dependencies for Core Functionality ===
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(
    FATAL_ERROR
      "Python3 not found it is required to generate the kernel sources.")
endif()

find_package(Thrust REQUIRED)

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

# === CUTLASS Configuration ===
if(FLASHINFER_CUTLASS_DIR)
  list(APPEND CMAKE_PREFIX_PATH ${FLASHINFER_CUTLASS_DIR})
endif()

if(FLASHINFER_CUTLASS_DIR)
  # Add CUTLASS include directories directly
  include_directories(${FLASHINFER_CUTLASS_DIR}/include)
  include_directories(${FLASHINFER_CUTLASS_DIR}/tools/util/include)

  message(STATUS "Using CUTLASS from ${FLASHINFER_CUTLASS_DIR}")
else()
  message(
    FATAL_ERROR "FLASHINFER_CUTLASS_DIR must be set to the path of CUTLASS")
endif()
