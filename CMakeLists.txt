cmake_minimum_required(VERSION 3.26.1)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Load CMake modules and utilities
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils")

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils")

# Include versioning and utilities
include(ProjectVersion)
include(Utils)
find_package(Git REQUIRED)
set_current_release()

project(
  flashinfer
  LANGUAGES CXX
  VERSION ${PROJECT_VERSION}
  DESCRIPTION "Fast Attention Algorithms for LLM Inference")

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==== RPATH CONFIGURATION ====
# Set Linux RPATH base
set(FLASHINFER_RPATH_BASE "$ORIGIN")

# Always build with install RPATH to avoid needs for LD_LIBRARY_PATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Include the link path in RPATH by default for better dependency resolution
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Skip the build tree rpath - we only care about install rpath
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)

# cmake-format: off
# Compiler flags - defined as lists for cleaner management
set(WARNING_FLAGS
    "-Wall"
    "-Wextra"
    "-Winit-self"
    "-Wno-switch-bool"
    "-Wunused-function"
    "-Wuninitialized"
    "-Wmissing-declarations"
    "-Wno-unused-parameter"
    "-fdiagnostics-color=auto")

set(SECURITY_FLAGS
    "-fstack-protector"
    "-fstack-protector-all"
    "-fpic"
    "-fPIC"
    "-D_FORTIFY_SOURCE=2"
    "-Wformat"
    "-Wformat-security"
    "-fno-delete-null-pointer-checks"
    "-fwrapv")

# cmake-format: on

# Join the lists with spaces to form the flag strings
string(JOIN " " WARNING_FLAGS ${WARNING_FLAGS})
string(JOIN " " SECURITY_FLAGS ${SECURITY_FLAGS})

# Combine warning and security flags for C++ compiler
string(CONCAT FLASHINFER_CXXFLAGS "${WARNING_FLAGS} " "${SECURITY_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLASHINFER_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} ${FLASHINFER_CXXFLAGS} -O0 -ggdb3 -DDEBUG")

# Load custom config if available
if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
elseif(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
  include(${CMAKE_SOURCE_DIR}/config.cmake)
endif()

include(GNUInstallDirs)

# Include consolidated option definitions
include(cmake/Options.cmake)

# Setup dependencies based on enabled options
include(cmake/Dependencies.cmake)

# Define components for build and installation
include(cmake/Components.cmake)

# Translate all data type configure options into preprocessor definitions
if(FLASHINFER_ENABLE_FP8)
  add_compile_definitions(FLASHINFER_ENABLE_FP8)

  # Pass the FP8 format flags if enabled
  if(FLASHINFER_ENABLE_FP8_E4M3)
    add_compile_definitions(FLASHINFER_ENABLE_FP8_E4M3)
  endif()

  if(FLASHINFER_ENABLE_FP8_E5M2)
    add_compile_definitions(FLASHINFER_ENABLE_FP8_E5M2)
  endif()
endif()

# Also ensure F16 and BF16 flags are set
if(FLASHINFER_ENABLE_BF16)
  add_compile_definitions(FLASHINFER_ENABLE_BF16)
endif()

if(FLASHINFER_ENABLE_F16)
  add_compile_definitions(FLASHINFER_ENABLE_F16)
endif()

if(FLASHINFER_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CXX_STANDARD 20)
  message(STATUS "CUDA language enabled")
  include(ConfigureCUDAArchitectures)

  # cmake-format: off
  set(FLASHINFER_NVCC_FLAGS
    "-O3"
    "--threads=1"
    "-Xfatbin=-compress-all"
    "-use_fast_math"
    "--expt-relaxed-constexpr"
    "-lineinfo"
    "-Xptxas=-v")
  # cmake-format: on

  # Validate that FLASHINFER_CUDA_ARCHITECTURES contains only supported values.
  # Only SM75 and later are supported.
  set(FLASHINFER_MINIMUM_SUPPORTED_CUDA_ARCH 75)
  validate_arch_list_against_min_supported_arch(
    FLASHINFER_CUDA_ARCHITECTURES FLASHINFER_MINIMUM_SUPPORTED_CUDA_ARCH)

  # Update the FLASHINFER_NVCC_FLAGS with the gencode flags based on the
  # FLASHINFER_CUDA_ARCHITECTURES list.
  generate_cuda_arch_flags("${FLASHINFER_CUDA_ARCHITECTURES}" cuda_arch_flags)
  list(APPEND FLASHINFER_NVCC_FLAGS ${cuda_arch_flags})
  string(JOIN " " FLASHINFER_NVCC_FLAGS_STR ${FLASHINFER_NVCC_FLAGS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${FLASHINFER_NVCC_FLAGS_STR}")
  set(CMAKE_CUDA_FLAGS_DEBUG
      "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -g -lineinfo -DDEBUG")
endif()

if(FLASHINFER_ENABLE_HIP)
  enable_language(HIP)
  find_package(HIP REQUIRED)
  message(STATUS "HIP language enabled")
  include(ConfigureHIPArchitectures)

  # cmake-format: off
  set(FLASHINFER_HIPCC_FLAGS
    "-O3"
    "-ffast-math"
    "-Wno-deprecated-declarations"
    "--offload-compress"
    "-fclang-abi-compat=17"
    "-DCUDA_HAS_FP16=1"
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1"
    "-D__HIP_PLATFORM_AMD__=1"
    "-D__HIP_ROCclr__=1"
    "-fno-gpu-rdc"
  )
  # cmake-format: on

  if(NOT FLASHINFER_HIP_ARCHITECTURES OR FLASHINFER_HIP_ARCHITECTURES STREQUAL
                                         "")
    set(detected_hip_archs "")
    detect_hip_architectures(detected_hip_archs)

    if(detected_hip_archs)
      set(FLASHINFER_HIP_ARCHITECTURES
          ${detected_hip_archs}
          CACHE STRING "HIP architectures" FORCE)
      message(
        STATUS
          "Auto-detected HIP architectures: ${FLASHINFER_HIP_ARCHITECTURES}")
    else()
      # No architectures detected, use safe default
      set(FLASHINFER_HIP_ARCHITECTURES
          "gfx942"
          CACHE STRING "HIP architectures" FORCE)
      message(
        STATUS "Failed to auto-detect HIP architectures, using default: gfx942")
    endif()
  else()
    message(
      STATUS
        "Using user-specified HIP architectures: ${FLASHINFER_HIP_ARCHITECTURES}"
    )
  endif()

  # Validate that FLASHINFER_HIP_ARCHITECTURES contains only supported values.
  # Only gfx942 is supported for now.
  set(FLASHINFER_SUPPORTED_HIP_ARCH "gfx942")

  foreach(arch ${FLASHINFER_HIP_ARCHITECTURES})
    if(NOT arch IN_LIST FLASHINFER_SUPPORTED_HIP_ARCH)
      message(
        FATAL_ERROR
          "Unsupported HIP architecture: ${arch} only gfx942 is supported")
    endif()
  endforeach()

  # Generate HIP arch flags
  generate_hip_arch_flags("${FLASHINFER_HIP_ARCHITECTURES}" hip_arch_flags)
  message(STATUS "HIP architecture flags: ${hip_arch_flags}")
  set(CMAKE_HIP_ARCHITECTURES
      ${FLASHINFER_HIP_ARCHITECTURES}
      CACHE STRING "HIP architectures for all targets" FORCE)

  foreach(flag ${hip_arch_flags})
    list(APPEND FLASHINFER_HIPCC_FLAGS ${flag})
  endforeach()
  string(JOIN " " FLASHINFER_HIPCC_FLAGS_STR ${FLASHINFER_HIPCC_FLAGS})
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${FLASHINFER_HIPCC_FLAGS_STR}")
  set(CMAKE_HIP_FLAGS_DEBUG
      "${CMAKE_HIP_FLAGS_DEBUG} -O0 -g2 -DDEBUG -fdebug-info-for-profiling")

  set(CMAKE_HIP_ARCHITECTURES
      "gfx942"
      CACHE STRING "HIP architectures for all targets" FORCE)

endif()

# Add library subdirectories
add_subdirectory(libflashinfer)

if(FLASHINFER_TVM_BINDING)
  add_subdirectory(tvm_binding)
endif()

if(FLASHINFER_BUILD_WHEELS)
  message(STATUS "Building Flashinfer PyTorch extension")
  add_subdirectory(flashinfer)
endif()

# Generate the flashinferConfig.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/flashinferConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer)

if(NOT FLASHINFER_BUILD_WHEELS)
  # Install CMake configuration
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer
    COMPONENT Headers)
endif()
