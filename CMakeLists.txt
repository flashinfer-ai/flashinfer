cmake_minimum_required(VERSION 3.26.1)

# Load CMake modules and utilities
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils")

# Include versioning and utilities
include(ProjectVersion)
include(Utils)
find_package(Git REQUIRED)
set_current_release()

project(
  flashinfer
  LANGUAGES CUDA CXX
  VERSION ${PROJECT_VERSION}
  DESCRIPTION "Fast Attention Algorithms for LLM Inference")

set(CMAKE_CXX_STANDARD 20)
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

set(FLASHINFER_NVCC_FLAGS
    "-O3"
    "--threads=1"
    "-Xfatbin=-compress-all"
    "-use_fast_math"
    "--expt-relaxed-constexpr"
    "-lineinfo"
    "-Xptxas=-v")
# cmake-format: on

# Join the lists with spaces to form the flag strings
string(JOIN " " WARNING_FLAGS ${WARNING_FLAGS})
string(JOIN " " SECURITY_FLAGS ${SECURITY_FLAGS})
string(JOIN " " FLASHINFER_NVCC_FLAGS ${FLASHINFER_NVCC_FLAGS})

# Combine warning and security flags for C++ compiler
string(CONCAT FLASHINFER_CXXFLAGS "${WARNING_FLAGS} " "${SECURITY_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLASHINFER_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} ${FLASHINFER_CXXFLAGS} -O0 -ggdb3 -DDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${FLASHINFER_NVCC_FLAGS}")

# Load custom config if available
if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
elseif(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
  include(${CMAKE_SOURCE_DIR}/config.cmake)
endif()

include(GNUInstallDirs)

include(ConfigureCUDAArchitectures)

# Include consolidated option definitions
include(cmake/Options.cmake)

# Setup dependencies based on enabled options
include(cmake/Dependencies.cmake)

# Define components for build and installation
include(cmake/Components.cmake)

# Validate that FLASHINFER_CUDA_ARCHITECTURES contains only supported values.
# Only SM75 and later are supported.
set(FLASHINFER_MINIMUM_SUPPORTED_CUDA_ARCH 75)
validate_arch_list_against_min_supported_arch(
  FLASHINFER_CUDA_ARCHITECTURES FLASHINFER_MINIMUM_SUPPORTED_CUDA_ARCH)

# Update the FLASHINFER_NVCC_FLAGS with the gencode flags based on the
# FLASHINFER_CUDA_ARCHITECTURES list.
generate_cuda_arch_flags("${FLASHINFER_CUDA_ARCHITECTURES}" cuda_arch_flags)
list(APPEND FLASHINFER_NVCC_FLAGS ${cuda_arch_flags})

# Add library subdirectories
add_subdirectory(libflashinfer)

if(FLASHINFER_TVM_BINDING)
  add_subdirectory(tvm_binding)
endif()

if(FLASHINFER_BUILD_WHEELS)
  message(STATUS "Building PyTorch CUDA extensions")
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
