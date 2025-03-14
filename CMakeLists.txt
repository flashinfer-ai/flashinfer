cmake_minimum_required(VERSION 3.23.1)

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
  VERSION ${PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# cmake-format: off
# Compiler flags
string(
  CONCAT WARNING_FLAGS
         "-Wall "
         "-Wextra "
         "-Winit-self "
         "-Wno-switch-bool "
         "-Wunused-function "
         "-Wuninitialized "
         "-Wmissing-declarations "
         "-Wno-unused-parameter "
         "-fdiagnostics-color=auto ")

string(
  CONCAT SECURITY_FLAGS
         "-fstack-protector "
         "-fstack-protector-all "
         "-fpic "
         "-fPIC "
         "-D_FORTIFY_SOURCE=2 "
         "-Wformat "
         "-Wformat-security "
         "-fno-delete-null-pointer-checks "
         "-fwrapv ")

string(
  CONCAT FLASHINFER_NVCC_FLAGS
         "-O3 "
         "--threads=1 "
         "-Xfatbin=-compress-all "
         "-use_fast_math "
         "--expt-relaxed-constexpr ")
# cmake-format: on

string(CONCAT FLASHINFER_CXXFLAGS "${WARNING_FLAGS}" "${SECURITY_FLAGS}")

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

# Include consolidated option definitions
include(cmake/Options.cmake)

# Setup dependencies based on enabled options
include(cmake/Dependencies.cmake)

# Define components for build and installation
include(cmake/Components.cmake)

# Add library subdirectories
add_subdirectory(libflashinfer)

if(FLASHINFER_TVM_BINDING)
  add_subdirectory(tvm_binding)
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

# Install CMake configuration
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer
  COMPONENT Headers)
