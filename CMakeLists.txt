cmake_minimum_required(VERSION 3.23.1)

# Load CMake modules to search for Thrust
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils")

# Include versioning utility
include(ProjectVersion)
find_package(Git REQUIRED)
set_current_release()

project(
  flashinfer
  LANGUAGES CUDA CXX
  VERSION ${PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include(Utils)

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

file(GLOB_RECURSE FLASHINFER_GENERATORS
     ${PROJECT_SOURCE_DIR}/aot_build_utils/*.py)

# NOTE: do not modify this file to change option values. You can create a
# config.cmake at build folder and add set(OPTION VALUE) to override these build
# options. Alternatively, use cmake -DOPTION=VALUE through command-line.

# Individual C++ components of flashinfer
flashinfer_option(FLASHINFER_TVM_BINDING
                  "Whether to compile tvm binding or not." OFF)
flashinfer_option(FLASHINFER_BUILD_KERNELS
                  "Build and install kernel libraries." OFF)
flashinfer_option(FLASHINFER_DISTRIBUTED
                  "Whether to install distributed all_reduce kernel." OFF)

# Define available components before config setup
set(FLASHINFER_COMPONENTS "Headers")
if(FLASHINFER_BUILD_KERNELS)
  list(APPEND FLASHINFER_COMPONENTS "Kernels")

  # Only add TVMBinding if kernels are built and TVM binding is enabled
  if(FLASHINFER_TVM_BINDING)
    list(APPEND FLASHINFER_COMPONENTS "TVMBinding")
  endif()
endif()

if(FLASHINFER_DISTRIBUTED)
  list(APPEND FLASHINFER_COMPONENTS "Distributed")
endif()

add_subdirectory(libflashinfer)

if(FLASHINFER_TVM_BINDING)
  add_subdirectory(tvm_bindings)
endif(FLASHINFER_TVM_BINDING)

# Generate the flashinferConfig.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# Pass component information to the config template
set(FLASHINFER_CONFIG_COMPONENTS "${FLASHINFER_COMPONENTS}")

# Install the config files with Headers component
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/flashinferConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer
  COMPONENT Headers)
