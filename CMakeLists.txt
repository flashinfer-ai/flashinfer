cmake_minimum_required(VERSION 3.23.1)
project(flashinfer CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Load CMake modules to search for Thrust
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(cmake/utils/Utils.cmake)

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
flashinfer_option(FLASHINFER_TVM_BINDING
                  "Whether to compile tvm binding or not." OFF)
flashinfer_option(FLASHINFER_TVM_SOURCE_DIR
                  "The path to tvm for building tvm binding." "")

add_subdirectory(libflashinfer)

if(FLASHINFER_TVM_BINDING)
  add_subdirectory(tvm_bindings)
endif(FLASHINFER_TVM_BINDING)
