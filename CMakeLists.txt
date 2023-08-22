cmake_minimum_required(VERSION 3.18)
project(flashinfer CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(FLASHINFER_USE_FP8 ON CACHE STRING "Whether compile fp8 kernels or not.")
set(FLASHINFER_TVM_BINDING ON CACHE STRING "Whether to compile tvm binding or not.")

set(CMAKE_CUDA_ARCHITECTURES native)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
add_subdirectory(3rdparty/nvbench)
add_subdirectory(3rdparty/googletest)
find_package(Thrust REQUIRED)

set(
  FLASHINFER_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include
)

if(FLASHINFER_USE_FP8)
  message(STATUS "Compile fp8 kernels.")
  add_definitions(-DFLASHINFER_USE_FP8)
endif(FLASHINFER_USE_FP8)

file(GLOB_RECURSE BENCHMAKR_SRCS ${PROJECT_SOURCE_DIR}/src/bench.cu)
add_executable(flashinfer_benchmark ${BENCHMAKR_SRCS})
target_include_directories(flashinfer_benchmark PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_include_directories(flashinfer_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
target_link_libraries(flashinfer_benchmark PRIVATE nvbench::main)

file(GLOB_RECURSE TEST_SRCS ${PROJECT_SOURCE_DIR}/src/test.cu)
add_executable(flashinfer_tests ${TEST_SRCS})
target_include_directories(flashinfer_tests PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_include_directories(flashinfer_tests PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_link_libraries(flashinfer_tests PRIVATE gtest gtest_main)


if(FLASHINFER_TVM_BINDING)
  message(STATUS "Compile tvm binding.")
  file(GLOB_RECURSE TVM_BINDING_SRCS ${PROJECT_SOURCE_DIR}/src/tvm_wrapper.cu)
  add_library(tvm_binding OBJECT ${TVM_BINDING_SRCS})
  target_include_directories(tvm_binding PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(tvm_binding PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/tvm/include)
  target_include_directories(tvm_binding PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include)
  target_include_directories(tvm_binding PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include)
  target_compile_options(tvm_binding PRIVATE -Xcompiler=-fPIC)
endif(FLASHINFER_TVM_BINDING)