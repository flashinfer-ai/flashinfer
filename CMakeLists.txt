cmake_minimum_required(VERSION 3.18)
project(flashinfer CUDA CXX)

include(cmake/utils/Utils.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
flashinfer_option(FLASHINFER_ENABLE_FP8 "Whether to compile fp8 kernels or not." ON)
flashinfer_option(FLASHINFER_PREFILL "Whether to compile prefill kernel tests/benchmarks or not." ON)
flashinfer_option(FLASHINFER_DECODE "Whether to compile decode kernel tests/benchmarks or not." ON)
flashinfer_option(FLASHINFER_TVM_BINDING "Whether to compile tvm binding or not." OFF)

set(CMAKE_CUDA_ARCHITECTURES native)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
add_subdirectory(3rdparty/nvbench)
add_subdirectory(3rdparty/googletest)
find_package(Thrust REQUIRED)

set(
  FLASHINFER_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include
)

if(FLASHINFER_ENABLE_FP8)
  message(STATUS "Compile fp8 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8)
endif(FLASHINFER_ENABLE_FP8)

if (FLASHINFER_DECODE)
  message(STATUS "Compile decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/bench_decode.cu)
  add_executable(bench_decode ${BENCH_DECODE_SRCS})
  target_include_directories(bench_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_decode PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_decode PRIVATE nvbench::main)

  message(STATUS "Compile decode kernel tests.")
  file(GLOB_RECURSE TEST_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/test_decode.cu)
  add_executable(test_decode ${TEST_DECODE_SRCS})
  target_include_directories(test_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_decode PRIVATE gtest gtest_main)
endif(FLASHINFER_DECODE)

if (FLASHINFER_PREFILL)
  message(STATUS "Compile prefill kernel benchmarks")
  file(GLOB_RECURSE BENCH_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/bench_prefill.cu)
  add_executable(bench_prefill ${BENCH_PREFILL_SRCS})
  target_include_directories(bench_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_prefill PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_prefill PRIVATE nvbench::main)

  message(STATUS "Compile prefill kernel tests.")
  file(GLOB_RECURSE TEST_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/test_prefill.cu)
  add_executable(test_prefill ${TEST_PREFILL_SRCS})
  target_include_directories(test_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_prefill PRIVATE gtest gtest_main)
endif(FLASHINFER_PREFILL)

if(FLASHINFER_TVM_BINDING)
  message(STATUS "Compile tvm binding.")
  file(GLOB_RECURSE TVM_BINDING_SRCS ${PROJECT_SOURCE_DIR}/src/tvm_wrapper.cu)
  add_library(tvm_binding OBJECT ${TVM_BINDING_SRCS})
  target_compile_definitions(tvm_binding PRIVATE -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
  target_include_directories(tvm_binding PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(tvm_binding PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/tvm/include)
  target_include_directories(tvm_binding PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include)
  target_include_directories(tvm_binding PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include)
  target_compile_options(tvm_binding PRIVATE -Xcompiler=-fPIC)
endif(FLASHINFER_TVM_BINDING)