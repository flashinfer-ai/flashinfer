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
flashinfer_option(FLASHINFER_PAGE "Whether to compile page kernel tests/benchmarks or not." ON)
flashinfer_option(FLASHINFER_TVM_BINDING "Whether to compile tvm binding or not." OFF)
flashinfer_option(FLASHINFER_TVM_HOME "The path to tvm for building tvm binding." "")
# "native" is a special value for CMAKE_CUDA_ARCHITECTURES which means use the architectures of the host's GPU.
# it's new in CMake 3.24, if you are using an older of CMake or you want to use a different value, you can
# set its value through config.cmake or -DCMAKE_CUDA_ARCHITECTURES=... through command-line.
# Supported CUDA architctures include 80;86;89;90
flashinfer_option(FLASHINFER_CUDA_ARCHITECTURES "CUDA architectures of tests/benchmarks" native)

set(CMAKE_CUDA_ARCHITECTURES ${FLASHINFER_CUDA_ARCHITECTURES})

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
  message(STATUS "Compile single decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/bench_single_decode.cu)
  add_executable(bench_single_decode ${BENCH_DECODE_SRCS})
  target_include_directories(bench_single_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_single_decode PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_single_decode PRIVATE nvbench::main)

  message(STATUS "Compile single decode kernel tests.")
  file(GLOB_RECURSE TEST_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/test_single_decode.cu)
  add_executable(test_single_decode ${TEST_DECODE_SRCS})
  target_include_directories(test_single_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_single_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_single_decode PRIVATE gtest gtest_main)

  message(STATUS "Compile batch decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/bench_batch_decode.cu)
  add_executable(bench_batch_decode ${BENCH_DECODE_SRCS})
  target_include_directories(bench_batch_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_batch_decode PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_batch_decode PRIVATE nvbench::main)

  message(STATUS "Compile batch decode kernel tests.")
  file(GLOB_RECURSE TEST_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/test_batch_decode.cu)
  add_executable(test_batch_decode ${TEST_DECODE_SRCS})
  target_include_directories(test_batch_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_batch_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_batch_decode PRIVATE gtest gtest_main)
endif(FLASHINFER_DECODE)

if (FLASHINFER_PREFILL)
  message(STATUS "Compile single prefill kernel benchmarks")
  file(GLOB_RECURSE BENCH_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/bench_single_prefill.cu)
  add_executable(bench_single_prefill ${BENCH_PREFILL_SRCS})
  target_include_directories(bench_single_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_single_prefill PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_single_prefill PRIVATE nvbench::main)

  message(STATUS "Compile single prefill kernel tests.")
  file(GLOB_RECURSE TEST_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/test_single_prefill.cu)
  add_executable(test_single_prefill ${TEST_PREFILL_SRCS})
  target_include_directories(test_single_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_single_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_single_prefill PRIVATE gtest gtest_main)

  message(STATUS "Compile batch prefill kernel tests.")
  file(GLOB_RECURSE TEST_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/test_batch_prefill.cu)
  add_executable(test_batch_prefill ${TEST_PREFILL_SRCS})
  target_include_directories(test_batch_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_batch_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_batch_prefill PRIVATE gtest gtest_main)
endif(FLASHINFER_PREFILL)

if (FLASHINFER_PAGE)
  message(STATUS "Compile page kernel tests.")
  file(GLOB_RECURSE TEST_PAGE_SRCS ${PROJECT_SOURCE_DIR}/src/test_page.cu)
  add_executable(test_page ${TEST_PAGE_SRCS})
  target_include_directories(test_page PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_page PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_page PRIVATE gtest gtest_main)
endif(FLASHINFER_PAGE)

if(FLASHINFER_TVM_BINDING)
  message(STATUS "Compile tvm binding.")
  if(NOT FLASHINFER_TVM_HOME STREQUAL "")
    set(TVM_HOME_SET ${FLASHINFER_TVM_HOME})
  elseif(DEFINED ENV{TVM_HOME})
    set(TVM_HOME_SET $ENV{TVM_HOME})
  else()
    message(FATAL_ERROR "Error: Cannot find TVM. Please set the path to TVM by 1) adding `-DFLASHINFER_TVM_HOME=path/to/tvm` in the cmake command, or 2) setting the environment variable `TVM_HOME` to the tvm path.")
  endif()
  message(STATUS "FlashInfer uses TVM home ${TVM_HOME_SET}.")

  file(GLOB_RECURSE TVM_BINDING_SRCS ${PROJECT_SOURCE_DIR}/src/tvm_wrapper.cu)
  add_library(flashinfer_tvm OBJECT ${TVM_BINDING_SRCS})
  target_compile_definitions(flashinfer_tvm PRIVATE -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
  target_include_directories(flashinfer_tvm PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(flashinfer_tvm PRIVATE ${TVM_HOME_SET}/include)
  target_include_directories(flashinfer_tvm PRIVATE ${TVM_HOME_SET}/3rdparty/dlpack/include)
  target_include_directories(flashinfer_tvm PRIVATE ${TVM_HOME_SET}/3rdparty/dmlc-core/include)
  target_compile_options(flashinfer_tvm PRIVATE -Xcompiler=-fPIC)
endif(FLASHINFER_TVM_BINDING)
