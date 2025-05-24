cmake_minimum_required(VERSION 3.23.1)
project(flashinfer CUDA CXX)

include(cmake/utils/Utils.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 not found.")
endif()

# NOTE: do not modify this file to change option values. You can create a
# config.cmake at build folder and add set(OPTION VALUE) to override these build
# options. Alternatively, use cmake -DOPTION=VALUE through command-line.
flashinfer_option(FLASHINFER_ENABLE_FP8
                  "Whether to compile fp8 kernels or not." ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E4M3
                  "Whether to compile fp8_e4m3 kernels or not." ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E5M2
                  "Whether to compile fp8_e5m2 kernels or not." ON)
flashinfer_option(FLASHINFER_ENABLE_F16
                  "Whether to compile f16 kernels or not." ON)
flashinfer_option(FLASHINFER_ENABLE_BF16
                  "Whether to compile bf16 kernels or not." ON)
flashinfer_option(
  FLASHINFER_PREFILL
  "Whether to compile prefill kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_DECODE "Whether to compile decode kernel tests/benchmarks or not."
  OFF)
flashinfer_option(FLASHINFER_PAGE
                  "Whether to compile page kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_CASCADE
  "Whether to compile cascade kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_SAMPLING
  "Whether to compile sampling kernel tests/benchmarks or not." OFF)
flashinfer_option(
  FLASHINFER_NORM
  "Whether to compile normalization kernel tests/benchmarks or not." OFF)
flashinfer_option(FLASHINFER_FASTDIV_TEST
                  "Whether to compile fastdiv kernel tests or not." OFF)
flashinfer_option(FLASHINFER_FASTDEQAUNT_TEST
                  "Whether to compile fast dequant kernel tests or not." OFF)

# The following configurations can impact the binary size of the generated
# library
flashinfer_option(FLASHINFER_GEN_HEAD_DIMS "Head dims to enable" 64 128 256)
flashinfer_option(FLASHINFER_GEN_POS_ENCODING_MODES "Pos encodings to enable" 0
                  1 2)
flashinfer_option(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS
                  "QK reductions to enable" OFF)
flashinfer_option(FLASHINFER_GEN_MASK_MODES "Mask modes to enable" 0 1 2)

if(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(
    STATUS "CMAKE_CUDA_ARCHITECTURES set to ${FLASHINFER_CUDA_ARCHITECTURES}.")
  set(CMAKE_CUDA_ARCHITECTURES ${FLASHINFER_CUDA_ARCHITECTURES})
else(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES is ${CMAKE_CUDA_ARCHITECTURES}")
endif(DEFINED FLASHINFER_CUDA_ARCHITECTURES)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
if(FLASHINFER_PREFILL
   OR FLASHINFER_DECODE
   OR FLASHINFER_PAGE
   OR FLASHINFER_CASCADE
   OR FLASHINFER_SAMPLING
   OR FLASHINFER_NORM)
  message(STATUS "NVBench and GoogleTest enabled")
  add_subdirectory(3rdparty/nvbench)
  add_subdirectory(3rdparty/googletest)
endif(
  FLASHINFER_PREFILL
  OR FLASHINFER_DECODE
  OR FLASHINFER_PAGE
  OR FLASHINFER_CASCADE
  OR FLASHINFER_SAMPLING
  OR FLASHINFER_NORM)
find_package(Thrust REQUIRED)

set(FLASHINFER_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

if(FLASHINFER_ENABLE_FP8)
  set(FLASHINFER_ENABLE_FP8_E4M3 ON)
  set(FLASHINFER_ENABLE_FP8_E5M2 ON)
endif(FLASHINFER_ENABLE_FP8)

if(FLASHINFER_ENABLE_FP8_E4M3)
  message(STATUS "Compile fp8_e4m3 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8_E4M3)
endif(FLASHINFER_ENABLE_FP8_E4M3)

if(FLASHINFER_ENABLE_FP8_E5M2)
  message(STATUS "Compile fp8_e5m2 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8_E5M2)
endif(FLASHINFER_ENABLE_FP8_E5M2)

if(FLASHINFER_ENABLE_BF16)
  message(STATUS "Compile bf16 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_BF16)
endif(FLASHINFER_ENABLE_BF16)

# generate kernel inst
set(HEAD_DIMS ${FLASHINFER_GEN_HEAD_DIMS})
set(POS_ENCODING_MODES ${FLASHINFER_GEN_POS_ENCODING_MODES})
set(USE_FP16_QK_REDUCTIONS ${FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS})
set(MASK_MODES ${FLASHINFER_GEN_MASK_MODES})

set(SM90_ALLOWED_HEAD_DIMS "64,64" "128,128" "256,256" "192,128")
set(HEAD_DIMS_SM90 "")

foreach(DIM_VAL ${HEAD_DIMS})
  string(CONCAT TUPLE_VAL "${DIM_VAL}" "," "${DIM_VAL}")
  list(FIND SM90_ALLOWED_HEAD_DIMS ${TUPLE_VAL} RESULT)
  if(NOT ${RESULT} EQUAL -1)
    list(APPEND HEAD_DIMS_SM90 ${TUPLE_VAL})
  endif(NOT ${RESULT} EQUAL -1)
endforeach(DIM_VAL)

foreach(TUPLE_VAL ${SM90_ALLOWED_HEAD_DIMS})
  string(REPLACE "," ";" HEAD_DIMS_LIST ${TUPLE_VAL})
  list(GET HEAD_DIMS_LIST 0 K)
  list(GET HEAD_DIMS_LIST 1 V)
  if(NOT K EQUAL V)
    list(APPEND HEAD_DIMS_SM90 ${TUPLE_VAL})
  endif(NOT K EQUAL V)
endforeach(TUPLE_VAL)

list(REMOVE_DUPLICATES HEAD_DIMS_SM90)

# log options
message(STATUS "FLASHINFER_HEAD_DIMS=${HEAD_DIMS}")
message(STATUS "FLASHINFER_POS_ENCODING_MODES=${POS_ENCODING_MODES}")
message(STATUS "FLASHINFER_USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
message(STATUS "FLASHINFER_MASK_MODES=${MASK_MODES}")

# Log SM90_ALLOWED_HEAD_DIMS and HEAD_DIMS_SM90
message(STATUS "SM90_ALLOWED_HEAD_DIMS=${SM90_ALLOWED_HEAD_DIMS}")
message(STATUS "HEAD_DIMS_SM90=${HEAD_DIMS_SM90}")

set(GENERATED_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/generated)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/generated)

if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  # ----------------------------- Dependencies -------------------------------#
  include(FetchContent)

  set(BOOST_ENABLE_CMAKE ON)
  FetchContent_Declare(boost_math
                       GIT_REPOSITORY https://github.com/boostorg/math.git)
  FetchContent_MakeAvailable(boost_math)
  # --------------------------------------------------------------------------#
  set(USE_FP16_QK_REDUCTIONS "true")
  message(STATUS "USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
else(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  set(USE_FP16_QK_REDUCTIONS "false")
  message(STATUS "USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
endif(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)

set(AOT_GENERATE_COMMAND
    ${Python3_EXECUTABLE} -m aot_build_utils.generate --path
    ${GENERATED_SOURCE_DIR} --head_dims ${HEAD_DIMS} --pos_encoding_modes
    ${POS_ENCODING_MODES} --use_fp16_qk_reductions ${USE_FP16_QK_REDUCTIONS}
    --mask_modes ${MASK_MODES} --enable_f16 ${FLASHINFER_ENABLE_F16}
    --enable_bf16 ${FLASHINFER_ENABLE_BF16} --enable_fp8_e4m3
    ${FLASHINFER_ENABLE_FP8_E4M3} --enable_fp8_e5m2
    ${FLASHINFER_ENABLE_FP8_E5M2})

set(AOT_GENERATE_DISPATCH_INC_COMMAND
    ${Python3_EXECUTABLE} -m aot_build_utils.generate_dispatch_inc --path
    "${GENERATED_SOURCE_DIR}/dispatch.inc" --head_dims ${HEAD_DIMS}
    --head_dims_sm90 ${HEAD_DIMS_SM90} --pos_encoding_modes
    ${POS_ENCODING_MODES} --use_fp16_qk_reductions ${USE_FP16_QK_REDUCTIONS}
    --mask_modes ${MASK_MODES})

execute_process(COMMAND ${AOT_GENERATE_COMMAND}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
execute_process(COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE FLASHINFER_GENERATORS
     ${PROJECT_SOURCE_DIR}/aot_build_utils/*.py)
file(GLOB_RECURSE DECODE_KERNELS_SRCS
     ${PROJECT_SOURCE_DIR}/src/generated/*decode_head*.cu)
file(GLOB_RECURSE PREFILL_KERNELS_SRCS
     ${PROJECT_SOURCE_DIR}/src/generated/*prefill_head*.cu)
file(GLOB_RECURSE DISPATCH_INC_FILE
     ${PROJECT_SOURCE_DIR}/src/generated/dispatch.inc)

add_custom_command(
  OUTPUT ${DECODE_KERNELS_SRCS} ${PREFILL_KERNELS_SRCS} ${DISPATCH_INC_FILE}
  COMMAND ${AOT_GENERATE_COMMAND}
  COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
  DEPENDS ${FLASHINFER_GENERATORS}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Generating kernel sources"
  VERBATIM)
add_custom_target(dispatch_inc DEPENDS ${DISPATCH_INC_FILE})

string(CONCAT CXX_FLAGS "-fpic " "-fPIC ")

string(CONCAT NVCC_FLAGS "-O3 " "--threads=1 " "-Xfatbin=-compress-all "
              "-use_fast_math " "--expt-relaxed-constexpr ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

add_library(decode_kernels STATIC ${DECODE_KERNELS_SRCS})
target_include_directories(decode_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  target_link_libraries(decode_kernels PRIVATE Boost::math)
endif(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)

add_library(prefill_kernels STATIC ${PREFILL_KERNELS_SRCS})
target_include_directories(prefill_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  add_definitions(-DFP16_QK_REDUCTION_SUPPORTED)
  target_link_libraries(prefill_kernels PRIVATE Boost::math)
endif(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)

if(FLASHINFER_DECODE)
  message(STATUS "Compile single decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_SRCS
       ${PROJECT_SOURCE_DIR}/src/bench_single_decode.cu)
  add_executable(bench_single_decode ${BENCH_DECODE_SRCS})
  target_include_directories(bench_single_decode
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_single_decode
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  add_dependencies(bench_single_decode dispatch_inc)
  target_link_libraries(bench_single_decode
                        PRIVATE nvbench::main decode_kernels prefill_kernels)
  target_compile_options(bench_single_decode PRIVATE -Wno-switch-bool)

  message(STATUS "Compile single decode kernel tests.")
  file(GLOB_RECURSE TEST_DECODE_SRCS
       ${PROJECT_SOURCE_DIR}/src/test_single_decode.cu)
  add_executable(test_single_decode ${TEST_DECODE_SRCS})
  target_include_directories(test_single_decode
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(
    test_single_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  add_dependencies(test_single_decode dispatch_inc)
  target_link_libraries(test_single_decode PRIVATE gtest gtest_main
                                                   decode_kernels)
  target_compile_options(test_single_decode PRIVATE -Wno-switch-bool)

  message(STATUS "Compile batch decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_SRCS
       ${PROJECT_SOURCE_DIR}/src/bench_batch_decode.cu)
  add_executable(bench_batch_decode ${BENCH_DECODE_SRCS})
  target_include_directories(bench_batch_decode
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_batch_decode
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  add_dependencies(bench_batch_decode dispatch_inc)
  target_link_libraries(bench_batch_decode PRIVATE nvbench::main decode_kernels
                                                   prefill_kernels)
  target_compile_options(bench_batch_decode PRIVATE -Wno-switch-bool)

  message(STATUS "Compile batch mla decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_MLA_SRCS
       ${PROJECT_SOURCE_DIR}/src/bench_batch_decode_mla.cu)
  add_executable(bench_batch_decode_mla ${BENCH_DECODE_MLA_SRCS})
  target_include_directories(bench_batch_decode_mla
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_batch_decode_mla
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  add_dependencies(bench_batch_decode_mla dispatch_inc)
  target_link_libraries(bench_batch_decode_mla PRIVATE nvbench::main
                                                       decode_kernels)
  target_compile_options(bench_batch_decode_mla PRIVATE -Wno-switch-bool)

  message(STATUS "Compile batch decode kernel tests.")
  file(GLOB_RECURSE TEST_DECODE_SRCS
       ${PROJECT_SOURCE_DIR}/src/test_batch_decode.cu)
  add_executable(test_batch_decode ${TEST_DECODE_SRCS})
  target_include_directories(test_batch_decode
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(
    test_batch_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  add_dependencies(test_batch_decode dispatch_inc)
  target_link_libraries(test_batch_decode PRIVATE gtest gtest_main
                                                  decode_kernels)
  target_compile_options(test_batch_decode PRIVATE -Wno-switch-bool)
endif(FLASHINFER_DECODE)

if(FLASHINFER_PREFILL)
  message(STATUS "Compile single prefill kernel benchmarks")
  file(GLOB_RECURSE BENCH_PREFILL_SRCS
       ${PROJECT_SOURCE_DIR}/src/bench_single_prefill.cu)
  add_executable(bench_single_prefill ${BENCH_PREFILL_SRCS})
  target_include_directories(bench_single_prefill
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_single_prefill
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  add_dependencies(bench_single_prefill dispatch_inc)
  target_link_libraries(bench_single_prefill PRIVATE nvbench::main
                                                     prefill_kernels)
  target_compile_options(bench_single_prefill PRIVATE -Wno-switch-bool)

  message(STATUS "Compile single prefill kernel tests.")
  file(GLOB_RECURSE TEST_PREFILL_SRCS
       ${PROJECT_SOURCE_DIR}/src/test_single_prefill.cu)
  add_executable(test_single_prefill ${TEST_PREFILL_SRCS})
  target_include_directories(test_single_prefill
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(
    test_single_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  add_dependencies(test_single_prefill dispatch_inc)
  target_link_libraries(test_single_prefill PRIVATE gtest gtest_main
                                                    prefill_kernels)
  target_compile_options(test_single_prefill PRIVATE -Wno-switch-bool)

  message(STATUS "Compile batch prefill kernel benchmarks.")
  file(GLOB_RECURSE BENCH_PREFILL_SRCS
       ${PROJECT_SOURCE_DIR}/src/bench_batch_prefill.cu)
  add_executable(bench_batch_prefill ${BENCH_PREFILL_SRCS})
  target_include_directories(bench_batch_prefill
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_batch_prefill
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  add_dependencies(bench_batch_prefill dispatch_inc)
  target_link_libraries(bench_batch_prefill PRIVATE nvbench::main
                                                    prefill_kernels)
  target_compile_options(bench_batch_prefill PRIVATE -Wno-switch-bool)

  message(STATUS "Compile batch prefill kernel tests.")
  file(GLOB_RECURSE TEST_PREFILL_SRCS
       ${PROJECT_SOURCE_DIR}/src/test_batch_prefill.cu)
  add_executable(test_batch_prefill ${TEST_PREFILL_SRCS})
  target_include_directories(test_batch_prefill
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(
    test_batch_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  add_dependencies(test_batch_prefill dispatch_inc)
  target_link_libraries(test_batch_prefill PRIVATE gtest gtest_main
                                                   prefill_kernels)
  target_compile_options(test_batch_prefill PRIVATE -Wno-switch-bool)
endif(FLASHINFER_PREFILL)

if(FLASHINFER_PAGE)
  message(STATUS "Compile page kernel tests.")
  file(GLOB_RECURSE TEST_PAGE_SRCS ${PROJECT_SOURCE_DIR}/src/test_page.cu)
  add_executable(test_page ${TEST_PAGE_SRCS})
  target_include_directories(test_page PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_page PRIVATE ${gtest_SOURCE_DIR}/include
                                               ${gtest_SOURCE_DIR})
  target_link_libraries(test_page PRIVATE gtest gtest_main)
  target_compile_options(test_page PRIVATE -Wno-switch-bool)
endif(FLASHINFER_PAGE)

if(FLASHINFER_CASCADE)
  message(STATUS "Compile cascade kernel benchmarks.")
  file(GLOB_RECURSE BENCH_CASCADE_SRCS
       ${PROJECT_SOURCE_DIR}/src/bench_cascade.cu)
  add_executable(bench_cascade ${BENCH_CASCADE_SRCS})
  target_include_directories(bench_cascade PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_cascade
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  add_dependencies(bench_cascade dispatch_inc)
  target_link_libraries(bench_cascade PRIVATE nvbench::main decode_kernels
                                              prefill_kernels)
  target_compile_options(bench_cascade PRIVATE -Wno-switch-bool)

  message(STATUS "Compile cascade kernel tests.")
  file(GLOB_RECURSE TEST_CASCADE_SRCS ${PROJECT_SOURCE_DIR}/src/test_cascade.cu)
  add_executable(test_cascade ${TEST_CASCADE_SRCS})
  target_include_directories(test_cascade PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_cascade PRIVATE ${gtest_SOURCE_DIR}/include
                                                  ${gtest_SOURCE_DIR})
  add_dependencies(test_cascade dispatch_inc)
  target_link_libraries(test_cascade PRIVATE gtest gtest_main decode_kernels
                                             prefill_kernels)
  target_compile_options(test_cascade PRIVATE -Wno-switch-bool)
endif(FLASHINFER_CASCADE)

if(FLASHINFER_SAMPLING)
  message(STATUS "Compile sampling kernel benchmarks.")
  file(GLOB_RECURSE BENCH_SAMPLING_SRCS
       ${PROJECT_SOURCE_DIR}/src/bench_sampling.cu)
  add_executable(bench_sampling ${BENCH_SAMPLING_SRCS})
  target_include_directories(bench_sampling PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_sampling
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_sampling PRIVATE nvbench::main)
  target_compile_options(bench_sampling PRIVATE -Wno-switch-bool)

  message(STATUS "Compile sampling kernel tests.")
  file(GLOB_RECURSE TEST_SAMPLING_SRCS
       ${PROJECT_SOURCE_DIR}/src/test_sampling.cu)
  add_executable(test_sampling ${TEST_SAMPLING_SRCS})
  target_include_directories(test_sampling PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_sampling PRIVATE ${gtest_SOURCE_DIR}/include
                                                   ${gtest_SOURCE_DIR})
  target_link_libraries(test_sampling PRIVATE gtest gtest_main)
  target_compile_options(test_sampling PRIVATE -Wno-switch-bool)
endif(FLASHINFER_SAMPLING)

if(FLASHINFER_NORM)
  message(STATUS "Compile normalization kernel benchmarks.")
  file(GLOB_RECURSE BENCH_NORM_SRCS ${PROJECT_SOURCE_DIR}/src/bench_norm.cu)
  add_executable(bench_norm ${BENCH_NORM_SRCS})
  target_include_directories(bench_norm PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_norm
                             PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_norm PRIVATE nvbench::main)
  target_compile_options(bench_norm PRIVATE -Wno-switch-bool)

  message(STATUS "Compile normalization kernel tests.")
  file(GLOB_RECURSE TEST_NORM_SRCS ${PROJECT_SOURCE_DIR}/src/test_norm.cu)
  add_executable(test_norm ${TEST_NORM_SRCS})
  target_include_directories(test_norm PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_norm PRIVATE ${gtest_SOURCE_DIR}/include
                                               ${gtest_SOURCE_DIR})
  target_link_libraries(test_norm PRIVATE gtest gtest_main)
  target_compile_options(test_norm PRIVATE -Wno-switch-bool)
endif(FLASHINFER_NORM)

if(FLASHINFER_FASTDIV_TEST)
  message(STATUS "Compile fastdiv test.")
  file(GLOB_RECURSE TEST_FASTDIV_SRCS ${PROJECT_SOURCE_DIR}/src/test_fastdiv.cu)
  add_executable(test_fastdiv ${TEST_FASTDIV_SRCS})
  target_include_directories(test_fastdiv PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_fastdiv PRIVATE ${gtest_SOURCE_DIR}/include
                                                  ${gtest_SOURCE_DIR})
  target_link_libraries(test_fastdiv PRIVATE gtest gtest_main)
endif(FLASHINFER_FASTDIV_TEST)

if(FLASHINFER_FASTDEQUANT_TEST)
  message(STATUS "Compile fast dequant test.")
  file(GLOB_RECURSE TEST_FAST_DEQUANT_SRCS
       ${PROJECT_SOURCE_DIR}/src/test_fast_dequant.cu)
  add_executable(test_fast_dequant ${TEST_FAST_DEQUANT_SRCS})
  target_include_directories(test_fast_dequant
                             PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(
    test_fast_dequant PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_fast_dequant PRIVATE gtest gtest_main)
endif(FLASHINFER_FASTDEQUANT_TEST)
