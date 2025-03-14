cmake_minimum_required(VERSION 3.23.1)
project(
  "libflashinfer"
  LANGUAGES CUDA CXX
  DESCRIPTION "C++ Kernel Library for LLM Serving")

cmake_policy(SET CMP0177 NEW)
include(FetchContent)

# ------------------------------ Dependencies ---------------------------------#

FetchContent_Declare(
  mscclpp
  GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
  GIT_TAG 11e62024d3eb190e005b4689f8c8443d91a6c82e)
FetchContent_MakeAvailable(mscclpp)

set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(boost_math
                     GIT_REPOSITORY https://github.com/boostorg/math.git)
FetchContent_MakeAvailable(boost_math)

# -----------------------------------------------------------------------------#

find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(
    FATAL_ERROR
      "Python3 not found it is required to generate the kernel sources.")
endif()

find_package(Thrust REQUIRED)

# NOTE: do not modify this file to change option values. You can create a
# config.cmake at build folder and add set(OPTION VALUE) to override these build
# options. Alternatively, use cmake -DOPTION=VALUE through command-line.
# cmake-format: off
flashinfer_option(FLASHINFER_ENABLE_FP8      "Whether to compile fp8 kernels or not."         ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E4M3 "Whether to compile fp8_e4m3 kernels or not."    ON)
flashinfer_option(FLASHINFER_ENABLE_FP8_E5M2 "Whether to compile fp8_e5m2 kernels or not."    ON)
flashinfer_option(FLASHINFER_ENABLE_F16      "Whether to compile f16 kernels or not."         ON)
flashinfer_option(FLASHINFER_ENABLE_BF16     "Whether to compile bf16 kernels or not."        ON)
flashinfer_option(FLASHINFER_UNITTESTS       "Whether to configure libflashinfer unit tests" OFF)
flashinfer_option(FLASHINFER_DIST_UNITTESTS  "Whether to configure libflashinfer distributed unit tests" OFF)
flashinfer_option(FLASHINFER_CXX_BENCHMARKS  "Whether to configure libflashinfer cxx benchmarks" OFF)
# The following configurations can impact the binary size of the generated library
flashinfer_option(FLASHINFER_GEN_HEAD_DIMS              "Head dims to enable" 128 256)
flashinfer_option(FLASHINFER_GEN_POS_ENCODING_MODES     "Pos encodings to enable" 0)
# FIXME: CAUTION!!! Turning on this option will cause build failures. (refer #806, #936)
flashinfer_option(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS "QK reductions to enable" OFF)
flashinfer_option(FLASHINFER_GEN_MASK_MODES             "Mask modes to enable" 0 1 2)
# cmake-format: on

if(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(
    STATUS "CMAKE_CUDA_ARCHITECTURES set to ${FLASHINFER_CUDA_ARCHITECTURES}.")
  set(CMAKE_CUDA_ARCHITECTURES ${FLASHINFER_CUDA_ARCHITECTURES})
else(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES is ${CMAKE_CUDA_ARCHITECTURES}")
endif(DEFINED FLASHINFER_CUDA_ARCHITECTURES)

set(FLASHINFER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FLASHINFER_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)

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

if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  set(USE_FP16_QK_REDUCTIONS "true")
else(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  set(USE_FP16_QK_REDUCTIONS "false")
endif(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)

# generate kernel inst
set(HEAD_DIMS ${FLASHINFER_GEN_HEAD_DIMS})
set(POS_ENCODING_MODES ${FLASHINFER_GEN_POS_ENCODING_MODES})
set(MASK_MODES ${FLASHINFER_GEN_MASK_MODES})

# log options
message(STATUS "FLASHINFER_HEAD_DIMS=${HEAD_DIMS}")
message(STATUS "FLASHINFER_POS_ENCODING_MODES=${POS_ENCODING_MODES}")
message(STATUS "FLASHINFER_USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
message(STATUS "FLASHINFER_MASK_MODES=${MASK_MODES}")

set(GENERATED_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/generated)
cmake_path(GET GENERATED_SOURCE_DIR PARENT_PATH GENERATED_SOURCE_DIR_ROOT)
file(MAKE_DIRECTORY ${GENERATED_SOURCE_DIR})

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
    --head_dims_sm90 "64,64" # FIXME Make this configurable.
    --pos_encoding_modes ${POS_ENCODING_MODES} --use_fp16_qk_reductions
    ${USE_FP16_QK_REDUCTIONS} --mask_modes ${MASK_MODES})

execute_process(COMMAND ${AOT_GENERATE_COMMAND}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
execute_process(COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE DECODE_KERNELS_SRCS ${GENERATED_SOURCE_DIR}/*decode_head*.cu)
file(GLOB_RECURSE PREFILL_KERNELS_SRCS
     ${GENERATED_SOURCE_DIR}/*prefill_head*.cu)
file(GLOB_RECURSE DISPATCH_INC_FILE ${GENERATED_SOURCE_DIR}/dispatch.inc)

add_custom_command(
  OUTPUT ${DECODE_KERNELS_SRCS} ${PREFILL_KERNELS_SRCS} ${DISPATCH_INC_FILE}
  COMMAND ${AOT_GENERATE_COMMAND}
  COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
  DEPENDS ${FLASHINFER_GENERATORS}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Generating kernel sources"
  VERBATIM)
add_custom_target(dispatch_inc DEPENDS ${DISPATCH_INC_FILE})

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
  CONCAT NVCC_FLAGS
         "-O3 "
         "--threads=1 "
         "-Xfatbin=-compress-all "
         "-use_fast_math "
         "-DFLASHINFER_ENABLE_F16 "
         "--expt-relaxed-constexpr ")
string(CONCAT FLASHINFER_CXXFLAGS "${WARNING_FLAGS}" "${SECURITY_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLASHINFER_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} ${FLASHINFER_CXXFLAGS} -O0 -ggdb3 -DDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

add_library(decode_kernels STATIC ${DECODE_KERNELS_SRCS})
target_include_directories(decode_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_link_libraries(decode_kernels PRIVATE Boost::math)

add_library(prefill_kernels STATIC ${PREFILL_KERNELS_SRCS})
target_include_directories(prefill_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_link_libraries(prefill_kernels PRIVATE Boost::math)

if(FLASHINFER_UNITTESTS)
  add_subdirectory(tests)
endif(FLASHINFER_UNITTESTS)

if(FLASHINFER_CXX_BENCHMARKS)
  add_subdirectory(benchmarks)
endif(FLASHINFER_CXX_BENCHMARKS)
