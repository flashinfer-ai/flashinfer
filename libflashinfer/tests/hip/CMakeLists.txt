# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache 2.0

cmake_minimum_required(VERSION 3.21)

project(FlashInferTests LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5)
FetchContent_MakeAvailable(googletest)
get_target_property(gtest_include_dirs GTest::gtest
                    INTERFACE_INCLUDE_DIRECTORIES)

# GoogleTest has to be included so that gtest_discover_tests works Thread
# support is needed for gtest
find_package(Threads REQUIRED)

find_package(HIP REQUIRED)
set(CMAKE_HIP_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(HIP_HIPCC_FLAGS
    "${HIP_HIPCC_FLAGS} -O3 --offload-arch=gfx942 -D__HIP_NO_HALF_OPERATORS__=0 -D__HIP_NO_HALF_CONVERSIONS__=0"
)

# clang-format off Add flashinfer directory here Example:
# set(FLASHINFER_INCLUDE_DIRS /root/flashinfer/libflashinfer/include/)
set(FLASHINFER_INCLUDE_DIRS
    ""
    CACHE PATH "Path to FlashInfer include directory")
# clang-format on

include_directories(${HIP_INCLUDE_DIRS})
include_directories("/opt/rocm/include") # Add this explicit include path
include_directories(${FLASHINFER_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add definition to use HIP API
add_definitions(-D__HIP_PLATFORM_AMD__)

add_library(math STATIC ${FLASHINFER_INCLUDE_DIRS}/gpu_iface/math_ops.hpp)
add_library(layout STATIC ${FLASHINFER_INCLUDE_DIRS}/gpu_iface/layout.cuh)
add_library(pos_enc STATIC
            ${FLASHINFER_INCLUDE_DIRS}/flashinfer/attention/generic/pos_enc.cuh)
add_library(state STATIC
            ${FLASHINFER_INCLUDE_DIRS}/flashinfer/attention/generic/state.cuh)
add_library(cascade STATIC
            ${FLASHINFER_INCLUDE_DIRS}/flashinfer/attention/generic/cascade.cuh)
add_library(decode STATIC
            ${FLASHINFER_INCLUDE_DIRS}/flashinfer/attention/generic/decode.cuh)
add_library(page STATIC
            ${FLASHINFER_INCLUDE_DIRS}/flashinfer/attention/generic/page.cuh)
add_library(mma_ops STATIC ${FLASHINFER_INCLUDE_DIRS}/gpu_iface/mma_ops.hpp)

set(CMAKE_HIP_FLAGS
    "${CMAKE_HIP_FLAGS} -O3 -g -D__HIP_PLATFORM_AMD__ -D__HIP_NO_HALF_OPERATORS__=0 -D__HIP_NO_HALF_CONVERSIONS__=0"
    -DPLATFORM_HIP_DEVICE)

set_target_properties(
  math
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

set_target_properties(
  layout
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

set_target_properties(
  pos_enc
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

set_target_properties(
  state
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

set_target_properties(
  cascade
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

set_target_properties(
  decode
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

set_target_properties(
  page
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

set_target_properties(
  mma_ops
  PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
             HIP_SEPARABLE_COMPILATION ON
             LINKER_LANGUAGE HIP)

target_link_libraries(math PUBLIC hip::host hip::device)
target_link_libraries(layout PUBLIC hip::host hip::device)
target_link_libraries(pos_enc PUBLIC hip::host hip::device)
target_link_libraries(state PUBLIC hip::host hip::device)
target_link_libraries(cascade PUBLIC hip::host hip::device)
target_link_libraries(page PUBLIC hip::host hip::device)
target_link_libraries(
  decode
  PUBLIC hip::host
         hip::device
         cascade
         state
         pos_enc
         layout
         math)

target_link_libraries(mma_ops PUBLIC hip::host hip::device)

message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_math ${CMAKE_CURRENT_SOURCE_DIR}/test_math.cpp)
target_link_libraries(test_math PRIVATE math "${TORCH_LIBRARIES}" gtest
                                        gtest_main pthread)

add_executable(test_pos_enc ${CMAKE_CURRENT_SOURCE_DIR}/test_pos_enc.cpp)
target_link_libraries(test_pos_enc PRIVATE pos_enc "${TORCH_LIBRARIES}" gtest
                                           gtest_main pthread)

add_executable(test_cascade ${CMAKE_CURRENT_SOURCE_DIR}/test_cascade.cpp)
target_link_libraries(test_cascade PRIVATE state cascade "${TORCH_LIBRARIES}"
                                           gtest gtest_main pthread)

add_executable(test_page ${CMAKE_CURRENT_SOURCE_DIR}/test_page.cpp)
target_link_libraries(test_page PRIVATE page "${TORCH_LIBRARIES}" layout gtest
                                        gtest_main pthread)

add_executable(test_single_decode
               ${CMAKE_CURRENT_SOURCE_DIR}/test_single_decode.cpp)
target_link_libraries(test_single_decode PRIVATE decode "${TORCH_LIBRARIES}"
                                                 gtest gtest_main pthread)

add_executable(test_batch_decode
               ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_decode.cpp)
target_link_libraries(test_batch_decode PRIVATE decode "${TORCH_LIBRARIES}"
                                                gtest gtest_main pthread)

add_executable(test_mfma_fp32_16x16x16fp16
               ${CMAKE_CURRENT_SOURCE_DIR}/test_mfma_fp32_16x16x16fp16.cpp)
target_link_libraries(
  test_mfma_fp32_16x16x16fp16 PRIVATE mma_ops "${TORCH_LIBRARIES}" gtest
                                      gtest_main pthread)

add_executable(test_rowsum ${CMAKE_CURRENT_SOURCE_DIR}/test_rowsum.cpp)
target_link_libraries(test_rowsum PRIVATE mma_ops "${TORCH_LIBRARIES}" gtest
                                          gtest_main pthread)

add_executable(
  test_transpose_4x4_half_registers
  ${CMAKE_CURRENT_SOURCE_DIR}/test_transpose_4x4_half_registers.cpp)
target_link_libraries(
  test_transpose_4x4_half_registers PRIVATE mma_ops "${TORCH_LIBRARIES}" gtest
                                            gtest_main pthread)

if(HIP_FOUND)
  set_target_properties(test_math PROPERTIES HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(test_pos_enc PROPERTIES HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(test_cascade PROPERTIES HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(test_page PROPERTIES HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(test_single_decode PROPERTIES HIP_SEPARABLE_COMPILATION
                                                      ON)
  set_target_properties(test_batch_decode PROPERTIES HIP_SEPARABLE_COMPILATION
                                                     ON)
  set_target_properties(test_mfma_fp32_16x16x16fp16
                        PROPERTIES HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(test_transpose_4x4_half_registers
                        PROPERTIES HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(test_rowsum PROPERTIES HIP_SEPARABLE_COMPILATION ON)
endif()

enable_testing()
add_test(NAME MathTest COMMAND test_math)
add_test(NAME PosEncTest COMMAND test_pos_enc)
add_test(NAME CascadeTest COMMAND test_cascade)
add_test(NAME PageTest COMMAND test_page)
add_test(NAME SingleDecodeTest COMMAND test_single_decode)
add_test(NAME BatchDecodeTest COMMAND test_batch_decode)
add_test(NAME test_mfma_fp32_16x16x16fp16 COMMAND test_mfma_fp32_16x16x16fp16)
add_test(NAME test_transpose_4x4_half_registers
         COMMAND test_transpose_4x4_half_registers)
add_test(NAME test_rowsum COMMAND test_rowsum)
