cmake_minimum_required(VERSION 3.20)
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
flashinfer_option(FLASHINFER_ENABLE_BF16 "Whether to compile bf16 kernels or not." ON)
flashinfer_option(FLASHINFER_PREFILL "Whether to compile prefill kernel tests/benchmarks or not." OFF)
flashinfer_option(FLASHINFER_DECODE "Whether to compile decode kernel tests/benchmarks or not." OFF)
flashinfer_option(FLASHINFER_PAGE "Whether to compile page kernel tests/benchmarks or not." OFF)
flashinfer_option(FLASHINFER_CASCADE "Whether to compile cascade kernel tests/benchmarks or not." OFF)
flashinfer_option(FLASHINFER_TVM_BINDING "Whether to compile tvm binding or not." OFF)
flashinfer_option(FLASHINFER_TVM_HOME "The path to tvm for building tvm binding." "")

if(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES set to ${FLASHINFER_CUDA_ARCHITECTURES}.")
  set(CMAKE_CUDA_ARCHITECTURES ${FLASHINFER_CUDA_ARCHITECTURES})
else(DEFINED FLASHINFER_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES is ${CMAKE_CUDA_ARCHITECTURES}")
endif(DEFINED FLASHINFER_CUDA_ARCHITECTURES)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
if(FLASHINFER_PREFILL OR FLASHINFER_DECODE OR FLASHINFER_PAGE OR FLASHINFER_CASCADE)
  message(STATUS "NVBench and GoogleTest enabled")
  add_subdirectory(3rdparty/nvbench)
  add_subdirectory(3rdparty/googletest)
endif(FLASHINFER_PREFILL OR FLASHINFER_DECODE OR FLASHINFER_PAGE OR FLASHINFER_CASCADE)
find_package(Thrust REQUIRED)

set(
  FLASHINFER_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include
)

if(FLASHINFER_ENABLE_FP8)
  message(STATUS "Compile fp8 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8)
endif(FLASHINFER_ENABLE_FP8)

if(FLASHINFER_ENABLE_BF16)
  message(STATUS "Compile bf16 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_BF16)
endif(FLASHINFER_ENABLE_BF16)

# generate kernel inst
set (GROUP_SIZES 1 4 6 8)
set (HEAD_DIMS 64 128 256)
set (KV_LAYOUTS 0 1)
set (POS_ENCODING_MODES 0 1 2)
set (ALLOW_FP16_QK_REDUCTIONS "false" "true")
set (CAUSALS "false" "true")
set (DECODE_DTYPES "f16")
set (PREFILL_DTYPES "f16")
set (DECODE_F8_DTYPES)
set (IDTYPES "i32")

if(FLASHINFER_ENABLE_FP8)
  list(APPEND DECODE_DTYPES "e4m3" "e5m2")
  list(APPEND DECODE_F8_DTYPES "e4m3" "e5m2")
endif(FLASHINFER_ENABLE_FP8)

if(FLASHINFER_ENABLE_BF16)
  list(APPEND DECODE_DTYPES "bf16")
  list(APPEND PREFILL_DTYPES "bf16")
endif(FLASHINFER_ENABLE_BF16)

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/src/generated)

# single decode kernel inst generation
foreach(group_size IN LISTS GROUP_SIZES)
  foreach(head_dim IN LISTS HEAD_DIMS)
    foreach(kv_layout IN LISTS KV_LAYOUTS)
      foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
        foreach(dtype IN LISTS DECODE_DTYPES)
          set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/single_decode_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_dtypein_${dtype}_dtypeout_${dtype}.cu)
          add_custom_command(
            OUTPUT ${generated_kernel_src}
            COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_single_decode_inst.py ${generated_kernel_src}
            DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_single_decode_inst.py
            COMMENT "Generating additional source file ${generated_kernel_src}"
            VERBATIM
          )
          list(APPEND single_decode_kernels_src ${generated_kernel_src})
        endforeach(dtype)

        # fp8 in, fp16 out
        foreach(dtype IN LISTS DECODE_F8_DTYPES)
          set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/single_decode_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_dtypein_${dtype}_dtypeout_f16.cu)
          add_custom_command(
            OUTPUT ${generated_kernel_src}
            COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_single_decode_inst.py ${generated_kernel_src}
            DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_single_decode_inst.py
            COMMENT "Generating additional source file ${generated_kernel_src}"
            VERBATIM
          )
          list(APPEND single_decode_kernels_src ${generated_kernel_src})         
        endforeach(dtype)
      endforeach(pos_encoding_mode)
    endforeach(kv_layout)
  endforeach(head_dim)
endforeach(group_size)

# batch decode kernel inst generation
foreach(group_size IN LISTS GROUP_SIZES)
  foreach(head_dim IN LISTS HEAD_DIMS)
    foreach(kv_layout IN LISTS KV_LAYOUTS)
      foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
        # paged kv-cache
        foreach(idtype IN LISTS IDTYPES)
          foreach(dtype IN LISTS DECODE_DTYPES)
            set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_decode_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_dtypein_${dtype}_dtypeout_${dtype}_idtype_${idtype}.cu)
            add_custom_command(
              OUTPUT ${generated_kernel_src}
              COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_decode_inst.py ${generated_kernel_src}
              DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_decode_inst.py
              COMMENT "Generating additional source file ${generated_kernel_src}"
              VERBATIM
            )
            list(APPEND batch_decode_kernels_src ${generated_kernel_src})
          endforeach(dtype)
          
          # fp8 in, fp16 out
          foreach(dtype IN LISTS DECODE_FP8_DTYPES)
            set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_decode_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_dtypein_${dtype}_dtypeout_f16_idtype_${idtype}.cu)
            add_custom_command(
              OUTPUT ${generated_kernel_src}
              COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_decode_inst.py ${generated_kernel_src}
              DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_decode_inst.py
              COMMENT "Generating additional source file ${generated_kernel_src}"
              VERBATIM
            )
            list(APPEND batch_decode_kernels_src ${generated_kernel_src})
          endforeach()
        endforeach(idtype)

        # padded kv-cache
        foreach(dtype IN LISTS DECODE_DTYPES)
          set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_padded_decode_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_dtypein_${dtype}_dtypeout_${dtype}.cu)
          add_custom_command(
            OUTPUT ${generated_kernel_src}
            COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_padded_decode_inst.py ${generated_kernel_src}
            DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_padded_decode_inst.py
            COMMENT "Generating additional source file ${generated_kernel_src}"
            VERBATIM
          )
          list(APPEND batch_decode_kernels_src ${generated_kernel_src})
        endforeach(dtype)
        
        # padded kv-cache, fp8 in, fp16 out
        foreach(dtype IN LISTS DECODE_FP8_DTYPES)
          set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_padded_decode_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_dtypein_${dtype}_dtypeout_f16.cu)
          add_custom_command(
            OUTPUT ${generated_kernel_src}
            COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_padded_decode_inst.py ${generated_kernel_src}
            DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_padded_decode_inst.py
            COMMENT "Generating additional source file ${generated_kernel_src}"
            VERBATIM
          )
          list(APPEND batch_decode_kernels_src ${generated_kernel_src})
        endforeach()
      endforeach(pos_encoding_mode)
    endforeach(kv_layout)
  endforeach(head_dim)
endforeach(group_size)

add_library(decode_kernels STATIC ${single_decode_kernels_src} ${batch_decode_kernels_src})
target_include_directories(decode_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_compile_options(decode_kernels PRIVATE -Xcompiler=-fPIC --fatbin-options -compress-all)

# single prefill kernel inst generation
foreach(group_size IN LISTS GROUP_SIZES)
  foreach(head_dim IN LISTS HEAD_DIMS)
    foreach(kv_layout IN LISTS KV_LAYOUTS)
      foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
        foreach(allow_fp16_qk_reduction IN LISTS ALLOW_FP16_QK_REDUCTIONS)
          foreach(causal IN LISTS CAUSALS)
            foreach(dtype IN LISTS PREFILL_DTYPES)
              set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/single_prefill_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_causal_${causal}_dtypein_${dtype}_dtypeout_${dtype}.cu)
              add_custom_command(
                OUTPUT ${generated_kernel_src}
                COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_single_prefill_inst.py ${generated_kernel_src}
                DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_single_prefill_inst.py
                COMMENT "Generating additional source file ${generated_kernel_src}"
                VERBATIM
              )
              list(APPEND single_prefill_kernels_src ${generated_kernel_src})
            endforeach(dtype)
          endforeach(causal)
        endforeach(allow_fp16_qk_reduction)
      endforeach(pos_encoding_mode)
    endforeach(kv_layout)
  endforeach(head_dim)
endforeach(group_size)

# batch paged prefill kernel inst generation
foreach(group_size IN LISTS GROUP_SIZES)
  foreach(head_dim IN LISTS HEAD_DIMS)
    foreach(kv_layout IN LISTS KV_LAYOUTS)
      foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
        foreach(allow_fp16_qk_reduction IN LISTS ALLOW_FP16_QK_REDUCTIONS)
          foreach(causal IN LISTS CAUSALS)
            foreach(dtype IN LISTS PREFILL_DTYPES)
              foreach(idtype IN LISTS IDTYPES)
                set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_paged_prefill_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_causal_${causal}_dtypein_${dtype}_dtypeout_${dtype}_idtype_${idtype}.cu)
                add_custom_command(
                  OUTPUT ${generated_kernel_src}
                  COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_prefill_inst.py ${generated_kernel_src}
                  DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_paged_prefill_inst.py
                  COMMENT "Generating additional source file ${generated_kernel_src}"
                  VERBATIM
                )
                list(APPEND batch_paged_prefill_kernels_src ${generated_kernel_src})    
              endforeach(idtype)
            endforeach(dtype)
          endforeach(causal)
        endforeach(allow_fp16_qk_reduction)
      endforeach(pos_encoding_mode)
    endforeach(kv_layout)
  endforeach(head_dim)
endforeach(group_size)

# batch ragged prefill kernel inst generation
foreach(group_size IN LISTS GROUP_SIZES)
  foreach(head_dim IN LISTS HEAD_DIMS)
    foreach(kv_layout IN LISTS KV_LAYOUTS)
      foreach(pos_encoding_mode IN LISTS POS_ENCODING_MODES)
        foreach(allow_fp16_qk_reduction IN LISTS ALLOW_FP16_QK_REDUCTIONS)
          foreach(causal IN LISTS CAUSALS)
            foreach(dtype IN LISTS PREFILL_DTYPES)
              foreach(idtype IN LISTS IDTYPES)
                set(generated_kernel_src ${PROJECT_SOURCE_DIR}/src/generated/batch_ragged_prefill_group_${group_size}_head_${head_dim}_layout_${kv_layout}_posenc_${pos_encoding_mode}_fp16qkred_${allow_fp16_qk_reduction}_causal_${causal}_dtypein_${dtype}_dtypeout_${dtype}_idtype_${idtype}.cu)
                add_custom_command(
                  OUTPUT ${generated_kernel_src}
                  COMMAND python ${PROJECT_SOURCE_DIR}/python/generate_batch_ragged_prefill_inst.py ${generated_kernel_src}
                  DEPENDS ${PROJECT_SOURCE_DIR}/python/generate_batch_ragged_prefill_inst.py
                  COMMENT "Generating additional source file ${generated_kernel_src}"
                  VERBATIM
                )
                list(APPEND batch_ragged_prefill_kernels_src ${generated_kernel_src})    
              endforeach(idtype)
            endforeach(dtype)
          endforeach(causal)
        endforeach(allow_fp16_qk_reduction)
      endforeach(pos_encoding_mode)
    endforeach(kv_layout)
  endforeach(head_dim)
endforeach(group_size)

add_library(prefill_kernels STATIC ${single_prefill_kernels_src} ${batch_paged_prefill_kernels_src} ${batch_ragged_prefill_kernels_src})
target_include_directories(prefill_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_compile_options(prefill_kernels PRIVATE -Xcompiler=-fPIC --fatbin-options -compress-all)

if (FLASHINFER_DECODE)
  message(STATUS "Compile single decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/bench_single_decode.cu)
  add_executable(bench_single_decode ${BENCH_DECODE_SRCS})
  target_include_directories(bench_single_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_single_decode PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_single_decode PRIVATE nvbench::main decode_kernels prefill_kernels)

  message(STATUS "Compile single decode kernel tests.")
  file(GLOB_RECURSE TEST_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/test_single_decode.cu)
  add_executable(test_single_decode ${TEST_DECODE_SRCS})
  target_include_directories(test_single_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_single_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_single_decode PRIVATE gtest gtest_main decode_kernels)

  message(STATUS "Compile batch decode kernel benchmarks.")
  file(GLOB_RECURSE BENCH_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/bench_batch_decode.cu)
  add_executable(bench_batch_decode ${BENCH_DECODE_SRCS})
  target_include_directories(bench_batch_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_batch_decode PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_batch_decode PRIVATE nvbench::main decode_kernels prefill_kernels)

  message(STATUS "Compile batch decode kernel tests.")
  file(GLOB_RECURSE TEST_DECODE_SRCS ${PROJECT_SOURCE_DIR}/src/test_batch_decode.cu)
  add_executable(test_batch_decode ${TEST_DECODE_SRCS})
  target_include_directories(test_batch_decode PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_batch_decode PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_batch_decode PRIVATE gtest gtest_main decode_kernels)
endif(FLASHINFER_DECODE)

if (FLASHINFER_PREFILL)
  message(STATUS "Compile single prefill kernel benchmarks")
  file(GLOB_RECURSE BENCH_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/bench_single_prefill.cu)
  add_executable(bench_single_prefill ${BENCH_PREFILL_SRCS})
  target_include_directories(bench_single_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_single_prefill PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_single_prefill PRIVATE nvbench::main prefill_kernels)

  message(STATUS "Compile single prefill kernel tests.")
  file(GLOB_RECURSE TEST_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/test_single_prefill.cu)
  add_executable(test_single_prefill ${TEST_PREFILL_SRCS})
  target_include_directories(test_single_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_single_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_single_prefill PRIVATE gtest gtest_main prefill_kernels)

  message(STATUS "Compile batch prefill kernel tests.")
  file(GLOB_RECURSE TEST_PREFILL_SRCS ${PROJECT_SOURCE_DIR}/src/test_batch_prefill.cu)
  add_executable(test_batch_prefill ${TEST_PREFILL_SRCS})
  target_include_directories(test_batch_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_batch_prefill PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_batch_prefill PRIVATE gtest gtest_main prefill_kernels)
endif(FLASHINFER_PREFILL)

if (FLASHINFER_PAGE)
  message(STATUS "Compile page kernel tests.")
  file(GLOB_RECURSE TEST_PAGE_SRCS ${PROJECT_SOURCE_DIR}/src/test_page.cu)
  add_executable(test_page ${TEST_PAGE_SRCS})
  target_include_directories(test_page PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_page PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_page PRIVATE gtest gtest_main)
endif(FLASHINFER_PAGE)

if (FLASHINFER_CASCADE)
  message(STATUS "Compile cascade kernel benchmarks.")
  file(GLOB_RECURSE BENCH_CASCADE_SRCS ${PROJECT_SOURCE_DIR}/src/bench_cascade.cu)
  add_executable(bench_cascade ${BENCH_CASCADE_SRCS})
  target_include_directories(bench_cascade PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(bench_cascade PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
  target_link_libraries(bench_cascade PRIVATE nvbench::main decode_kernels prefill_kernels)

  message(STATUS "Compile cascade kernel tests.")
  file(GLOB_RECURSE TEST_CASCADE_SRCS ${PROJECT_SOURCE_DIR}/src/test_cascade.cu)
  add_executable(test_cascade ${TEST_CASCADE_SRCS})
  target_include_directories(test_cascade PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(test_cascade PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  target_link_libraries(test_cascade PRIVATE gtest gtest_main decode_kernels prefill_kernels)
endif(FLASHINFER_CASCADE)

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
  target_link_libraries(flashinfer_tvm PRIVATE decode_kernels prefill_kernels)
  target_include_directories(flashinfer_tvm PRIVATE ${FLASHINFER_INCLUDE_DIR})
  target_include_directories(flashinfer_tvm PRIVATE ${TVM_HOME_SET}/include)
  target_include_directories(flashinfer_tvm PRIVATE ${TVM_HOME_SET}/3rdparty/dlpack/include)
  target_include_directories(flashinfer_tvm PRIVATE ${TVM_HOME_SET}/3rdparty/dmlc-core/include)
  target_compile_options(flashinfer_tvm PRIVATE -Xcompiler=-fPIC -diag-suppress "1305")
endif(FLASHINFER_TVM_BINDING)
