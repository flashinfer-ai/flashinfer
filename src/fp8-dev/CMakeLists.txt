cmake_minimum_required(VERSION 3.23.1)
project(flashinfer CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90a)

############ Modify the following paths to your own paths ############
set(FA3_INCLUDE_DIR /home/ylzhao/flash-attention/hopper)
set(LIBTORCH_INCLUDE_DIR /home/ylzhao/libtorch)
############ Modify the above paths to your own paths ############

find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 not found.")
endif()

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_INCLUDE_DIR})
find_package(Torch REQUIRED)
find_package(Thrust REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(${CMAKE_SOURCE_DIR}/../../3rdparty/nvbench ${CMAKE_BINARY_DIR}/nvbench_build)
add_subdirectory(${CMAKE_SOURCE_DIR}/../../3rdparty/googletest ${CMAKE_BINARY_DIR}/googletest_build)

set(FLASHINFER_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../include)
set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../3rdparty/cutlass/include ${CMAKE_SOURCE_DIR}/../../3rdparty/cutlass/tools/util/include)
set(NVBENCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../3rdparty/nvbench)

# build flashinfer library
add_library(FLASHINFER_LIB STATIC ${CMAKE_SOURCE_DIR}/flashinfer_ops.cu)
set_target_properties(FLASHINFER_LIB PROPERTIES CMAKE_CUDA_ARCHITECTURES "90a")
target_compile_options(FLASHINFER_LIB PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: 
        --expt-extended-lambda 
        --use_fast_math
        --compiler-options -fPIC
        --generate-code=arch=compute_90a,code=sm_90a
    >
)
target_include_directories(FLASHINFER_LIB PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_include_directories(FLASHINFER_LIB PRIVATE ${CUTLASS_INCLUDE_DIR})

# build FA3 library
file(GLOB FA3_IMPL_FILES ${FA3_INCLUDE_DIR}/flash_fwd_*.cu)
add_library(FA3_LIB STATIC ${FA3_IMPL_FILES})
set_target_properties(FA3_LIB PROPERTIES CMAKE_CUDA_ARCHITECTURES "90a")
target_compile_options(FA3_LIB PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: 
        --expt-extended-lambda 
        --use_fast_math
        --compiler-options -fPIC
        --generate-code=arch=compute_90a,code=sm_90a
    >
)
target_include_directories(FA3_LIB PRIVATE ${CUTLASS_INCLUDE_DIR} ${FA3_INCLUDE_DIR})

# build benchmark FA3/FlashInfer/FP16
add_executable(bench_single_prefill ${CMAKE_SOURCE_DIR}/bench_single_prefill_sm90.cu)
target_compile_options(bench_single_prefill PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>  # Disables warnings in CUDA
    $<$<COMPILE_LANGUAGE:CXX>:-w>                   # Disables warnings in C++
    $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --generate-code=arch=compute_90a,code=sm_90a>
)
target_include_directories(bench_single_prefill PRIVATE ${FLASHINFER_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(bench_single_prefill PRIVATE ${CUTLASS_INCLUDE_DIR})
target_include_directories(bench_single_prefill PRIVATE ${NVBENCH_INCLUDE_DIR})
target_include_directories(bench_single_prefill PRIVATE ${FA3_INCLUDE_DIR})
target_link_libraries(bench_single_prefill PRIVATE nvbench::main ${TORCH_LIBRARIES} FA3_LIB)

# build test FA3 / FlashInfer / FP16 test cases
add_executable(test_single_prefill_fa3_sm90 ${CMAKE_SOURCE_DIR}/test_single_prefill_fa3_sm90.cu)
target_compile_options(test_single_prefill_fa3_sm90 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>  # Disables warnings in CUDA
    $<$<COMPILE_LANGUAGE:CXX>:-w>                   # Disables warnings in C++
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --generate-code=arch=compute_90a,code=sm_90a>
)
target_include_directories(test_single_prefill_fa3_sm90 PRIVATE ${CUTLASS_INCLUDE_DIR})
target_include_directories(test_single_prefill_fa3_sm90 PRIVATE ${FLASHINFER_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(test_single_prefill_fa3_sm90 PRIVATE ${FA3_INCLUDE_DIR})
target_link_libraries(test_single_prefill_fa3_sm90 PRIVATE ${TORCH_LIBRARIES} FA3_LIB FLASHINFER_LIB)