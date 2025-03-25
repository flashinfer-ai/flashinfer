# Set global paths and initialize test list
set(ALL_TEST_TARGETS "")

# Include centralized config utilities
include(ConfigureTargets)

# Include GoogleTest to enable test discovery
include(GoogleTest)
find_package(Threads REQUIRED)

# cmake-format: off
# === Core Tests Configuration ===

# Single and batch decode tests
configure_flashinfer_target(
  TARGET_NAME test_single_decode
  SOURCES "test_single_decode.cu"
  LINK_LIBS "decode_kernels"
  IS_GTEST ON
)

configure_flashinfer_target(
  TARGET_NAME test_batch_decode
  SOURCES "test_batch_decode.cu"
  LINK_LIBS "decode_kernels"
  IS_GTEST ON
)

# Single and batch prefill tests
configure_flashinfer_target(
  TARGET_NAME test_single_prefill
  SOURCES "test_single_prefill.cu"
  LINK_LIBS "prefill_kernels"
  IS_GTEST ON
)

configure_flashinfer_target(
  TARGET_NAME test_batch_prefill
  SOURCES "test_batch_prefill.cu"
  LINK_LIBS "prefill_kernels"
  IS_GTEST ON
)

# Other core tests
configure_flashinfer_target(
  TARGET_NAME test_page
  SOURCES "test_page.cu"
  IS_GTEST ON
)

configure_flashinfer_target(
  TARGET_NAME test_cascade
  SOURCES "test_cascade.cu"
  LINK_LIBS "decode_kernels;prefill_kernels"
  IS_GTEST ON
)

configure_flashinfer_target(
  TARGET_NAME test_sampling
  SOURCES "test_sampling.cu"
  IS_GTEST ON
)

configure_flashinfer_target(
  TARGET_NAME test_norm
  SOURCES "test_norm.cu"
  IS_GTEST ON
)

configure_flashinfer_target(
  TARGET_NAME test_fastdiv
  SOURCES "test_fastdiv.cu"
  IS_GTEST ON
)

configure_flashinfer_target(
  TARGET_NAME test_fast_dequant
  SOURCES "test_fast_dequant.cu"
  IS_GTEST ON
)

# === Distributed Test Configuration ===
if(FLASHINFER_DIST_UNITTESTS)
  set(DIST_INCLUDE_DIRS
      "${FLASHINFER_INCLUDE_DIR}"
      "${mscclpp_SOURCE_DIR}/include"
  )

  # Add spdlog include directory if available
  if(DEFINED SPDLOG_INCLUDE_DIR)
    list(APPEND DIST_INCLUDE_DIRS "${SPDLOG_INCLUDE_DIR}")
  elseif(TARGET spdlog::spdlog)
    # If found via find_package
    get_target_property(SPDLOG_INCLUDE_DIRS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND DIST_INCLUDE_DIRS "${SPDLOG_INCLUDE_DIRS}")
  endif()

  configure_flashinfer_target(
    TARGET_NAME test_sum_all_reduce
    SOURCES "test_sum_all_reduce.cu"
    LINK_LIBS "MPI::MPI_CXX;flashinfer::mscclpp"
    COMPILE_FLAGS "-DENABLE_MPI"
    INCLUDE_DIRS "${DIST_INCLUDE_DIRS}"
    IS_GTEST OFF
  )

  configure_flashinfer_target(
    TARGET_NAME test_attn_all_reduce
    SOURCES "test_attn_all_reduce.cu"
    LINK_LIBS "MPI::MPI_CXX;flashinfer::mscclpp"
    COMPILE_FLAGS "-DENABLE_MPI"
    IS_GTEST OFF
  )
endif()

# === FP8 Test Configuration ===
if(FLASHINFER_FP8_TESTS)
  # Set path to FP8 utilities
  set(FP8_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fp8)

  # Define FP8-specific CUDA compile flags - one per line for readability
  set(FP8_CUDA_FLAGS_LIST
      --expt-extended-lambda
      --use_fast_math
      --generate-code=arch=compute_90a,code=sm_90a
  )

  # Combine the list into a generator expression for CUDA language only
  string(JOIN ";" FP8_CUDA_FLAGS_JOINED "${FP8_CUDA_FLAGS_LIST}")
  set(FP8_CUDA_FLAGS "$<$<COMPILE_LANGUAGE:CUDA>:${FP8_CUDA_FLAGS_JOINED}>")

  # Define FP8-specific include directories
  set(FP8_INCLUDE_DIRS
      "${TORCH_INCLUDE_DIRS}"
      "${CUDA_INCLUDE_DIRS}"
      "${Python3_INCLUDE_DIRS}"
      "${FA3_INCLUDE_DIR}"
      "${CUTLASS_INCLUDE_DIRS}"
      "${FP8_UTILS_DIR}"
  )

  # Add the FP8 test with improved readability
  configure_flashinfer_target(
    TARGET_NAME test_single_prefill_fa3_sm90
    SOURCES "fp8/test_single_prefill_fa3_sm90.cu"
    LINK_LIBS "FA3_LIB;${TORCH_LIBRARIES}"
    COMPILE_FLAGS "${FP8_CUDA_FLAGS}"
    INCLUDE_DIRS "${FP8_INCLUDE_DIRS}"
  )
endif()

# cmake-format: on

# === Test Discovery and Targets ===
message(STATUS "All unit test targets: ${ALL_TEST_TARGETS}")

# Use GoogleTest's discover_tests to find all test cases
foreach(test_target IN LISTS ALL_TEST_TARGETS)
  if(TARGET ${test_target})
    gtest_discover_tests(${test_target})
  endif()
endforeach()

# Create target to build all tests
add_custom_target(build_tests)
add_dependencies(build_tests ${ALL_TEST_TARGETS})

# Setup "check" target similar to autotools
set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure)
add_custom_target(check COMMAND ${CMAKE_COMMAND} ${CMAKE_CTEST_COMMAND})
add_dependencies(check build_tests)

enable_testing()
