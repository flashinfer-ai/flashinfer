# ------------------------------ Dependencies ---------------------------------#

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5 # release-1.16.0
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_MakeAvailable(googletest)
# -----------------------------------------------------------------------------#

# GoogleTest has to be included so that gtest_discover_tests works
include(GoogleTest)
# Thread support is needed for gtest
find_package(Threads REQUIRED)

set(ALL_TEST_TARGETS "")

function(configure_unit_test)
  set(options OPTIONAL IS_GTEST)
  set(oneValueArgs TEST_TARGET_NAME)
  set(multiValueArgs TEST_SOURCES EXTRA_LINK_LIBS EXTRA_COMPILE_FLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")

  message(STATUS "Configure ${arg_TEST_TARGET_NAME} unit test")

  add_executable(${arg_TEST_TARGET_NAME} EXCLUDE_FROM_ALL ${arg_TEST_SOURCES})
  target_include_directories(
    ${arg_TEST_TARGET_NAME}
    PRIVATE ${FLASHINFER_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
            ${FLASHINFER_UTILS_INCLUDE_DIR} ${GENERATED_SOURCE_DIR_ROOT})
  # TODO: Why is this needed at all? The use of dispatch_inc has to be
  # understood. For now we need it as otherwise the generated sources do not
  # compile.
  add_dependencies(${arg_TEST_TARGET_NAME} dispatch_inc)
  foreach(extra_link_lib IN LISTS arg_EXTRA_LINK_LIBS)
    target_link_libraries(${arg_TEST_TARGET_NAME} PRIVATE ${extra_link_lib})
  endforeach(extra_link_lib)

  foreach(copt IN LISTS arg_EXTRA_COMPILE_FLAGS)
    target_compile_options(${arg_TEST_TARGET_NAME} PRIVATE ${copt})
  endforeach(copt)

  if(${arg_IS_GTEST})
    target_include_directories(
      ${arg_TEST_TARGET_NAME} PRIVATE ${gtest_SOURCE_DIR}/include
                                      ${gtest_SOURCE_DIR})
    target_link_libraries(${arg_TEST_TARGET_NAME} PRIVATE gtest gtest_main)
  elseif(${arg_IS_GTEST})
    add_test(NAME ${arg_TEST_TARGET_NAME} COMMAND ${arg_TEST_TARGET_NAME})
  endif(${arg_IS_GTEST})

  # Add the target to the ALL_TEST_TARGETS_LIST list so that it gets built when
  # the build_tests target is invoked.
  list(APPEND ALL_TEST_TARGETS "${arg_TEST_TARGET_NAME}")
  set(ALL_TEST_TARGETS
      "${ALL_TEST_TARGETS}"
      PARENT_SCOPE)
  message(STATUS "Configured ${arg_TEST_TARGET_NAME} unit test")

endfunction(configure_unit_test)

function(configure_distributed_unit_tests)
  find_package(MPI)

  if(NOT MPI_FOUND)
    message(
      WARNING "Cannot configure distributed unit tests as MPI package not found"
    )
    return()
  endif(NOT MPI_FOUND)

  # cmake-format: off
  configure_unit_test(
    TEST_TARGET_NAME test_sum_all_reduce
    TEST_SOURCES "test_sum_all_reduce.cu"
    EXTRA_LINK_LIBS "MPI::MPI_CXX;mscclpp;"
    EXTRA_COMPILE_FLAGS "-DENABLE_MPI"
    IS_GTEST OFF
  )
  configure_unit_test(
    TEST_TARGET_NAME test_attn_all_reduce
    TEST_SOURCES "test_attn_all_reduce.cu"
    EXTRA_LINK_LIBS "MPI::MPI_CXX;mscclpp;"
    EXTRA_COMPILE_FLAGS "-DENABLE_MPI"
    IS_GTEST OFF
  )
  # cmake-format: on
endfunction(configure_distributed_unit_tests)

# cmake-format: off
configure_unit_test(
  TEST_TARGET_NAME test_single_decode
  TEST_SOURCES "test_single_decode.cu"
  EXTRA_LINK_LIBS "decode_kernels"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_batch_decode
  TEST_SOURCES "test_batch_decode.cu"
  EXTRA_LINK_LIBS "decode_kernels"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_single_prefill
  TEST_SOURCES "test_single_prefill.cu"
  EXTRA_LINK_LIBS "prefill_kernels"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_batch_prefill
  TEST_SOURCES "test_batch_prefill.cu"
  EXTRA_LINK_LIBS "prefill_kernels"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_page
  TEST_SOURCES "test_page.cu"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_cascade
  TEST_SOURCES "test_cascade.cu"
  EXTRA_LINK_LIBS "decode_kernels;prefill_kernels"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_sampling
  TEST_SOURCES "test_sampling.cu"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_norm
  TEST_SOURCES "test_norm.cu"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_fastdiv
  TEST_SOURCES "test_fastdiv.cu"
  IS_GTEST ON
)
configure_unit_test(
  TEST_TARGET_NAME test_fast_dequant
  TEST_SOURCES "test_fast_dequant.cu"
  IS_GTEST ON
)
# cmake-format: on

if(FLASHINFER_DIST_UNITTESTS)
  configure_distributed_unit_tests()
endif(FLASHINFER_DIST_UNITTESTS)

# A top-level target to build all unit tests
add_custom_target(build_tests)

message(STATUS "All unit test targets: ${ALL_TEST_TARGETS}")

# gtest_discover_tests adds all discovered tests as ctest tests. FIXME: for some
# reason google_discover_tests does not add the ctests
foreach(test_trgt IN LISTS ${ALL_TEST_TARGETS})
  gtest_discover_tests(${test_trgt})
endforeach(test_trgt)

add_dependencies(build_tests ${ALL_TEST_TARGETS})
# Emulate autotools like `make check` target to run all ctests FIXME does not
# work as gtest_discover_tests is failing for some reason
set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure)
add_custom_target(check COMMAND ${CMAKE_COMMAND} ${CMAKE_CTEST_COMMAND})
add_dependencies(check build_tests)

enable_testing()
