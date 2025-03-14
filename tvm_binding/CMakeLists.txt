message(STATUS "Compile tvm binding.")

if(NOT FLASHINFER_TVM_SOURCE_DIR STREQUAL "")
  set(TVM_SOURCE_DIR_SET ${FLASHINFER_TVM_SOURCE_DIR})
elseif(DEFINED ENV{TVM_SOURCE_DIR})
  set(TVM_SOURCE_DIR_SET $ENV{TVM_SOURCE_DIR})
elseif(DEFINED ENV{TVM_HOME}) # for backward compatibility
  set(TVM_SOURCE_DIR_SET $ENV{TVM_HOME})
else()
  message(
    FATAL_ERROR
      "Error: Cannot find TVM. Please set the path to TVM by                       \
      1) adding `-DFLASHINFER_TVM_SOURCE_DIR=path/to/tvm` in the cmake command, or \
      2) setting the environment variable `TVM_SOURCE_DIR` to the tvm path.")
endif()
message(STATUS "FlashInfer uses TVM home ${TVM_SOURCE_DIR_SET}.")

file(GLOB_RECURSE TVM_BINDING_SRCS ${PROJECT_SOURCE_DIR}/src/tvm_wrapper.cu)
add_library(flashinfer_tvm OBJECT ${TVM_BINDING_SRCS})
target_compile_definitions(flashinfer_tvm PRIVATE -DDMLC_USE_LOGGING_LIBRARY=
                                                  \<tvm/runtime/logging.h\>)
target_link_libraries(flashinfer_tvm PRIVATE decode_kernels prefill_kernels)
target_include_directories(flashinfer_tvm PRIVATE ${FLASHINFER_INCLUDE_DIR})
target_include_directories(flashinfer_tvm PRIVATE ${TVM_SOURCE_DIR_SET}/include)
target_include_directories(
  flashinfer_tvm PRIVATE ${TVM_SOURCE_DIR_SET}/3rdparty/dlpack/include)
target_include_directories(
  flashinfer_tvm PRIVATE ${TVM_SOURCE_DIR_SET}/3rdparty/dmlc-core/include)
add_dependencies(flashinfer_tvm dispatch_inc)
target_compile_options(flashinfer_tvm PRIVATE -Xcompiler=-fPIC -diag-suppress
                                              "1305" -Wno-switch-bool)

export(
  EXPORT flashinferTVMBindingsTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/flashinferTVMBindingsTargets.cmake"
  NAMESPACE flashinfer::)

install(
  EXPORT flashinferTVMBindingsTargets
  FILE flashinferTVMBindingsTargets.cmake
  NAMESPACE flashinfer::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer)
