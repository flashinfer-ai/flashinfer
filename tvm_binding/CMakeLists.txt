message(STATUS "Configuring TVM binding")

# TVM source directory configuration
flashinfer_option(FLASHINFER_TVM_SOURCE_DIR
                  "The path to TVM for building TVM binding." "")

# Check if kernel libraries are available - required for TVM bindings
if(NOT FLASHINFER_BUILD_KERNELS)
  message(
    FATAL_ERROR
      "TVM bindings require kernel libraries. Please enable FLASHINFER_BUILD_KERNELS."
  )
endif()

# Resolve TVM source directory
if(NOT FLASHINFER_TVM_SOURCE_DIR STREQUAL "")
  set(TVM_SOURCE_DIR_SET ${FLASHINFER_TVM_SOURCE_DIR})
elseif(DEFINED ENV{TVM_SOURCE_DIR})
  set(TVM_SOURCE_DIR_SET $ENV{TVM_SOURCE_DIR})
elseif(DEFINED ENV{TVM_HOME}) # for backward compatibility
  set(TVM_SOURCE_DIR_SET $ENV{TVM_HOME})
else()
  message(
    FATAL_ERROR
      "Error: Cannot find TVM. Please set the path to TVM by either:
      1) Adding `-DFLASHINFER_TVM_SOURCE_DIR=path/to/tvm` in the CMake command, or
      2) Setting the environment variable `TVM_SOURCE_DIR` to the TVM path.")
endif()
message(STATUS "FlashInfer uses TVM located at: ${TVM_SOURCE_DIR_SET}")

# Collect TVM binding source files
file(GLOB_RECURSE TVM_BINDING_SRCS ${PROJECT_SOURCE_DIR}/src/tvm_wrapper.cu)

# Create the TVM binding library
add_library(flashinfer_tvm OBJECT ${TVM_BINDING_SRCS})

# Configure compilation options
target_compile_definitions(flashinfer_tvm PRIVATE -DDMLC_USE_LOGGING_LIBRARY=
                                                  \<tvm/runtime/logging.h\>)
target_compile_options(flashinfer_tvm PRIVATE -Xcompiler=-fPIC -diag-suppress
                                              "1305" -Wno-switch-bool)

# Add dependencies
target_link_libraries(flashinfer_tvm PRIVATE decode_kernels prefill_kernels)
add_dependencies(flashinfer_tvm dispatch_inc)

# Set include directories
target_include_directories(
  flashinfer_tvm
  PRIVATE ${FLASHINFER_INCLUDE_DIR} ${TVM_SOURCE_DIR_SET}/include
          ${TVM_SOURCE_DIR_SET}/3rdparty/dlpack/include
          ${TVM_SOURCE_DIR_SET}/3rdparty/dmlc-core/include)

# Export the targets for the build tree
export(
  EXPORT flashinferTVMBindingTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/flashinferTVMBindingTargets.cmake"
  NAMESPACE flashinfer::)

# Install the library
install(
  TARGETS flashinfer_tvm
  EXPORT flashinferTVMBindingTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT TVMBinding)

# Install the export set for the install tree
install(
  EXPORT flashinferTVMBindingTargets
  FILE flashinferTVMBindingTargets.cmake
  NAMESPACE flashinfer::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flashinfer
  COMPONENT TVMBinding)
