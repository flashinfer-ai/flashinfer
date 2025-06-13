# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache 2.0

#[=======================================================================[.rst:
.. command:: flashinfer_diagnose_target

  Provides detailed diagnostic information for a CMake target, useful for
  debugging build issues.

  .. code-block:: cmake

    flashinfer_diagnose_target(target_name)

  ``target_name`` (required)
    The name of the CMake target to diagnose.

  This function outputs comprehensive information about the target, including:

  - Compile definitions, options, and flags
  - Link options and libraries
  - Include directories
  - Interface properties
  - Source file properties
  - PyTorch-specific flags and settings

  The diagnostic information is printed to the CMake output using message(STATUS).
  This is particularly useful for debugging complex build issues related to
  compiler flags, include paths, and library dependencies.

  Example usage:

  .. code-block:: cmake

    add_library(my_lib SHARED my_source.cpp)
    flashinfer_diagnose_target(my_lib)
#]=======================================================================]
function(flashinfer_diagnose_target target_name)
  message(
    STATUS "============= BUILD DIAGNOSTICS: ${target_name} =============")

  # === TARGET PROPERTIES ===
  message(STATUS "--- Target Properties ---")
  foreach(prop COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FLAGS LINK_OPTIONS
               INCLUDE_DIRECTORIES)
    get_target_property(VALUE ${target_name} ${prop})
    message(STATUS "${prop}: ${VALUE}")
  endforeach()

  # === INTERFACE FLAGS ===
  message(STATUS "--- Interface Flags ---")
  get_target_property(interface_flags ${target_name} INTERFACE_COMPILE_OPTIONS)
  if(interface_flags)
    foreach(flag ${interface_flags})
      message(STATUS "Interface flag: ${flag}")
    endforeach()
  else()
    message(STATUS "No interface flags")
  endif()

  # Check interface flags from dependencies
  get_target_property(interface_libs ${target_name} INTERFACE_LINK_LIBRARIES)
  if(interface_libs)
    message(STATUS "Interface libraries: ${interface_libs}")
    foreach(lib ${interface_libs})
      if(TARGET ${lib})
        get_target_property(flags ${lib} INTERFACE_COMPILE_OPTIONS)
        if(flags)
          message(STATUS "Library ${lib} interface flags:")
          foreach(flag ${flags})
            message(STATUS "  - ${flag}")
          endforeach()
        endif()
      endif()
    endforeach()
  else()
    message(STATUS "No interface libraries")
  endif()

  # === FLAG SOURCES ===
  message(STATUS "--- Flag Sources ---")

  # Direct compile options
  get_target_property(direct_flags ${target_name} COMPILE_OPTIONS)
  message(STATUS "Direct COMPILE_OPTIONS: ${direct_flags}")

  # Source file properties
  get_target_property(sources ${target_name} SOURCES)
  message(STATUS "Source files:")
  foreach(source ${sources})
    get_source_file_property(src_flags ${source} COMPILE_FLAGS)
    if(src_flags)
      message(STATUS "  ${source}: ${src_flags}")
    else()
      message(STATUS "  ${source}: no custom flags")
    endif()
  endforeach()

  # Linked libraries
  get_target_property(libs ${target_name} LINK_LIBRARIES)
  if(libs)
    message(STATUS "Linked libraries: ${libs}")
    foreach(lib ${libs})
      if(TARGET ${lib})
        get_target_property(lib_flags ${lib} INTERFACE_COMPILE_OPTIONS)
        if(lib_flags)
          message(STATUS "Library ${lib} interface flags: ${lib_flags}")
        endif()
      endif()
    endforeach()
  endif()

  # Global flags
  message(STATUS "Global CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
  message(STATUS "Global CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  if(CMAKE_CUDA_FLAGS)
    message(STATUS "Global CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
  endif()

  # === PYTORCH-SPECIFIC FLAGS ===
  message(STATUS "--- PyTorch-Specific Flags ---")
  foreach(
    lib
    torch
    torch_cpu
    torch_cuda
    torch_hip
    c10
    c10_cuda
    c10_hip)
    if(TARGET ${lib})
      message(STATUS "PyTorch library: ${lib}")

      get_target_property(defs ${lib} INTERFACE_COMPILE_DEFINITIONS)
      if(defs)
        message(STATUS "  Definitions: ${defs}")
      endif()

      get_target_property(opts ${lib} INTERFACE_COMPILE_OPTIONS)
      if(opts)
        message(STATUS "  Options: ${opts}")
      endif()

      get_target_property(features ${lib} INTERFACE_COMPILE_FEATURES)
      if(features)
        message(STATUS "  Features: ${features}")
      endif()
    endif()
  endforeach()

  message(
    STATUS "===============================================================")
endfunction()
