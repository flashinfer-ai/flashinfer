function(_add_torch_extension)
  set(options PY_LIMITED_API)
  set(oneValueArgs EXT_NAME MIN_PYTHON_ABI)
  set(multiValueArgs SOURCES LINK_LIBS LINK_LIB_DIRS COMPILE_FLAGS INCLUDE_DIRS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")

  # Validate required arguments
  if(NOT DEFINED arg_EXT_NAME)
    message(FATAL_ERROR "EXT_NAME is required")
  endif()

  if(NOT DEFINED arg_SOURCES)
    message(FATAL_ERROR "SOURCES is required")
  endif()

  # Create the Python module
  add_library(${arg_EXT_NAME} MODULE ${arg_SOURCES})

  # Set position-independent code flag (required for Python modules)
  set_target_properties(${arg_EXT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Find Python include directories
  if(arg_PY_LIMITED_API)
    # For limited API, we need a specific version format
    if(DEFINED arg_MIN_PYTHON_ABI)
      # Parse the version string (format: 3.8, 3.9, etc.)
      string(REPLACE "." ";" VERSION_LIST ${arg_MIN_PYTHON_ABI})
      list(GET VERSION_LIST 0 MAJOR)
      list(GET VERSION_LIST 1 MINOR)

      if(MAJOR EQUAL 3 AND MINOR EQUAL 8)
        set(PY_VERSION_HEX "0x03080000")
      elseif(MAJOR EQUAL 3 AND MINOR EQUAL 9)
        set(PY_VERSION_HEX "0x03090000")
      elseif(MAJOR EQUAL 3 AND MINOR EQUAL 10)
        set(PY_VERSION_HEX "0x030A0000")
      elseif(MAJOR EQUAL 3 AND MINOR EQUAL 11)
        set(PY_VERSION_HEX "0x030B0000")
      else()
        set(PY_VERSION_HEX "0x03080000") # Default to 3.8
      endif()
    else()
      # Default to Python 3.8
      set(PY_VERSION_HEX "0x03080000")
    endif()

    target_compile_definitions(${arg_EXT_NAME}
                               PRIVATE "Py_LIMITED_API=${PY_VERSION_HEX}")
  endif()

  # Explicitly add Python include directories
  target_include_directories(${arg_EXT_NAME} PRIVATE ${Python_INCLUDE_DIRS})

  # Add include directories from arguments
  if(DEFINED arg_INCLUDE_DIRS)
    target_include_directories(${arg_EXT_NAME} PRIVATE ${arg_INCLUDE_DIRS})
  endif()

  # Add library directories
  if(DEFINED arg_LINK_LIB_DIRS)
    target_link_directories(${arg_EXT_NAME} PRIVATE ${arg_LINK_LIB_DIRS})
  endif()

  # Common PyTorch libraries for all extensions
  set(COMMON_LIBS c10 torch torch_cpu)

  # Prepare libraries to link
  set(LIBS_TO_LINK ${arg_LINK_LIBS} ${COMMON_LIBS})

  # Handle PY_LIMITED_API and conditional libraries
  if(arg_PY_LIMITED_API)
    # Handle configurable Python ABI version
    if(DEFINED arg_MIN_PYTHON_ABI)
      # Parse the version string (format: 3.8, 3.9, etc.)
      string(REPLACE "." ";" VERSION_LIST ${arg_MIN_PYTHON_ABI})
      list(GET VERSION_LIST 0 MAJOR)
      list(GET VERSION_LIST 1 MINOR)

      # Convert to hex format: 0x030X0000
      math(EXPR PY_VERSION_HEX "(${MAJOR} << 24) | (${MINOR} << 16)")
      # Format it as hex with leading 0x
      string(REGEX REPLACE "^(.+)$" "0x\\1" PY_VERSION_HEX "${PY_VERSION_HEX}")
    else()
      # Default to Python 3.8
      set(PY_VERSION_HEX "0x03080000")
    endif()

    target_compile_definitions(${arg_EXT_NAME}
                               PRIVATE "Py_LIMITED_API=${PY_VERSION_HEX}")
  else()
    # Add torch_python only if not using limited API
    list(APPEND LIBS_TO_LINK torch_python)
  endif()

  # Add link libraries
  if(DEFINED LIBS_TO_LINK)
    target_link_libraries(${arg_EXT_NAME} PRIVATE ${LIBS_TO_LINK})
  endif()

  # Add compile flags
  if(DEFINED arg_COMPILE_FLAGS)
    target_compile_options(${arg_EXT_NAME} PRIVATE ${arg_COMPILE_FLAGS})
  endif()

  set(CXX11_ABI 0)
  if(FLASHINFER_USE_CXX11_ABI)
    set(CXX11_ABI 1)
  endif()

  target_compile_definitions(
    ${arg_EXT_NAME}
    PRIVATE TORCH_EXTENSION_NAME=${arg_EXT_NAME}
            PYBIND11_COMPILER_TYPE="_gcc"
            PYBIND11_STDLIB="_libstdcpp"
            PYBIND11_BUILD_ABI="_cxxabi1011"
            NDEBUG
            TORCH_API_INCLUDE_EXTENSION_H
            _GLIBCXX_USE_CXX11_ABI=${CXX11_ABI})

  if(arg_PY_LIMITED_API)
    set_target_properties(
      ${arg_EXT_NAME} PROPERTIES SUFFIX ".abi3${CMAKE_SHARED_MODULE_SUFFIX}")
  endif()

  # Use only the simple, standard linking flags PyTorch uses (no exotic flags)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
      ${arg_EXT_NAME}
      PRIVATE
      "LINKER:--allow-shlib-undefined"
      "LINKER:--gc-sections"
      "LINKER:-O2"
      "LINKER:--sort-common"
      "LINKER:--as-needed"
      "LINKER:-z,relro"
      "LINKER:-z,now"
      "LINKER:--disable-new-dtags")
  endif()

  # Set RPATH the same way PyTorch does
  set_target_properties(
    ${arg_EXT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
                               INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")

  # Set up Python module naming conventions (no lib prefix)
  set_target_properties(${arg_EXT_NAME} PROPERTIES PREFIX "")

endfunction()

#[=======================================================================[.rst:
.. command:: add_cuda_torch_extension

  Creates a Python module that extends PyTorch with CUDA support.

  .. code-block:: cmake

    add_cuda_torch_extension(
      EXT_NAME module_name
      SOURCES src1.cpp src2.cu [...]
      [INCLUDE_DIRS dir1 dir2 [...]]
      [LINK_LIBS lib1 lib2 [...]]
      [LINK_LIB_DIRS dir1 dir2 [...]]
      [COMPILE_FLAGS flag1 flag2 [...]]
      [DLINK]
      [DLINK_LIBS lib1 lib2 [...]]
      [DLINK_LIB_DIRS dir1 dir2 [...]]
      [PY_LIMITED_API]
      [MIN_PYTHON_ABI "3.8"]
    )

  ``EXT_NAME`` (required)
    Name of the extension module without a file extension (e.g. "my_ext").

  ``SOURCES`` (required)
    List of C++/CUDA source files for the extension.

  ``INCLUDE_DIRS``
    Additional include directories for compiling the extension.

  ``LINK_LIBS``
    Additional libraries to link against (beyond the automatic PyTorch libraries).
    The module automatically links against: c10, torch, torch_cpu, cudart, c10_cuda,
    torch_cuda and (unless PY_LIMITED_API is specified) torch_python.

  ``LINK_LIB_DIRS``
    Additional library directories to search when linking.

  ``COMPILE_FLAGS``
    Additional compiler flags for building the extension sources.
    The module automatically adds platform-specific flags for Python extension
    compatibility (-Xfatbin -compress-all) and on Linux (-fno-gnu-unique).

  ``DLINK``
    Enable CUDA device link-time optimization (device code linking).

  ``DLINK_LIBS``
    Additional device libraries to link against during device linking.
    Implies DLINK.

  ``DLINK_LIB_DIRS``
    Additional library directories for device linking.

  ``PY_LIMITED_API``
    Build with Python's limited API for improved binary compatibility
    across Python versions. When enabled, torch_python is not linked.

  ``MIN_PYTHON_ABI``
    Minimum Python version for limited API compatibility (default: "3.8").
    Only relevant when PY_LIMITED_API is set.
#]=======================================================================]
function(add_cuda_torch_extension)
  # Parse arguments to get extension name and all other args
  set(options PY_LIMITED_API DLINK) # DLINK remains here
  set(oneValueArgs EXT_NAME MIN_PYTHON_ABI)
  set(multiValueArgs
      SOURCES
      LINK_LIBS
      DLINK_LIBS
      LINK_LIB_DIRS
      DLINK_LIB_DIRS
      COMPILE_FLAGS
      INCLUDE_DIRS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")
  set(CUDA_SPECIFIC_LIBS cudart c10_cuda torch_cuda)

  # Add library paths
  if(NOT DEFINED arg_LINK_LIB_DIRS)
    set(arg_LINK_LIB_DIRS "")
  endif()

  # Add CUDA libraries
  if(DEFINED arg_LINK_LIBS)
    list(APPEND arg_LINK_LIBS ${CUDA_SPECIFIC_LIBS})
  else()
    set(arg_LINK_LIBS ${CUDA_SPECIFIC_LIBS})
  endif()

  # Add CUDA-specific compile flags - only keeping extension-specific ones
  if(NOT DEFINED arg_COMPILE_FLAGS)
    set(arg_COMPILE_FLAGS "")
  endif()

  if(NOT DEFINED arg_PY_LIMITED_API)
    set(arg_PY_LIMITED_API OFF)
  endif()

  # Add extension-specific binary flags
  list(APPEND arg_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin>
       $<$<COMPILE_LANGUAGE:CUDA>:-compress-all>)

  # Add Linux-specific flags for Python compatibility
  if(UNIX AND NOT APPLE)
    list(APPEND arg_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>
         $<$<COMPILE_LANGUAGE:CUDA>:-fno-gnu-unique>)
  endif()

  # Create the extension first
  # cmake-format: off
  _add_torch_extension(
    EXT_NAME ${arg_EXT_NAME}
    SOURCES ${arg_SOURCES}
    LINK_LIBS ${arg_LINK_LIBS}
    LINK_LIB_DIRS ${arg_LINK_LIB_DIRS}
    COMPILE_FLAGS ${arg_COMPILE_FLAGS}
    INCLUDE_DIRS ${arg_INCLUDE_DIRS}
    MIN_PYTHON_ABI ${arg_MIN_PYTHON_ABI}
    PY_LIMITED_API ${arg_PY_LIMITED_API}
  )
  # cmake-format: on
  # Handle device link time optimization (moved from helper function)
  if(arg_DLINK OR DEFINED arg_DLINK_LIBS)
    set(DLINK_FLAGS "-dlink")

    # Add library directories to dlink flags
    foreach(dir ${arg_DLINK_LIB_DIRS})
      list(APPEND DLINK_FLAGS "-L${dir}")
    endforeach()

    # Add libraries to dlink flags
    foreach(lib ${arg_DLINK_LIBS})
      list(APPEND DLINK_FLAGS "-l${lib}")
    endforeach()

    # Add device link time optimization if CUDA version >= 11.2
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
      list(APPEND DLINK_FLAGS "-dlto")
    endif()

    # Set CUDA separable compilation
    set_target_properties(
      ${arg_EXT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                 CUDA_DLINK_OPTIONS "${DLINK_FLAGS}")
  endif()
endfunction()

#[=======================================================================[.rst:
.. command:: add_hip_torch_extension

  Creates a Python module that extends PyTorch with AMD ROCm/HIP support.

  .. code-block:: cmake

    add_hip_torch_extension(
      EXT_NAME module_name
      SOURCES src1.cpp src2.hip [...]
      [INCLUDE_DIRS dir1 dir2 [...]]
      [LINK_LIBS lib1 lib2 [...]]
      [LINK_LIB_DIRS dir1 dir2 [...]]
      [COMPILE_FLAGS flag1 flag2 [...]]
      [PY_LIMITED_API]
      [MIN_PYTHON_ABI "3.8"]
    )

  ``EXT_NAME`` (required)
    Name of the extension module without a file extension (e.g. "my_ext").

  ``SOURCES`` (required)
    List of C++/HIP source files for the extension.

  ``INCLUDE_DIRS``
    Additional include directories for compiling the extension.

  ``LINK_LIBS``
    Additional libraries to link against (beyond the automatic PyTorch libraries).
    The module automatically links against: c10, torch, torch_cpu, amdhip64,
    c10_hip, torch_hip and (unless PY_LIMITED_API is specified) torch_python.

  ``LINK_LIB_DIRS``
    Additional library directories to search when linking.

  ``COMPILE_FLAGS``
    Additional compiler flags for building the extension sources.

  ``PY_LIMITED_API``
    Build with Python's limited API for improved binary compatibility
    across Python versions. When enabled, torch_python is not linked.

  ``MIN_PYTHON_ABI``
    Minimum Python version for limited API compatibility (default: "3.8").
    Only relevant when PY_LIMITED_API is set.
#]=======================================================================]
function(add_hip_torch_extension)
  set(options PY_LIMITED_API) # Removed DLINK from here
  set(oneValueArgs EXT_NAME MIN_PYTHON_ABI)
  set(multiValueArgs SOURCES LINK_LIBS LINK_LIB_DIRS COMPILE_FLAGS INCLUDE_DIRS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")

  # HIP specific libraries
  set(HIP_SPECIFIC_LIBS amdhip64 c10_hip torch_hip)

  # Add HIP libraries
  if(DEFINED arg_LINK_LIBS)
    list(APPEND arg_LINK_LIBS ${HIP_SPECIFIC_LIBS})
  else()
    set(arg_LINK_LIBS ${HIP_SPECIFIC_LIBS})
  endif()

  if(NOT DEFINED arg_PY_LIMITED_API)
    set(arg_PY_LIMITED_API OFF)
  endif()

  _add_torch_extension(
    EXT_NAME
    ${arg_EXT_NAME}
    SOURCES
    ${arg_SOURCES}
    LINK_LIBS
    ${arg_LINK_LIBS}
    LINK_LIB_DIRS
    ${arg_LINK_LIB_DIRS}
    COMPILE_FLAGS
    ${arg_COMPILE_FLAGS}
    INCLUDE_DIRS
    ${arg_INCLUDE_DIRS}
    MIN_PYTHON_ABI
    ${arg_MIN_PYTHON_ABI}
    PY_LIMITED_API
    ${arg_PY_LIMITED_API})
endfunction()
