# Check if a ROCM_PATH envvar is set and verify it exists. If a valid ROCM_PATH
# exists it will be used to load the HIP cmake modules.
if(DEFINED ENV{ROCM_PATH})
  if(EXISTS $ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  else()
    message(
      FATAL_ERROR
        "ROCM_PATH environment variable is set to $ENV{ROCM_PATH} that "
        " does not exist.\n"
        "Set a valid ROCM_PATH or unset the ROCM_PATH environment variable")
  endif()
endif()

if(NOT ROCM_PATH)
  if(UNIX AND EXISTS /opt/rocm)
    set(ROCM_PATH /opt/rocm)
  else()
    message(
      FATAL_ERROR
        "No ROCm installation found. Set a valid ROCM_PATH or install ROCm to "
        "the default location (/opt/rocm).")
  endif()
endif()

# Update CMAKE_MODULE_PATH and CMAKE_PREFIX_PATH to ensure we can find the HIP
# package and other ROCM components
if(UNIX)
  set(CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip ${CMAKE_MODULE_PATH})
  list(APPEND CMAKE_PREFIX_PATH${ROCM_PATH})
endif()
