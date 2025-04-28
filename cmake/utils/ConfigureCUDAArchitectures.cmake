# Detects available CUDA architectures on which the build is running and returns
# a list of detected architectures or empty list if none were found.
function(detect_cuda_architectures output_var)
  set(detected_archs "")

  if(CMAKE_CUDA_COMPILER_LOADED)
    # Try to detect CUDA devices
    execute_process(
      COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
      OUTPUT_VARIABLE gpu_detect_output
      ERROR_VARIABLE gpu_detect_error
      RESULT_VARIABLE gpu_detect_result
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(gpu_detect_result EQUAL 0)
      # Parse the architecture values (sm_XX)
      string(REGEX MATCHALL "sm_([0-9]+)" arch_matches "${gpu_detect_output}")
      foreach(match ${arch_matches})
        string(REGEX REPLACE "sm_" "" arch "${match}")
        list(APPEND detected_archs "${arch}")
      endforeach()

      if(detected_archs)
        list(REMOVE_DUPLICATES detected_archs)
        list(SORT detected_archs)
        message(STATUS "Detected CUDA architectures: ${detected_archs}")
      else()
        message(STATUS "No CUDA architectures detected via --list-gpu-arch")
      endif()
    else()
      message(STATUS "Failed to detect CUDA architectures: ${gpu_detect_error}")
    endif()
  else()
    message(STATUS "CUDA compiler not loaded, architecture detection skipped")
  endif()

  # Return all CUDA archs that were found or an empty list is none were found
  set(${output_var}
      "${detected_archs}"
      PARENT_SCOPE)
endfunction()

# Function to generate NVCC architecture flags
function(generate_cuda_arch_flags arch_list output_var)
  set(cuda_arch_flags "")

  foreach(arch ${arch_list})
    if(arch EQUAL 90)
      # Special SM90a flag for Hopper architecture
      list(APPEND cuda_arch_flags
           "--generate-code=arch=compute_90a,code=sm_90a")
    else()
      # Standard architecture flags
      list(APPEND cuda_arch_flags
           "--generate-code=arch=compute_${arch},code=sm_${arch}")
    endif()
  endforeach()

  # Return the flags
  set(${output_var}
      "${cuda_arch_flags}"
      PARENT_SCOPE)
endfunction()

function(validate_arch_list_against_min_supported_arch arch_list
         min_supported_arch)

  set(MIN_CUDA_ARCH 999)
  foreach(arch ${arch_list})
    # Strip any suffixes like 'a' or '+PTX'
    string(REGEX MATCH "^[0-9]+" arch_num "${arch}")
    if(arch_num LESS MIN_CUDA_ARCH)
      set(MIN_CUDA_ARCH ${arch_num})
    endif()
  endforeach()

  if(MIN_CUDA_ARCH LESS ${min_supported_arch})
    message(
      FATAL_ERROR
        "FlashInfer supports only SM${min_supported_arch} or higher. "
        "Unsupported architecture SM${MIN_CUDA_ARCH} detected in "
        "FLASHINFER_CUDA_ARCHITECTURES.")
  endif()

endfunction()
