# Function to detect available HIP architectures
function(detect_hip_architectures output_var)
  set(detected_archs "")

  # Check if HIP language is enabled and compiler is available
  if(CMAKE_HIP_COMPILER)
    message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
    set(ROCM_AGENT_ENUMERATOR "rocm_agent_enumerator")

    # Check platform
    execute_process(
      COMMAND ${ROCM_AGENT_ENUMERATOR}
      OUTPUT_VARIABLE rocm_devices
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(rocm_devices)
      # Parse architecture values
      string(REGEX MATCHALL "gfx[0-9a-z]+" arch_matches "${rocm_devices}")
      foreach(match ${arch_matches})
        if(NOT "${match}" STREQUAL "gfx000")
          list(APPEND detected_archs "${match}")
        endif()
      endforeach()
      if(detected_archs)
        list(REMOVE_DUPLICATES detected_archs)
        message(STATUS "Detected HIP architectures: ${detected_archs}")
      else()
        message(
          STATUS "No valid GPU architectures detected (found: ${arch_matches})")
      endif()
    else()
      message(STATUS "No HIP architectures detected automatically")
    endif()

  endif()

  set(${output_var}
      "${detected_archs}"
      PARENT_SCOPE)

endfunction()

# Function to generate HIP architecture flags
function(generate_hip_arch_flags arch_list output_var)
  set(hip_arch_flags "")

  foreach(arch ${arch_list})
    list(APPEND hip_arch_flags "--offload-arch=${arch}")
  endforeach()

  set(${output_var}
      "${hip_arch_flags}"
      PARENT_SCOPE)
endfunction()
