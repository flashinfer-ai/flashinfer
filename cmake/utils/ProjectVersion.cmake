# Function to determine the project version from git tags
function(set_current_release)
  set(PROJECT_VERSION
      "0.0.0"
      PARENT_SCOPE)
  set(PROJECT_VERSION_FULL
      "0.0.0"
      PARENT_SCOPE)

  # Use git describe to get latest tag name
  if(GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
      RESULT_VARIABLE result
      OUTPUT_VARIABLE VERSION_FROM_GIT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Check if git command succeeded
    if(result EQUAL 0)
      set(VERSION_LOCAL ${VERSION_FROM_GIT})

      # Remove non-numeric prefix (e.g., "v1.2.3" -> "1.2.3")
      if(VERSION_LOCAL MATCHES "^[^0-9]+([0-9].*)$")
        string(REGEX REPLACE "^[^0-9]+" "" VERSION_LOCAL ${VERSION_LOCAL})
        message(
          STATUS
            "Removed non-numeric prefix from version: ${VERSION_FROM_GIT} -> ${VERSION_LOCAL}"
        )
      endif()

      # Validate format (should be at least MAJOR.MINOR)
      if(NOT VERSION_LOCAL MATCHES "^[0-9]+\\.[0-9]+")
        message(
          WARNING
            "Git tag '${VERSION_FROM_GIT}' doesn't appear to be a valid version number. Using default 0.0.0"
        )
        set(VERSION_LOCAL "0.0.0")
      endif()

      # Store the clean numeric version (X.Y.Z) before adding any suffixes
      # Extract X.Y.Z components only
      if(VERSION_LOCAL MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(VERSION_NUMERIC ${CMAKE_MATCH_1})
      else()
        # If it's X.Y format, default to X.Y.0
        if(VERSION_LOCAL MATCHES "^([0-9]+\\.[0-9]+)$")
          set(VERSION_NUMERIC "${VERSION_LOCAL}.0")
        else()
          set(VERSION_NUMERIC "0.0.0")
        endif()
      endif()

      # Save full version starting with numeric base
      set(VERSION_FULL ${VERSION_LOCAL})

      # Check if there are commits since the tag
      execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long
        OUTPUT_VARIABLE GIT_DESCRIBE_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        RESULT_VARIABLE GIT_LONG_RESULT)

      if(GIT_LONG_RESULT EQUAL 0)
        # Extract commit count since tag
        string(REGEX MATCH "-([0-9]+)-g" COMMIT_COUNT_MATCH
                     "${GIT_DESCRIBE_OUTPUT}")
        if(COMMIT_COUNT_MATCH)
          string(REGEX REPLACE ".*-([0-9]+)-g.*" "\\1" COMMIT_COUNT
                               "${GIT_DESCRIBE_OUTPUT}")
          if(NOT COMMIT_COUNT EQUAL 0)
            # Append dev suffix for commits after tag (to full version only)
            set(VERSION_FULL "${VERSION_FULL}.dev${COMMIT_COUNT}")
            message(STATUS "Development version detected: ${VERSION_FULL}")
          endif()
        endif()
      endif()

      # Add version suffix if specified (to full version only)
      if(DEFINED FLASHINFER_VERSION_SUFFIX
         AND NOT "${FLASHINFER_VERSION_SUFFIX}" STREQUAL "")
        set(VERSION_FULL "${VERSION_FULL}+${FLASHINFER_VERSION_SUFFIX}")
        message(STATUS "Adding version suffix: ${VERSION_FULL}")
      endif()

      # Set both versions in parent scope
      set(PROJECT_VERSION
          ${VERSION_NUMERIC}
          PARENT_SCOPE)
      set(PROJECT_VERSION_FULL
          ${VERSION_FULL}
          PARENT_SCOPE)

      message(STATUS "CMake version: ${VERSION_NUMERIC}")
      message(STATUS "Full version: ${VERSION_FULL}")
    else()
      message(STATUS "Git describe failed, using default versions 0.0.0")
    endif()
  endif(GIT_FOUND)
endfunction(set_current_release)
