# Function to determine the project version from git tags
function(set_current_release)
  set(PROJECT_VERSION
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
      set(PROJECT_VERSION
          ${VERSION_LOCAL}
          PARENT_SCOPE)
      message(STATUS "PROJECT_VERSION: ${VERSION_LOCAL}")
    else()
      message(STATUS "Git describe failed, using default PROJECT_VERSION 0.0.0")
    endif()
  endif(GIT_FOUND)
endfunction(set_current_release)
