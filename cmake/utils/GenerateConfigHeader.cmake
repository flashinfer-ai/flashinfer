# Helper macro for standard variable handling
macro(_add_define SECTION)
  if(_value STREQUAL "ON"
     OR _value STREQUAL "TRUE"
     OR _value STREQUAL "1")
    string(APPEND ${SECTION} "#define ${_var} 1\n")
  elseif(
    _value STREQUAL "OFF"
    OR _value STREQUAL "FALSE"
    OR _value STREQUAL "0")
    string(APPEND ${SECTION} "#define ${_var} 0\n")
  elseif(_value MATCHES "^[0-9]+$")
    string(APPEND ${SECTION} "#define ${_var} ${_value}\n")
  else()
    string(REPLACE "\"" "\\\"" _escaped_value "${_value}")
    string(APPEND ${SECTION} "#define ${_var} \"${_escaped_value}\"\n")
  endif()
endmacro()

# helper function for flag formatting
function(_format_flags_with_line_breaks flag_string output_var)
  string(STRIP "${flag_string}" clean_value)
  string(REPLACE " " ";" flags_list "${clean_value}")

  set(formatted_value "\"")
  set(first_flag TRUE)

  foreach(flag ${flags_list})
    if(first_flag)
      string(APPEND formatted_value "${flag}")
      set(first_flag FALSE)
    else()
      string(APPEND formatted_value " \\\n    ${flag}")
    endif()
  endforeach()

  string(APPEND formatted_value "\"")
  set(${output_var}
      "${formatted_value}"
      PARENT_SCOPE)
endfunction()

function(flashinfer_generate_config_header)
  # Parse function arguments
  set(options "")
  set(oneValueArgs SOURCE_DIR BINARY_DIR INSTALL_DIR COMPONENT)
  set(multiValueArgs EXCLUDE_PATTERNS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # Set defaults with more concise syntax
  if(NOT ARG_SOURCE_DIR)
    set(ARG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  if(NOT ARG_BINARY_DIR)
    set(ARG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  if(NOT ARG_INSTALL_DIR)
    set(ARG_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/flashinfer)
  endif()
  if(NOT ARG_COMPONENT)
    set(ARG_COMPONENT Headers)
  endif()

  # Standard exclusion patterns
  set(EXCLUDE_PATTERNS
      ".*_DIR$"
      ".*_PATH$"
      ".*_FOUND$"
      ".*_FILE$"
      "FLASHINFER_CONFIG_DEFINES"
      "FLASHINFER_ALL_OPTIONS"
      ".*GENERATED.*DIR.*"
      ".*SOURCE.*ROOT.*")

  # Add user exclusions
  list(APPEND EXCLUDE_PATTERNS ${ARG_EXCLUDE_PATTERNS})

  # Ensure output directory exists
  file(MAKE_DIRECTORY "${ARG_BINARY_DIR}/include/flashinfer")

  # Get and filter FLASHINFER variables
  get_cmake_property(_variableNames VARIABLES)
  list(FILTER _variableNames INCLUDE REGEX "^FLASHINFER_[A-Z0-9_]+$")

  # Apply exclude patterns
  foreach(_pattern ${EXCLUDE_PATTERNS})
    list(FILTER _variableNames EXCLUDE REGEX "${_pattern}")
  endforeach()

  # Clean up variable list
  list(SORT _variableNames)
  list(REMOVE_DUPLICATES _variableNames)

  # Initialize section buffers
  set(VERSION_DEFINES "")
  set(COMPILER_FLAGS_DEFINES "")
  set(FEATURE_DEFINES "")
  set(BUILD_CONFIG_DEFINES "")

  # Process each variable by category
  foreach(_var ${_variableNames})
    if(DEFINED ${_var})
      set(_value "${${_var}}")

      # Categorize and process variables
      if(_var MATCHES "^FLASHINFER_VERSION_")
        _add_define(VERSION_DEFINES)
      elseif(_var MATCHES "FLAGS$" OR _var MATCHES "_ARGS$")
        # Special handling for compiler flags
        _format_flags_with_line_breaks("${_value}" _formatted_value)

        string(APPEND COMPILER_FLAGS_DEFINES
               "#define ${_var} \\\n    ${_formatted_value}\n\n")
        # Feature flags
      elseif(
        _var MATCHES "^FLASHINFER_ENABLE_"
        OR _var MATCHES "_USE_"
        OR _var MATCHES "^FLASHINFER_GEN_")
        _add_define(FEATURE_DEFINES)
        # Everything else
      else()
        _add_define(BUILD_CONFIG_DEFINES)
      endif()
    endif()
  endforeach()

  # Build unified defines with section headers
  set(FLASHINFER_CONFIG_DEFINES "// Version info\n${VERSION_DEFINES}\n")

  if(NOT "${COMPILER_FLAGS_DEFINES}" STREQUAL "")
    string(APPEND FLASHINFER_CONFIG_DEFINES
           "// Compiler and build flags\n${COMPILER_FLAGS_DEFINES}")
  endif()

  if(NOT "${FEATURE_DEFINES}" STREQUAL "")
    string(APPEND FLASHINFER_CONFIG_DEFINES
           "// Feature configuration\n${FEATURE_DEFINES}\n")
  endif()

  if(NOT "${BUILD_CONFIG_DEFINES}" STREQUAL "")
    string(APPEND FLASHINFER_CONFIG_DEFINES
           "// Build configuration\n${BUILD_CONFIG_DEFINES}")
  endif()

  # Generate and install the configuration file
  set(FLASHINFER_CONFIG_DEFINES
      "${FLASHINFER_CONFIG_DEFINES}"
      PARENT_SCOPE)

  configure_file("${ARG_SOURCE_DIR}/include/flashinfer/configure.h.in"
                 "${ARG_BINARY_DIR}/include/flashinfer/configure.h" @ONLY)

  install(
    FILES "${ARG_BINARY_DIR}/include/flashinfer/configure.h"
    DESTINATION "${ARG_INSTALL_DIR}"
    COMPONENT "${ARG_COMPONENT}")

  message(STATUS "Generated configure.h")
endfunction()
