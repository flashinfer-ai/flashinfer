# Computed the head dims for Nvidia sm90 devices based on the
# FLASHINFER_GEN_HEAD_DIMS and FLASHINFER_SM90_ALLOWED_HEAD_DIMS options
function(flashinfer_compute_sm90_head_dims)
  set(oneValueArgs RESULT)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT DEFINED ARG_RESULT)
    message(
      FATAL_ERROR "flashinfer_compute_sm90_head_dims: missing RESULT argument")
  endif()

  set(HEAD_DIMS_SM90 "")

  foreach(DIM_VAL ${FLASHINFER_GEN_HEAD_DIMS})
    string(CONCAT TUPLE_VAL "${DIM_VAL}" "," "${DIM_VAL}")
    list(FIND FLASHINFER_SM90_ALLOWED_HEAD_DIMS ${TUPLE_VAL} RESULT)
    if(NOT ${RESULT} EQUAL -1)
      list(APPEND HEAD_DIMS_SM90 ${TUPLE_VAL})
    endif()
  endforeach()

  foreach(TUPLE_VAL ${FLASHINFER_SM90_ALLOWED_HEAD_DIMS})
    string(REPLACE "," ";" HEAD_DIMS_LIST ${TUPLE_VAL})
    list(GET HEAD_DIMS_LIST 0 K)
    list(GET HEAD_DIMS_LIST 1 V)
    if(NOT K EQUAL V)
      list(APPEND HEAD_DIMS_SM90 ${TUPLE_VAL})
    endif()
  endforeach()

  list(REMOVE_DUPLICATES HEAD_DIMS_SM90)
  set(${ARG_RESULT}
      ${HEAD_DIMS_SM90}
      PARENT_SCOPE)
endfunction()
