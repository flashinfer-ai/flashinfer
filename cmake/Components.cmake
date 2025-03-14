# Define the component structure
set(FLASHINFER_COMPONENTS "Headers")

if(FLASHINFER_BUILD_KERNELS)
  list(APPEND FLASHINFER_COMPONENTS "Kernels")

  if(FLASHINFER_TVM_BINDING)
    list(APPEND FLASHINFER_COMPONENTS "TVMBinding")
  endif()
endif()

if(FLASHINFER_DISTRIBUTED)
  list(APPEND FLASHINFER_COMPONENTS "Distributed")
endif()

# Setup component-specific build flags
macro(add_component_flags component)
  add_definitions(-DFLASHINFER_COMPONENT_${component})
endmacro()

# For each enabled component, add compile-time flags
foreach(comp ${FLASHINFER_COMPONENTS})
  add_component_flags(${comp})
endforeach()
