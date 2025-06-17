function(flashinfer_configure_prebuilt_uris)
    message(STATUS "Configuring prebuilt URIs")
    get_property(PREBUILT_URI_LIST GLOBAL PROPERTY FLASHINFER_PREBUILT_URIS)
    set(PYTHON_URI "")
    message(STATUS "PREBUILT_URI_LIST: ${PREBUILT_URI_LIST}")

    string(REPLACE ";" "\", \"" list_items "${PREBUILT_URI_LIST}")
    set(PYTHON_URI "${list_items}")

    message(STATUS "PYTHON_URI: ${PYTHON_URI}")
    
    set(TEMPLATE_FILE "${CMAKE_SOURCE_DIR}/templates/__aot_prebuilt_uris__.py.in")    
    set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/flashinfer/__aot_prebuilt_uris__.py")
    set(INSTALL_DIR "flashinfer")

    configure_file("${TEMPLATE_FILE}" "${OUTPUT_FILE}" @ONLY)

    install(FILES "${OUTPUT_FILE}" DESTINATION "${INSTALL_DIR}")
endfunction()
