set(SOURCE_FILES
    ultra.h
    ultra.cc
    ultra_replace_passes.cc
    mutator_passes.cc
    native_library.cc
    schema.cc
)

add_library(ultra SHARED STATIC ${SOURCE_FILES})
target_link_libraries(ultra "${TORCH_LIBRARIES}")

install(TARGETS ultra DESTINATION ${ULTRA_INSTALL_LIB_DIR})
install(FILES ultra.h DESTINATION ${ULTRA_INSTALL_INCLUDE_DIR})