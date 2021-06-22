add_subdirectory(ultra)
#add_subdirectory(opt)

set(SOURCE_FILES synthetic_forward.cc)

add_executable(synthetic_forward ${SOURCE_FILES})
target_link_libraries(synthetic_forward ultra)
install(TARGETS synthetic_forward DESTINATION ${ULTRA_INSTALL_BIN_DIR})
