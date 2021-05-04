set(SOURCE_FILES
    outs.h
    outs.cc
)

add_library(outs STATIC ${SOURCE_FILES})
target_link_libraries(outs "${TORCH_LIBRARIES}")

install(TARGETS outs DESTINATION ${ULTRA_INSTALL_LIB_DIR})
install(FILES outs.h DESTINATION ${ULTRA_INSTALL_INCLUDE_DIR})