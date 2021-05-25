get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
add_executable(${ProjectId} "${ProjectId}.cc")
target_link_libraries(${ProjectId} ultra)
add_test(NAME ${ProjectId} COMMAND sh ${ULTRA_SCRIPTS}/test.sh ${ProjectId})
