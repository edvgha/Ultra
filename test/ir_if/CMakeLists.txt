get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
add_executable(unit_test "${ProjectId}.cc")
target_link_libraries(unit_test ultra)
add_test(NAME ${ProjectId} COMMAND sh ${ULTRA_SCRIPTS}/test.sh)