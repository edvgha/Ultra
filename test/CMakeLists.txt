include_directories(${ULTRA_HEADERS_DIR})
include_directories(${TORCH_INCLUDE_DIRS})

set(UNIT_SOURCE_FILES unit_tests.cc)
add_executable(unit_tests ${UNIT_SOURCE_FILES})
add_test(NAME unit_tests COMMAND unit_tests)
target_link_libraries(unit_tests ultra)

set(MODULE_SOURCE_FILES module_tests.cc)
add_executable(module_tests ${MODULE_SOURCE_FILES})
add_test(NAME module_tests COMMAND module_tests)
target_link_libraries(module_tests ultra)

add_executable(exe_graph execute_graph.cc)
add_test(NAME exe_graph COMMAND exe_graph)
target_link_libraries(exe_graph ultra)
