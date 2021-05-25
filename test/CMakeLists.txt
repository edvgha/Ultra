include_directories(${ULTRA_HEADERS_DIR})
include_directories(${TORCH_INCLUDE_DIRS})

set(MODULE_SOURCE_FILES module_tests.cc)
add_executable(module_tests ${MODULE_SOURCE_FILES})
add_test(NAME module_tests COMMAND module_tests)
target_link_libraries(module_tests ultra)

add_executable(exe_graph execute_graph.cc)
add_test(NAME exe_graph COMMAND exe_graph)
target_link_libraries(exe_graph ultra)

# Add unit test dir
add_subdirectory(ir_if)
add_subdirectory(ir_if_1)
add_subdirectory(ir_if_2)
add_subdirectory(ir_if_3)
add_subdirectory(ir_if_4)
add_subdirectory(ir_for)
add_subdirectory(ir_for_for)
add_subdirectory(ir_for_for_if)
add_subdirectory(ir_list_pack_unpack)
add_subdirectory(ir_tuple_pack_unpack)
