set(PROJECT_WORKSPACE ${PROJECT_SOURCE_DIR}/testM)
add_executable(testM synthetic_main.cpp ${PROJECT_WORKSPACE}/forward.h ${PROJECT_WORKSPACE}/forward.cpp)
target_link_libraries(testM "${TORCH_LIBRARIES}")