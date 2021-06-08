set(PROJECT_WORKSPACE ${PROJECT_SOURCE_DIR}/Demo_LLD6)

add_custom_command(
    OUTPUT ${PROJECT_WORKSPACE}/forward.h ${PROJECT_WORKSPACE}/forward.cpp
    COMMAND ./../src/synthetic_forward ${PROJECT_WORKSPACE}
    DEPENDS synthetic_forward
    COMMENT "Running generator"
    VERBATIM
)

set_source_files_properties(${PROJECT_WORKSPACE}/forward.h PROPERTIES GENERATED TRUE)
set_source_files_properties(${PROJECT_WORKSPACE}/forward.cpp PROPERTIES GENERATED TRUE)

add_executable(synthetic_run_lld6 synthetic_main.cpp ${PROJECT_WORKSPACE}/forward.h ${PROJECT_WORKSPACE}/forward.cpp)
target_link_libraries(synthetic_run_lld6 "${TORCH_LIBRARIES}")
