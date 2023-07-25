project(flash-attention LANGUAGES CXX CUDA)

find_package(Git QUIET REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)

set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)

add_subdirectory(flash_attn)
