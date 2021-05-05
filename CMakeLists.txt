cmake_minimum_required(VERSION 3.18.2 FATAL_ERROR)
project(Ultra CXX)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/ext/libtorch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++17 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(ULTRA_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(ULTRA_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(ULTRA_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(ULTRA_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/ultra ${PROJECT_SOURCE_DIR}/src/out)

include_directories(${ULTRA_INSTALL_INCLUDE_DIR})
include_directories(${ULTRA_HEADERS_DIR})

add_subdirectory(src)
enable_testing()
add_subdirectory(test)

add_subdirectory(DeepAndWide)
add_subdirectory(Resnet18)
