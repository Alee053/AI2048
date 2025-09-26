cmake_minimum_required(VERSION 3.15)

project(Fast2048Cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(searcher
        Fast2048.cpp
        bindings.cpp
        RandomUtil.cpp
        ExpectimaxSearcher.cpp
)

target_compile_features(searcher PRIVATE cxx_std_17)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../twenty_forty_eight_ai/utils)

add_executable(game_tester main.cpp Fast2048.cpp RandomUtil.cpp)
target_compile_features(game_tester PRIVATE cxx_std_17)