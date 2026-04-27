cmake_minimum_required(VERSION 3.15)

project(Fast2048Cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(_searcher_cpp
        Fast2048.cpp
        bindings.cpp
        RandomUtil.cpp
        ExpectimaxSearcher.cpp
)

target_compile_features(_searcher_cpp PRIVATE cxx_std_17)


if(CMAKE_CONFIGURATION_TYPES)
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG}
            ${CMAKE_SOURCE_DIR}/../twenty_forty_eight_ai/utils)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}
            ${CMAKE_SOURCE_DIR}/../twenty_forty_eight_ai/utils)
    endforeach()
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../twenty_forty_eight_ai/utils)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../twenty_forty_eight_ai/utils)
endif()

install(
    TARGETS _searcher_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../twenty_forty_eight_ai/utils/
    RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../twenty_forty_eight_ai/utils/
)
