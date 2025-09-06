# Set the minimum required version of CMake.
cmake_minimum_required(VERSION 3.15)

# Set the name of the C++ project.
project(Fast2048Cpp)

# --- Find Python and pybind11 (The Modern Way) ---

# Explicitly find a Python 3 installation.
# This is the modern approach recommended by CMake.
# It ensures we find both the interpreter program and the development
# libraries needed for linking, and will fail here if they're not found.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Now, with Python found, find the pybind11 package.
# It will use the already-discovered Python environment to locate itself.
find_package(pybind11 REQUIRED)

# --- Create the Python Module ---
# This part remains the same.
pybind11_add_module(fast2048_cpp
        Fast2048.cpp
        bindings.cpp
        RandomUtil.cpp
        main.cpp
)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(fast2048_cpp PRIVATE -static-libgcc -static-libstdc++)
endif()


# --- Configure the Build ---
# (The rest of your file stays the same)
target_compile_features(fast2048_cpp PRIVATE cxx_std_17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../src)

# (Your add_executable for the game_tester also stays the same)
add_executable(game_tester main.cpp Fast2048.cpp RandomUtil.cpp)
target_compile_features(game_tester PRIVATE cxx_std_17)