# ------------------------------------------------------------
# Tutorial overview:
# This build script tells CMake how to compile the C++ project.
# ------------------------------------------------------------

# Minimum CMake version needed to configure this project.
cmake_minimum_required(VERSION 3.10)
# Project name used by build tools/IDE integrations.
project(hogwild_project)

# Use C++17 language features.
set(CMAKE_CXX_STANDARD 17)
# Fail configure step if compiler cannot provide C++17.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow source files to include headers from the include/ folder.
include_directories(include)

# Build one executable named "hogwild" from these source files.
add_executable(hogwild
    src/main.cpp
    src/dataset.cpp
    src/model.cpp
    src/metrics.cpp
    src/sgd.cpp
    src/utils.cpp
)
