# CMake = the standard C++ build tool (think Maven for C++). scikit-build-core runs it during `uv sync`.
cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Python we are building for, and pybind11's CMake helpers (installed as a build requirement).
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Compile src/cpp/promptclean.cpp into a Python extension module named `_native`
# (the file will be called _native.cpython-312-<platform>.so) ...
pybind11_add_module(_native src/cpp/promptclean.cpp)

# ... and put it inside the `promptclean` package, next to __init__.py.
install(TARGETS _native DESTINATION promptclean)
