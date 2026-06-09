cmake_minimum_required(VERSION 3.15)
project(cmethods)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get pybind11 from Python
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_dir})

find_package(Eigen3 5.0.0 REQUIRED NO_MODULE)
find_package(pybind11 REQUIRED)

pybind11_add_module(cmethods cmethods.cpp
    cmethods.h)

target_link_libraries(cmethods PRIVATE Eigen3::Eigen)
target_link_options(cmethods PRIVATE
    -static-libgcc
    -static-libstdc++
)

target_compile_options(cmethods PRIVATE
                -O3
                -march=native
)