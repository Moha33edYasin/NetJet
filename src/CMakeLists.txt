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

# find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(cmethods cmethods.cpp)

# target_link_libraries(cmethods PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(cmethods PRIVATE -O3 -march=native)
# target_compile_options(cmethods PRIVATE /O2 /arch:AVX2 /fp:fast)