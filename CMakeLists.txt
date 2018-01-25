cmake_minimum_required(VERSION 3.8)
project(lineFitting)

set(CMAKE_CXX_STANDARD 11)

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "/home/lightol/Desktop/g2o/cmake_modules")

find_package(G2O REQUIRED)

include_directories(${G20_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(lineFitting ${SOURCE_FILES})

target_link_libraries(lineFitting ${OpenCV_LIBS} g2o_core g2o_stuff)