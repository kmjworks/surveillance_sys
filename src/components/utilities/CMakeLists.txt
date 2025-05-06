cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

set(ROS_UTILS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ROS")

add_library(utilities INTERFACE
    "${ROS_UTILS_PATH}/EventLoopTimeKeeper.hpp"
)

target_include_directories(utilities INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(utilities INTERFACE cxx_std_20)