cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_compile_options(-Wall -Wextra -Wpedantic)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")

set(HARRIER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CAM_Harrier36X")
set(SERIAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../utilities/SerialComms")

add_library(harrierUtils STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/VISCA/ViscaProtocol.cpp"
    "${HARRIER_PATH}/Harrier36X.cpp"
    "${SERIAL_PATH}/Serial.cpp"
)

target_include_directories(harrierUtils PRIVATE
    "${SERIAL_PATH}/"
    "${CMAKE_CURRENT_SOURCE_DIR}/VISCA"
    "${HARRIER_PATH}/"
    ${Boost_INCLUDE_DIRS}
)


target_link_libraries(harrierUtils
    fmt::fmt
    ${Boost_LIBRARIES}
)


target_compile_features(harrierUtils PRIVATE cxx_std_20)