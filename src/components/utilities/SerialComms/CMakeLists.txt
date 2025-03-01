cmake_minimum_required(VERSION 3.10)
project(SerialCommsUtility)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
message(STATUS "Boost version: ${Boost_VERSION}")

include_directories("../SerialComms")
add_executable(serialUtils ${CMAKE_SOURCE_DIR} Serial.cpp)
target_link_libraries(serialUtils fmt::fmt ${Boost_LIBRARIES})

target_compile_features(serialUtils PRIVATE cxx_std_17)