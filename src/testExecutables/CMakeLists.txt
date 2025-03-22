cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD_REQUIRED 17)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")

include_directories(${Boost_INCLUDE_DIR} "../components/Harrier36X/VISCA" "../components/Harrier36X/CAM_Harrier36X")

add_executable(harrierBringUpTest "Harrier36X/harrierBringup/main.cpp")
target_link_libraries(harrierBringUpTest harrierUtils)
target_compile_features(harrierBringUpTest PRIVATE cxx_std_17)