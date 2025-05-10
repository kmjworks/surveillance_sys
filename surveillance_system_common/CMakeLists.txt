cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_common)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

find_package(CUDA REQUIRED)
find_package(OpenCV 4.6.0 REQUIRED COMPONENTS core dnn imgproc cudaimgproc cudabgsegm)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS
    src
    src/DeepSORT/incl
    src/ThreadSafeQueue
    src/ROS
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs
)

include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_subdirectory(src/DeepSORT)

add_library(${PROJECT_NAME}
  src/Serial/Serial.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  Threads::Threads
  Eigen3::Eigen
  nvinfer
  nvinfer_plugin
  nvonnxparser
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)