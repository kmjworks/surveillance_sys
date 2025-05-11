cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_motion_detection_old)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  vision_msgs
  image_transport
  cv_bridge
)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)


catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs vision_msgs image_transport cv_bridge
  DEPENDS OpenCV TensorRT CUDA
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(${PROJECT_NAME}
  src/motionDetectionNode_trt.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  nvinfer
  nvinfer_plugin
  nvonnxparser
)

add_executable(motion_detector_node_old src/main.cpp)
target_link_libraries(motion_detector_node_old
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  nvinfer
  nvinfer_plugin
  nvonnxparser
  ${CUDA_LIBRARIES}
)

target_compile_features(motion_detector_node_old PUBLIC cxx_std_20)

install(TARGETS ${PROJECT_NAME} motion_detector_node_old
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

