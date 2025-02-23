cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_motion_detection)

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

# Find TensorRT using custom FindTensorRT.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../valvesysteem)
find_package(TensorRT REQUIRED)

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
  ${TensorRT_LIBRARIES}
  ${CUDA_LIBRARIES}
)

add_executable(motion_detector_node src/main.cpp)
target_link_libraries(motion_detector_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TensorRT_LIBRARIES}
  ${CUDA_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME} motion_detector_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)