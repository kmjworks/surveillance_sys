cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_motion_tracking)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  vision_msgs
  image_transport
  cv_bridge
  surveillance_system_common
  message_filters
)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(fmt REQUIRED)

# Find TensorRT using custom FindTensorRT.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../valvesysteem)
find_package(TensorRT REQUIRED)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs vision_msgs image_transport cv_bridge surveillance_system_common message_filters
  DEPENDS OpenCV TensorRT CUDA fmt
)

include_directories(
  src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(${PROJECT_NAME}
  src/motionTrackingNode.cpp
  src/components/FeatureExtractor.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TensorRT_LIBRARIES}
  ${CUDA_LIBRARIES}
  fmt::fmt
)

add_executable(motion_tracking_node src/main.cpp)
target_link_libraries(motion_tracking_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TensorRT_LIBRARIES}
  ${CUDA_LIBRARIES}
  fmt::fmt
)

install(TARGETS ${PROJECT_NAME} motion_tracking_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)