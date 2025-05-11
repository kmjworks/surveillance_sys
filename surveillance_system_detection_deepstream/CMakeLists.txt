cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_detection_deepstream)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(GStreamer REQUIRED)
find_package(DeepstreamAppCommon REQUIRED)
find_package(NVDS REQUIRED)
find_package(CUDA REQUIRED)
find_package(X11 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(yaml-cpp REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_check_modules(JSON-GLIB REQUIRED json-glib-1.0)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  vision_msgs
  image_transport
  cv_bridge
  surveillance_system_common
)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs vision_msgs image_transport cv_bridge surveillance_system_common
  DEPENDS OpenCV TensorRT CUDA
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${GSTREAMER_VIDEO_INCLUDE_DIRS}
  ${GSTREAMER_RTSPSERVER_INCLUDE_DIRS}
  ${GLIB_INCLUDE_DIRS}
  ${JSON-GLIB_INCLUDE_DIRS}
  ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}
  ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}/../src       
  ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}/../src/deepstream-yaml 
  ${NVDS_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${X11_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
  src
)


add_library(${PROJECT_NAME}
  src/detectorNode.cpp
  src/deepstream_app_config_parser.c
  src/deepstream_app_config_parser_yaml.cpp
  src/deepstream_app_main.c
  src/deepstream_app.c
  ${DEEPSTREAM_APP_COMMON_SRCS} 
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GSTREAMER_VIDEO_LIBRARIES}
  ${GSTREAMER_RTSPSERVER_LIBRARIES}
  ${GLIB_LIBRARIES}
  ${JSON-GLIB_LIBRARIES}
  ${NVDS_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CMAKE_DL_LIBS}
  ${X11_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  m
)
add_executable(motion_detector_node 
    src/main.cpp
)

target_link_libraries(motion_detector_node
    ${PROJECT_NAME}
)
target_compile_features(motion_detector_node PUBLIC cxx_std_20)

install(TARGETS ${PROJECT_NAME} motion_detector_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)