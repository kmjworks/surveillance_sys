cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_pipeline)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  surveillance_system_common
  surveillance_system_msgs
)

find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0>=1.8)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0>=1.8)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0>=1.8)
pkg_check_modules(GLIB REQUIRED glib-2.0)


catkin_package(
  INCLUDE_DIRS src
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs image_transport cv_bridge surveillance_system_common surveillance_system_msgs
  DEPENDS OpenCV
)

include_directories(
  src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${GST_APP_INCLUDE_DIRS}
  ${GST_VIDEO_INCLUDE_DIRS}
  ${GLIB_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/pipelineNode.cpp
  src/components/pipelineInternal.cpp
  src/components/pipelineInitialDetectionLite.cpp
  src/components/harrierCaptureSrc.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(pipeline_node src/main.cpp)
target_link_libraries(pipeline_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  ${GST_APP_LIBRARIES}
  ${GST_VIDEO_LIBRARIES}
  ${GLIB_LIBRARIES}
)

target_compile_features(pipeline_node PUBLIC cxx_std_20)

install(TARGETS ${PROJECT_NAME} pipeline_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)