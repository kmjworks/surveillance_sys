cmake_minimum_required(VERSION 3.10)
project(surveillance_system VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compiler warnings and export compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  vision_msgs
  std_msgs
  message_generation
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc cudaimgproc cudabgsegm)
find_package(Threads REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0>=1.8)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0>=1.8)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0>=1.8)
pkg_check_modules(GLIB REQUIRED glib-2.0)


include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)


add_message_files(
  FILES
  motion_event.msg
  motion_detection_events_array.msg
  pipeline_metrics.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

# # Generate services in the 'srv' folder
# add_service_files(
# FILES
# Service1.srv
# Service2.srv
# )

# # Generate actions in the 'action' folder
# add_action_files(
# FILES
# Action1.action
# Action2.action
# )

# # Generate added messages and services with any dependencies listed here
# generate_messages(
# DEPENDENCIES
# sensor_msgs#   std_msgs
# )

# ###############################################
# # Declare ROS dynamic reconfigure parameters ##
# ###############################################

# # To declare and build dynamic reconfigure parameters within this
# # package, follow these steps:
# # * In the file package.xml:
# #   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
# # * In this file (CMakeLists.txt):
# #   * add "dynamic_reconfigure" to
# #     find_package(catkin REQUIRED COMPONENTS ...)
# #   * uncomment the "generate_dynamic_reconfigure_options" section below
# #     and list every .cfg file to be processed

# # Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
# cfg/DynReconf1.cfg
# cfg/DynReconf2.cfg
# )

# ##################################
# # catkin specific configuration ##
# ##################################
# # The catkin_package macro generates cmake config files for your package
# # Declare things to be passed to dependent projects
# # INCLUDE_DIRS: uncomment this if your package contains header files
# # LIBRARIES: libraries you create in this project that dependent projects also need
# # CATKIN_DEPENDS: catkin_packages dependent projects also need
# # DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cv_bridge image_transport roscpp rospy sensor_msgs std_msgs
  DEPENDS OpenCV
)

# ##########
# # Build ##
# ##########

# # Specify additional locations of header files
# # Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${GST_APP_INCLUDE_DIRS}
  ${GST_VIDEO_INCLUDE_DIRS}
  ${GLIB_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  ${GLIB_LIBRARY_DIRS}
  ${GSTREAMER_LIBRARY_DIRS}
  ${GST_APP_LIBRARY_DIRS}
  ${GST_VIDEO_LIBRARY_DIRS}
)

set(COMMON_DEPS
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Threads::Threads
)

set(GSTREAMER_DEPS
  ${GSTREAMER_LIBRARIES}
  ${GST_APP_LIBRARIES}
  ${GST_VIDEO_LIBRARIES}
  ${GLIB_LIBRARIES}
)


# # Declare a C++ executable
# # With catkin_make all packages are built within a single CMake context
# # The recommended prefix ensures that target names across packages don't collide
add_executable(motion_detection_node 
  src/nodes/motionDetectionNode/motionDetectionNode.cpp 
  src/nodes/motionDetectionNode/main.cpp
)
add_dependencies(motion_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(motion_detection_node
  ${COMMON_DEPS}
  fmt::fmt
)

add_executable(camera_node 
  src/nodes/cameraNode/cameraNode.cpp
)
add_dependencies(camera_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(camera_node 
  ${COMMON_DEPS}
)


add_executable(video_pipeline_node 
  src/nodes/pipelineNode/main.cpp
  src/nodes/pipelineNode/pipelineNode.cpp 
  src/nodes/pipelineNode/components/pipelineInternal.cpp 
  src/nodes/pipelineNode/components/pipelineInitialDetection.cpp
  src/nodes/pipelineNode/components/harrierCaptureSrc.cpp  
)
add_dependencies(video_pipeline_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(video_pipeline_node 
  ${COMMON_DEPS}
  ${GSTREAMER_DEPS}
)

add_executable(diagnostics_node 
  src/nodes/diagnosticsNode/diagnosticsNode.cpp 
  src/nodes/diagnosticsNode/main.cpp
)
add_dependencies(diagnostics_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(diagnostics_node 
  ${COMMON_DEPS}
)

install(TARGETS
  motion_detection_node
  camera_node
  video_pipeline_node
  diagnostics_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}  
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

catkin_install_python(PROGRAMS scripts/yolo_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(pipeline_test
    test/test_pipeline.cpp
    src/nodes/pipelineNode/components/pipelineInternal.cpp
    src/nodes/pipelineNode/components/pipelineInitialDetection.cpp
  )
  target_link_libraries(pipeline_test
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )
endif()

# ############
# # Testing ##
# ############

# # Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_surveillance_system.cpp)
# if(TARGET ${PROJECT_NAME}-test)
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

# # Add folders to be run by python nosetests
# catkin_add_nosetests(test)
