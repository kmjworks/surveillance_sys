cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  camera_status.msg
  diagnostic_event.msg
  key_value.msg
  motion_detection_events_array.msg
  motion_event.msg
  pipeline_metrics.msg
  storage_status.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  CenterCameraOnPoint.srv
)

## Generate added messages and services with any dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    message_runtime 
    std_msgs 
    sensor_msgs 
    geometry_msgs
    vision_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)