cmake_minimum_required(VERSION 3.0.2)
project(surveillance_system_diagnostics)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  surveillance_system_msgs
  surveillance_system_common
)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs surveillance_system_msgs surveillance_system_common
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/diagnosticsNode.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

add_executable(diagnostics_node src/main.cpp)
target_link_libraries(diagnostics_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME} diagnostics_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)