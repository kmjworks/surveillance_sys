execute_process(COMMAND "/home/kmj/vision_opencv/cv_bridge/build/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/kmj/vision_opencv/cv_bridge/build/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
