#include <ros/ros.h>
#include <surveillance_system_camera/cameraNode.hpp>

int main(int argc, char **argv) {
    ros::init(argc, argv, "camera_node");
    CameraNode node;
    ros::spin();
    return 0;
}