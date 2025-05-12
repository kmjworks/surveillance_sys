#include <ros/ros.h>
#include "motionDetectionNode_trt.hpp"

int main(int argc, char** argv) {
    return 0;
    ros::init(argc, argv, "motion_detector_node_old");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    try {
        MotionDetectionNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL_STREAM("Fatal error in YoloTRTNode: " << e.what());
        return 1;
    }
    return 0;
}