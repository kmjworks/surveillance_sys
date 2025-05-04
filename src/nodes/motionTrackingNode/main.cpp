#include "ros/node_handle.h"
#include "motionTrackingNode.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "motion_tracking_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    try {
        MotionTrackingNode node(nh, private_nh);
        ROS_INFO("[MotionTrackingNode] Motion Tracking started");
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL_STREAM("Exception in motion tracking node: " << e.what());
        return 1;
    }
    
    return 0;
}