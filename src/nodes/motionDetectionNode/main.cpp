#include <ros/ros.h>
#include "motionDetectionNode.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "motion_detector_node");

    // Create a motion detector node with the default frame difference strategy
    surveillance_system::MotionDetectorNode<> detector;

    // Spin to process callbacks
    ros::spin();

    return 0;
}
