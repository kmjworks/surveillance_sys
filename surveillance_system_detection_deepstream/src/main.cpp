#include "detectorNode.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "motion_detector_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    try {
        DetectorNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL_STREAM("[DetectorNode] Fatal error: " << e.what());
        return 1;
    }
    return 0;
}
