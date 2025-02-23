#include "surveillance_system_capture/captureNode.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "capture_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    CaptureNode capture(nh, private_nh);
        
    ros::spin();
    return 0;
}