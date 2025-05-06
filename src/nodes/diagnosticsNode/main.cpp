#include "ros/node_handle.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "diagnostics_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
        
    ros::spin();
    return 0;
}