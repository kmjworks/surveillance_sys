#include "ros/node_handle.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "surveillance_system/systemNode");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
        
    ros::spin();
    return 0;
}