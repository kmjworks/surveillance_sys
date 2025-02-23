#include "pipelineNode.hpp"
#include "ros/node_handle.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "pipeline_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    PipelineNode pipelineNode(nh, private_nh);
    if (!pipelineNode.initializePipelineNode()) {
        ROS_ERROR("Failed to initialize pipeline node");
        return 1;
    }
        
    ros::spin();
    return 0;
}