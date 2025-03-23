#include "pipelineNode.hpp"
#include "ros/node_handle.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "pipeline_node");
    ros::NodeHandle nh;

    PipelineNode pipelineNode(nh);

    if (!pipelineNode.initialize()) {
        ROS_ERROR("Failed to initialize pipeline. Exiting.");
        return 1;
    }
    
    pipelineNode.run();
    
    ros::spin();
    return 0;
}