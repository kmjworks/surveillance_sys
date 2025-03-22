#include "components/pipeline.hpp"
#include "pipelineNode.hpp"
#include "ros/node_handle.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "pipeline_node");
    ros::NodeHandle nh;

    PipelineNode pipelineNode(nh);

    ros::spin();

    return 0;
}