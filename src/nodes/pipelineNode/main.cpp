#include "components/pipeline.hpp"
#include "pipelineNode.hpp"
#include "ros/node_handle.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "pipelineNode");
    ros::NodeHandle nh("~");
    PipelineNode pipeline(nh);

    ros::spin();

    return 0;
}