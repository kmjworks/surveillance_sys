#include "captureNode.hpp"
#include "EventLoopTimeKeeper"

CaptureNode::CaptureNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh(nh), private_nh(pnh) {

}