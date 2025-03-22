#include "pipelineNode.hpp"
#include "opencv2/videoio.hpp"
#include "sensor_msgs/Image.h"
#include "surveillance_system/motion_event.h"

PipelineNode(ros::NodeHandle& nh) : PipelineBase(100, 1920, 1080), nh(nh) {
    nh.param("motion_threshold", detectionThresholds.motionThreshold, 25.0);
    nh.param("min_motion_area", detectionThresholds.minimalMotionArea, 500);

    nh.param("show_debug_frames", detectionInternalState.showDebugFrames, false);
    nh.param<std::string>("output_path", detectionInternalState.outputPath, "recordings");

    detectionThresholds.motionDetected = false;

    detectionInternalState.videoWriterStatus = false;
    detectionInternalState.backgroundSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);

    rosInterface.pub_processedFrames = nh.advertise<sensor_mgs::Image>("pipeline/runtime_processedFrames");
    rosInterface.pub_motionEvents = nh.advertise<surveillance_system::motion_event>("pipeline/runtime_motionEvents");

    rosInterface.sub_motionEvents = nh.subscribe("motion_detection/runtime_motionEvents", 10, &PipelineNode::cb_motionEvent, this);
    rosInterface.sub_rawImageData = nh.subscribe("camera_node/runtime_rawImageData", 1, &PipelineNode::cb_imageData, this);

    std::string mkdir_cmd = "mkdir -p " + detectionInternalState.outputPath;
    system(mkdir_cmd.c_str());

    ROS_INFO("[+] Pipeline initialized.");
}


void PipelineNode::cb_imageData(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv::Mat frame = cv_bridge::toCvShare
    }
}