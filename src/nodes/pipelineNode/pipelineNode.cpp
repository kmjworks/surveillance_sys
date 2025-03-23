#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInternal.hpp"
#include "components/pipelineInitialDetection.hpp"

PipelineNode::PipelineNode(ros::NodeHandle& nh) : nodeHandle(nh), pipelineRunning(false),
    errorHandling{3, 5} {
    
    rosInterface.motionPublisher = nodeHandle.advertise<sensor_msgs::Image>("pipeline/runtime_potentialMotionEvents", 10);
    rosInterface.errorPublisher = nodeHandle.advertise<std_msgs::String>("pipeline/runtime_videoSrcErrors", 10);
    rosInterface.processedFramePublisher = nodeHandle.advertise<sensor_msgs::Image>("pipeline/runtime_processedFrames", 10);

    loadPipelineParams();

    captureSrcRaw = std::make_unique<HarrierCaptureSrc>(state.devicePath, state.frameRate);
    pipelineInternal = std::make_unique<PipelineInternal>(state.nightMode, state.showDebugFrames);
    initialMotionDetection = std::make_unique<PipelineInitialDetection>(state.motionPublishingRate);    
}

PipelineNode::~PipelineNode() {
    pipelineCleanup();
}

void PipelineNode::loadPipelineParams() {
    nodeHandle.param<std::string>("video_src", state.devicePath, "/dev/video0");
    nodeHandle.param<int>("frame_rate", state.frameRate, 30);
    nodeHandle.param<bool>("show_debug_frames", state.showDebugFrames, false);
    nodeHandle.param<int>("buffer_size", state.bufferSize, 100);
    nodeHandle.param<std::string>("output_path", state.outputPath, "/tmp/sur_sys_recordings");
    nodeHandle.param<bool>("night_mode", state.nightMode, false);
    nodeHandle.param<int>("motion_publish_rate", state.motionPublishingRate, 1);
    nodeHandle.param<int>("camera_retry_count", errorHandling.cameraHandleRetryCount, 3);
    nodeHandle.param<int>("camera_retry_delay", errorHandling.cameraHandleRetryDelay, 5);

    ROS_INFO("Pipeline parameters: ");
    ROS_INFO(" Device path: %s", state.devicePath.c_str());
    ROS_INFO(" Frame rate: %d", state.frameRate);
    ROS_INFO(" Night mode: %s", state.nightMode ? "enabled" : "disabled");
    ROS_INFO(" Motion publishing rate: %d hz", state.motionPublishingRate);
}

bool PipelineNode::initialize() {
    int retryCount = 0;
    bool cameraInitialized = false;

    while (retryCount < errorHandling.cameraHandleRetryCount && !cameraInitialized) {
        ROS_INFO("Initializing camera, attempt %d of %d", retryCount+1, errorHandling.cameraHandleRetryCount);

        cameraInitialized = captureSrcRaw->initialize();

    if (!cameraInitialized) {
            ROS_WARN("Failed to initialize camera, retrying in %d seconds", 
                    errorHandling.cameraHandleRetryDelay);
            ros::Duration(errorHandling.cameraHandleRetryDelay).sleep();
            retryCount++;
        }
    }

    if(!cameraInitialized) {
        std::string errorMsg = "Failed to initialize camera after " + 
                              std::to_string(errorHandling.cameraHandleRetryCount) + " attempts";
        ROS_ERROR("%s", errorMsg.c_str());
        publishError(errorMsg);
        return false;
    }

    if(!pipelineInternal->initialize()) {
        std::string errorMsg = "Failed to initialize pipeline internal components";
        ROS_ERROR("%s", errorMsg.c_str());
        publishError(errorMsg);
        return false;
    }

    if(initialMotionDetection->initialize()) {
        std::string errorMsg = "Failed to initialize motion detector";
        ROS_ERROR("%s", errorMsg.c_str());
        publishError(errorMsg);
        return false;
    }

    ROS_INFO("Pipeline initialized.");
    return true;
}

void PipelineNode::run() {
    if(pipelineRunning) {
        ROS_WARN("Pipeline already running.");
        return;
    }

    pipelineRunning = true;
    pipelineProcessingThread = std::thread(&PipelineNode::pipelineProcessingLoop, this);
    ROS_INFO("Pipeline started.");
}


