#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInternal.hpp"
#include "components/pipelineInitialDetection.hpp"

PipelineNode::PipelineNode(ros::NodeHandle& nh, ros::NodeHandle& privateHandle) :
    nh(nh), nh_priv(privateHandle), pipelineRunning(false) {
    
    params.devicePath = "/dev/video0";
    params.outputPath = "~/sursys_recordings";

    params.frameRate = 30;
    params.nightMode = false;
    params.showDebugFrames = true;
    params.motionSamplingRate = 1;
    params.bufferSize = 100;    

}

PipelineNode::~PipelineNode() {
    shutdown();
}

bool PipelineNode::initializePipelineNode() {
    ROS_INFO("Initializing Pipeline Node...");

    loadParameters();

    rosInterface.pub_motionEvents = nh.advertise<sensor_msgs::Image>("pipeline/runtime_potentialMotionEvents", 10);
    rosInterface.pub_processedFrames = nh.advertise<sensor_msgs::Image>("pipeline/runtime_processedFrames", 10);
    rosInterface.pub_runtimeErrors = nh.advertise<std_msgs::String>("pipeline/runtime_errors", 10);

    components.cameraSrc = std::make_unique<pipeline::HarrierCaptureSrc>(params.devicePath, params.frameRate, params.nightMode);
    components.pipelineInternal = std::make_unique<pipeline::PipelineInternal>(params.frameRate, params.nightMode);
    components.pipelineIntegratedMotionDetection = std::make_unique<pipeline::PipelineInitialDetection>(params.motionSamplingRate);

    if(!components.cameraSrc->initializeRawSrcForCapture()) {
        publishError("Failed to initialize camera after multiple attempts.");
        return false;
    }

    startProcessingThread();

    ROS_INFO("Pipeline Node initialized.");
    return true;
}

void PipelineNode::loadParameters() {
    nh_priv.param<std::string>("device_path", params.devicePath, params.devicePath);
    nh_priv.param<int>("frame_rate", params.frameRate, params.frameRate);
    nh_priv.param<bool>("night_mode", params.nightMode, params.nightMode);
    nh_priv.param<bool>("show_debug_frames", params.showDebugFrames, params.showDebugFrames);
    nh_priv.param<int>("motion_sampling_rate", params.motionSamplingRate, params.motionSamplingRate);
    nh_priv.param<int>("buffer_size", params.bufferSize, params.bufferSize);
    nh_priv.param<std::string>("output_path", params.outputPath, params.outputPath);

    ROS_INFO("Pipeline parameters loaded.");
}


void PipelineNode::publishFrame(const cv::Mat& frame, bool motionPresence) {
    cv_bridge::CvImage cvImage;
    cvImage.encoding = (frame.channels() == 1 ? "mono8" : "bgr8");
    cvImage.image = frame;

    rosInterface.pub_processedFrames.publish(cvImage.toImageMsg());

    if(motionPresence) {
        rosInterface.pub_motionEvents.publish(cvImage.toImageMsg());
    }
}

void PipelineNode::publishError(const std::string& errorMsg) {
    std_msgs::String msg;
    msg.data = errorMsg;
    rosInterface.pub_runtimeErrors.publish(msg);

    ROS_ERROR_THROTTLE(10, "%s", errorMsg.c_str());
}

void PipelineNode::startProcessingThread() {
    pipelineRunning = true;
    pipelineProcessingThread = std::thread(&PipelineNode::processFrames, this);
}

void PipelineNode::stopProcessingThread() {
    pipelineRunning = false;
    if(pipelineProcessingThread.joinable()) pipelineProcessingThread.join();
}

void PipelineNode::processFrames() {
    cv::Mat raw, processed;
    bool motionPresence = false;
    std::vector<cv::Rect> motionRects;

    while(pipelineRunning) {
        if(!components.cameraSrc->captureFrameFromSrc(raw)) {
            publishError("Frame capture failed.");
            ros::Duration(0.5).sleep();
            continue;
        }
        
        {
            std::lock_guard<std::mutex> lock(frameMtx);
            processed = components.pipelineInternal->processFrame(raw);
            motionPresence = components.pipelineIntegratedMotionDetection->detectedPotentialMotion(processed, motionRects);
        }

        if(motionPresence && !motionRects.empty()) {
            for(const auto& rect : motionRects) {
                cv::rectangle(processed, rect, cv::Scalar(0,255,0), 2);
            }
        }

        publishFrame(processed, motionPresence);

        ros::Duration(1.0 / params.frameRate).sleep();
    }
}

void PipelineNode::shutdown() {
    ROS_INFO("Shutting down pipeline node.");

    if(components.cameraSrc) {
        components.cameraSrc->releasePipeline();
    }

    ROS_INFO("Pipeline node shutdown complete.");
}
