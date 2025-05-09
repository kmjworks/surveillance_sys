#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInitialDetectionLite.hpp"
#include "components/pipelineInternal.hpp"

// #include "surveillance_system/motion_detection_events_array.h"
//  #include "surveillance_system/motion_event.h"
// #include <sensor_msgs/RegionOfInterest.h>
// #include <geometry_msgs/Point.h>

PipelineNode::PipelineNode(ros::NodeHandle& nh, ros::NodeHandle& privateHandle) :
    nh(nh), nh_priv(privateHandle), imageTransport(nh), pipelineRunning(false) {
    ROS_INFO("Initializing Pipeline Node...");

    params.motionSamplingRate = 1;
    params.bufferSize = 100;
    params.motionMinAreaPx = 800;
    params.motionDownScale = 0.5f;
    params.motionHistory = 120;

    loadParameters(nh_priv, params);
}

PipelineNode::~PipelineNode() { 
    shutdown(); 
}

bool PipelineNode::initializePipelineNode() {

    ROS_INFO("Initializing Pipeline Node...");
    rosInterface.pub_motionEvents = imageTransport.advertise("pipeline/runtime_potentialMotionEvents", 1);
    rosInterface.pub_processedFrames = imageTransport.advertise("pipeline/runtime_processedFrames", 1);
    rosInterface.pub_runtimeErrors = nh.advertise<std_msgs::String>("pipeline/runtime_errors", 10);

    components.cameraSrc = std::make_unique<pipeline::HarrierCaptureSrc>(params.devicePath, params.frameRate, params.nightMode);
    components.pipelineInternal = std::make_unique<pipeline::PipelineInternal>(params.frameRate, params.nightMode);
    components.pipelineIntegratedMotionDetection = std::make_unique<pipeline::PipelineInitialDetectionLite>(params.motionMinAreaPx, params.motionDownScale, params.motionHistory, params.motionSamplingRate);
    components.rawFrameQueue.initialize(params.bufferSize);

    if (!components.cameraSrc->initializeRawSrcForCapture()) {
        publishError("Failed to initialize camera after multiple attempts.");
        ROS_ERROR("Failed to initialize camera source.");
        return false;
    }

    startWorkerThreads();

    ROS_INFO("Pipeline Node initialized.");
    return true;
}

void PipelineNode::loadParameters(ros::NodeHandle& nh_priv, pipeline::ConfigurationParameters& params) {
    nh_priv.param<std::string>("device_path", params.devicePath, "/dev/video0");
    nh_priv.param<int>("frame_rate", params.frameRate, 30);
    nh_priv.param<bool>("night_mode", params.nightMode, params.nightMode);
    nh_priv.param<bool>("show_debug_frames", params.showDebugFrames, params.showDebugFrames);
    nh_priv.param<int>("motion_sampling_rate", params.motionSamplingRate, params.motionSamplingRate);
    nh_priv.param<std::string>("output_path", params.outputPath, params.outputPath);
    nh_priv.param<int>("buffer_size", params.bufferSize, 10);
    nh_priv.param<int>("motion_detector/min_area_px", params.motionMinAreaPx, params.motionMinAreaPx);
    nh_priv.param<float>("motion_detector/downscale", params.motionDownScale, params.motionDownScale);
    nh_priv.param<int>("motion_detector/history", params.motionHistory, params.motionHistory);



    ROS_INFO("Pipeline parameters loaded.");
    return;
}

void PipelineNode::publishRawFrame(const cv::Mat& frame, const ros::Time& timestamp) {
    try {
        cv_bridge::CvImage cvImage;
        cvImage.encoding = (frame.channels() == 1 ? sensor_msgs::image_encodings::MONO8 : sensor_msgs::image_encodings::BGR8);
        cvImage.image = frame;
        cvImage.header.stamp = timestamp;
        cvImage.header.frame_id = "camera_link";

        rosInterface.pub_processedFrames.publish(cvImage.toImageMsg());
    } catch (const cv_bridge::Exception& e) {
        publishError("cv_bridge exception during frame publishing : " + std::string(e.what()));
    } catch (const std::exception& e) {
        publishError("Runtime exception during frame publishing : " + std::string(e.what()));
    }
}

void PipelineNode::publishMotionEventFrame(const cv::Mat& frame, const ros::Time& timestamp) {
    try {
        cv_bridge::CvImage img;
        img.encoding = (frame.channels() == 1 ? sensor_msgs::image_encodings::MONO8 : sensor_msgs::image_encodings::BGR8);
        img.image = frame;
        img.header.stamp = timestamp;
        img.header.frame_id = "camera_link_motion";

        rosInterface.pub_motionEvents.publish(img.toImageMsg());

    } catch (const cv_bridge::Exception& e) {
        publishError("cv_bridge exception during frame publishing : " + std::string(e.what()));
    } catch (const std::exception& e) {
        publishError("Runtime exception during frame publishing : " + std::string(e.what()));
    }
}

void PipelineNode::publishError(const std::string& errorMsg) {
    ROS_ERROR_STREAM_THROTTLE(5.0, "[PipelineNode] " << errorMsg);
    std_msgs::String msg;
    msg.data = errorMsg;
    rosInterface.pub_runtimeErrors.publish(msg);
}

void PipelineNode::startWorkerThreads() {
    ROS_INFO("[PipelineNode] Starting pipeline worker threads.");
    captureThread = std::thread(&PipelineNode::captureLoop, this);
    processingThread = std::thread(&PipelineNode::processingLoop, this);
    pipelineRunning = true;
    ROS_INFO("[PipelineNode] Pipeline worker threads started.");
}

void PipelineNode::stopWorkerThreads() {
    pipelineRunning = false;
    ROS_INFO("[PipelineNode] Stopping Pipeline worker threads.");

    components.rawFrameQueue.stopWaitingThreads();
    if (captureThread.joinable()) captureThread.join();
    if (processingThread.joinable()) processingThread.join();
    ROS_INFO("[PipelineNode] Pipeline worker threads stopped.");
}

void PipelineNode::captureLoop() {
    ROS_INFO("[PipelineNode - Capture Thread] Thread started.");
    cv::Mat rawFrame;
    ros::Rate loopRate(params.frameRate > 0 ? params.frameRate : 30);

    while (pipelineRunning && ros::ok()) {
        ros::Time captureTimestamp = ros::Time::now();
        bool captureStatus = components.cameraSrc->captureFrameFromSrc(rawFrame);

        if (!pipelineRunning)
            break;

        if (captureStatus) {
            if (!rawFrame.empty()) {
                pipeline::FrameData data{rawFrame.clone(), captureTimestamp};
                if (!components.rawFrameQueue.try_push(std::move(data))) {
                    ROS_WARN_THROTTLE(2.0, "[PipelineNode - Capture Thread] Raw frame queue full, dropping frame.");
                }
            } else {
                ROS_WARN_THROTTLE(2.0, "[PipelineNode - Capture Thread] Captured empty frame.");
            }
        } else {
            publishError("[PipelineNode - Capture Thread] Frame capture failed.");
        }
    }
    ROS_INFO("[PipelineNode - Capture Thread] Exiting.");
}


void PipelineNode::processingLoop() {
    ROS_INFO("[PipelineNode - Processing Thread] Thread started.");
    cv::Mat processedFrame;
    cv::Mat frameForMotionDetection;
    std::vector<cv::Rect> motionRects;
    bool motionPresence = false;

    while(pipelineRunning && ros::ok()) {
        std::optional<pipeline::FrameData> dataOpt = components.rawFrameQueue.pop();

        if(!dataOpt.has_value()) {
            ROS_WARN("[PipelineNode - Processing Thread] Queue pop returned nullopt unexpectedly.");
            break;
        }

        if(!pipelineRunning) break;

        pipeline::FrameData data = std::move(dataOpt.value());
        if(data.frame.empty()) {
            ROS_WARN("[PipelineNode - Processing Thread] Received empty frame from queue.");
            continue;
        }

        // publishRawFrame(data.frame, data.timestamp); // Debug
        // publishMotionEventFrame(data.frame, data.timestamp);
        processedFrame = components.pipelineInternal->processFrame(data.frame);
        if(processedFrame.empty()) ROS_WARN("[PipelineNode - Processing Thread] Frame processing resulted in empty frame.");
        
        motionPresence = components.pipelineIntegratedMotionDetection->detect(processedFrame, motionRects);
        
        
        publishMotionEventFrame(data.frame, data.timestamp);
        
    }
}


void PipelineNode::shutdown() {
    ROS_INFO("Shutting down pipeline node.");

    stopWorkerThreads();

    if(components.cameraSrc) {
        components.cameraSrc->releasePipeline();
    }
    components.cameraSrc.reset();
    components.pipelineInternal.reset();
    components.pipelineIntegratedMotionDetection.reset();

    ROS_INFO("[PipelineNode] Shutdown complete.");
}
