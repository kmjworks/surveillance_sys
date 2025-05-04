#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInternal.hpp"
#include "components/pipelineInitialDetectionLite.hpp"

//#include "surveillance_system/motion_detection_events_array.h"
// #include "surveillance_system/motion_event.h"
//#include <sensor_msgs/RegionOfInterest.h>
//#include <geometry_msgs/Point.h>

PipelineNode::PipelineNode(ros::NodeHandle& nh, ros::NodeHandle& privateHandle) :
    nh(nh), nh_priv(privateHandle), imageTransport(nh), rawFrameQueue(loadParameters(nh_priv, configuration)), pipelineRunning(false) {
    
    configuration.devicePath = "/dev/video0";
    configuration.outputPath = "~/sursys_recordings";

    configuration.frameRate = 30;
    configuration.nightMode = false;
    configuration.showDebugFrames = true;
    configuration.motionSamplingRate = 1;
    configuration.bufferSize = 100;
    configuration.motionMinAreaPx = 800;
    configuration.motionDownScale = 0.5f;
    configuration.motionHistory = 120;    

}

PipelineNode::~PipelineNode() {
    shutdown();
}

bool PipelineNode::initializePipelineNode() {
    ROS_INFO("Initializing Pipeline Node...");
    rosInterface.pub_motionEvents = imageTransport.advertise("pipeline/runtime_potentialMotionEvents", 1);
    rosInterface.pub_processedFrames = imageTransport.advertise("pipeline/runtime_processedFrames", 1);
    rosInterface.pub_runtimeErrors = nh.advertise<std_msgs::String>("pipeline/runtime_errors", 10);

    components.cameraSrc = std::make_unique<pipeline::HarrierCaptureSrc>(configuration.devicePath, configuration.frameRate, configuration.nightMode);
    components.pipelineInternal = std::make_unique<pipeline::PipelineInternal>(configuration.frameRate, configuration.nightMode);
    components.pipelineIntegratedMotionDetection = std::make_unique<pipeline::PipelineInitialDetectionLite>(configuration.motionMinAreaPx, configuration.motionDownScale, configuration.motionHistory, configuration.motionSamplingRate);

    if(!components.cameraSrc->initializeRawSrcForCapture()) {
        publishError("Failed to initialize camera after multiple attempts.");
        ROS_ERROR("Failed to initialize camera source.");
        return false;
    }

    startWorkerThreads();

    ROS_INFO("Pipeline Node initialized.");
    return true;
}

int PipelineNode::loadParameters(ros::NodeHandle& nh_priv, pipeline::ConfigurationParameters& configuration) {
    nh_priv.param<std::string>("device_path", configuration.devicePath, "/dev/video0");
    nh_priv.param<int>("frame_rate", configuration.frameRate, 30);
    nh_priv.param<bool>("night_mode", configuration.nightMode, configuration.nightMode);
    nh_priv.param<bool>("show_debug_frames", configuration.showDebugFrames, configuration.showDebugFrames);
    nh_priv.param<int>("motion_sampling_rate", configuration.motionSamplingRate, configuration.motionSamplingRate);
    nh_priv.param<std::string>("output_path", configuration.outputPath, configuration.outputPath);
    nh_priv.param<int>("buffer_size", configuration.bufferSize, 10);
    nh_priv.param<int>("motion_detector/min_area_px", configuration.motionMinAreaPx, configuration.motionMinAreaPx);
    nh_priv.param<float>("motion_detector/downscale", configuration.motionDownScale, configuration.motionDownScale);
    nh_priv.param<int>("motion_detector/history", configuration.motionHistory, configuration.motionHistory);

    ROS_INFO("Pipeline parameters loaded.");
    return configuration.bufferSize;
}

void PipelineNode::publishRawFrame(const cv::Mat& frame, const ros::Time& timestamp) {
    try {
        cv_bridge::CvImage cvImage;
        cvImage.encoding = (frame.channels() == 1 ? sensor_msgs::image_encodings::MONO8 : sensor_msgs::image_encodings::BGR8);
        cvImage.image = frame; 
        cvImage.header.stamp = timestamp;
        cvImage.header.frame_id = "camera_link";

        rosInterface.pub_processedFrames.publish(cvImage.toImageMsg());
    } catch (const cv_bridge::Exception &e) {
        publishError("cv_bridge exception during frame publishing : " + std::string(e.what()));
    } catch (const std::exception &e) {
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

    } catch (const cv_bridge::Exception &e) {
        publishError("cv_bridge exception during frame publishing : " + std::string(e.what()));
    } catch (const std::exception &e) {
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
    pipelineRunning = true;
    workerThreads.captureThread = std::thread(&PipelineNode::captureLoop, this);
    workerThreads.processingThread = std::thread(&PipelineNode::processingLoop, this);
    ROS_INFO("[PipelineNode] Pipeline worker threads started.");
}

void PipelineNode::stopWorkerThreads() {
    pipelineRunning = false;

    ROS_INFO("[PipelineNode] Stopping Pipeline worker threads.");

    rawFrameQueue.stopWaitingThreads();
    if (workerThreads.captureThread.joinable()) workerThreads.captureThread.join();
    if (workerThreads.processingThread.joinable()) workerThreads.processingThread.join();

    ROS_INFO("[PipelineNode] Pipeline worker threads stopped.");
}

void PipelineNode::captureLoop() {
    ROS_INFO("[PipelineNode - Capture Thread] Thread started.");
    ros::Rate loopRate(configuration.frameRate > 0 ? configuration.frameRate : 30);
    cv::Mat rawFrame;
        

    while (pipelineRunning && ros::ok()) {
        ros::Time captureTimestamp = ros::Time::now();
        bool captureStatus = components.cameraSrc->captureFrameFromSrc(rawFrame);

        if (!pipelineRunning)
            break;

        if (captureStatus) {
            if (!rawFrame.empty()) {
                pipeline::FrameData data{rawFrame.clone(), captureTimestamp};
                if (!rawFrameQueue.try_push(std::move(data))) {
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

    ros::Time lastProcessTime = ros::Time::now();
    double avgProcessTime = 0.0;
    int frameCount = 0;

    while(pipelineRunning && ros::ok()) {
        std::optional<pipeline::FrameData> dataOpt = rawFrameQueue.pop();

        if(!dataOpt.has_value()) {
            if(pipelineRunning) {
                ROS_WARN_THROTTLE(5.0, "[PipelineNode - Processing Thread] Queue pop timeout.");
            }
            continue;
        }

        if(!pipelineRunning) break;

        pipeline::FrameData data = std::move(dataOpt.value());
        if(data.frame.empty()) {
            ROS_WARN_THROTTLE(5.0, "[PipelineNode - Processing Thread] Received empty frame from queue.");
            continue;
        }
        ros::Time startTime = ros::Time::now();

        // publishRawFrame(data.frame, data.timestamp); // Debug
        // publishMotionEventFrame(data.frame, data.timestamp);
        processedFrame = components.pipelineInternal->processFrame(data.frame);
        if(processedFrame.empty()) {
            ROS_WARN_THROTTLE(5.0, "[PipelineNode - Processing Thread] Frame processing resulted in empty frame.");
            continue;
        }
        
        motionPresence = components.pipelineIntegratedMotionDetection->detect(processedFrame, motionRects);
        
        if(motionPresence) publishMotionEventFrame(data.frame, data.timestamp);
        
        ros::Time endTime = ros::Time::now();
        double processingTime = (endTime - startTime).toSec();
        avgProcessTime = (avgProcessTime * frameCount + processingTime) / (frameCount + 1);
        ++frameCount;

        if(frameCount % 100 == 0) {
            double queueFillRate = 100.0 * rawFrameQueue.getQueueSize() / configuration.bufferSize;
            ROS_INFO("[PipelineNode - Processing Thread] Stats: Avg process time: %.3f s, Queue fill: %.1f%%", avgProcessTime, queueFillRate);
        }
    }

    ROS_INFO("[PipelineNode - Processing Thread] Exiting. Processed %d frames.", frameCount);
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

    ROS_INFO("Pipeline node shutdown complete.");
}
