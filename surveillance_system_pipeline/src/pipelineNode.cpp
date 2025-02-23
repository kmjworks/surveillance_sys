#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInternal.hpp"
#include "components/pipelineInitialDetectionLite.hpp"
#include <std_msgs/String.h>

PipelineNode::PipelineNode(ros::NodeHandle& nh, ros::NodeHandle& privateHandle)
    : nh(nh), nh_priv(privateHandle), imageTransport(nh), pipelineRunning(false) {
    
    ROS_INFO("[PipelineNode] Initializing");
}

PipelineNode::~PipelineNode() {
    shutdown();
    ROS_INFO("[PipelineNode] Destroyed");
}

int PipelineNode::loadParameters(ros::NodeHandle& nh_priv, pipeline::ConfigurationParameters& parametersToLoad) {
    // Load ROS parameters with defaults
    nh_priv.param<int>("frame_rate", parametersToLoad.frameRate, 10);
    nh_priv.param<int>("motion_sampling_rate", parametersToLoad.motionSamplingRate, 3);
    nh_priv.param<int>("buffer_size", parametersToLoad.bufferSize, 100);
    nh_priv.param<int>("motion_min_area_px", parametersToLoad.motionMinAreaPx, 500);
    nh_priv.param<float>("motion_down_scale", parametersToLoad.motionDownScale, 0.5);
    nh_priv.param<int>("motion_history", parametersToLoad.motionHistory, 50);
    nh_priv.param<bool>("night_mode", parametersToLoad.nightMode, false);
    nh_priv.param<bool>("show_debug_frames", parametersToLoad.showDebugFrames, true);
    nh_priv.param<std::string>("device_path", parametersToLoad.devicePath, "/dev/video0");
    nh_priv.param<std::string>("output_path", parametersToLoad.outputPath, "/tmp/surveillance_output");
    
    // Log parameters
    ROS_INFO("[PipelineNode] Loaded parameters:");
    ROS_INFO("[PipelineNode] - Frame rate: %d", parametersToLoad.frameRate);
    ROS_INFO("[PipelineNode] - Motion sampling rate: %d", parametersToLoad.motionSamplingRate);
    ROS_INFO("[PipelineNode] - Buffer size: %d", parametersToLoad.bufferSize);
    ROS_INFO("[PipelineNode] - Motion min area (px): %d", parametersToLoad.motionMinAreaPx);
    ROS_INFO("[PipelineNode] - Motion down scale: %.2f", parametersToLoad.motionDownScale);
    ROS_INFO("[PipelineNode] - Motion history frames: %d", parametersToLoad.motionHistory);
    ROS_INFO("[PipelineNode] - Night mode: %s", parametersToLoad.nightMode ? "true" : "false");
    ROS_INFO("[PipelineNode] - Show debug frames: %s", parametersToLoad.showDebugFrames ? "true" : "false");
    ROS_INFO("[PipelineNode] - Device path: %s", parametersToLoad.devicePath.c_str());
    ROS_INFO("[PipelineNode] - Output path: %s", parametersToLoad.outputPath.c_str());
    
    return 0;
}

bool PipelineNode::initializePipelineNode() {
    // Load parameters
    if (loadParameters(nh_priv, params) != 0) {
        ROS_ERROR("[PipelineNode] Failed to load parameters");
        return false;
    }
    
    // Initialize ROS interfaces
    rosInterface.pub_runtimeErrors = nh.advertise<std_msgs::String>("pipeline/runtime_errors", 10);
    rosInterface.pub_processedFrames = imageTransport.advertise("pipeline/runtime_processedFrames", 1);
    rosInterface.pub_motionEvents = imageTransport.advertise("pipeline/runtime_potentialMotionEvents", 1);
    
    // Initialize components
    try {
        // Initialize frame queue
        components.rawFrameQueue.initialize(params.bufferSize);
        
        // Initialize camera source
        components.cameraSrc = std::make_unique<pipeline::HarrierCaptureSrc>(params.devicePath);
        if (!components.cameraSrc->isInitialized()) {
            publishError("Failed to initialize camera source");
            return false;
        }
        
        // Initialize pipeline processing components
        components.pipelineInternal = std::make_unique<pipeline::PipelineInternal>(
            params.frameRate, params.nightMode);
        
        components.pipelineIntegratedMotionDetection = std::make_unique<pipeline::PipelineInitialDetectionLite>(
            params.motionMinAreaPx, params.motionDownScale, params.motionHistory);
        
        // Start worker threads
        startWorkerThreads();
        
        ROS_INFO("[PipelineNode] Successfully initialized");
        return true;
        
    } catch (const std::exception& e) {
        std::string errorMsg = std::string("Exception during initialization: ") + e.what();
        publishError(errorMsg);
        ROS_ERROR("[PipelineNode] %s", errorMsg.c_str());
        return false;
    }
}

void PipelineNode::startWorkerThreads() {
    if (!pipelineRunning) {
        pipelineRunning = true;
        captureThread = std::thread(&PipelineNode::captureLoop, this);
        processingThread = std::thread(&PipelineNode::processingLoop, this);
        ROS_INFO("[PipelineNode] Worker threads started");
    }
}

void PipelineNode::stopWorkerThreads() {
    if (pipelineRunning) {
        pipelineRunning = false;
        
        components.rawFrameQueue.stopWaitingThreads();
        
        if (captureThread.joinable()) {
            captureThread.join();
        }
        
        if (processingThread.joinable()) {
            processingThread.join();
        }
        
        ROS_INFO("[PipelineNode] Worker threads stopped");
    }
}

void PipelineNode::captureLoop() {
    ROS_INFO("[PipelineNode] Capture thread started");
    
    ros::Rate loopRate(params.frameRate);
    int reconnectAttempts = 0;
    
    while (pipelineRunning && ros::ok()) {
        cv::Mat frame;
        
        if (components.cameraSrc->captureFrame(frame)) {
            reconnectAttempts = 0;
            
            if (!frame.empty()) {
                pipeline::FrameData frameData;
                frameData.frame = frame;
                frameData.timestamp = ros::Time::now();
                
                // Add frame to queue
                components.rawFrameQueue.try_push(frameData);
            }
        } else {
            // Camera disconnected or error
            reconnectAttempts++;
            
            if (reconnectAttempts % 10 == 0) {
                publishError("Camera capture failed, attempting to reconnect");
                components.cameraSrc->reconnect();
            }
        }
        
        loopRate.sleep();
    }
    
    ROS_INFO("[PipelineNode] Capture thread stopped");
}

void PipelineNode::processingLoop() {
    ROS_INFO("[PipelineNode] Processing thread started");
    
    int frameCounter = 0;
    
    while (pipelineRunning && ros::ok()) {
        std::optional<pipeline::FrameData> frameData = components.rawFrameQueue.pop();
        
        if (!frameData.has_value()) {
            // Queue was empty or interrupted
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (frameData->frame.empty()) {
            continue;
        }
        
        try {
            // Process frame through pipeline
            cv::Mat processedFrame = components.pipelineInternal->processFrame(frameData->frame);
            
            if (processedFrame.empty()) {
                continue;
            }
            
            // Publish processed frame
            publishRawFrame(processedFrame, frameData->timestamp);
            
            // Run motion detection at a lower rate to save compute
            frameCounter++;
            if (frameCounter % params.motionSamplingRate == 0) {
                cv::Mat motionMask, annotatedFrame;
                
                bool motionDetected = components.pipelineIntegratedMotionDetection->detectMotion(
                    processedFrame, motionMask, annotatedFrame);
                
                if (motionDetected) {
                    // Publish frame with detected motion
                    publishMotionEventFrame(annotatedFrame, frameData->timestamp);
                }
            }
            
        } catch (const std::exception& e) {
            std::string errorMsg = std::string("Exception during processing: ") + e.what();
            publishError(errorMsg);
            ROS_ERROR("[PipelineNode] %s", errorMsg.c_str());
        }
    }
    
    ROS_INFO("[PipelineNode] Processing thread stopped");
}

void PipelineNode::publishRawFrame(const cv::Mat& frame, const ros::Time& timestamp) {
    if (frame.empty()) return;
    
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
        std_msgs::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = timestamp;
    
    rosInterface.pub_processedFrames.publish(msg);
}

void PipelineNode::publishMotionEventFrame(const cv::Mat& frame, const ros::Time& timestamp) {
    if (frame.empty()) return;
    
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
        std_msgs::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = timestamp;
    
    rosInterface.pub_motionEvents.publish(msg);
}

void PipelineNode::publishError(const std::string& errorMsg) {
    std_msgs::String msg;
    msg.data = errorMsg;
    rosInterface.pub_runtimeErrors.publish(msg);
}

void PipelineNode::shutdown() {
    stopWorkerThreads();
    
    // Shutdown components
    components.pipelineIntegratedMotionDetection.reset();
    components.pipelineInternal.reset();
    components.cameraSrc.reset();
}