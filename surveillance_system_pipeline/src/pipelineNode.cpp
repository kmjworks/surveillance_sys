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
    components.cudaPreprocessor = std::make_unique<cuda_components::CUDAPreprocessor>(1920, 1080, 10);
    components.rawFrameQueue.initialize(params.bufferSize);

    int frameW = 1920;
    int frameYH = 1080;
    int nv12BufH = frameYH * 3 / 2;
    int matNV12Comp = CV_8UC1;
    int poolSize = 3;

    try {
        
        components.hostFramePool.reserve(poolSize);
        for (int i = 0; i < poolSize; ++i) {
            components.hostFramePool.emplace_back(nv12BufH, frameW, matNV12Comp, cv::cuda::HostMem::PAGE_LOCKED);
        }
    } catch (const cv::Exception& e) {
        publishError("OpenCV Exception during pinned memory allocation: " + std::string(e.what()));
        ROS_ERROR("OpenCV Exception during pinned memory allocation: %s", e.what());
        return false;
    }


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

void PipelineNode::publishRawFrame(const  cv::Mat& frame, const ros::Time& timestamp) {
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
    ros::Rate loopRate(params.frameRate > 0 ? params.frameRate : 30);


    while (pipelineRunning && ros::ok()) {
        ros::Time captureTimestamp = ros::Time::now();
        cv::Mat frame = components.hostFramePool[components.bufferIdx].createMatHeader();

        bool capture = components.cameraSrc->captureFrameFromSrc(frame);

        if(capture && !frame.empty()) {
            pipeline::FrameData data; 
            frame.copyTo(data.frame); data.timestamp = captureTimestamp;
            components.bufferIdx = (components.bufferIdx + 1) % components.hostFramePool.size();
            if(not components.rawFrameQueue.try_push(std::move(data))) {
                ROS_WARN_THROTTLE(1.0, "Queue full, dropping frame");
            }
        }

        loopRate.sleep();
    }

    ROS_INFO("[PipelineNode - Capture Thread] Exiting.");
}


void PipelineNode::processingLoop() {
    ROS_INFO("[PipelineNode - Processing Thread] Thread started.");

    while(pipelineRunning && ros::ok()) {
        std::optional<pipeline::FrameData> dataOpt = components.rawFrameQueue.pop();
        if (!pipelineRunning && (!dataOpt.has_value() || dataOpt->frame.empty())) break;
        
        if (dataOpt->frame.type() != CV_8UC1) {
            ROS_ERROR_STREAM_THROTTLE(1.0, "[PipelineNode - Processing Thread] Invalid frame type for NV12 input: "
                                      << dataOpt->frame.type() << ". Expected CV_8UC1. Frame will be skipped.");
            continue;
        }



        try {
            cv::Mat outputBGR;
            cv::cvtColor(dataOpt->frame, outputBGR, cv::COLOR_YUV2BGR_NV12); // extremely slow
            publishMotionEventFrame(outputBGR, dataOpt->timestamp);

        } catch (const cv::Exception& e) {
            ROS_ERROR_STREAM_THROTTLE(1.0, "[PipelineNode - Processing Thread] OpenCV (CUDA) Exception: " << e.what()
                                       << " | Input frame dims: " << dataOpt->frame.rows << "x" << dataOpt->frame.cols
                                       << " type: " << dataOpt->frame.type());
            continue;
        }
        
    }

    ROS_INFO("[PipelineNode - Processing Thread] Exiting.");
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
