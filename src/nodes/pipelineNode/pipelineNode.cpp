#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInternal.hpp"
#include "components/pipelineInitialDetectionLite.hpp"

//#include "surveillance_system/motion_detection_events_array.h"
// #include "surveillance_system/motion_event.h"
//#include <sensor_msgs/RegionOfInterest.h>
//#include <geometry_msgs/Point.h>

PipelineNode::PipelineNode(ros::NodeHandle& nh, ros::NodeHandle& privateHandle) :
    nh(nh), nh_priv(privateHandle), pipelineRunning(false), rawFrameQueue(10) {
    
    params.devicePath = "/dev/video0";
    params.outputPath = "~/sursys_recordings";

    params.frameRate = 30;
    params.nightMode = false;
    params.showDebugFrames = true;
    params.motionSamplingRate = 1;
    params.bufferSize = 100;
    params.motionMinAreaPx = 800;
    params.motionDownScale = 0.5f;
    params.motionHistory = 120;    

}

PipelineNode::~PipelineNode() {
    shutdown();
}

bool PipelineNode::initializePipelineNode() {
    ROS_INFO("Initializing Pipeline Node...");

    loadParameters();

    rosInterface.pub_motionEvents = nh.advertise<sensor_msgs::Image>("pipeline/runtime_potentialMotionEvents", 1);
    rosInterface.pub_processedFrames = nh.advertise<sensor_msgs::Image>("pipeline/runtime_processedFrames", 1);
    rosInterface.pub_runtimeErrors = nh.advertise<std_msgs::String>("pipeline/runtime_errors", 10);


    components.cameraSrc = std::make_unique<pipeline::HarrierCaptureSrc>(params.devicePath, params.frameRate, params.nightMode);
    components.pipelineInternal = std::make_unique<pipeline::PipelineInternal>(params.frameRate, params.nightMode);
    components.pipelineIntegratedMotionDetection = std::make_unique<pipeline::PipelineInitialDetectionLite>(params.motionMinAreaPx, params.motionDownScale, params.motionHistory, params.motionSamplingRate);

    if(!components.cameraSrc->initializeRawSrcForCapture()) {
        publishError("Failed to initialize camera after multiple attempts.");
        ROS_ERROR("Failed to initialize camera source.");
        return false;
    }

    rawFrameQueue = ThreadSafeQueue<FrameData>(params.bufferSize);

    pipelineRunning = true;
    startWorkerThreads();

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

    nh_priv.param<int>("motion_detector/min_area_px", params.motionMinAreaPx, params.motionMinAreaPx);
    nh_priv.param<float>("motion_detector/downscale", params.motionDownScale, params.motionDownScale);
    nh_priv.param<int>("motion_detector/history", params.motionHistory, params.motionHistory);

    ROS_INFO("Pipeline parameters loaded.");
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

        rosInterface.pub_processedFrames.publish(img.toImageMsg());
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


void PipelineNode::shutdown() {
    ROS_INFO("Shutting down pipeline node.");

    if(components.cameraSrc) {
        components.cameraSrc->releasePipeline();
    }

    ROS_INFO("Pipeline node shutdown complete.");
}
