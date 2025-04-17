#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInternal.hpp"
#include "components/pipelineInitialDetectionLite.hpp"

//#include "surveillance_system/motion_detection_events_array.h"
// #include "surveillance_system/motion_event.h"
//#include <sensor_msgs/RegionOfInterest.h>
//#include <geometry_msgs/Point.h>

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
    components.pipelineIntegratedMotionDetection = std::make_unique<pipeline::PipelineInitialDetectionLite>(800, 0.5f, 120, 2);

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


void PipelineNode::publishFrame(const cv::Mat& frame, const ros::Time& timestamp) {
    try {
        cv_bridge::CvImage cvImgForMsgPrep;
        cvImgForMsgPrep.encoding = (frame.channels() == 1 ? "mono8" : "bgr8");
        cvImgForMsgPrep.image = frame;
        cvImgForMsgPrep.header.stamp = timestamp;
        cvImgForMsgPrep.header.frame_id = "Harrier36X-Initial-Detection";

        rosInterface.pub_processedFrames.publish(cvImgForMsgPrep.toImageMsg());
    } catch (const cv_bridge::Exception &e) {
        publishError("cv_bridge exception: " + std::string(e.what()));
    }
    /* 
        Should probably handle a std::exception as well? 
        Hmm, come to think of it - I think it's not the right place for this since I will eventually
        make a separate component for handling system level errors - the pipeline assumes that everything is
        functioning on its side and it doesn't (and doesn't have to) know about system-wide status (that's a task for the master)
    */
}

void PipelineNode::publishError(const std::string& errorMsg) {
    std_msgs::String msg;
    msg.data = errorMsg;
    rosInterface.pub_runtimeErrors.publish(msg);

    ROS_ERROR_THROTTLE(10, "%s", errorMsg.c_str());
}

void PipelineNode::startProcessingThread() {
    if(!pipelineRunning) pipelineRunning = true;
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
    ros::Rate processingLoop_rate(30);

    while(pipelineRunning && ros::ok()) {

        /*
            Potentially move the capture time to the gstreamer pipeline closer to the raw src (caps)
            and attach it as metadata.

            'frameCaptureTime' is not instantenous here and is guaranteed to have some kind of a latency 
            because of the kernel's scheduling, frame data transmission via USB and so on.
            But for testing and prototyping the inital system, it's an acceptable solution for a temporal one. 

            TODO: Determine the frame acquisition latency and subtract it or monitor it as a compounding
            offset
        */
        ros::Time frameCaptureTime = ros::Time::now();

        if(!components.cameraSrc->captureFrameFromSrc(raw)) {
            publishError("Frame capture failed.");
            ros::Duration(0.5).sleep();
            continue;
        }

        if(raw.empty()) {
            ROS_WARN_THROTTLE(2.0, "Empty frame captured, skpping processing.");
            continue;
        }
    
        
        {
            std::lock_guard<std::mutex> lock(frameMtx);
            processed = components.pipelineInternal->processFrame(raw);
            
            if(!processed.empty()) {
                motionPresence = components.pipelineIntegratedMotionDetection->detect(processed, motionRects);

                if(motionPresence && !motionRects.empty()) {
                    for(const auto& rect : motionRects) {
                        cv::rectangle(processed, rect, cv::Scalar(0,255,0), 2);
                    }
                }
            }
            
        }

        if(!processed.empty()) {
            publishFrame(processed, frameCaptureTime);
        }

        if(motionPresence) { 
            cv_bridge::CvImage cvImageMotion; 
            cvImageMotion.encoding = (processed.channels() == 1 ? "mono8" : "bgr8");
            cvImageMotion.image = processed;
            cvImageMotion.header.stamp = frameCaptureTime;
            cvImageMotion.header.frame_id = "Harrier36X-Initial-Detection";
            rosInterface.pub_motionEvents.publish(cvImageMotion.toImageMsg());
        }
        processingLoop_rate.sleep();
    }
}

void PipelineNode::shutdown() {
    ROS_INFO("Shutting down pipeline node.");

    if(components.cameraSrc) {
        components.cameraSrc->releasePipeline();
    }

    ROS_INFO("Pipeline node shutdown complete.");
}
