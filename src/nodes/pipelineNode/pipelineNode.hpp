#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <mutex>
#include <atomic>
#include <thread>
#include <string>
#include <memory>

#include "components/utilities/ThreadSafeQueue/ThreadSafeQueue.hpp"

namespace pipeline {
    class HarrierCaptureSrc;
    class PipelineInternal;
    class PipelineInitialDetectionLite;

    struct ROSInterface {
        ros::Publisher pub_runtimeErrors;
        image_transport::ImageTransport imageTransport;
        image_transport::Publisher pub_processedFrames;
        image_transport::Publisher pub_motionEvents;

    };

    struct PipelineComponents {
        std::unique_ptr<HarrierCaptureSrc> cameraSrc;
        std::unique_ptr<PipelineInternal> pipelineInternal;
        std::unique_ptr<PipelineInitialDetectionLite> pipelineIntegratedMotionDetection;
    };

    struct ConfigurationParameters {
        int frameRate;
        int motionSamplingRate;
        int bufferSize;
        int motionMinAreaPx;
        float motionDownScale;
        int motionHistory;

        bool nightMode;
        bool showDebugFrames;
        
        std::string devicePath;
        std::string outputPath;
    };

    struct FrameData {
        cv::Mat frame;
        ros::Time timestamp;
    };
}

class PipelineNode {
    public:
        PipelineNode(ros::NodeHandle& nh, ros::NodeHandle& privateHandle);
        ~PipelineNode();

        bool initializePipelineNode();
        void shutdown();

    private:
        ros::NodeHandle& nh;
        ros::NodeHandle& nh_priv;

        pipeline::ROSInterface rosInterface;
        pipeline::PipelineComponents components;
        pipeline::ConfigurationParameters params;
        ThreadSafeQueue<pipeline::FrameData> rawFrameQueue;

        std::thread captureThread;
        std::thread processingThread;
        std::atomic<bool> pipelineRunning;

        void publishMotionEventFrame(const cv::Mat& frame, const ros::Time& timestamp);
        void publishRawFrame(const cv::Mat& frame, const ros::Time& timestamp);
        void publishError(const std::string& errorMsg);

        void loadParameters();
        void processFrames();
        void captureLoop();
        void startWorkerThreads();
        void stopWorkerThreads();


};