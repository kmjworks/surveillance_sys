#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <string>
#include <memory>


namespace pipeline {
    class HarrierCaptureSrc;
    class PipelineInternal;
    class PipelineInitialDetection;

    struct ROSInterface {
        ros::Publisher pub_motionEvents;
        ros::Publisher pub_processedFrames;
        ros::Publisher pub_runtimeErrors;
    };

    struct PipelineComponents {
        std::unique_ptr<HarrierCaptureSrc> cameraSrc;
        std::unique_ptr<PipelineInternal> pipelineInternal;
        std::unique_ptr<PipelineInitialDetection> pipelineIntegratedMotionDetection;
    };

    struct ConfigurationParameters {
        int frameRate;
        int motionSamplingRate;
        int bufferSize;

        bool nightMode;
        bool showDebugFrames;
        
        std::string devicePath;
        std::string outputPath;
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

        std::atomic<bool> pipelineRunning;
        std::mutex frameMtx;
        std::thread pipelineProcessingThread;

        void loadParameters();
        void processFrames();
        void publishFrame(const cv::Mat& frame, const ros::Time& timestamp);
        void publishError(const std::string& errorMsg);
        void startProcessingThread();
        void stopProcessingThread();


};