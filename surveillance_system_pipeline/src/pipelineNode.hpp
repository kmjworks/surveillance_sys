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

#include "ThreadSafeQueue/ThreadSafeQueue.hpp"
#include "components/cuda/cudaPreprocessor.hpp"

namespace pipeline {
    class HarrierCaptureSrc;
    class PipelineInternal;
    class PipelineInitialDetectionLite;

    struct ROSInterface {
        ros::Publisher pub_runtimeErrors;
        image_transport::Publisher pub_processedFrames;
        image_transport::Publisher pub_motionEvents;
    };

    struct FrameData {
        cv::Mat frame;
        ros::Time timestamp;
    };

    struct PipelineComponents {
        std::unique_ptr<HarrierCaptureSrc> cameraSrc;
        std::unique_ptr<PipelineInternal> pipelineInternal;
        std::unique_ptr<PipelineInitialDetectionLite> pipelineIntegratedMotionDetection;
        std::unique_ptr<cuda_components::CUDAPreprocessor> cudaPreprocessor;
        ThreadSafeQueue<FrameData> rawFrameQueue;

        cv::cuda::Stream stream;
        cv::cuda::GpuMat gpuFrameBuffer[2];
        std::vector<cv::cuda::HostMem> hostFramePool;
        int bufferIdx = 0;
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

    
    
}

class PipelineNode {
    public:
        PipelineNode(ros::NodeHandle& nh, ros::NodeHandle& privateHandle);
        ~PipelineNode();

        bool initializePipelineNode();
        /*
            loadParameters made public for unit testing
        */
        void loadParameters(ros::NodeHandle& nh_priv, pipeline::ConfigurationParameters& parametersToLoad); // hack
        void shutdown();

    private:
        ros::NodeHandle& nh;
        ros::NodeHandle& nh_priv;
        image_transport::ImageTransport imageTransport;
        
        pipeline::ROSInterface rosInterface;
        pipeline::ConfigurationParameters params;
        pipeline::PipelineComponents components;

        std::thread captureThread;
        std::thread processingThread;
        std::atomic<bool> pipelineRunning;
        
        void publishMotionEventFrame(const cv::Mat& frame, const ros::Time& timestamp);
        void publishRawFrame(const cv::Mat& frame, const ros::Time& timestamp);
        void publishError(const std::string& errorMsg);
        void captureLoop();
        void processingLoop();
        void startWorkerThreads();
        void stopWorkerThreads();

};
