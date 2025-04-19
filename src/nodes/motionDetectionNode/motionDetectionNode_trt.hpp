#pragma once
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <string>
#include <vector>

extern nvinfer1::ILogger& gLogger;

namespace motion_detection {
    struct ROSInterface {
        image_transport::Subscriber sub_imageSrc;
        ros::Publisher pub_detectedMotion;
        ros::Publisher pub_vizDebug;
    };
    
    struct RuntimeConfiguration {
        std::string enginePath;
        float confidenceThreshold{0.6F};
        int inputWidth{640};
        int inputHeight{640};
        void* gpuBuffers[2]{};
        size_t outputSize{0}; 
    };
    
    struct TensorRTInterface {
        std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime{nullptr, [](nvinfer1::IRuntime* p){ if (p) p->destroy(); }};
        std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine{nullptr, [](nvinfer1::ICudaEngine* p){ if (p) p->destroy(); }};
        std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> ctx{nullptr, [](nvinfer1::IExecutionContext* p){ if (p) p->destroy(); }};

        cudaStream_t stream{};
    };
    
    struct DebugConfiguration {
        bool enableViz{true};
        cv_bridge::CvImage vizImg;
    };    
}

class MotionDetectionNode {
public:
    MotionDetectionNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~MotionDetectionNode();

private:
    ros::NodeHandle nh;
    ros::NodeHandle private_nh;
    image_transport::ImageTransport imageTransport;
    motion_detection::ROSInterface rosInterface;
    motion_detection::TensorRTInterface engineInterface;
    motion_detection::RuntimeConfiguration runtimeConfiguration;
    motion_detection::DebugConfiguration runtimeDebugConfiguration;

    void initEngine();
    void imageCb(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat preProcess(const cv::Mat& img);
    std::vector<vision_msgs::Detection2D> postProcess(const float* out);
};
