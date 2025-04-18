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

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING)
            ROS_INFO_STREAM_NAMED("TensorRT", msg);
    }
};

class MotionDetectionNode {
public:
    MotionDetectionNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~MotionDetectionNode();

private:
    image_transport::ImageTransport imageTransport;
    image_transport::Subscriber sub_imageSrc;
    ros::Publisher pub_detectedMotion;
    ros::Publisher pub_vizDebug;
    cv_bridge::CvImage vizImg;

    std::string enginePath;
    float confidenceThreshold{0.4F};
    bool enableViz{true};
    int inputWidth{640};
    int inputHeight{640};

  
    TrtLogger gLogger;
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* ctx{nullptr};
    cudaStream_t stream{};
    void* gpuBuffers[2]{};
    size_t outputSize{0};  

    void imageCb(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat preProcess(const cv::Mat& img);
    std::vector<vision_msgs::Detection2D> postProcess(const float* out);
};
