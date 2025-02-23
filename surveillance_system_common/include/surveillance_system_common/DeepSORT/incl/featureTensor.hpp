#pragma once

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "utilities/model.hpp"


namespace inference {
    struct TensorRTInterface {
        std::unique_ptr<nvinfer1::IRuntime> runtime = nullptr;
        std::unique_ptr<nvinfer1::ICudaEngine> engine = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext> context = nullptr;
    };
    
    using inputBuffer = float* const;
    using outputBuffer = float* const;
}

struct ImageProperties {
    const int maxBatchSize;
    const cv::Size imgShape;
    const int featureDim;
};

struct CudaStreamProperties {
    int curBatchSize;
    const int inputStreamSize, outputStreamSize;
    bool initFlag;
    float* const inputBuffer;
    float* const outputBuffer;
    void* buffers[2];
    cudaStream_t cudaStream;
    int inputIndex, outputIndex;  
};

struct bgrFormat {
    std::array<float, 3> means;
    std::array<float, 3> std;
    const std::string inputName, outputName;
};

class FeatureTensor {
    public:
        FeatureTensor(const int maxBatchSize, const cv::Size& imgShape, const int featureDim, int gpuID, nvinfer1::ILogger* gLogger);
        ~FeatureTensor();

        bool getRectsFeature(const cv::Mat& image, model_internal::DETECTIONS& det);
        bool getRectsFeature(model_internal::DETECTIONS& det);
        int getResult(float*& buffer);

        void loadEngine(const std::string& enginePath);
        void loadOnnxRuntime(const std::string& onnxPath);
        void runInference(std::vector<cv::Mat>& imageMats);
    
    private:
        void initResource();
        void runInference(float* inputBuffer, float* outputBuffer);
        void convertMatToStream(std::vector<cv::Mat>& imgMats, float* stream);
        void decodeStreamToDet(float* stream, model_internal::DETECTIONS& det);

        inference::TensorRTInterface engineInterface; 
        ImageProperties imgInternal;
        CudaStreamProperties cudaStream;
        nvinfer1::ILogger* gLogger;
        bgrFormat format;

};