#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h> 
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory> 
#include <stdexcept>
#include <numeric> 
#include <algorithm>


struct NvInferDeleteHandler {
    template<typename T>
    void operator()(T* obj) const {
        if(obj) {
            obj->destroy();
        }
    }
};

struct ModelProperties {
    int input_c_ = 3;
    int inputHeight = 128;
    int inputWidth = 64;
    int featureDim = 0;     
    int maxBatchSize = 1; 

    const int EXPECTED_H = 128;
    const int EXPECTED_W = 64;
    const int EXPECTED_C = 3;
};

template <typename T>
using NvUniquePtr = std::unique_ptr<T, NvInferDeleteHandler>;

class FeatureExtractor {
    public:
        explicit FeatureExtractor(const std::string& enginePath);
        ~FeatureExtractor();

        FeatureExtractor(const FeatureExtractor&) = delete;
        FeatureExtractor& operator=(const FeatureExtractor&) = delete;

        bool getApperanceFeatures(const std::vector<cv::Mat>& imagePatches, std::vector<std::vector<float>>& features);
        int getInputHeight() const { return modelProperties.inputHeight; }
        int getInputWidth() const { return modelProperties.inputWidth; }
        int getFeatureDims() const { return modelProperties.featureDim; }
        int getMaxBatchSize() const { return modelProperties.maxBatchSize; }

    private:
        NvUniquePtr<nvinfer1::IRuntime> runtime = nullptr;
        NvUniquePtr<nvinfer1::ICudaEngine> engine = nullptr;
        NvUniquePtr<nvinfer1::IExecutionContext> context = nullptr;
        cudaStream_t stream = nullptr;

        void* gpuBuffers[2];                
        std::vector<float> hostOutputBuffer;
        int inputBindingIndex = -1;
        int outputBindingIndex = -1;
        size_t inputBufferSize = 0;   
        size_t outputBufferSize = 0; 

        ModelProperties modelProperties;

        bool preprocessMatBatch(const std::vector<cv::Mat>& batchPatches, std::vector<float>& hostBuffer);
};