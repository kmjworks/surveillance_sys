#include "FeatureExtractor.hpp"

#include <ros/ros.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string> 
#include <stdexcept> 
#include <fmt/core.h>


class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info messages
        if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;

        switch (severity) {
            case Severity::kINTERNAL_ERROR: ROS_ERROR("[TRT Internal Error] %s", msg); break;
            case Severity::kERROR:          ROS_ERROR("[TRT Error] %s", msg); break;
            case Severity::kWARNING:        ROS_WARN("[TRT Warning] %s", msg); break;
            default:                        ROS_INFO("[TRT Log] %s", msg); break;
        }
    }
} gLogger;

FeatureExtractor::FeatureExtractor(const std::string& enginePath) {
    ROS_INFO("[FeatureExtractor] Initializing with engine: %s", enginePath.c_str());
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (not engineFile.good()) throw std::runtime_error("FE: Failed to open engine file: " + enginePath);
    engineFile.seekg(0, std::ios::end); size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg); std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);

    if (not engineFile) throw std::runtime_error("FE: Failed to read engine file: " + enginePath);
    ROS_INFO("[FeatureExtractor] Engine file read (%zu bytes).", engineSize);

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (not runtime) throw std::runtime_error("FE: Failed to create TRT Runtime.");

    engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr));
    if (not engine) throw std::runtime_error("FE: Failed to deserialize TRT Engine.");
    ROS_INFO("[FeatureExtractor] Engine deserialized.");

    if (engine->getNbBindings() != 2) throw std::runtime_error("FE: Expected 2 bindings, found " + std::to_string(engine->getNbBindings()));
    inputBindingIndex = engine->getBindingIndex("input");   
    outputBindingIndex = engine->getBindingIndex("output");
    if (inputBindingIndex < 0) throw std::runtime_error("FE: Input binding 'input' not found.");
    if (outputBindingIndex < 0) throw std::runtime_error("FE: Output binding 'output' not found.");

    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputBindingIndex);
    if (inputDims.nbDims != 4) throw std::runtime_error("FE: Expected 4 input dims (NCHW), found " + std::to_string(inputDims.nbDims));
    modelProperties.maxBatchSize = inputDims.d[0] > 0 ? inputDims.d[0] : 32; 
    modelProperties.input_c_ = inputDims.d[1];
    modelProperties.inputHeight = inputDims.d[2];
    modelProperties.inputWidth = inputDims.d[3];

    if (modelProperties.input_c_ != modelProperties.EXPECTED_C || modelProperties.inputHeight != modelProperties.EXPECTED_H || modelProperties.inputWidth != modelProperties.EXPECTED_W) {
        std::string dimensionErr = fmt::format("Engine input dimensions mismatch. Expected CHW = {}x{}x{}, engine requires = {}x{}x{}",
            modelProperties.EXPECTED_C, modelProperties.EXPECTED_H, modelProperties.EXPECTED_W, 
            modelProperties.input_c_, modelProperties.inputHeight, modelProperties.inputWidth); 
        ROS_ERROR("[FeatureExtractor] %s", dimensionErr.c_str());
        throw std::runtime_error("FE: " + dimensionErr);
    }

    size_t singleInputSize = static_cast<size_t>(modelProperties.input_c_) * modelProperties.inputHeight * modelProperties.inputWidth * sizeof(float);
    inputBufferSize = static_cast<size_t>(modelProperties.maxBatchSize) * singleInputSize;

    // Calculate output buffer size and feature dim 
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputBindingIndex);
    if (outputDims.nbDims != 2) { throw std::runtime_error("FE: Expected 2 output dims (Batch, FeatureDim), found " + std::to_string(outputDims.nbDims)); }
    if (outputDims.d[0] != -1 && outputDims.d[0] != inputDims.d[0]) { throw std::runtime_error("FE: Output batch dim doesn't match input."); }
    modelProperties.featureDim = outputDims.d[1]; 
    size_t singleOutputSize = static_cast<size_t>(modelProperties.featureDim) * sizeof(float);
    outputBufferSize = static_cast<size_t>(modelProperties.maxBatchSize) * singleOutputSize;
    hostOutputBuffer.resize(static_cast<size_t>(modelProperties.maxBatchSize) * modelProperties.featureDim); 

    ROS_INFO("[FeatureExtractor] Verified Input (NCHW): %dx%dx%dx%d, Output: %dx%d, Max Batch: %d",
             modelProperties.maxBatchSize, modelProperties.input_c_, modelProperties.inputHeight, modelProperties.inputWidth,
             modelProperties.maxBatchSize, modelProperties.featureDim, modelProperties.maxBatchSize);


    context.reset(engine->createExecutionContext());
    if (not context) throw std::runtime_error("FE: Failed to create TRT Context.");
    if (cudaStreamCreate(&stream) != cudaSuccess) throw std::runtime_error("FE: Failed to create CUDA stream.");


    if (cudaMalloc(&gpuBuffers[inputBindingIndex], inputBufferSize) != cudaSuccess) {
        cudaStreamDestroy(stream); stream = nullptr; throw std::runtime_error("FE: Failed cudaMalloc for input.");
    }
    if (cudaMalloc(&gpuBuffers[outputBindingIndex], outputBufferSize) != cudaSuccess) {
        cudaFree(gpuBuffers[inputBindingIndex]); gpuBuffers[inputBindingIndex] = nullptr;
        cudaStreamDestroy(stream); stream = nullptr; throw std::runtime_error("FE: Failed cudaMalloc for output.");
    }

    ROS_INFO("[FeatureExtractor] Initialization complete.");
}

FeatureExtractor::~FeatureExtractor() {
    if(gpuBuffers[inputBindingIndex]) {
        cudaFree(gpuBuffers[inputBindingIndex]);
        gpuBuffers[inputBindingIndex] = nullptr;
    }

    if(gpuBuffers[outputBindingIndex]) {
        cudaFree(gpuBuffers[outputBindingIndex]);
        gpuBuffers[outputBindingIndex] = nullptr;
    }

    if(stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

bool FeatureExtractor::preprocessMatBatch(const std::vector<cv::Mat>& batchPatches, std::vector<float>& hostBuffer) {
    hostBuffer.clear();
    if(batchPatches.empty()) return true;

    hostBuffer.resize(batchPatches.size() * modelProperties.input_c_ * modelProperties.inputHeight * modelProperties.inputWidth);
    float* bufferPtr = hostBuffer.data();
    size_t singleImageArea = static_cast<size_t>(modelProperties.inputHeight);

    for (const auto& patch : batchPatches) {
        if (patch.empty()) {
            ROS_ERROR("[FeatureExtractor] Encountered empty patch during preprocessing.");
            return false;
        }

        cv::Mat resizedPatch;
        cv::resize(patch,  resizedPatch, cv::Size(modelProperties.inputWidth, modelProperties.inputHeight), 0, 0, cv::INTER_LINEAR);

        cv::Mat floatPatch;
        resizedPatch.convertTo(floatPatch, CV_32FC3, 1.0 / 255.0);

        float* currentImgBufferStart = bufferPtr;
        for (int c = 0; c < modelProperties.input_c_; ++c) {
            for (int h = 0; h < modelProperties.inputHeight; ++h) {
                for (int w = 0; w < modelProperties.inputWidth; ++w) {
                    bufferPtr[c * singleImageArea + h * modelProperties.inputWidth + w] = floatPatch.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        bufferPtr += static_cast<size_t>(modelProperties.input_c_) * singleImageArea;
    }
    return true;
}

bool FeatureExtractor::getApperanceFeatures(const std::vector<cv::Mat>& imagePatches, std::vector<std::vector<float>>& features) {
    features.clear();
    if(imagePatches.empty()) {
        ROS_DEBUG("[FeatureExtractor] getFeatures called with empty patch list.");
        return true; 
    }

    int nPatches = static_cast<int>(imagePatches.size());
    features.resize(nPatches);

    int nBatches = (nPatches + modelProperties.maxBatchSize - 1) / modelProperties.maxBatchSize;

    std::vector<float> hostInputBuffer;

    for(int batchIndex = 0; batchIndex < nBatches; ++batchIndex) {
        int startingIndex = batchIndex * modelProperties.maxBatchSize;
        int batchSizeActual = std::min(modelProperties.maxBatchSize, nPatches - startingIndex);
        if (batchSizeActual <= 0) continue;

        std::vector<cv::Mat> currentBatchPatches(imagePatches.begin() + startingIndex, imagePatches.begin() + startingIndex + batchSizeActual);

        if(not preprocessMatBatch(currentBatchPatches, hostInputBuffer)) {
            ROS_ERROR("[FeatureExtractor] Preprocessing failed for batch %d.", batchIndex);
            features.clear();
            return false;   
        }

        size_t inputBytes = batchSizeActual * modelProperties.input_c_ * modelProperties.inputHeight * modelProperties.inputWidth * sizeof(float);
        size_t outputBytes = batchSizeActual * modelProperties.featureDim * sizeof(float);

        cudaError_t cudaStatus;
        cudaStatus = cudaMemcpyAsync(gpuBuffers[inputBindingIndex], hostInputBuffer.data(), inputBytes, cudaMemcpyHostToDevice, stream);
        if (cudaStatus != cudaSuccess) { 
            ROS_ERROR("[FE] H2D Err: %s", cudaGetErrorString(cudaStatus)); 
            cudaStreamSynchronize(stream); 
            features.clear(); 
            return false; 
        }

        bool status = context->enqueueV2(gpuBuffers, stream, nullptr);
        if (!status) { 
            ROS_ERROR("[FE] Inference Err batch %d", batchIndex); 
            cudaStreamSynchronize(stream); 
            features.clear(); 
            return false; 
        }
    
        cudaStatus = cudaMemcpyAsync(hostOutputBuffer.data(), gpuBuffers[outputBindingIndex], outputBytes, cudaMemcpyDeviceToHost, stream);
        if (cudaStatus != cudaSuccess) { 
            ROS_ERROR("[FE] D2H Err: %s", cudaGetErrorString(cudaStatus)); 
            cudaStreamSynchronize(stream); 
            features.clear(); 
            return false; 
        }

        cudaStatus = cudaStreamSynchronize(stream);
        if (cudaStatus != cudaSuccess) { 
            ROS_ERROR("[FE] Sync Err: %s", cudaGetErrorString(cudaStatus)); 
            features.clear(); 
            return false; 
        }

        for (int i = 0; i < batchSizeActual; ++i) {
            int outputIndex = startingIndex + i;
            float* batchOutputStart = hostOutputBuffer.data() + i * modelProperties.featureDim;
            features[outputIndex].resize(modelProperties.featureDim);
            features[outputIndex].assign(batchOutputStart, batchOutputStart + modelProperties.featureDim);
        }
        ROS_DEBUG("[FeatureExtractor] Processed batch %d/%d (%d patches)", batchIndex+1, nBatches, batchSizeActual);
    }
    return true;
}