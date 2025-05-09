#include "../incl/featureTensor.hpp"
#include <fstream>
#include "ros/console.h"


#define INPUTSTREAM_SIZE (maxBatchSize*3*imgShape.area())
#define OUTPUTSTREAM_SIZE (maxBatchSize*featureDim)
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::stringstream ss; \
        ss << "CUDA error: " << cudaGetErrorString(error); \
        throw std::runtime_error(ss.str()); \
    } \
} while(0)

FeatureTensor::FeatureTensor(const int maxBatchSize, const cv::Size& imgShape, const int featureDim, int gpuID, nvinfer1::ILogger* gLogger) :
    engineInterface{nullptr, nullptr, nullptr},
    imgInternal{maxBatchSize, imgShape, featureDim}, cudaStream{0, INPUTSTREAM_SIZE, OUTPUTSTREAM_SIZE, false, new float[INPUTSTREAM_SIZE], new float[OUTPUTSTREAM_SIZE], {nullptr, nullptr}, nullptr, 0, 0}, 
    gLogger(gLogger),
    format{{0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}, "output", "input"} {
    
    cudaSetDevice(gpuID);
}

FeatureTensor::~FeatureTensor() {
    try {
        if(cudaStream.initFlag) cudaStreamSynchronize(cudaStream.cudaStream);

        delete[] cudaStream.inputBuffer; 
        delete[] cudaStream.outputBuffer;

        if(cudaStream.initFlag) {
            if(cudaStream.buffers[cudaStream.inputIndex]) {
                cudaFree(cudaStream.buffers[cudaStream.inputIndex]);
                cudaStream.buffers[cudaStream.inputIndex] = nullptr;
            }

            if(cudaStream.buffers[cudaStream.outputIndex]) {
                cudaFree(cudaStream.buffers[cudaStream.outputIndex]);
                cudaStream.buffers[cudaStream.outputIndex] = nullptr;
            }
            
            cudaStreamDestroy(cudaStream.cudaStream);
            cudaStream.cudaStream = nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "[FeatureTensor] Exception during destruction: " << e.what() << std::endl;
    }
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, model_internal::DETECTIONS& det) {
    std::vector<cv::Mat> mats;
    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        rect.width = rect.height * 0.5;
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));
        cv::Mat tempMat = img(rect).clone();
        cv::resize(tempMat, tempMat, imgInternal.imgShape);
        mats.push_back(tempMat);
    }
    runInference(mats);
    decodeStreamToDet(cudaStream.outputBuffer, det);
    return true;
}

bool FeatureTensor::getRectsFeature(model_internal::DETECTIONS& det) {
    return true;
}

void FeatureTensor::loadEngine(const std::string& enginePath) {
    
    // Deserialize model
    engineInterface.runtime.reset(nvinfer1::createInferRuntime(*gLogger));
    assert(engineInterface.runtime != nullptr);
    std::ifstream engineStream(enginePath, std::ios::binary);
    std::string engineCache("");
    while (engineStream.peek() != EOF) {
        std::stringstream buffer;
        buffer << engineStream.rdbuf();
        engineCache.append(buffer.str());
    }
    engineStream.close();

    engineInterface.engine.reset(engineInterface.runtime->deserializeCudaEngine(engineCache.data(), engineCache.size(), nullptr));
    ROS_INFO("[FeatureExtractor] Engine deserialized.");
    assert(engineInterface.engine != nullptr);

    engineInterface.context.reset(engineInterface.engine->createExecutionContext());
    assert(engineInterface.context != nullptr);
    initResource();
} 

void FeatureTensor::loadOnnxRuntime(const std::string& onnxPath) {
    auto builder = nvinfer1::createInferBuilder(*gLogger);
    assert(builder != nullptr);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);
    auto config = builder->createBuilderConfig();
    assert(config != nullptr);

    auto profile = builder->createOptimizationProfile();
    nvinfer1::Dims dims = nvinfer1::Dims4{1, 3, imgInternal.imgShape.height, imgInternal.imgShape.width};
    profile->setDimensions(format.inputName.c_str(),
                nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(format.inputName.c_str(),
                nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{imgInternal.maxBatchSize, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(format.inputName.c_str(),
                nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{imgInternal.maxBatchSize, dims.d[1], dims.d[2], dims.d[3]});
    config->addOptimizationProfile(profile);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, *gLogger);
    assert(parser != nullptr);
    auto parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    assert(parsed);
    config->setMaxWorkspaceSize(1 << 20);
    engineInterface.engine.reset(builder->buildEngineWithConfig(*network, *config));
    assert(engineInterface.engine != nullptr);
    engineInterface.context.reset(engineInterface.engine->createExecutionContext());
    assert(engineInterface.context != nullptr);
    initResource();
}

int FeatureTensor::getResult(float*& buffer) {
    if (buffer != nullptr)
        delete buffer;
    int curStreamSize = cudaStream.curBatchSize*imgInternal.featureDim;
    buffer = new float[curStreamSize];
    for (int i = 0; i < curStreamSize; ++i) {
        buffer[i] = cudaStream.outputBuffer[i];
    }
    return curStreamSize;
}

void FeatureTensor::runInference(std::vector<cv::Mat>& imgMats) {
    convertMatToStream(imgMats, cudaStream.inputBuffer);
    runInference(cudaStream.inputBuffer, cudaStream.outputBuffer);
}

void FeatureTensor::initResource() {
    cudaStream.inputIndex = engineInterface.engine->getBindingIndex(format.inputName.c_str());
    cudaStream.outputIndex = engineInterface.engine->getBindingIndex(format.outputName.c_str());

    
    CUDA_CHECK(cudaStreamCreate(&cudaStream.cudaStream));

    cudaStream.buffers[cudaStream.inputIndex] = cudaStream.inputBuffer;
    cudaStream.buffers[cudaStream.outputIndex] = cudaStream.outputBuffer;
    
    CUDA_CHECK(cudaMalloc(&cudaStream.buffers[cudaStream.inputIndex], cudaStream.inputStreamSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cudaStream.buffers[cudaStream.outputIndex], cudaStream.outputStreamSize * sizeof(float)));
    
    ROS_INFO("[FeatureExtractor] Initialization complete - CUDA streams initialized.");

    cudaStream.initFlag = true;
}

void FeatureTensor::runInference(float* inputBuffer, float* outputBuffer) {   
    
    CUDA_CHECK(cudaMemcpyAsync(cudaStream.buffers[cudaStream.inputIndex], inputBuffer, cudaStream.inputStreamSize * sizeof(float), cudaMemcpyHostToDevice, cudaStream.cudaStream));
    nvinfer1::Dims4 inputDims{cudaStream.curBatchSize, 3, imgInternal.imgShape.height, imgInternal.imgShape.width};
    engineInterface.context->setBindingDimensions(0, inputDims);
    
    engineInterface.context->enqueueV2(cudaStream.buffers, cudaStream.cudaStream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(outputBuffer, cudaStream.buffers[cudaStream.outputIndex], cudaStream.outputStreamSize * sizeof(float), cudaMemcpyDeviceToHost, cudaStream.cudaStream));
    cudaError_t syncStatus = cudaStreamSynchronize(cudaStream.cudaStream);
    if(syncStatus != cudaSuccess) ROS_ERROR("[FeatureTensor] cudaStreamSynchronize Error: %s", cudaGetErrorString(syncStatus));
}

void FeatureTensor::convertMatToStream(std::vector<cv::Mat>& imgMats, float* stream) {
    int imgArea = imgInternal.imgShape.area();
    cudaStream.curBatchSize = imgMats.size();
    if (cudaStream.curBatchSize > imgInternal.maxBatchSize) {
        std::cout << "[WARNING]::Batch size overflow, input will be truncated!" << "\n";
        cudaStream.curBatchSize = imgInternal.maxBatchSize;
    }
    for (int batch = 0; batch < cudaStream.curBatchSize; ++batch) {
        const cv::Mat& tempMat = imgMats[batch];
        int i = 0; 
        for (int row = 0; row < imgInternal.imgShape.height; ++row) {
            uchar* uc_pixel = tempMat.data + row * tempMat.step;
            for (int col = 0; col < imgInternal.imgShape.width; ++col) {
                stream[batch * 3 * imgArea + i] = ((float)uc_pixel[0] / 255.0 - format.means[0]) / format.std[0];
                stream[batch * 3 * imgArea + i + imgArea] = ((float)uc_pixel[1] / 255.0 - format.means[1]) / format.std[1];
                stream[batch * 3 * imgArea + i + 2 * imgArea] = ((float)uc_pixel[2] / 255.0 - format.means[2]) / format.std[2];
                uc_pixel += 3;
                ++i;
            }
        }
    }
}

void FeatureTensor::decodeStreamToDet(float* stream, model_internal::DETECTIONS& det) {
    int i = 0;

    //ROS_INFO("decodeStreamToDet - Processing %zu detections with batch size %d", det.size(), cudaStream.curBatchSize);

    for(DETECTION_ROW& dbox : det) {
        for(int j = 0; j < imgInternal.featureDim; ++j) {
            dbox.feature[j] = stream[i*imgInternal.featureDim + j];
            ++i;
        }
    }

    //ROS_INFO("decodeStreamToDet - Completed processing %d detections", i);
}
