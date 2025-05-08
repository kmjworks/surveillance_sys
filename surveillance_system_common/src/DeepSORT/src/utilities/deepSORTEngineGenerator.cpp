#include "../../incl/utilities/deepSORTEngineGenerator.hpp"
#include <cassert>
#include <memory.h> 
#include <fstream>

DeepSortEngineGenerator::DeepSortEngineGenerator(nvinfer1::ILogger* gLogger) {
    this->gLogger = gLogger;
}

DeepSortEngineGenerator::~DeepSortEngineGenerator() {}

void DeepSortEngineGenerator::setFP16(bool state) {
    this->useFP16 = state;
}

void DeepSortEngineGenerator::createEngine(std::string onnxPath, std::string enginePath) {
    // Load onnx model
    auto builder = nvinfer1::createInferBuilder(*gLogger);
    assert(builder != nullptr);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);
    auto config = builder->createBuilderConfig();
    assert(config != nullptr);

    auto profile = builder->createOptimizationProfile();
    nvinfer1::Dims dims = nvinfer1::Dims4{1, 3, engine_properties::IMG_HEIGHT, engine_properties::IMG_WIDTH};
    profile->setDimensions(engine_properties::INPUT_NAME.c_str(),
                nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(engine_properties::INPUT_NAME.c_str(),
                nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{engine_properties::MAX_BATCH_SIZE, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(engine_properties::INPUT_NAME.c_str(),
                nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{engine_properties::MAX_BATCH_SIZE, dims.d[1], dims.d[2], dims.d[3]});
    config->addOptimizationProfile(profile);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, *gLogger);
    assert(parser != nullptr);
    auto parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    assert(parsed);
    if (useFP16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMaxWorkspaceSize(1 << 20);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    nvinfer1::IHostMemory* modelStream = engine->serialize();
    std::string serializeStr;
    std::ofstream serializeOutputStream;
    serializeStr.resize(modelStream->size());
    memcpy((void*)serializeStr.data(), modelStream->data(), modelStream->size());
    serializeOutputStream.open(enginePath);
    serializeOutputStream << serializeStr;
    serializeOutputStream.close();
}