#pragma once

#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>


namespace engine_properties {
    const int IMG_HEIGHT = 128;
    const int IMG_WIDTH = 64;
    const int MAX_BATCH_SIZE = 128;
    const std::string INPUT_NAME("input");
};


class DeepSortEngineGenerator {
public:
    DeepSortEngineGenerator(nvinfer1::ILogger* gLogger);
    ~DeepSortEngineGenerator();

public:
    void setFP16(bool state);
    void createEngine(std::string onnxPath, std::string enginePath);

private: 
    std::string modelPath, enginePath;
    nvinfer1::ILogger* gLogger;  
    bool useFP16; 
};