#pragma once 

#include <iostream>
#include <opencv2/opencv.hpp>
#include "utilities/dataTypes.hpp"
#include <vector>

class DeepSort {
    public:
        DeepSort(std::string enginePath, int batchSize, int featureDimensions, int gpuIdentifier, ILogger* gLogger);
        ~DeepSort();

        void sort(cv::Mat& frame, std::vector<DetectB>)

}