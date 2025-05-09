#pragma once 

#include <opencv2/opencv.hpp>
#include "utilities/dataTypes.hpp"
#include "featureTensor.hpp"
#include "Tracker.hpp"
#include <vector>
#include <memory> 

class DeepSort {
    public:
        DeepSort(std::string enginePath, int batchSize, int featureDimensions, int gpuIdentifier, nvinfer1::ILogger* gLogger);
        ~DeepSort();

        void sort(cv::Mat& frame, std::vector<DetectionBox>& dets);

    private:
        void sort(cv::Mat& frame, model_internal::DETECTIONS& detections);
        void sort(cv::Mat& frame, model_internal::DETECTIONSV2& detectionsv2);    
        void sort(std::vector<DetectionBox>& dets);
        void sort(model_internal::DETECTIONS& detections);
        void init();

        std::string enginePath;
        int batchSize;
        int featureDim;
        cv::Size imgShape;
        float confThres;
        float nmsThres;
        int maxBudget;
        float maxCosineDist;

        std::vector<motiontracker::RESULT_DATA> result;
        std::vector<std::pair<CLSCONF, tracking::DETECTBOX>> results;
        std::unique_ptr<Tracker> objTracker;
        std::unique_ptr<FeatureTensor> featureExtractor;
        nvinfer1::ILogger* gLogger;
        std::mutex resultMutex;
        
        int gpuID;
};
