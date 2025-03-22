#include "pipelineInitialDetection.hpp"
#include <ros/ros.h>

PipelineInitialDetection::PipelineInitialDetection(int publishRate) 
    : publishingRate(publishRate), motionParameters{15.0, 200}, isFirstFrame(true) {}


PipelineInitialDetection::~PipelineInitialDetection() {}

bool PipelineInitialDetection::initialize() {
    ROS_INFO("Motion detector initialized with a threshold of %.1f%%", motionParameters.threshold);
    return true;
}

bool PipelineInitialDetection::detectMotion(const cv::Mat& currentFrame) {
    if(isFirstFrame) {
        previousFrame = preprocessFrame(currentFrame);
        isFirstFrame = false;
        return false;
    }

    cv::Mat processedCurrentFrame = preprocessFrame(currentFrame);

    cv::Mat frameDifference; 
    cv::absdiff(previousFrame, processedCurrentFrame, frameDifference);

    cv::Mat thresholdedDifferenceVariance;
    cv::threshold(frameDifference, thresholdedDifferenceVariance, 25, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> detectionContours;
    cv::findContours(thresholdedDifferenceVariance, detectionContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int potentialAnomalies = 0; 
    double totalFrameSceneArea = currentFrame.rows * currentFrame.cols;
    double changedSceneArea = 0; 

    for(const auto& contour : detectionContours) {
        double area = cv::contourArea(contour);
        if(area > motionParameters.threshold) {
            ++potentialAnomalies;
            changedSceneArea += area;
        }
    }

    processedCurrentFrame.copyTo(previousFrame);

    double changeInPercentage = (changedSceneArea / totalFrameSceneArea) * 100.0;

    return (changeInPercentage > motionParameters.threshold);
}

cv::Mat PipelineInitialDetection::preprocessFrame(const cv::Mat& frame) {
    cv::Mat processedFrame;
    /*
        Grayscale conversion if it's not done already
    */
    if(frame.channels() > 1) {
        cv::cvtColor(frame, processedFrame, cv::COLOR_BGR2GRAY);
    } else {
        processedFrame = frame.clone();
    }

    /*
        Slightly trivial, maybe this should be removed - slight Gaussian blur applied for noise 
        reduction but I don't think there's a huge benefit derived from this aside from perhaps
        extra processing overhead
    */
    cv::GaussianBlur(processedFrame, processedFrame, cv::Size(5,5), 0);

    return processedFrame;
}

void PipelineInitialDetection::setThresholdForDetection(double threshold) {
    {
        std::lock_guard<std::mutex> lock(motionDetectionMtx);
        motionParameters.threshold = threshold; 
        ROS_INFO("Motion threshold set to: %.1f%%", motionParameters.threshold);
    }
}

double PipelineInitialDetection::getThreshold() const {
    return motionParameters.threshold;
}



