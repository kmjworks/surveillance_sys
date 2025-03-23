#include "pipelineInitialDetection.hpp"
#include <ros/ros.h>

PipelineInitialDetection::PipelineInitialDetection(int publishRate) 
    : publishingRate(publishRate), motionParameters{5.0, 50}, isFirstFrame(true) {

        for(int i = 0; i < 5; ++i) {
            potentialMotionEventHistory.push_back(false);
        }
    }


PipelineInitialDetection::~PipelineInitialDetection() {}

bool PipelineInitialDetection::initialize() {

    try {
        previousFrame = cv::Mat::zeros(480, 640, CV_8UC1);

        isFirstFrame = true;
        potentialMotionEventHistory.clear();

        for(int i = 0; i < 5; ++i) {
            potentialMotionEventHistory.push_back(false);
        }
    } catch (const std::exception &e) {
        ROS_ERROR("Execption in motion detector initialization: %s", e.what());
        return false;
    }

    ROS_INFO("Motion detector initialized with a threshold of %.1f%%", motionParameters.threshold);
    return true;
}

bool PipelineInitialDetection::initialize(int minThresholdArea, double motionThreshold) {
    motionParameters.minArea = minThresholdArea;
    motionParameters.threshold = motionThreshold;
    return initialize();
}

bool PipelineInitialDetection::detectMotion(const cv::Mat& currentFrame) {
    std::lock_guard<std::mutex> lock(motionDetectionMtx);

    if(currentFrame.empty()) {
        ROS_WARN("Empty frame passed to motion detector.");
        return false;
    }

    bool motionDetected = detectFrameDiff(currentFrame);

    potentialMotionEventHistory.push_back(motionDetected);
    if(potentialMotionEventHistory.size() > 5) {
        potentialMotionEventHistory.pop_front();
    }

    return motionDetected;
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
    cv::GaussianBlur(processedFrame, processedFrame, cv::Size(3,3), 0);

    return processedFrame;
}

bool PipelineInitialDetection::detectFrameDiff(const cv::Mat& currentFrame) {
    if(isFirstFrame) {
        previousFrame = preprocessFrame(currentFrame);
        isFirstFrame = false;
        return false;
    }

    // Process current frame with more efficient preprocessing
    cv::Mat processedCurrentFrame = preprocessFrame(currentFrame);

    // Compute frame difference
    cv::Mat frameDifference;
    cv::absdiff(previousFrame, processedCurrentFrame, frameDifference);

    // Threshold the difference image
    cv::Mat thresholded;
    cv::threshold(frameDifference, thresholded, 20, 255, cv::THRESH_BINARY);
    
    // Calculate changed pixels directly without contour finding for better performance
    // This avoids the expensive contour finding operation
    double totalArea = currentFrame.rows * currentFrame.cols;
    double changeArea = cv::countNonZero(thresholded);
    
    // Update previous frame for next comparison (use move assignment if possible)
    processedCurrentFrame.copyTo(previousFrame);

    // Calculate change percentage and determine if motion is detected
    double changePercentage = (changeArea / totalArea) * 100.0;
    bool motionDetected = (changePercentage > motionParameters.threshold);

    ROS_DEBUG_THROTTLE(1, "Motion detection: change percentage = %.2f%%, threshold = %.2f%%", 
        changePercentage, motionParameters.threshold);

    return motionDetected;
}

void PipelineInitialDetection::setThresholdForDetection(double threshold) {
    std::lock_guard<std::mutex> lock(motionDetectionMtx);
    motionParameters.threshold = threshold; 
    ROS_INFO("Motion threshold set to: %.1f%%", motionParameters.threshold);
    
}

double PipelineInitialDetection::getThreshold() const {
    return motionParameters.threshold;
}
