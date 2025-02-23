#include "pipelineInternal.hpp"
#include <ros/ros.h>

namespace pipeline {

PipelineInternal::PipelineInternal(int frameRate, bool nightMode) {
    state.frameRate = frameRate;
    state.nightMode = nightMode;
    ROS_INFO("[PipelineInternal] Initialized with frameRate=%d, nightMode=%s", 
              frameRate, nightMode ? "true" : "false");
}

PipelineInternal::~PipelineInternal() {
    ROS_INFO("[PipelineInternal] Shutting down");
}

cv::Mat PipelineInternal::processFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        ROS_WARN_THROTTLE(5.0, "[PipelineInternal] Empty frame received");
        return cv::Mat();
    }

    cv::Mat processedFrame = preprocessFrame(frame);
    return processedFrame;
}

cv::Mat PipelineInternal::convertFormat(const cv::Mat& frame) {
    if (frame.empty()) return cv::Mat();
    
    cv::Mat result;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, result, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 3) {
        result = frame.clone();
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, result, cv::COLOR_BGRA2BGR);
    }
    
    return result;
}

cv::Mat PipelineInternal::preprocessFrame(const cv::Mat& frame) {
    cv::Mat result = convertFormat(frame);
    
    if (state.nightMode) {
        // Apply night mode enhancements
        cv::GaussianBlur(result, result, cv::Size(5, 5), 0);
        cv::convertScaleAbs(result, result, 1.5, 10);
    }
    
    return result;
}

} // namespace pipeline