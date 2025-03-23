#include "pipelineInternal.hpp"
#include <ros/ros.h>
#include <chrono>
#include <iomanip>
#include <sstream>

PipelineInternal::PipelineInternal(bool nightMode, bool debugMode)
    : internalState{nightMode, debugMode, true, true} {

    updateTimestamp();
}

PipelineInternal::~PipelineInternal() {

}

bool PipelineInternal::initialize() {
    ROS_INFO("Pipeline internal processing initialized");
    ROS_INFO("  Night mode: %s", internalState.nightMode ? "enabled" : "disabled");
    ROS_INFO("  Debug mode: %s", internalState.debugMode ? "enabled" : "disabled");
    return true;
}

cv::Mat PipelineInternal::processFrame(const cv::Mat& frame) {
    updateTimestamp();

    if(frame.empty()) {
        ROS_WARN("Empty frame passed to pipeline for processing.");
        return cv::Mat();
    }

    cv::Mat processedFrame = frame.clone();

    if(internalState.denoiseEnabled) {
        processedFrame = applyDenoising(processedFrame);
    }

    if(internalState.autoBrightnessCorrection) {
        if(processedFrame.channels() == 3) {
            double alpha = 1.2;
            int beta = 5;
            processedFrame.convertTo(processedFrame, -1, alpha, beta);
        }
        processedFrame = applyBrightnessCorrection(processedFrame);
    }

    if(internalState.nightMode) {
        processedFrame = convertForNightModeCompatibility(processedFrame);
    }

    if(internalState.debugMode) {
        processedFrame = latchDebugInfo(processedFrame);
    }

    return processedFrame;
}

cv::Mat PipelineInternal::convertForNightModeCompatibility(const cv::Mat& frame) {
    cv::Mat grayscaleFrame;

    if(frame.channels() == 1) {
        return frame.clone();
    }

    cv::cvtColor(frame, grayscaleFrame, cv::COLOR_BGR2GRAY);

    return grayscaleFrame;
}

cv::Mat PipelineInternal::applyDenoising(const cv::Mat& frame) {
    cv::Mat denoisedFrame;
    
    // Replace extremely slow NLMeans denoising with much faster bilateral filter
    // or median blur which are better suited for real-time processing
    if(frame.channels() == 3) {
        // Using bilateral filter - preserves edges while removing noise
        cv::bilateralFilter(frame, denoisedFrame, 5, 75, 75);
    } else {
        // For grayscale images, median blur works well and is fast
        cv::medianBlur(frame, denoisedFrame, 5);
    }

    return denoisedFrame;
}

cv::Mat PipelineInternal::applyBrightnessCorrection(const cv::Mat& frame) {
    cv::Mat correctedFrame;

    // Optimize CLAHE algorithm parameters for better performance
    // Reduce the size of the grid and clip limit
    if(frame.channels() == 3) {
        // For color images, apply CLAHE only to luminance channel
        // which is faster than processing all channels
        cv::Mat hsvFrame;
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
        
        std::vector<cv::Mat> hsvChannels(3);
        cv::split(hsvFrame, hsvChannels);
        
        // Use a smaller grid for better performance (4x4 instead of 8x8)
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(4, 4));
        clahe->apply(hsvChannels[2], hsvChannels[2]); // V channel (brightness)
        
        cv::merge(hsvChannels, hsvFrame);
        cv::cvtColor(hsvFrame, correctedFrame, cv::COLOR_HSV2BGR);
    } else {
        // For grayscale, use smaller grid size for faster processing
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(4, 4));
        clahe->apply(frame, correctedFrame);
    }

    return correctedFrame;
}

cv::Mat PipelineInternal::latchDebugInfo(const cv::Mat& frame) {
    cv::Mat debugFrame = frame.clone();

    cv::putText(debugFrame, internalState.timestamp, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    std::string modeText = internalState.nightMode ? "Night Mode" : "Day Mode";
    cv::putText(debugFrame, modeText, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    return debugFrame;
}

void PipelineInternal::updateTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);

    std::stringstream formattedTime;

    formattedTime << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    internalState.timestamp = formattedTime.str();
}

void PipelineInternal::setNightMode(bool enabled) {
    internalState.nightMode = enabled;
    ROS_INFO("Night mode %s", enabled ? "enabled" : "disabled");
}

bool PipelineInternal::isNightModeEnabled() const {
    return internalState.nightMode;
}

cv::Mat PipelineInternal::processFrameRealtime(const cv::Mat& frame) {
    if(frame.empty()) {
        return cv::Mat();
    }

    cv::Mat processedFrame = frame.clone();

    if(internalState.denoiseEnabled) {
        cv::GaussianBlur(processedFrame, processedFrame, cv::Size(3, 3), 0);
    }

    if(internalState.autoBrightnessCorrection) {
        processedFrame.convertTo(processedFrame, -1, 1.1, 5);
    }

    if(internalState.nightMode) {
        processedFrame = convertForNightModeCompatibility(processedFrame);
    }

    if(internalState.debugMode) {
        processedFrame = latchDebugInfo(processedFrame);
    }

    return processedFrame;
}