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

    if(frame.channels() == 3) {
        cv::fastNlMeansDenoisingColored(frame, denoisedFrame, 5, 5, 7, 21);
    } else {
        cv::fastNlMeansDenoising(frame, denoisedFrame, 5, 7, 21);
    }

    return denoisedFrame;
}

cv::Mat PipelineInternal::applyBrightnessCorrection(const cv::Mat& frame) {
    cv::Mat correctedFrame;

    if(frame.channels() == 3) {
        cv::Mat labFrame;
        cv::cvtColor(frame, labFrame, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> labChannels(3);
        cv::split(labFrame, labChannels);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(labChannels[0], labChannels[0]);

        cv::merge(labChannels, labFrame);

        cv::cvtColor(labFrame, correctedFrame, cv::COLOR_Lab2BGR);
    } else {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
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