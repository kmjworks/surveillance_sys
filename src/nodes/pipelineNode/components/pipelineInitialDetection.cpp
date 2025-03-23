#include "pipelineInitialDetection.hpp"
#include <ros/ros.h>

namespace pipeline {

    PipelineInitialDetection::PipelineInitialDetection(int samplingRate) :
        state{samplingRate, 0, 0.03} {}


    PipelineInitialDetection::~PipelineInitialDetection() {}

    bool PipelineInitialDetection::detectedPotentialMotion(const cv::Mat& frame) {
        if(frame.empty()) return false;

        std::lock_guard<std::mutex> lock(frameMtx);
        cv::Mat currentFrame = prepareFrameForDifferencing(frame);

        if(previousFrame.empty()) {
            currentFrame.copyTo(previousFrame);
            return false;
        }

        double diff = calculateFrameDifference(currentFrame, previousFrame);

        state.frameCounter = (state.frameCounter + 1) % state.samplingRate;

        if(state.frameCounter == 0) {
            currentFrame.copyTo(previousFrame);
        }

        bool motionDetected = (diff > state.motionThreshold);

        if(motionDetected) {
            ROS_DEBUG_STREAM("Motion detected between two frames. Difference (%): " << diff);
        }

        return motionDetected;
    }
    
    cv::Mat PipelineInitialDetection::prepareFrameForDifferencing(const cv::Mat& frame) {
        cv::Mat resultingFrame;

        if(frame.channels() > 1) {
            cv::cvtColor(frame, resultingFrame, cv::COLOR_BGR2GRAY);
        } else {
            frame.copyTo(resultingFrame);
        }

        cv::GaussianBlur(resultingFrame, resultingFrame, cv::Size(21, 21), 0);

        return resultingFrame;
    }

    double PipelineInitialDetection::calculateFrameDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame) {
        cv::Mat diff;
        cv::absdiff(currentFrame, previousFrame, diff);

        cv::Mat thresholdedDifference;
        cv::threshold(diff, thresholdedDifference, 25, 255, cv::THRESH_BINARY);

        int nonzeroPx = cv::countNonZero(thresholdedDifference);

        double frameSize = currentFrame.rows * currentFrame.cols;

        return static_cast<double>(nonzeroPx) / frameSize;
    }
}