#include "pipelineInitialDetection.hpp"
#include <ros/ros.h>

namespace pipeline {

    PipelineInitialDetection::PipelineInitialDetection(int samplingRate) :
        state{samplingRate, 0, 0.03, 500, 0.8} {}


    PipelineInitialDetection::~PipelineInitialDetection() {}

    bool PipelineInitialDetection::detectedPotentialMotion(const cv::Mat& frame) {
        std::vector<cv::Rect> unused;
        return detectedPotentialMotion(frame, unused);
    }

    bool PipelineInitialDetection::detectedPotentialMotion(const cv::Mat& frame, std::vector<cv::Rect>& motionRects) {
        if(frame.empty()) {
            return false;
        }

        std::lock_guard<std::mutex> lock(frameMtx);
        cv::Mat currentFrame = prepareFrameForDifferencing(frame);

        if(previousFrame.empty()) {
            currentFrame.copyTo(previousFrame);
            return false;
        }

        cv::Mat binaryDifference = getThresholdedDifference(currentFrame, previousFrame);

        findMotionRegions(binaryDifference, motionRects);

        state.frameCounter = (state.frameCounter +1) % state.samplingRate;
        if(state.frameCounter == 0) {
            currentFrame.copyTo(previousFrame);
        }

        bool motionDetected = !motionRects.empty();

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

    cv::Mat PipelineInitialDetection::getThresholdedDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame) {
        cv::Mat diff, thresholdedDifference;

        cv::absdiff(currentFrame, previousFrame, diff);
        cv::threshold(diff, thresholdedDifference, 25, 255, cv::THRESH_BINARY);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(thresholdedDifference,  thresholdedDifference, cv::MORPH_CLOSE, kernel);

        return thresholdedDifference;
    }
    
    void PipelineInitialDetection::findMotionRegions(const cv::Mat& thresholdedDifference, std::vector<cv::Rect>& motionRects) {
        motionRects.clear();

        std::vector<std::vector<cv::Point>> contoursForDetection;
        cv::findContours(thresholdedDifference.clone(), contoursForDetection, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for(const auto& contour : contoursForDetection) {
            double area = cv::contourArea(contour);

            if(area < state.minArea) continue;

            cv::Rect boundingRect = cv::boundingRect(contour);

            double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
            if (aspectRatio > state.aspectRatioThreshold && aspectRatio < 1.0/state.aspectRatioThreshold) 
                continue;

            motionRects.push_back(boundingRect);
        }
    }
}