#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>

namespace pipeline {

    struct DetectionParameters {
        int samplingRate;
        int frameCounter;
        double motionThreshold;
        double minArea;
        double aspectRatioThreshold;
    };

    class PipelineInitialDetection {
        public:
            PipelineInitialDetection(int samplingRate = 1);
            ~PipelineInitialDetection();

            bool detectedPotentialMotion(const cv::Mat& frame);
            bool detectedPotentialMotion(const cv::Mat& frame, std::vector<cv::Rect>& motionRects);

        private:
            DetectionParameters state;
            cv::Mat previousFrame;
            std::mutex frameMtx;

            double calculateFrameDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame);
            cv::Mat getThresholdedDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame);
            void findMotionRegions(const cv::Mat& thresholdedDifference, std::vector<cv::Rect>& motionRects);
            cv::Mat prepareFrameForDifferencing(const cv::Mat& frame);
            
    };
}