#pragma once

#include <opencv2/opencv.hpp>
#include <queue>

namespace pipeline {

    struct DetectionParameters {
        int minAreaPx;        // Minimum area in pixels for a motion region to be considered
        float downScale;      // Downscale factor for processing (0.5 = half resolution)
        int historyLength;    // How many frames to keep in history for background subtraction
    };

    class PipelineInitialDetectionLite {
        public:
            PipelineInitialDetectionLite(int minAreaPixels, float downscaleFactor, int history);
            ~PipelineInitialDetectionLite();

            bool detectMotion(const cv::Mat& frame, cv::Mat& outputMotionMask, cv::Mat& annotatedFrame);
            
        private:
            cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor;
            cv::Mat fgMask;
            
            DetectionParameters params;
            bool initialized;
            std::queue<cv::Mat> frameHistory;

            void preprocessForDetection(const cv::Mat& input, cv::Mat& output);
            void postprocessMask(cv::Mat& mask);
    };
}