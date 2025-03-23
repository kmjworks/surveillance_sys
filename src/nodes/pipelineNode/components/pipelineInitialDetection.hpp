#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>

namespace pipeline {

    struct DetectionParameters {
        int samplingRate;
        int frameCounter;
        double motionThreshold;
    };

    class PipelineInitialDetection {
        public:
            PipelineInitialDetection(int samplingRate = 1);
            ~PipelineInitialDetection();

            bool detectedPotentialMotion(const cv::Mat& frame);

        private:
            DetectionParameters state;
            cv::Mat previousFrame;
            std::mutex frameMtx;

            double calculateFrameDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame);
            cv::Mat prepareFrameForDifferencing(const cv::Mat& frame);
    };
}