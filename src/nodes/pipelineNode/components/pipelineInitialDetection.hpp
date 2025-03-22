#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <mutex>

namespace internal {
    struct MotionParams {
        double threshold;
        double minArea;
    };
}

class PipelineInitialDetection {
    public:
        explicit PipelineInitialDetection(int publishRate = 1);
        ~PipelineInitialDetection();
        bool initialize();

        bool detectMotion(const cv::Mat& currentFrame);

        void setThresholdForDetection(double threshold);
        double getThreshold() const;

    private:
        int publishingRate;
        internal::MotionParams motionParameters;

        cv::Mat previousFrame;
        bool isFirstFrame;

        std::deque<bool> potentialMotionEventHistory;
        std::mutex motionDetectionMtx;

        bool detectFrameDiff(const cv::Mat& currentFrame);
        cv::Mat preprocessFrame(const cv::Mat& frame);
};