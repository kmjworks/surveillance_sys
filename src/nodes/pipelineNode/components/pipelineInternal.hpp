#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace internal {
    struct State {
        bool nightMode; 
        bool debugMode;
        bool autoBrightnessCorrection;
        bool denoiseEnabled;

        std::string timestamp;
    };
}

class PipelineInternal {
    public:
        PipelineInternal(bool nightMode = false, bool debugMode = false);
        ~PipelineInternal();

        bool initialize();

        cv::Mat processFrame(const cv::Mat& frame);
        cv::Mat processFrameRealtime(const cv::Mat& frame);

        void setNightMode(bool enabled);
        bool isNightModeEnabled() const;

    private:
        internal::State internalState;

        cv::Mat convertForNightModeCompatibility(const cv::Mat& frame);
        cv::Mat applyDenoising(const cv::Mat& frame);
        cv::Mat applyBrightnessCorrection(const cv::Mat& frame);
        cv::Mat latchDebugInfo(const cv::Mat& frame);
        
        void updateTimestamp();
};