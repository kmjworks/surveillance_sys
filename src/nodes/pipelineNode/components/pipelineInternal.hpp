#pragma once 

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>

namespace pipeline {

    struct State {
        int frameRate;
        bool nightMode;
    };

    class PipelineInternal {
        public:
            PipelineInternal(int frameRate, bool nightMode);
            ~PipelineInternal();

            cv::Mat processFrame(const cv::Mat& frame);

        private:
            State state;

            cv::Mat convertFormat(const cv::Mat& frame);
            cv::Mat preprocessFrame(const cv::Mat& frame);

    };
}