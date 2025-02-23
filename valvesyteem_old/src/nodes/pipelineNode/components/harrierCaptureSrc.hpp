#pragma once 

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <string>
#include <atomic>


namespace pipeline {
    
    struct SrcState {
        int frameRate;
        bool nightMode;
        int retryCount;
    };

    struct Elements {
        GstElement* pipeline;
        GstElement* appsink;
    };

    class HarrierCaptureSrc {
        public:
            HarrierCaptureSrc(const std::string& devicePath, int frameRate, bool nightMode);
            ~HarrierCaptureSrc();

            bool initializeRawSrcForCapture();
            bool captureFrameFromSrc(cv::Mat& frame);
            void releasePipeline();

        private:
            std::string devicePath;
            std::atomic<bool> isPipelineInitialized;
            SrcState harrierState;
            Elements pipelineElements;

            bool initPipeline();
            bool initCameraSrc();

            void cleanup();
            bool retryOnFail();
            bool isNightMode(GstSample *sample);

    };
}