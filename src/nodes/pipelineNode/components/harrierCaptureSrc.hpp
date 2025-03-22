#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <atomic>

namespace internal {

    struct FrameParams {
        int frameRate;
        int height;
        int width;
        std::string format;
    };

}


class HarrierCaptureSrc {
    public:
        HarrierCaptureSrc(const std::string& devicePath, int frameRate);
        ~HarrierCaptureSrc();

        bool initialize();

        bool captureFrame(cv::Mat& frame);

        std::string getFrameFormat() const;

        bool isNightModeDetected();

    private:
        std::string devicePath;
        internal::FrameParams frameParams;

        GstElement* pipeline;
        GstElement* appsink;

        std::atomic<bool> isInitialized;
        std::atomic<bool> isNightMode;

        bool buildPipeline();
        void releasePipelineResources();
        
        static GstFlowReturn cb_newFrameSample(GstElement* sink, gpointer data);
        cv::Mat gstSampleToCvMat(GstSample* sample);
};