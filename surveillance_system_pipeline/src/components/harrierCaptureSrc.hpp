#pragma once 

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <string>
#include <atomic>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>


namespace pipeline {

    enum class CodecType {
        NONE = 0,
        H264 = 1,
        H265 = 2,
        MJPEG = 3
    };
    
    struct SrcState {
        int frameRate;
        int retryCount;
        bool nightMode;
        int bitrate; 
        bool useCompression;
        CodecType codecType;
        
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
