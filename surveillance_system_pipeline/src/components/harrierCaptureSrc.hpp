#pragma once

#include <opencv2/videoio.hpp>
#include <string>
#include <mutex>
#include <atomic>

namespace pipeline {
    
    class HarrierCaptureSrc {
        public:
            HarrierCaptureSrc(const std::string& devicePath);
            ~HarrierCaptureSrc();

            bool isInitialized() const { return initialized; }
            bool captureFrame(cv::Mat& outFrame);
            bool reconnect();

        private:
            std::string devicePath;
            cv::VideoCapture cap;
            std::mutex capMutex;
            std::atomic<bool> initialized{false};

            bool openDevice();
            bool configureDevice();
    };
}