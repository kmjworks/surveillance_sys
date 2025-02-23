#include "harrierCaptureSrc.hpp"
#include <ros/ros.h>

namespace pipeline {

HarrierCaptureSrc::HarrierCaptureSrc(const std::string& devicePath) : devicePath(devicePath) {
    ROS_INFO("[HarrierCaptureSrc] Initializing with device path: %s", devicePath.c_str());
    
    if (openDevice()) {
        configureDevice();
        initialized = true;
        ROS_INFO("[HarrierCaptureSrc] Successfully initialized video capture");
    } else {
        ROS_ERROR("[HarrierCaptureSrc] Failed to open video capture device: %s", devicePath.c_str());
        initialized = false;
    }
}

HarrierCaptureSrc::~HarrierCaptureSrc() {
    std::lock_guard<std::mutex> lock(capMutex);
    if (cap.isOpened()) {
        cap.release();
    }
    ROS_INFO("[HarrierCaptureSrc] Released video capture device");
}

bool HarrierCaptureSrc::openDevice() {
    std::lock_guard<std::mutex> lock(capMutex);
    
    // Close if already open
    if (cap.isOpened()) {
        cap.release();
    }
    
    // Try to open with string path (might be a URL)
    if (devicePath.find("://") != std::string::npos) {
        return cap.open(devicePath);
    }
    
    // Otherwise try as device index
    try {
        int deviceIndex = std::stoi(devicePath);
        return cap.open(deviceIndex);
    } catch (const std::exception& e) {
        // If string->int conversion fails, try opening as a string path
        return cap.open(devicePath);
    }
}

bool HarrierCaptureSrc::configureDevice() {
    std::lock_guard<std::mutex> lock(capMutex);
    
    if (!cap.isOpened()) {
        return false;
    }
    
    // Configure capture properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    return true;
}

bool HarrierCaptureSrc::captureFrame(cv::Mat& outFrame) {
    std::lock_guard<std::mutex> lock(capMutex);
    
    if (!cap.isOpened()) {
        return false;
    }
    
    if (!cap.read(outFrame)) {
        ROS_WARN_THROTTLE(5.0, "[HarrierCaptureSrc] Failed to read frame");
        return false;
    }
    
    if (outFrame.empty()) {
        ROS_WARN_THROTTLE(5.0, "[HarrierCaptureSrc] Empty frame captured");
        return false;
    }
    
    return true;
}

bool HarrierCaptureSrc::reconnect() {
    ROS_INFO("[HarrierCaptureSrc] Attempting to reconnect to device: %s", devicePath.c_str());
    
    if (openDevice()) {
        configureDevice();
        initialized = true;
        ROS_INFO("[HarrierCaptureSrc] Successfully reconnected to video capture device");
        return true;
    }
    
    ROS_ERROR("[HarrierCaptureSrc] Failed to reconnect to video capture device");
    initialized = false;
    return false;
}

} // namespace pipeline