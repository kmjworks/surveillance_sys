#include "captureNodeInternal.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <boost/format.hpp>
#include <utility>
#include <ros/ros.h>

using namespace std::chrono;

CaptureNodeInternal::CaptureNodeInternal(const std::string& saveDirectoryPath, int imageQueueSize, std::function<void(const std::string&, int, const std::vector<std::string>&)> diagnosticCallback) 
: diagnosticsHandler(std::move(diagnosticCallback)) {
    internalProcessingState.saveDirectoryPath = saveDirectoryPath;
    imageQueue.initialize(imageQueueSize);

    boost::filesystem::path directory(internalProcessingState.saveDirectoryPath);
    if(not boost::filesystem::exists(directory)) boost::filesystem::create_directories(directory);
    std::vector<std::string> values; 
    values.push_back(saveDirectoryPath);
    diagnosticsHandler("Capture-save directory created", 0, values);
}

CaptureNodeInternal::~CaptureNodeInternal() {
    stopInternal();
}

void CaptureNodeInternal::startInternal() {
    if (not internalProcessingState.running) {
        internalProcessingState.running = true;
        internalProcessingState.processingThread= std::thread(&CaptureNodeInternal::processImageQueue, this);
    }
}

void CaptureNodeInternal::stopInternal() {
    if (internalProcessingState.running) {
        internalProcessingState.running = false;
        if(internalProcessingState.processingThread.joinable()) internalProcessingState.processingThread.join();
    }
}

void CaptureNodeInternal::convertTimestampStr(char* buffer, std::tm* timestamp) {
    std::strftime(buffer, 100, "%Y%m%d_%H%M%S", timestamp);
}

void CaptureNodeInternal::processImage(const sensor_msgs::ImageConstPtr& msg, const std::string& source) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        
        capture_internal::ImageData imageData;
        imageData.image = cv_ptr->image;
        imageData.source = source;
        imageData.timestamp = std::chrono::system_clock::now();
        imageData.identifier = static_cast<int>(msg->header.seq);
        
        if (imageQueue.try_push(imageData)) {
            diagnosticsHandler("Image queue full, dropping frame", 1, {});
        }
    } catch (cv_bridge::Exception& e) {
        std::vector<std::string> values;
        values.emplace_back(e.what());
        diagnosticsHandler("Failed to convert image", 2, values);
    }
}

void CaptureNodeInternal::saveImage(const capture_internal::ImageData& imageData) {
    auto tNow = system_clock::to_time_t(imageData.timestamp);
    std::tm* tmNow = std::localtime(&tNow);
    char timestampStr[100];
    convertTimestampStr(timestampStr, tmNow);

    auto msNow = duration_cast<milliseconds>(imageData.timestamp.time_since_epoch()) % 1000;

    std::string targetFilename = (boost::format("%s_%s_%03d_%d.jpg") % imageData.source % timestampStr % msNow.count() % imageData.identifier).str();

    boost::filesystem::path savePath = boost::filesystem::path(internalProcessingState.saveDirectoryPath) / targetFilename;
    std::vector<std::string> values;

    try {
        cv::imwrite(savePath.string(), imageData.image);    
        values.emplace_back(savePath.string());
        diagnosticsHandler("Image saved successfully", 0, values);

    } catch (const cv::Exception& e) {
        values.emplace_back(e.what());
        diagnosticsHandler("Failed to save image", 2, values);
    }
}

void CaptureNodeInternal::processImageQueue() {
    while(internalProcessingState.running) {
        auto imageData = imageQueue.pop();
        
        if(imageData) {
            saveImage(*imageData);
        } else {
            std::this_thread::sleep_for(milliseconds(10));
        }
    }
}

capture_internal::StorageMetrics CaptureNodeInternal::calculateStorageCapability() {
    capture_internal::StorageMetrics metrics;

    try {
        boost::filesystem::path path(internalProcessingState.saveDirectoryPath); 
        boost::filesystem::space_info spaceInfo = boost::filesystem::space(path);

        metrics.totalStorage = spaceInfo.capacity;
        metrics.availableStorage = spaceInfo.available;
        metrics.usedPercentage = 100.0 * (static_cast<double>(spaceInfo.capacity - spaceInfo.available) / static_cast<double>(spaceInfo.capacity));
    } catch (const boost::filesystem::filesystem_error& e) {
        std::vector<std::string> values;
        values.push_back(e.what());
        diagnosticsHandler("Failed to get storage metrics.", 2, values);

        metrics = {0, 0, 0.0};

    }

    return metrics;
}





