#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <thread>

#include "opencv2/core/mat.hpp"
#include "sensor_msgs/Image.h"

#include "ThreadSafeQueue.hpp"

namespace capture_internal {
    struct StorageMetrics {
        uint64_t totalStorage;
        uint64_t availableStorage;
        double usedPercentage;
    };

    struct ImageData {
        cv::Mat image;
        std::string source;
        std::chrono::system_clock::time_point timestamp;
        int identifier;
    };

    struct State {
        std::atomic<bool> running{false};
        std::thread processingThread;
        std::string saveDirectoryPath;
    };

}

class CaptureNodeInternal {
    public:
        CaptureNodeInternal(const std::string& saveDirectoryPath,int imageQueueSize,std::function<void(const std::string&, int, const std::vector<std::string>&)> diagnosticCallback);
        CaptureNodeInternal(const CaptureNodeInternal&) = delete;
        CaptureNodeInternal& operator=(const CaptureNodeInternal&) = delete;

        ~CaptureNodeInternal();

        void startInternal();
        void stopInternal();
        void processImage(const sensor_msgs::ImageConstPtr& msg, const std::string& imgSource);

        capture_internal::StorageMetrics calculateStorageCapability();
    private:
        capture_internal::State internalProcessingState;
        std::function<void(const std::string&, int, const std::vector<std::string>&)> diagnosticsHandler;
        ThreadSafeQueue<capture_internal::ImageData> imageQueue;

        void convertTimestampStr(char* buffer, std::tm* timestamp);
        void imageCallback(const sensor_msgs::ImageConstPtr& msg, const std::string& source);
        void saveImage(const capture_internal::ImageData& imageData);
        void processImageQueue();
};