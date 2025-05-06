#pragma once

#include <memory>
#include <thread>

#include "ros/node_handle.h"
#include "image_transport/subscriber.h"

#include "ROS/EventLoopTimeKeeper.hpp"
#include "components/captureNodeInternal.hpp"

namespace capture {
    struct ROSInterface {
        ros::Publisher pub_StorageData;
        ros::Publisher pub_runtimeStatus;
        image_transport::Subscriber sub_detectedMotionVisualized;
        image_transport::Subscriber sub_trackedMotionVisualized;
    };
    
    struct StorageMetrics {
        uint64_t totalStorage;
        uint64_t availableStorage;
        double usedPercentage;
    };

    struct ConfigurationParameters {
        std::string saveDirectoryPath;
        int imageQueueSize;
    };

    struct State {
        std::atomic<bool> running{false};
        std::thread periodicThread;
    };
}

class CaptureNode {
    public:
        CaptureNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
        CaptureNode(const CaptureNode&) = delete;
        CaptureNode& operator=(const CaptureNode&) = delete;

        ~CaptureNode();

    private: 
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;
        capture::ROSInterface rosInterface;
        capture::ConfigurationParameters configuration;
        capture::State state; 

        std::unique_ptr<CaptureNodeInternal> internalInterface;
        
        void initROSIO();
        void loadParameters();
        void initComponents();
        
        void detectionImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void trackingImageCallback(const sensor_msgs::ImageConstPtr& msg);
        void diagnosticCallback(const std::string& description, int severity, const std::vector<std::string>& values);
        
        void checkPeriodics();
        void updateAndPublishStorageMetrics();

};  