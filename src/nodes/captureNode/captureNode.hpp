#pragma once 
#include "image_transport/subscriber.h"
#include "ros/init.h"
#include "ros/node_handle.h"

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
        capture::StorageMetrics metrics;

};  