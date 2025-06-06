#include <image_transport/image_transport.h>
#include <ros/ros.h>

#include <surveillance_system_msgs/storage_status.h>
#include <surveillance_system_msgs/diagnostic_event.h>

#include "ROS/EventLoopTimeKeeper.hpp"
#include "captureNode.hpp"

using namespace std::this_thread;
using namespace std::chrono;

CaptureNode::CaptureNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh(nh), private_nh(pnh) {
    loadParameters();
    initROSIO();
    //initComponents();

    internalInterface = std::make_unique<CaptureNodeInternal>(configuration.saveDirectoryPath, configuration.imageQueueSize, 
        [this](const std::string& description, int severity, const std::vector<std::string>& values) {
            this->diagnosticCallback(description, severity, values);
        }
    );

    internalInterface->startInternal();
    
    
    state.periodicThread = std::thread(&CaptureNode::checkPeriodics, this);
    state.running = true;

    ROS_INFO("[CaptureNode] Node initialized successfully");
}

CaptureNode::~CaptureNode() {
    state.running = false; 
    if(state.periodicThread.joinable()) state.periodicThread.join();
}

void CaptureNode::loadParameters() {
    private_nh.param<std::string>("save_directory", configuration.saveDirectoryPath, "/tmp/surveillance_captures");
    private_nh.param<int>("image_queue_size", configuration.imageQueueSize, 100);

    ROS_INFO_STREAM("[CaptureNode] Save directory set to: " << configuration.saveDirectoryPath);
    ROS_INFO_STREAM("[CaptureNode] Image queue size set to: " << configuration.imageQueueSize);
}

void CaptureNode::initROSIO() {
    image_transport::ImageTransport it(nh);

    rosInterface.pub_StorageData = nh.advertise<surveillance_system_msgs::storage_status>("capture/runtime_storageMetrics", 10);
    rosInterface.pub_StorageData = nh.advertise<surveillance_system_msgs::diagnostic_event>("capture/runtime_diagnostics", 10);
    
    rosInterface.sub_trackedMotionVisualized = it.subscribe("motion_tracker/runtime_trackedObjects", 5,&CaptureNode::trackingImageCallback, this);
    rosInterface.sub_detectedMotionVisualized = it.subscribe("yolo/runtime_detectionVisualizationDebug", 5,&CaptureNode::detectionImageCallback, this);
}

void CaptureNode::initComponents() {
    internalInterface = std::make_unique<CaptureNodeInternal>(configuration.saveDirectoryPath, configuration.imageQueueSize, 
        [this](const std::string& description, int severity, const std::vector<std::string>& values) {
            this->diagnosticCallback(description, severity, values);
        }
    );

    internalInterface->startInternal();
}

void CaptureNode::detectionImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    internalInterface->processImage(msg, "detection");
}

void CaptureNode::trackingImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    internalInterface->processImage(msg, "detection-tracking");
}

void CaptureNode::diagnosticCallback(const std::string& description, int severity, const std::vector<std::string>& values) {
    surveillance_system_msgs::diagnostic_event event; 
    event.event_description = description;
    event.event_severity = severity;
    event.origin_node = "capture_node";

    event.values.clear(); size_t i = 0;

    do {
        surveillance_system_msgs::key_value kv;
        kv.key = "value_" + std::to_string(i);
        kv.value = values[i];
        event.values.emplace_back(kv);
        ++i;
    } while(i < values.size());
    
    rosInterface.pub_runtimeStatus.publish(event);
}

void CaptureNode::checkPeriodics() {
    EventLoopTimeKeeper metricsUpdate(0.1);

    while(state.running && ros::ok()) {
        if(metricsUpdate.shouldRun()) {
            updateAndPublishStorageMetrics();
        }
        sleep_for(milliseconds(50));
    }
}

void CaptureNode::updateAndPublishStorageMetrics() {
    auto metrics = internalInterface->calculateStorageCapability();

    surveillance_system_msgs::storage_status msg;
    msg.storage_total_bytes = metrics.totalStorage;
    msg.storage_available_bytes = metrics.availableStorage;
    msg.storage_usage_percentage = metrics.usedPercentage;

    rosInterface.pub_StorageData.publish(msg);
}