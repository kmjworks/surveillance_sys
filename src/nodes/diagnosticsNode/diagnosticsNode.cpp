#include "diagnosticsNode.hpp"
 

DiagnosticsNode::DiagnosticsNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh(nh), private_nh(pnh) {
    
    loadParameters();
    initROSIO();


}

DiagnosticsNode::~DiagnosticsNode() {
    while(!stopThreads());
}

void DiagnosticsNode::loadParameters() {
    private_nh.param<int>("queue_size", configuration.queueSize, 100);
    private_nh.param<int>("priority_queue_size", configuration.priorityQueueSize, 10);
    return;
}

void DiagnosticsNode::initROSIO() {
    rosInterface.sub_cameraStatus = nh.subscribe("pipeline/runtime_status", 10, &DiagnosticsNode::diagnosticsReceiveHandler, this);
    return;
}

void DiagnosticsNode::initComponents() {
    ROS_INFO("[DiagnosticsNode] Starting diagnostic node's worker threads.");
    diagnosticsRunning = true;
    components.processingThread = std::thread(&DiagnosticsNode::publishingLoop, this);
    components.publishingThread = std::thread(&DiagnosticsNode::severityProcessingLoop, this); 
    ROS_INFO("[DiagnosticsNode] Diagnostic node's worker threads started.");
}

bool DiagnosticsNode::stopThreads() {
    diagnosticsRunning = false;

    components.diagnosticSeverePriorityQueue.stopWaitingThreads();
    components.diagnosticEventQueue.stopWaitingThreads();

    if(components.processingThread.joinable()) components.processingThread.join();
    if(components.publishingThread.joinable()) components.publishingThread.join();

    ROS_INFO("[DiagnosticsNode] Diagnostic node's worker threads stopped.");

    return true;
}



void DiagnosticsNode::diagnosticsReceiveHandler(const surveillance_system::diagnostic_eventConstPtr& data) {
    surveillance_system::diagnostic_event diagnosticMsg;
    diagnosticMsg.event_description = data->event_description;
    diagnosticMsg.event_severity = data->event_severity;
    diagnosticMsg.origin_node = data->origin_node;
    diagnosticMsg.values = data->values;

    switch(diagnosticMsg.event_severity) {
        case 0:
        case 1:
            components.diagnosticEventQueue.try_push(diagnosticMsg);
            break;
        case 2:
        case 3:
            components.diagnosticSeverePriorityQueue.try_push(diagnosticMsg);
            break;
        default:
            break;
    }
}

void DiagnosticsNode::severityProcessingLoop() {
    ROS_INFO("[Diagnostics Node] Severity processing thread started.");
    ros::Rate loopRate(100);
    while(diagnosticsRunning && ros::ok()) {
        std::optional<surveillance_system::diagnostic_event> checkForSeverity = components.diagnosticSeverePriorityQueue.pop();
        if(not checkForSeverity.has_value()) continue;

        publishDiagnosticsData(checkForSeverity.value());

    }
}

void DiagnosticsNode::publishingLoop() {
    ROS_INFO("[Diagnostics Node] Publishing thread started.");
    ros::Rate loopRate(100);
    while(diagnosticsRunning && ros::ok()) {
        std::optional<surveillance_system::diagnostic_event> checkForData = components.diagnosticEventQueue.pop();
        if(not checkForData.has_value()) continue;
        
        publishDiagnosticsData(checkForData.value());
    }
}

void DiagnosticsNode::publishDiagnosticsData(const surveillance_system::diagnostic_event& diagnosticData) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        try {
            rosInterface.pub_diagnosticData.publish(diagnosticData);
        } catch (const std::exception& e) {
            ROS_ERROR_THROTTLE(5.0, "[DiagnosticsNode] Publishing error caught : %s", e.what());
        }
    }
}
