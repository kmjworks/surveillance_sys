#include <cstdint>
#include <string>
#include <thread>
#include <surveillance_system/diagnostic_event.h>
#include "ros/node_handle.h"

#include "ThreadSafeQueue.hpp"

using DetectionEventCount = int64_t;
using MotionEventCount = int64_t;
using Runtime = int64_t; 

namespace diagnostics {
    struct ROSInterface {
        ros::Publisher pub_diagnosticData;

        ros::Subscriber sub_cameraStatus;
        ros::Subscriber sub_pipelineStatus;
        ros::Subscriber sub_captureStatus;
        ros::Subscriber sub_motionDetectorStatus;
        ros::Subscriber sub_motionTrackerStatus;
        
    };

    struct ConfigurationParameters {
        int queueSize;
        int priorityQueueSize;
    };

    struct Components {
        ThreadSafeQueue<surveillance_system::diagnostic_event> diagnosticEventQueue;
        ThreadSafeQueue<surveillance_system::diagnostic_event> diagnosticSeverePriorityQueue;

        std::thread publishingThread;
        std::thread processingThread;
    }; 



}

class DiagnosticsNode {
    public:
        DiagnosticsNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
        DiagnosticsNode(const DiagnosticsNode&) = delete;
        DiagnosticsNode& operator=(const DiagnosticsNode&) = delete;

        ~DiagnosticsNode();

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;
        std::atomic<bool> diagnosticsRunning;
        std::mutex mtx;

        diagnostics::ROSInterface rosInterface;
        diagnostics::ConfigurationParameters configuration;
        surveillance_system::diagnostic_event dataMsg;
        diagnostics::Components components;
        
        void initROSIO();
        void loadParameters();
        void initComponents();
        bool stopThreads();

        void diagnosticsReceiveHandler(const surveillance_system::diagnostic_eventConstPtr& data);
        void publishDiagnosticsData(const surveillance_system::diagnostic_event& diagnosticData);

        void publishingLoop();
        void severityProcessingLoop();

        
};