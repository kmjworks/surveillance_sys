#include <cstdint>
#include <string>

using DetectionEventCount = int64_t;
using MotionEventCount = int64_t;
using Runtime = int64_t; 

namespace internal {
    struct DiagnosticMetrics {
        MotionEventCount motionEvents;
        DetectionEventCount detectionEvents;
        Runtime uptimeInSeconds;
    };
}


class DiagnosticNode {
    public: 
        DiagnosticNode(ros::NodeHandle nh); 

    private: 
        std::string diagnosticsLogPath;
        internal::DiagnosticMetrics diagnostics;

};