#include "../../../components/DeepSORT/incl/DeepSORT.hpp" 
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "sensor_msgs/Image.h"
#include "vision_msgs/Detection2DArray.h"


namespace motion_tracking {

    struct DeepSortConfig {
        int batchSize = 128;
        int featureDim = 256;
        int gpuIdentifier = 0;
    };
    
    struct MotionTrackerConfig {
        std::string deepSortEnginePath;
        DeepSortConfig deepSortConfiguration;
        int personClassIdentifier = 0;
        int queueSize = 10;
        int detectorInputWidth = 640;
        int detectorInputHeight = 640;
    };
    
    struct MotionTrackerDebugConfig {
        bool enableTrackingViz = false;
    };
    
    struct MotionTrackerROSInterface {
        ros::Publisher pub_trackedObjects;
        message_filters::Subscriber<sensor_msgs::Image> sub_rawImages;
        message_filters::Subscriber<vision_msgs::Detection2DArray> sub_detection;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, vision_msgs::Detection2DArray> SyncPolicy;
        std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync;
    };

    struct MotionTrackerComponents {
        std::unique_ptr<DeepSort> deepSort;
        std::shared_ptr<nvinfer1::ILogger> gLogger;
    };
    
}
