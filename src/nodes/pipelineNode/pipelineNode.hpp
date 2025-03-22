#include <queue>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/videoio.hpp"

#include <ros/ros.h>
#include "ros/console.h"
#include "ros/node_handle.h"
#include "ros/subscriber.h"

#include "components/pipeline.hpp"

#include <sensor_msgs/Image.h>
#include <surveillance_system/motion_event.h>

namespace internal {
struct ROSInterface {
    ros::Subscriber sub_motionEvents;
    ros::Subscriber sub_rawImageData;
    ros::Publisher pub_processedFrames;
    ros::Publisher pub_motionEvents;
};

struct MotionDetectionThresholds {
    double motionThreshold; 
    int minimalMotionArea;
    bool motionDetected;
};

struct MotionDetectionInternals {
    bool showDebugFrames; 
    bool videoWriterStatus;
    std::string outputPath;

    std::queue<cv::Mat> frameBuffer;
    cv::Ptr<cv::BackgroundSubtractorMOG2> backgroundSubtractor;
};

}  // namespace internal

class PipelineNode : public PipelineBase {
public:
    PipelineNode(ros::NodeHandle& nh);
    cv::VideoWriter videoWriterForSimulation;

private:
    ros::NodeHandle nh;
    internal::ROSInterface rosInterface;
    internal::MotionDetectionThresholds detectionThresholds;
    internal::MotionDetectionInternals detectionInternalState;
    ros::Time lastMotionEventRegistered;

    void videoWriterRecording(bool start, uint32_t motionEventID);
    void cb_imageData(const sensor_msgs::ImageConstPtr& msg);
    void cb_motionEvent(const surveillance_system::motion_event::ConstPtr& msg);
};