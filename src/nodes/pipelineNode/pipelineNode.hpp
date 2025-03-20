#include <queue>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/videoio.hpp"

#include <ros/ros.h>
#include "ros/console.h"
#include "ros/node_handle.h"
#include "ros/subscriber.h"

#include <sensor_msgs/Image.h>
#include <surveillance_system/motion_event.h>

namespace internal {
struct ROSInterface {
    ros::Subscriber sub_motionEvents;
    ros::Subscriber sub_rawImageData;
};
}  // namespace internal

class PipelineNode {
public:
    PipelineNode(ros::NodeHandle nh);
    cv::VideoWriter videoWriterForSimulation;
    internal::ROSInterface ROSInternalInterface;

private:
    ros::NodeHandle nh;
    bool videoWriterStatus;
    int videoWriterBufferSize;

    std::string outputPath;
    ros::Time lastMotionEventRegistered;
    std::queue<cv::Mat> frameBuffer;

    void videoWriterRecording(bool start, uint32_t motionEventID);
    void cb_imageData(const sensor_msgs::ImageConstPtr& msg);
    void cb_motionEvent(const surveillance_system::motion_event::ConstPtr& msg);
};