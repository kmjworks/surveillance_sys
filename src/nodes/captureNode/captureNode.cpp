#include <ros/ros.h>

class CaptureNode {
public:
    std::string captureLogPath;
    std::string captureMetricsPath;

private:
    ros::NodeHandle nh;
};