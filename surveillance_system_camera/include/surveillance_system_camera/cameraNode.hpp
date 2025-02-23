#ifndef CAMERA_NODE_HPP
#define CAMERA_NODE_HPP

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <surveillance_system_camera/mock_frames.hpp>

class CameraNode {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher image_pub_;
    ros::Timer timer_;
    MockFrames frame_generator_;

    int frame_width_;
    int frame_height_;
    int frame_rate_;

public:
    CameraNode();
    
private:
    void timerCallback(const ros::TimerEvent &);
};

#endif // CAMERA_NODE_HPP