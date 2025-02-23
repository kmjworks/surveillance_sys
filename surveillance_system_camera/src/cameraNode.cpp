#include <surveillance_system_camera/cameraNode.hpp>

CameraNode::CameraNode() : nh_("~"), it_(nh_) {
    nh_.param("frame_width", frame_width_, 1920);
    nh_.param("frame_height", frame_height_, 1080);
    nh_.param("frame_rate", frame_rate_, 30);

    frame_generator_ = MockFrames(frame_width_, frame_height_);

    image_pub_ = it_.advertise("image_raw", 1);

    timer_ = nh_.createTimer(ros::Duration(1.0 / frame_rate_), &CameraNode::timerCallback, this);

    ROS_INFO("Camera simulation initialized");
}

void CameraNode::timerCallback(const ros::TimerEvent &) {
    cv::Mat frame = frame_generator_.generateFrame();

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();

    msg->header.stamp = ros::Time::now();
    image_pub_.publish(msg);
}