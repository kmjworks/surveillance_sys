#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "../simulation/mock_frames.hpp"

class CameraNode
{
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
        CameraNode() : nh_("~"), it_(nh_) 
        {
            nh_.param("frame_width", frame_width_, 1920); 
            nh_.param("frame_height", frame_height_, 1080);
            nh_.param("frame_rate", frame_rate_, 30);
            

            frame_generator_ = MockFrames(frame_width_, frame_height_);

            image_pub_ = it_.advertise("image_raw", 1);

            timer_ = nh_.createTimer(
                ros::Duration(1.0/frame_rate_),
                &CameraNode::timerCallback,
                this
            );

            ROS_INFO("Camera simulation initialized");
        }

    private:
        void timerCallback(const ros::TimerEvent&)
        {
            cv::Mat frame = frame_generator_.generateFrame();

            sensor_msgs::ImagePtr msg = 
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame)
                .toImageMsg();

            msg->header.stamp = ros::Time::now();
            image_pub_.publish(msg);
        }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "camera_node");
    CameraNode node; 
    ros::spin();
    return 0; 
}
