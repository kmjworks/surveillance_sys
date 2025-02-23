#include <ros/ros.h>
#include <surveillance_system/MotionEvent.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include "opencv2/videoio.hpp"
#include "ros/console.h"
#include "ros/node_handle.h"


class RecordingManagerNode
{
    private:
        ros::NodeHandle nh_;
        ros::Subscriber event_sub_;
        ros::Subscriber image_sub_;
        std::queue<cv::Mat> frame_buffer_;
        cv::VideoWriter video_writer_;
        bool recording_;
        int buffer_size_;
        std::string output_path_;
        ros::Time last_event_time_;

    public:
        RecordingManagerNode() : nh_("~"), recording_(false)
        {
            nh_.param("buffer_size", buffer_size_, 30);
            nh_.param<std::string>("output_path", output_path_, "recordings");

            event_sub_ = nh_.subscribe("/motion_detector_node/motion_events", 10,
                &RecordingManagerNode::eventCallback, this); 
            image_sub_ = nh_.subscribe("/camera_node/image_raw", 1,
                &RecordingManagerNode::imageCallback, this);


            std::string mkdir_cmd = "mkdir -p " + output_path_;
            system(mkdir_cmd.c_str());

            ROS_INFO("Recording Manager initialized.");

        }
    
    private:
        void eventCallback(const surveillance_system::MotionEvent::ConstPtr& event)
        {
            last_event_time_ = ros::Time::now();

            if(!recording_)
            {
                startRecording(event->event_id);
            }
        }

        void imageCallback(const sensor_msgs::ImageConstPtr& msg)
        {
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

            if(recording_)
            {
                video_writer_.write(frame);
                /*
                    Stop frame capture if there is no motion for 3 seconds
                */
                if((ros::Time::now() - last_event_time_).toSec() > 3.0)
                {
                    stopRecording();
                }
            }
            else
            {
                frame_buffer_.push(frame.clone());
                if(frame_buffer_.size() > buffer_size_)
                {
                    frame_buffer_.pop();
                }
            }
        }

        void startRecording(uint32_t event_id)
        {
            std::string filename = output_path_ + "/event_" + 
                             std::to_string(event_id) + ".avi";

            video_writer_.open(filename,
                            cv::VideoWriter::fourcc('M','J','P','G'),
                            30.0, cv::Size(1920, 1080));

            while(!frame_buffer_.empty())
            {
                video_writer_.write(frame_buffer_.front());
                frame_buffer_.pop();
            }

            recording_ = true;
            ROS_INFO("Started recording to %s", filename.c_str());
        }

        void stopRecording()
        {
            video_writer_.release();
            recording_ = false; 
            ROS_INFO("Stopped recording");
        }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "recording_manager_node");
    RecordingManagerNode manager;
    ros::spin();
    return 0;
}