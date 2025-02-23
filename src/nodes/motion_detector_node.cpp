#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <surveillance_system/MotionEvent.h>
#include <image_transport/image_transport.h>
#include <vector>

class MotionDetectorNode
{
    private:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        ros::Subscriber image_sub_;
        ros::Publisher event_pub_;
        image_transport::Publisher debug_pub_;
        cv::Mat previous_frame_;
        bool first_frame_;
        double detection_threshold_;
        uint32_t event_counter_;

    public:
        MotionDetectorNode() : nh_("~"), first_frame_(true)
        {
            nh_.param("detection_threshold", detection_threshold_, 30.0);

            image_sub_ = nh_.subscribe("/camera_node/image_raw", 1,
            &MotionDetectorNode::imageCallback, this);

            event_pub_ = nh_.advertise<surveillance_system::MotionEvent>
                ("motion_events", 10);
            
            debug_pub_ = it_.advertise("debug_image", 1);

            ROS_INFO("Motion detector initalized.");
        }

    private:
        void imageCallback(const sensor_msgs::ImageConstPtr& msg)
        {
            try {
                cv::Mat current_frame = cv::bridge::toCvShare(msg, "bgr8")->image;
                cv::Mat debug_frame == curret_frame.clone();
                cv::Mat current_gray;


                // grayscale conversion
                cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);

                if (first_frame_) {
                    previous_frame_ = current_gray.clone();
                    first_frame_ = false;
                    return;
                }

                cv::Mat frame_diff;
                cv::absdiff(current_gray, previous_frame_, frame_diff);

                cv::Mat motion_mask;
                cv::threshold(frame_diff, motion_mask, detection_threshold_, 255, cv::THRESH_BINARY);
                
                /*
                    Contours of motion regions
                */
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(motion_mask, contours, cv::RETR_EXTERNAL,
                                cv::CHAIN_APPROX_SIMPLE);

                for(const auto& contour :: contours)
                {
                    if(cv::contourArea(contour) > 100)
                    {
                        cv::Rect bbox = cv::boundingRect(contour);
                        cv::Moments moments = cv::moments(contour);
                        
                        cv::rectangle(debug_frame, bbox, cv::Scalar(0, 255, 0), 2);

                        surveillance_system::MotionEvent event; 

                        event.header = msg->header;
                        event.event_id = ++event_counter_;
                    }
                }

                previous_frame_ = current_gray.clone();
            } 
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("ROS Image message conversion to OpenCV format failed: %s", 
                    e.what());
            }
        }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "motion_detector_node");
    MotionDetectorNode detector;
    ros::spin();
    return 0;
}