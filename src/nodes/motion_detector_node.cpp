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
        MotionDetectorNode() : nh_("~"), it_(nh_), first_frame_(true), event_counter_(0)
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
                cv::Mat current_frame = cv_bridge::toCvShare(msg, "bgr8")->image;
                cv::Mat debug_frame = current_frame.clone();
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
                cv::threshold(frame_diff, motion_mask, detection_threshold_, 
                    255, cv::THRESH_BINARY);
                
                /*
                    Contours of motion regions
                */
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(motion_mask, contours, cv::RETR_EXTERNAL,
                                cv::CHAIN_APPROX_SIMPLE);

                for(const auto& contour : contours)
                {
                    float area = cv::contourArea(contour);
                    if(area > 100)
                    {
                        cv::Rect bbox = cv::boundingRect(contour);
                        cv::Moments moments = cv::moments(contour);
                        
                        cv::rectangle(debug_frame, bbox, cv::Scalar(0, 255, 0), 2);

                        surveillance_system::MotionEvent event; 

                        event.header = msg->header;
                        event.event_id = ++event_counter_;
                        event.location.x = moments.m10 / moments.m00; 
                        event.location.y = moments.m01 / moments.m00;
                        event.area = area; 
                        event.confidence = area / (current_frame.rows * current_frame.cols);

                        event_pub_.publish(event);
                    }
                }

                sensor_msgs::ImagePtr debug_msg =
                    cv_bridge::CvImage(msg->header, "bgr8", debug_frame).toImageMsg();

                debug_pub_.publish(debug_msg);

                previous_frame_ = current_gray.clone();
            } 
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("ROS CV Bridge failed: %s", 
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