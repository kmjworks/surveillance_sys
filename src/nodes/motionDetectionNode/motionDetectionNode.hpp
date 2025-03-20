#ifndef MOTION_DETECTION_NODE_HPP
#define MOTION_DETECTION_NODE_HPP

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <surveillance_system/motion_event.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace surveillance_system {

/**
 * @brief Template class for motion detection algorithms
 */
template <typename FrameType = cv::Mat>
class MotionDetectionStrategy {
public:
    virtual ~MotionDetectionStrategy() = default;
    virtual cv::Mat detectMotion(const FrameType& current_frame, const FrameType& previous_frame,
                                 double threshold) const = 0;
};

/**
 * @brief Pretty basic motion detection by frame differencing
 */
template <typename FrameType = cv::Mat>
class FrameDifferenceStrategy : public MotionDetectionStrategy<FrameType> {
public:
    cv::Mat detectMotion(const FrameType& current_frame, const FrameType& previous_frame,
                         double threshold) const override {
        cv::Mat frame_diff;
        cv::absdiff(current_frame, previous_frame, frame_diff);

        cv::Mat motion_mask;
        cv::threshold(frame_diff, motion_mask, threshold, 255, cv::THRESH_BINARY);

        return motion_mask;
    }
};

/**
 * Class for processing motion events
 *
 * This class handles the detection of motion in video frames and
 * publishes motion events when significant motion is detected.
 */
template <typename MotionStrategy = FrameDifferenceStrategy<cv::Mat>>
class MotionDetectorNode {
public:
    /*
     * Construct a new Motion Detector Node
     */
    MotionDetectorNode();

    /*
     * Set the motion detection threshold
     */
    void setDetectionThreshold(double threshold);

    /**
     * Get the current motion detection thresholde
     */
    double getDetectionThreshold() const;

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
    MotionStrategy motion_strategy_;

    /**
     * @brief Callback for processing incoming image frames
     *
     * @param msg Image message from ROS
     */
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    /**
     * @brief Process motion contours and publish events
     *
     * @param contours Vector of detected motion contours
     * @param debug_frame Frame to draw debug information on
     * @param msg_header Header from the original image message
     */
    void processMotionContours(const std::vector<std::vector<cv::Point>>& contours,
                               cv::Mat& debug_frame, const std_msgs::Header& msg_header);
};

// Forward declaration of template implementation
template <typename MotionStrategy>
MotionDetectorNode<MotionStrategy>::MotionDetectorNode()
    : nh_("~"), it_(nh_), first_frame_(true), event_counter_(0) {
    nh_.param("detection_threshold", detection_threshold_, 30.0);

    image_sub_ = nh_.subscribe("/camera_node/image_raw", 1,
                               &MotionDetectorNode<MotionStrategy>::imageCallback, this);

    event_pub_ = nh_.advertise<surveillance_system::motion_event>("motion_events", 10);

    debug_pub_ = it_.advertise("debug_image", 1);

    ROS_INFO("Motion detector initialized.");
}

template <typename MotionStrategy>
void MotionDetectorNode<MotionStrategy>::setDetectionThreshold(double threshold) {
    detection_threshold_ = threshold;
}

template <typename MotionStrategy>
double MotionDetectorNode<MotionStrategy>::getDetectionThreshold() const {
    return detection_threshold_;
}

template <typename MotionStrategy>
void MotionDetectorNode<MotionStrategy>::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
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

        // Use the motion detection strategy to detect motion
        cv::Mat motion_mask =
            motion_strategy_.detectMotion(current_gray, previous_frame_, detection_threshold_);

        // Find contours of motion regions
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motion_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Process the detected motion contours
        processMotionContours(contours, debug_frame, msg->header);

        // Publish debug image
        sensor_msgs::ImagePtr debug_msg =
            cv_bridge::CvImage(msg->header, "bgr8", debug_frame).toImageMsg();
        debug_pub_.publish(debug_msg);

        // Update previous frame
        previous_frame_ = current_gray.clone();
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("ROS CV Bridge failed: %s", e.what());
    }
}

template <typename MotionStrategy>
void MotionDetectorNode<MotionStrategy>::processMotionContours(
    const std::vector<std::vector<cv::Point>>& contours, cv::Mat& debug_frame,
    const std_msgs::Header& msg_header) {
    for (const auto& contour : contours) {
        float area = cv::contourArea(contour);
        if (area > 100) {  // Filter small motion areas
            cv::Rect bbox = cv::boundingRect(contour);
            cv::Moments moments = cv::moments(contour);

            // Draw bounding box on debug frame
            cv::rectangle(debug_frame, bbox, cv::Scalar(0, 255, 0), 2);

            // Create and publish motion event
            surveillance_system::motion_event event;
            event.header = msg_header;
            event.event_id = ++event_counter_;
            event.location.x = moments.m10 / moments.m00;
            event.location.y = moments.m01 / moments.m00;
            event.area = area;
            event.confidence = area / (debug_frame.rows * debug_frame.cols);

            event_pub_.publish(event);
        }
    }
}

}  // namespace surveillance_system

#endif  // MOTION_DETECTION_NODE_HPP
