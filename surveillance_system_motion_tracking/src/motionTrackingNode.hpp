#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>

#include "components/motionTrackingInternal.hpp"
#include "DeepSORT/incl/utilities/dataTypes.hpp"
#include "components/asyncTracking.hpp"

class MotionTrackingNode : public VisualizationHandler {
    public:
        MotionTrackingNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
        ~MotionTrackingNode();

        MotionTrackingNode(const MotionTrackingNode&) = delete;
        MotionTrackingNode& operator=(const MotionTrackingNode&) = delete;

    private:
        ros::NodeHandle nh;
        ros::NodeHandle pnh;

        motion_tracking::MotionTrackerConfig configuration;
        motion_tracking::MotionTrackerDebugConfig debugConfiguration;
        motion_tracking::MotionTrackerROSInterface rosInterface;
        motion_tracking::MotionTrackerComponents components;
        std::unique_ptr<AsyncProcessingPipeline> asyncPipeline;
        
        void synchronizedCb(const sensor_msgs::ImageConstPtr& imageMsg, const vision_msgs::Detection2DArrayConstPtr& detections);
        void loadParams();
        void initROSIO();
        void initComponents();
        cv::Mat preprocessImage(cv::Mat& img, int inputWidth, int inputHeight);

        //void publishTrackingVisualization(cv::Mat& image, std::vector<DetectionBox>& boundingBoxes);
        void processTrackingVisualization(const cv::Mat& image, const std::vector<DetectionBox>& boundingBoxes, const ros::Time& timestamp) override;
};
