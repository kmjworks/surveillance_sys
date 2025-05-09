#include <exception>
#include <vector>
#include <NvInfer.h>

#include "motionTrackingNode.hpp"

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;
        switch (severity) {
            case Severity::kINTERNAL_ERROR: ROS_ERROR("[TRT Internal Error] %s", msg); break;
            case Severity::kERROR:          ROS_ERROR("[TRT Error] %s", msg); break;
            case Severity::kWARNING:        ROS_WARN("[TRT Warning] %s", msg); break;
            default:                        ROS_INFO("[TRT Log] %s", msg); break;
        }
    }
};

MotionTrackingNode::MotionTrackingNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh(nh), pnh(pnh) {
    loadParams();
    initComponents();
    initROSIO();
    ROS_INFO("[MotionTrackingNode] Initialization successful.");
}

MotionTrackingNode::~MotionTrackingNode() {
    try {
        asyncPipeline.reset();

        ROS_INFO("[MotionTrackingNode] Shutting down...");
        components.deepSort.reset();

        ROS_INFO("[MotionTrackingNode] Shutdown complete.");
    } catch (const std::exception& e) {
        ROS_ERROR("[MotionTrackingNode] Exception during shutdown: %s", e.what());
    }
}

void MotionTrackingNode::loadParams() {
    pnh.param<std::string>("deepsort_engine_path", configuration.deepSortEnginePath, "models/deepsort.engine");
    pnh.param<int>("deepsort_batch_size", configuration.deepSortConfiguration.batchSize, 32);
    pnh.param<int>("deepsort_feature_dim", configuration.deepSortConfiguration.featureDim, 256);
    pnh.param<int>("gpu_identifier", configuration.deepSortConfiguration.gpuIdentifier, 0);

    pnh.param<int>("detector_input_width", configuration.detectorInputWidth, 640); 
    pnh.param<int>("detector_input_height", configuration.detectorInputHeight, 640);
    pnh.param<int>("person_class_identifier", configuration.personClassIdentifier, 0);
    pnh.param<int>("queue_size", configuration.queueSize, 10);
    pnh.param<bool>("enable_tracking_viz", debugConfiguration.enableTrackingViz, false);

    ROS_INFO("[MotionTrackingNode] Params: Engine='%s', PersonID=%d, Batch=%d, FeatDim=%d, GpuID=%d",
        configuration.deepSortEnginePath.c_str(), configuration.personClassIdentifier,
        configuration.deepSortConfiguration.batchSize, configuration.deepSortConfiguration.featureDim, configuration.deepSortConfiguration.gpuIdentifier);
}

void MotionTrackingNode::initComponents() {
    components.gLogger = std::make_shared<TRTLogger>();

    try {
        components.deepSort = std::make_unique<DeepSort>(
            configuration.deepSortEnginePath,
            configuration.deepSortConfiguration.batchSize,
            configuration.deepSortConfiguration.featureDim,
            configuration.deepSortConfiguration.gpuIdentifier,
            components.gLogger.get()
        );
       
        asyncPipeline = std::make_unique<AsyncProcessingPipeline>(this);

        ROS_INFO("[MotionTrackingNode] DeepSort & Async pipeline initialized successfully.");

    } catch (const std::exception& e) {
        ROS_FATAL("[MotionTrackingNode] Failed to initialize DeepSort instance: %s. Shutting down.", e.what());
        ros::shutdown();
        throw; 
    }
}

void MotionTrackingNode::initROSIO() {
    std::string imageTopic, detectionTopic, trackedTopic;

    pnh.param<std::string>("image_topic", imageTopic, "pipeline/runtime_potentialMotionEvents");
    pnh.param<std::string>("detection_topic", detectionTopic, "yolo/runtime_detections");
    pnh.param<std::string>("tracked_topic", trackedTopic, "motion_tracker/runtime_trackedObjects");    
    
    rosInterface.pub_trackingObjectsViz = nh.advertise<sensor_msgs::Image>("motion_tracker/runtime_trackedObjectsViz", 10);
    rosInterface.pub_trackedObjects = nh.advertise<vision_msgs::Detection2DArray>(trackedTopic, configuration.queueSize);
    rosInterface.sub_rawImages.subscribe(nh, imageTopic, configuration.queueSize);
    rosInterface.sub_detection.subscribe(nh, detectionTopic, configuration.queueSize);
    rosInterface.sync = std::make_unique<message_filters::Synchronizer<motion_tracking::MotionTrackerROSInterface::SyncPolicy>>(motion_tracking::MotionTrackerROSInterface::SyncPolicy(configuration.queueSize), rosInterface.sub_rawImages, rosInterface.sub_detection);
    rosInterface.sync->registerCallback(
        boost::bind(&MotionTrackingNode::synchronizedCb, this, _1, _2)
    );
}

void MotionTrackingNode::synchronizedCb(const sensor_msgs::ImageConstPtr& imageMsg,
                                        const vision_msgs::Detection2DArrayConstPtr& detections) {
    ros::Time startTime = ros::Time::now();
    ROS_DEBUG("[MotionTrackingNode] Sync Callback triggered at %f", startTime.toSec());
    try {
        if (!components.deepSort) {
            ROS_ERROR_THROTTLE(
                5.0, "[MotionTrackingNode] DeepSort instance not ready. Skipping callback.");
            return;
        }

        cv::Mat imageConv;
        try {
            imageConv = cv_bridge::toCvShare(imageMsg, sensor_msgs::image_encodings::BGR8)->image;
        } catch (const cv_bridge::Exception& e) {
            ROS_ERROR("[MotionTrackingNode] cv_bridge exception: %s", e.what());
            return;
        }

        std::vector<DetectionBox> currentDetectionBoxes;
        currentDetectionBoxes.reserve(detections->detections.size());

        const float detectorWidth = static_cast<float>(configuration.detectorInputWidth);
        const float detectorHeight = static_cast<float>(configuration.detectorInputHeight);

        if (detectorWidth <= 0 || detectorHeight <= 0) {
            ROS_ERROR_THROTTLE(
                5.0, "[MotionTrackingNode] Invalid detector input dimensions configured: %dx%d",
                configuration.detectorInputWidth, configuration.detectorInputHeight);
            return;
        }

        const float scaleX = static_cast<float>(imageConv.cols) / detectorWidth;
        const float scaleY = static_cast<float>(imageConv.rows) / detectorHeight;

        for (const auto& det : detections->detections) {
            if (det.results.empty())
                continue;

            int classIdentifier = det.results[0].id;
            if (classIdentifier == configuration.personClassIdentifier) {
                float cx_det = det.bbox.center.x;
                float cy_det = det.bbox.center.y;
                float w_det = det.bbox.size_x;
                float h_det = det.bbox.size_y;

                if (w_det <= 0 || h_det <= 0)
                    continue;

                float cx_orig = cx_det * scaleX;
                float cy_orig = cy_det * scaleY;
                float w_orig = w_det * scaleX;
                float h_orig = h_det * scaleY;

                float x1 = cx_orig - w_orig / 2.0f;
                float y1 = cy_orig - h_orig / 2.0f;
                float x2 = cx_orig + w_orig / 2.0f;
                float y2 = cy_orig + h_orig / 2.0f;

                x1 = std::max(0.0f, std::min(x1, (float)imageConv.cols - 1.0f));
                y1 = std::max(0.0f, std::min(y1, (float)imageConv.rows - 1.0f));
                x2 = std::max(0.0f, std::min(x2, (float)imageConv.cols - 1.0f));
                y2 = std::max(0.0f, std::min(y2, (float)imageConv.rows - 1.0f));

                if (x2 <= x1 || y2 <= y1)
                    continue;

                currentDetectionBoxes.emplace_back(x1, y1, x2, y2, det.results[0].score,
                                                   (float)classIdentifier);
            }
        }

        ROS_DEBUG("[MotionTrackingNode] Prepared %zu person detections for DeepSORT.",
                  currentDetectionBoxes.size());

        if (currentDetectionBoxes.empty()) {
            vision_msgs::Detection2DArray emptyTrackedMsg;
            emptyTrackedMsg.header = imageMsg->header;
            rosInterface.pub_trackedObjects.publish(emptyTrackedMsg);
            ROS_DEBUG("[MotionTrackingNode] No person detections to track this cycle.");
            return;
        }

        cv::Mat imageCopy = imageConv.clone();

        try {
            components.deepSort->sort(imageCopy, currentDetectionBoxes);
            
            asyncPipeline->enqueueWork(imageCopy, currentDetectionBoxes, imageMsg->header.stamp);

        } catch (const std::exception& e) {
            ROS_ERROR_THROTTLE(5.0, "[MotionTrackingNode] Exception during DeepSORT sort: %s",
                               e.what());
            return;
        }

        vision_msgs::Detection2DArray trackingMsg;
        trackingMsg.header = imageMsg->header;

        for (const auto& trackedBox : currentDetectionBoxes) {
            if (trackedBox.trackIdentifier >= 0) {
                vision_msgs::Detection2D trackedDetection;
                trackedDetection.header = imageMsg->header;

                float w = trackedBox.x2 - trackedBox.x1;
                float h = trackedBox.y2 - trackedBox.y1;
                if (w <= 0 || h <= 0 || std::isnan(w) || std::isnan(h))
                    continue;

                trackedDetection.bbox.center.x = trackedBox.x1 + w / 2.0;
                trackedDetection.bbox.center.y = trackedBox.y1 + h / 2.0;
                trackedDetection.bbox.size_x = w;
                trackedDetection.bbox.size_y = h;

                vision_msgs::ObjectHypothesisWithPose hyp;
                hyp.id = static_cast<int>(trackedBox.classIdentifier);
                hyp.score = trackedBox.confidence;
                trackedDetection.results.push_back(hyp);

                trackingMsg.detections.push_back(trackedDetection);
            }
        }

        rosInterface.pub_trackedObjects.publish(trackingMsg);
        ros::Duration elapsed = ros::Time::now() - startTime;
        ROS_INFO("[MotionTrackingNode] Published %zu tracks (Callback time: %.4f s)",
                 trackingMsg.detections.size(), elapsed.toSec());

    } catch (const std::exception& e) {
        ROS_ERROR("[MotionTrackingNode] Unhandled exception in callback: %s", e.what());
    }
}



void MotionTrackingNode::processTrackingVisualization(const cv::Mat& image, const std::vector<DetectionBox>& boundingBoxes, const ros::Time& timestamp) {
    ros::Time startTime = ros::Time::now();
    cv::Mat visualizationImage = image.clone();
    for(const auto& box : boundingBoxes) {
        if (box.trackIdentifier >= 0) {
            cv::Point lt(box.x1, box.y1);
            cv::Point br(box.x2, box.y2);
            cv::rectangle(visualizationImage, lt, br, cv::Scalar(0, 0, 255), 2);
            
            std::string label = cv::format("ID:%d_x:%f_y:%f", static_cast<int>(box.trackIdentifier), (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2);
            cv::putText(visualizationImage, label, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }
    }

    try{
        cv_bridge::CvImage trackingOutputViz;
        trackingOutputViz.encoding = sensor_msgs::image_encodings::BGR8;
        trackingOutputViz.image = visualizationImage;
        trackingOutputViz.header.stamp = timestamp;
        rosInterface.pub_trackingObjectsViz.publish(trackingOutputViz.toImageMsg());
        
        ros::Duration elapsed = ros::Time::now() - startTime;
        ROS_DEBUG("Total visualization processing time: %.4f s", elapsed.toSec());
    } catch (const cv_bridge::Exception& e) {
        ROS_WARN("[MotionTrackingNode] cv_bridge publishing error: %s", e.what());
    }
} 
