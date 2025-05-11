#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection2D.h>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <thread>

#include <gstreamer-1.0/gst/gstpad.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "nvbufsurface.h"
#include "nvds_obj_encode.h"

#include "ThreadSafeQueue/ThreadSafeQueue.hpp"

extern "C" {
    #include "deepstream_app.h"
    #include "deepstream_config_file_parser.h"
}


namespace detector_internal {
    
    struct ROSInterface {
        ros::Publisher pub_imageSrc;
        ros::Publisher pub_detectedMotion;
        ros::Publisher pub_vizDebug;
    };

    struct FrameData {
        std::string frameIdentifier;
        std::pair<int, int> inputWidthAndHeight;
        std::pair<int, int> detectorWidthAndHeight;
    };

    struct PublishedFrameData {
        sensor_msgs::ImagePtr srcImage;
        cv::Mat vizImg;
        vision_msgs::Detection2DArray detections;
        std::string frameIdentifier;

        bool hasDetections;
        ros::Time timestamp;
    };

    struct PGIEData {
        NvDsBatchMeta* batchMeta;
        GstBuffer* buffer;
        std::vector<NvDsObjectMeta*> objects;
        ros::Time timestamp;
    };

    struct PublishingQueue {
        ThreadSafeQueue<PublishedFrameData> publishQueue;
        std::thread publishingThread;
        std::atomic<bool> publisherRunning{false};
    };

    struct PGIEQueue {
        ThreadSafeQueue<PGIEData> pgieQueue;
        std::thread pgieProcessingThread;
        std::atomic<bool> pgieQRunning{true};
    };

    struct Components {
        PublishingQueue pubQueue;
        PGIEQueue pgieQueue;
    };

    struct RuntimeConfiguration {
        std::string configurationFile;
        FrameData frameConfiguration;
        float confidenceThreshold;
        bool enableViz;
    };

    struct DeepStreamInterface {
        AppCtx *appCtx = nullptr;
        NvDsConfig *parsedConfig = nullptr;
        GMainLoop *loop = nullptr;
        gulong osd_probeIdentifier = 0;
        gulong pgie_probeIdentifier = 0;
    };

    struct State {
        std::thread pipelineThread;
        bool running = false;
    };

}


class DetectorNode {
    public: 
        DetectorNode(ros::NodeHandle& nh, ros::NodeHandle &pnh);
        ~DetectorNode();

        bool initPipeline();
        void stopPipeline();

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        detector_internal::ROSInterface rosInterface;
        detector_internal::DeepStreamInterface dsInterface;
        detector_internal::State state;
        detector_internal::RuntimeConfiguration runtimeConfiguration;
        detector_internal::Components components;
        
        bool initROSIO(ros::NodeHandle& nodeHandle);
        void processBuffer(GstBuffer *buf);
        void publishingHandler();
        void pgiePreprocess();

        static GstPadProbeReturn osdPadBufferProbe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
        static GstPadProbeReturn pgieSrcPadBufferProbe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
        static gboolean busCb(GstBus *bus, GstMessage *message, gpointer loop_to_quit_ptr);
        vision_msgs::Detection2DArray getDetectionArray(ros::Time timestamp, const std::string& frameId);
        
};