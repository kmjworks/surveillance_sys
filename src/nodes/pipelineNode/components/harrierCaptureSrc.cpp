#include "harrierCaptureSrc.hpp"
#include "gst/app/gstappsink.h"
#include "ros/console.h"

namespace pipeline {
    HarrierCaptureSrc::HarrierCaptureSrc(const std::string& devicePath, int frameRate, bool nightMode) 
        : devicePath(devicePath), isPipelineInitialized(false),
          harrierState{frameRate, nightMode, 0}, pipelineElements{nullptr, nullptr} {

            gst_init(nullptr, nullptr);
    }

    HarrierCaptureSrc::~HarrierCaptureSrc() {
        releasePipeline();
    }


    bool HarrierCaptureSrc::initializeRawSrcForCapture() {
        ROS_INFO("Initializing Harrier camera for capture.");

        if(initCameraSrc()) {
            isPipelineInitialized = true;
            ROS_INFO("Camera initialized");
            return true;
        }

        ROS_WARN("Camera initialization failed.");
        ros::Duration(5.0).sleep();

        ROS_ERROR("Failed to initialize camera (raw) after 3 attempts. Exiting..");
        return false;
    }

    bool HarrierCaptureSrc::initPipeline() {
        std::string pipelineStringBuildUp = 
        "v4l2src device=" + devicePath + " ! "
        "video/x-raw, format=YUY2, width=1920, height=1080, framerate=" + 
        std::to_string(harrierState.frameRate) + "/1 ! "
        "videoconvert ! video/x-raw, format=" + 
        (harrierState.nightMode ? "GRAY8" : "BGR") + " ! "
        "appsink name=sink";

        ROS_INFO_STREAM("Initializing pipeline - " << pipelineStringBuildUp);
        
        GError* error = nullptr;
        pipelineElements.pipeline = gst_parse_launch(pipelineStringBuildUp.c_str(), &error);

        if(error) {
            ROS_ERROR("Failed to initialize pipeline: %s", error->message);
            g_error_free(error);
            return false; 
        }

        pipelineElements.appsink = gst_bin_get_by_name(GST_BIN(pipelineElements.pipeline), "sink");

        if(!pipelineElements.appsink) {
            ROS_ERROR("Failed to get appsink element. Pipeline initialization failed");
            cleanup();
            return false;
        }

        gst_app_sink_set_drop(GST_APP_SINK(pipelineElements.appsink), true);
        gst_app_sink_set_max_buffers(GST_APP_SINK(pipelineElements.appsink), 1);

        GstStateChangeReturn ret = gst_element_set_state(pipelineElements.pipeline, GST_STATE_PLAYING);

        if(ret == GST_STATE_CHANGE_FAILURE) {
            ROS_ERROR("Pipeline failed to start.");
            cleanup();
            return false;
        }

        return (ret == GST_STATE_CHANGE_SUCCESS || 
            ret == GST_STATE_CHANGE_ASYNC || 
            ret == GST_STATE_CHANGE_NO_PREROLL);

    }

    bool HarrierCaptureSrc::initCameraSrc() {
        if (pipelineElements.pipeline) {
            cleanup();
        }
        
    
        return initPipeline();
    }

    bool HarrierCaptureSrc::captureFrameFromSrc(cv::Mat& frame) {
        if(!isPipelineInitialized or !pipelineElements.appsink) {
            return false;
        }

        GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(pipelineElements.appsink));

        if(!sample) {
            gst_sample_unref(sample);
            return false;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!buffer) {
            gst_sample_unref(sample);
            return false;
        }

        GstCaps* caps = gst_sample_get_caps(sample);
        if(!caps) {
            gst_sample_unref(sample);
            return false;
        }

        GstStructure* structure = gst_caps_get_structure(caps, 0);
        if(!structure) {
            gst_sample_unref(sample);
            return false;
        }

        int width, height;
        
        if (!gst_structure_get_int(structure, "width", &width) ||
            !gst_structure_get_int(structure, "height", &height)) {
                gst_sample_unref(sample);
                return false;
        }

        bool currentMode = (harrierState.nightMode or isNightMode(sample));

        GstMapInfo map;
        if(!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return false;
        }

        cv::Mat tmp(height, width, currentMode ? CV_8UC1 : CV_8UC3, (void*)map.data);
        tmp.copyTo(frame);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        return true;
    }

    bool HarrierCaptureSrc::isNightMode(GstSample *sample) {
        GstCaps *caps = gst_sample_get_caps(sample);
        if (not caps) return false;

        GstStructure* structure = gst_caps_get_structure(caps, 0);
        if (not structure) return false;

        const gchar* format = gst_structure_get_string(structure, "format");
        if(not format) return false; 

        return(g_strcmp0(format, "GRAY8") == 0);
    }

    void HarrierCaptureSrc::releasePipeline() {
        cleanup();
        isPipelineInitialized = false;
    }

    void HarrierCaptureSrc::cleanup() {
        if(pipelineElements.pipeline) {
            gst_element_set_state(pipelineElements.pipeline, GST_STATE_NULL);
            gst_object_unref(pipelineElements.pipeline);
            pipelineElements.pipeline = nullptr;
        }

        if(pipelineElements.appsink) {
            gst_object_unref(pipelineElements.appsink);
            pipelineElements.appsink = nullptr;
        }
    }

} // namespace pipeline