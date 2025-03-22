#include "harrierCaptureSrc.hpp"
#include <ros/ros.h>
#include <gst/app/gstappsink.h>


HarrierCaptureSrc::HarrierCaptureSrc(const std::string& devicePath, int frameRate) 
    : devicePath(devicePath), frameParams{frameRate, 1920, 1080, "YUYV"},
    isInitialized(false), isNightMode(false) {

    static bool gstInitialized = false; 
    if(!gstInitialized) {
        int argc = 0; 
        char** argv = nullptr;
        gst_init(&argc, &argv);
        gstInitialized = true; 
    }
}

HarrierCaptureSrc::~HarrierCaptureSrc() {
    releasePipelineResources();
}

bool HarrierCaptureSrc::initialize() {
    if(isInitialized) {
        ROS_INFO("HarrierCaptureSrc already initialized - nothing to do.");
        return true; 
    }


    if(!buildPipeline()) {
        ROS_ERROR("Failed to set up pipeline.");
        return false; 
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if(ret == GST_STATE_CHANGE_FAILURE) {
        ROS_ERROR("Failed to start pipeline.");
        releasePipelineResources();
        return false; 
    }

    ret = gst_element_get_state(pipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
    if(ret != GST_STATE_CHANGE_SUCCESS) {
        ROS_ERROR("Failed to start pipeline");
        releasePipelineResources();
        return false; 
    }

    isInitialized = true; 
    ROS_INFO("HarrierCaptureSrc initialized.");
    return true; 
}

bool HarrierCaptureSrc::buildPipeline() {
    std::string pipelineStr = 
        "v4l2src device=" + devicePath + 
        " ! video/x-raw,format=" + frameParams.format + 
        ",width=" + std::to_string(frameParams.width) + 
        ",height=" + std::to_string(frameParams.height) + 
        ",framerate=" + std::to_string(frameParams.frameRate) + "/1" + 
        " ! videoconvert ! video/x-raw,format=BGR" +
        " ! appsink name=sink";

    ROS_INFO("Pipeline: %s", pipelineStr.c_str());

    GError* error = nullptr;
    pipeline = gst_parse_launch(pipelineStr.c_str(), &error);

    if (error) {
        ROS_ERROR("Failed to create GStreamer pipeline: %s", error->message);
        g_error_free(error);
        return false;
    }

    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if(!appsink) {
        ROS_ERROR("Failed to get appsink element.");
        gst_object_unref(pipeline);
        pipeline = nullptr;
        return false;
    }

    
}