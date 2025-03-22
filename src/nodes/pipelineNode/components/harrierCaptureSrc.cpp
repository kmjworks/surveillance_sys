#include "harrierCaptureSrc.hpp"
#include <ros/ros.h>
#include <gst/app/gstappsink.h>


HarrierCaptureSrc::HarrierCaptureSrc(const std::string& devicePath, int frameRate) 
    : devicePath(devicePath), frameParams{frameRate, 1920, 1080, "YUYV"},
    pipeline(nullptr), appsink(nullptr), isInitialized(false), isNightMode(false),
    latestFrame(nullptr) {

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

    gst_app_sink_set_emit_signals(GST_APP_SINK(appsink), TRUE);
    gst_app_sink_set_drop(GST_APP_SINK(appsink), TRUE);
    gst_app_sink_set_max_buffers(GST_APP_SINK(appsink), 1);

    g_signal_connect(appsink, "new-sample", G_CALLBACK(cb_newFrameSample), this);

    return true;
}


bool HarrierCaptureSrc::captureFrame(cv::Mat& frame) {
    if(!isInitialized) {
        /* Shouldn't even hit this branch in the first place. */
        ROS_ERROR("HarrierCaptureSrc not initialized.");
        return false;
    }

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));

    {
        std::lock_guard<std::mutex> lock(frameSampleMtx);
        if(latestFrame) {
            sample = gst_sample_ref(latestFrame);
        }
    }

    if(!sample) {
        sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));

        ROS_WARN("Failed to pull a runtime sample from pipeline.");
        return false;
    }

    frame = gstSampleToCvMat(sample);
    gst_sample_unref(sample);

    /*
        Night-mode detection based on frame characteristics
        If the frame is already grayscale or has very low saturation, we can map it as a frame that was captured
        in "night-mode"
    */

    if(frame.empty() && frame.channels() == 3) {
        cv::Mat hsvFrame;
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

        cv::Scalar meanSaturation = cv::mean(hsvFrame);

        isNightMode = (meanSaturation[1] < 20) ? true : false; 
        
    }

    return !frame.empty();
}

cv::Mat HarrierCaptureSrc::gstSampleToCvMat(GstSample* sample) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    GstStructure* structure = gst_caps_get_structure(caps, 0);

    int width, height; 

    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);

    const gchar* format = gst_structure_get_string(structure, "format");

    GstMapInfo map; 
    if(!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        ROS_ERROR("Failed to map pipeline buffer.");
        return cv::Mat();
    }

    cv::Mat frame; 
    if(g_str_equal(format, "BGR")) {
        frame = cv::Mat(height, width, CV_8UC3, map.data).clone();
    } else if(g_str_equal(format, "GRAY8") or g_str_equal(format, "Y800")) {
        frame = cv::Mat(height, width, CV_8UC1, map.data).clone();
    } else {
        ROS_WARN("Unknown format: %s, trying to interpret as BGR", format);
        frame = cv::Mat(height, width, CV_8UC3, map.data).clone();
    }

    gst_buffer_unmap(buffer, &map);

    return frame;
}

GstFlowReturn HarrierCaptureSrc::cb_newFrameSample(GstElement* sink, gpointer data) {
    HarrierCaptureSrc* self = static_cast<HarrierCaptureSrc*>(data);

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));

    if(!sample) {
        return GST_FLOW_ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(self->frameSampleMtx);

        if(self->latestFrame) {
            gst_sample_unref(self->latestFrame);
        }

        self->latestFrame = sample;
    }

    return GST_FLOW_OK;
}

void HarrierCaptureSrc::releasePipelineResources() {
    if(pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);

        if(appsink) {
            gst_object_unref(appsink);
            appsink = nullptr;
        }
        
        gst_object_unref(pipeline);
        pipeline = nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(frameSampleMtx);
        
        if(latestFrame) {
            gst_sample_unref(latestFrame);
            latestFrame = nullptr;
        }

    }


    isInitialized = false;
}

std::string HarrierCaptureSrc::getFrameFormat() const {
    return frameParams.format;
}

bool HarrierCaptureSrc::isNightModeDetected() const {
    return isNightMode;
}