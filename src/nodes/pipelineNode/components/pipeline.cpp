#include "pipeline.hpp"

PipelineBase::PipelineBase(int bufferSize, unsigned int width, unsigned int height) 
    : bufferSize(30), width(1920), height(1080) {

        gst_init(NULL, NULL);

        baseElements.pipeline = NULL; 
        baseElements.appsrc = NULL; 
        baseElements.videoconvert = NULL;
        baseElements.encoder = NULL; 
        baseElements.muxer = NULL;
        baseElements.filesink = NULL; 

}

PipelineBase::~PipelineBase(void) {
    stopPipeline();

    while(!frameBuffer.empty()) {
        gst_buffer_unref(frameBuffer.front());
        frameBuffer.pop();
    }
}

bool PipelineBase::initPipeline(const std::string& sinkPath) {
    std::lock_guard<std::mutex> lock(pipelineMtx);

    /*
        Cleanup any existing pipelines or artifacts
    */
    stopPipeline();

    baseElements.pipeline = gst_pipeline_new("video-record-pipeline");
    baseElements.appsrc = gst_element_factory_make("appsrc", "source");
    baseElements.videoconvert = gst_element_factory_make("videoconvert", "converter");
    baseElements.encoder = gst_element_factory_make("x264enc", "encoder");
    baseElements.muxer = gst_element_factory_make("mp4mux", "muxer");
    baseElements.filesink = gst_element_factory_make("filesink", "sink");

    if (!baseElements.pipeline || !baseElements.appsrc ||
         !baseElements.videoconvert || !baseElements.encoder || 
         !baseElements.muxer || !baseElements.filesink) {
        ROS_ERROR("Failed to create GStreamer elements");
        return false;
    }

    g_object_set(G_OBJECT(baseElements.appsrc),
                "stream-type",0,
                "format", GST_FORMAT_TIME,
                "is_live", TRUE,
                "do-timestamp", FALSE, // Stalls the pipeline with real-time
                NULL); 

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "BGR",
                                        "width", G_TYPE_INT, width,
                                        "height", G_TYPE_INT, height,
                                        "framerate", G_TYPE_FRACTION, 30, 1,
                                        NULL);

    gst_app_src_set_caps(GST_APP_SRC(baseElements.appsrc), caps);
    gst_caps_unref(caps);


    g_object_set(G_OBJECT(baseElements.encoder),
                "bitrate", 2000, /* 2Mbps */
                "speed-preset", 1, /* Faster encoding, but less compression */
                "tune", 4, // Zero-latency, trivial
                NULL);
    
    g_object_set(G_OBJECT(baseElements.filesink), "location", filename.c_str(), NULL);


    gst_bin_add_many(GST_BIN(baseElements.pipeline), baseElements.appsrc, baseElements.videoconvert,
    baseElements.encoder, baseElements.muxer, baseElements.filesink, NULL);


    if(!gst_element_link_many(baseElements.appsrc, baseElements.videoconvert, baseElements.encoder,
        baseElements.muxer, baseElements.filesink, NULL)) {
            ROS_ERROR_THROTTLE(10, "GStreamer link failed.");
            gst_object_unref(baseElements.pipeline);
            return false;
        }

    GstStateChangeReturn ret = gst_element_set_state(baseElements.pipeline, GST_STATE_READY);
    if(ret == GST_STATE_CHANGE_FAILURE) {
        ROS_ERROR("Pipeline state change failed.");
        gst_object_unref(baseElements.pipeline);
        return false; 
    }

    recording = false; 
    return true;
}

bool PipelineBase::startCapture(void) {
    std::lock_guard<std::mutex> lock(pipelineMtx);

    if(!baseElements.pipeline) {
        ROS_ERROR("Pipeline not created - no capture without a running pipeline.");
        return false;
    }

    if(recording) {
        return true;
    }

    GstStateChangeReturn ret = gst_element_set_state(baseElements.pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            ROS_ERROR("Failed to start pipeline");
            return false;
        }

    while(!frameBuffer.empty())  {
        GstBuffer *buf = frameBuffer.front();
        frameBuffer.pop();

        GstFlowReturn flowReturn = gst_app_src_push_buffer(GST_APP_SRC(baseElements.appsrc));

        if(flowReturn != GST_FLOW_OK) {
            ROS_WARN("Failed to push buffer from queue - dropping frames: %d", flowReturn);
        }
    }

    recording = true; 
    return true;
}