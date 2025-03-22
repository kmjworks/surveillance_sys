#include "pipeline.hpp"
#include <cassert>

PipelineBase::PipelineBase(int bufSize, unsigned int frameWidth, unsigned int frameHeight, const std::string &devicePath) 
    : frameBufferInternals{std::queue<GstBuffer*>(), bufSize, frameWidth, frameHeight},
      source{devicePath, false, false} {

    gst_init(NULL, NULL);
    assert(bufSize > 30);

    elements.pipeline = nullptr;

    elements.v4l2src = nullptr; 
    elements.capsfilter = nullptr;
    elements.queue1 = nullptr;
    elements.videoconvert_in = nullptr;
    elements.videorate = nullptr; 
    elements.ratecaps = nullptr;
    elements.tee = nullptr;

    elements.queue_process = nullptr; 
    elements.appsink = nullptr; 

    elements.queue_record = nullptr; 
    elements.appsrc = nullptr;
    elements.videoconvert_out = nullptr;
    elements.encoder = nullptr;
    elements.queue_enc = nullptr;
    elements.muxer = nullptr; 
    elements.filesink = nullptr;

}

PipelineBase::~PipelineBase() {
    stopPipeline();

    while(!frameBufferInternals.frameBuffer.empty()) {
        gst_buffer_unref(frameBufferInternals.frameBuffer.front());
        frameBufferInternals.frameBuffer.pop();
    }
}

bool PipelineBase::initPipelineCapture() {
    /* Clean up an existing pipeline should there be one */
    stopPipeline();

    elements.pipeline = gst_pipeline_new("v4l2-capture-pipeline");

    elements.v4l2src = gst_element_factory_make("v4l2src", "source");
    elements.capsfilter = gst_element_factory_make("capsfilter", "caps");
    elements.queue1 = gst_element_factory_make("queue", "queue1");
    elements.videoconvert_in = gst_element_factory_make("videconvert", "convert_in");
    elements.videorate = gst_element_factory_make("videorate", "rate");
    elements.ratecaps = gst_element_factory_make("capsfilter", "rate_caps");
    elements.tee = gst_element_factory_make("tee", "tee");

    elements.queue_process = gst_element_factory_make("queue", "queue_process");
    elements.appsink = gst_element_factory_make("appsink", "sink");

    /* Check if elements were actually created */
    if (!elements.pipeline || !elements.v4l2src || !elements.capsfilter || 
        !elements.queue1 || !elements.videoconvert_in || !elements.videorate || !elements.ratecaps ||
        !elements.tee || !elements.queue_process || !elements.appsink) {
        ROS_ERROR_THROTTLE(10, "Failed to create GStreamer capture elements");
        return false;
    }

    g_object_set(G_OBJECT(elements.v4l2src), "device", source.srcDevicePath.c_str(), "io-mode", 2, NULL);

    g_object_set(G_OBJECT(elements.queue1),
                "max-size-buffers", 30,
                "max-size-bytes", 0,
                "max-size-time", 0,
                NULL);

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "YUYV",
                                        "width", G_TYPE_INT, frameBufferInternals.frameWidth,
                                        "height", G_TYPE_INT, frameBufferInternals.frameHeight,
                                        NULL);

    g_object_set(G_OBJECT(elements.capsfilter), "caps", caps, NULL);
    gst_caps_unref(caps);

    g_object_set(G_OBJECT(elements.videorate), "max-rate", 60, NULL);


    caps = gst_caps_new_simple("video/x-raw",
                     "format", G_TYPE_STRING, "I420",
                                "width", G_TYPE_INT, frameBufferInternals.frameWidth,
                                "height", G_TYPE_INT, frameBufferInternals.frameHeight,
                                "framerate", GST_TYPE_FRACTION, 60, 1,
                                NULL);

    g_object_set(G_OBJECT(elements.ratecaps), "caps", caps, NULL);
    gst_caps_unref(caps);

    /*
        Processing queue configuration
    */

    g_object_set(G_OBJECT(elements.queue_process),
                "max-size-buffers", 30,
                "max-size-bytes", 0,
                "max-size-times", 0,
                NULL);
    
                
    /* appsink configuration */

    g_object_set(G_OBJECT(elements.appsink),
                        "emit-signals", TRUE,
                        "max-buffers", 1,
                        "drop", TRUE, 
                        NULL);

    
    
    caps = gst_caps_new_simple("video/x-raw",
                     "format", G_TYPE_STRING, "BGR",
                                "width", G_TYPE_INT, frameBufferInternals.frameWidth,
                                "height", G_TYPE_INT, frameBufferInternals.frameHeight,
                                NULL);

    gst_app_sink_set_caps(GST_APP_SINK(elements.appsink), caps);
    gst_caps_unref(caps);

    g_signal_connect(elements.appsink, "new-sample", G_CALLBACK(cb_frameSample), this);

    gst_bin_add_many(GST_BIN(elements.pipeline),
                    elements.v4l2src, elements.capsfilter,
                    elements.queue1, elements.videoconvert_in,
                    elements.videorate, elements.ratecaps,
                    elements.tee, elements.queue_process,
                    elements.appsink, NULL);

    if(!gst_element_link_many(elements.v4l2src, elements.capsfilter,
                            elements.queue1, elements.videoconvert_in,
                            elements.videorate, elements.ratecaps,
                            elements.tee, NULL
                            )) {
        ROS_ERROR("Failed to link the first part of the frame capture pipeline.");
        gst_object_unref(elements.pipeline);
        return false;
    }

    GstPad *tee_process_pad = gst_element_get_request_pad(elements.tee, "src_%u");
    GstPad *queue_process_pad = gst_element_get_static_pad(elements.queue_process, "sink");

    if(gst_pad_link(tee_process_pad, queue_process_pad) != GST_PAD_LINK_OK) {
        ROS_ERROR_THROTTLE(10, "Failed to link tee to the pipeline processing queue.");
        gst_object_unref(tee_process_pad);
        gst_object_unref(queue_process_pad);
        gst_object_unref(elements.pipeline);
        return false;
    }

    gst_object_unref(tee_process_pad);
    gst_object_unref(queue_process_pad);

    if(!gst_element_link(elements.queue_process, elements.appsink)) {
        ROS_ERROR_THROTTLE(10, "Failed to link the pipeline's processing queue to appsink.");
        gst_object_unref(elements.pipeline);
        return false;
    }

    /* Set pipeline to READY state indicating that the preparing configuration succeeded. */

    GstStateChangeReturn ret = gst_element_set_state(elements.pipeline, GST_STATE_READY);

    if(ret == GST_STATE_CHANGE_FAILURE) {
        ROS_ERROR_THROTTLE(10, "Failed to set pipeline to READY state.");
        return false; 
    }

    return true;
}

bool PipelineBase::initPipelineRecording(const std::string &sinkPath) {
    if(!elements.pipeline || !elements.tee) {
        ROS_ERROR("Cannot initialize pipeline recording without a running capture pipeline.");
        return false;
    }

    elements.queue_record = gst_element_factory_make("queue", "queue_record");
    elements.appsrc = gst_element_factory_make("appsrc", "appsource");
    elements.videoconvert_out = gst_element_factory_make("videconvert", "convert_out");
    elements.encoder = gst_element_factory_make("x264enc", "encoder");
    elements.queue_enc = gst_element_factory_make("queue", "queue_enc");
    elements.muxer = gst_element_factory_make("mp4mux", "muxer");
    elements.filesink = gst_element_factory_make("filesink", "file_sink");

    if (!elements.queue_record || !elements.appsrc || !elements.videoconvert_out ||
        !elements.encoder || !elements.queue_enc || !elements.muxer || !elements.filesink) {
        ROS_ERROR("Failed to create GStreamer recording elements");
        return false;
    }

    
}