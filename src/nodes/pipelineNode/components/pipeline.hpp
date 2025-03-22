#pragma once
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <gst/gst.h>

#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <surveillance_system/motion_event.h>

#include <mutex>
#include <queue>
#include <string>

namespace internal {
    
    struct PipelineGstElements {
        GstElement *pipeline;

        /* Capture elements */
        GstElement *v4l2src;
        GstElement *capsfilter;
        GstElement *queue1;
        GstElement *videoconvert_in;
        GstElement *videorate;
        GstElement *ratecaps;
        GstElement *tee;

        /* Processing branch/elements */
        GstElement *queue_process;
        GstElement *appsink;

        /* Recording branch */
        GstElement *queue_record;
        GstElement *appsrc;
        GstElement *videoconvert_out;
        GstElement *encoder;
        GstElement *queue_enc;
        GstElement *muxer;
        GstElement *filesink;
    };

    struct PipelineFrameBuffer {
        std::queue<GstBuffer*> frameBuffer;

        int bufferSize;
        unsigned int frameWidth; 
        unsigned int frameHeight;
    };

    struct PipelineSrc {
        std::string srcDevicePath;

        bool pipelineRunning; 
        bool pipelineRecording;
    };
}

class PipelineBase {
public:
    PipelineBase(int bufSize, unsigned int frameWidth, unsigned int frameHeight, const std::string &devicePath = "/dev/video0");
    virtual ~PipelineBase();

    bool initPipelineCapture();
    bool initPipelineRecording(const std::string &sinkPath);

    bool startCapture();

    bool startRecording();
    bool stopRecording();

    void stopPipeline();

    void pushFrame(const cv::Mat &frame);

    cv::Mat gstBufferToMatFormat(GstBuffer* buffer, GstCaps* caps);

private:
    /*
        Callback for new frames
    */ 
    static GstFlowReturn cb_frameSample(GstAppSink *appsink, gpointer user_data);
    GstFlowReturn sampleHandler(GstSample* sample);

    virtual void processFrame(const cv::Mat &frame) = 0; 

    internal::PipelineGstElements elements;
    internal::PipelineFrameBuffer frameBufferInternals;
    internal::PipelineSrc source;


};
