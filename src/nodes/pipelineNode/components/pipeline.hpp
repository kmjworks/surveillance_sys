#include <cv_bridge/cv_bridge.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <surveillance_system/motion_event.h>
#include <mutex>
#include <queue>
#include <string>

class PipelineBase {
private:
    struct elements {
        GstElement *pipeline;
        GstElement *appsrc;
        GstElement *videoconvert;
        GstElement *encoder;
        GstElement *muxer;
        GstElement *filesink;
    };

    elements baseElements;
    bool pipelineRunning;
    std::mutex pipelineMtx;

    std::queue<GstBuffer *> frameBuffer;

    int bufferSize;
    unsigned int width, height;

    bool recording;

public:
    PipelineBase(int bufferSize, unsigned int width, unsigned int height);
    ~PipelineBase();

    bool initPipeline(const std::string &sinkPath);
    bool startCapture();
    void stopPipeline();

    void pushFrame(const cv::Mat &frame);
};
