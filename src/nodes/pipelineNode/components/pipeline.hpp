#include <ros/ros.h> 
#include <surveillance_system/MotionEvent.h>
#include <sensor_msgs/Image>
#include <cv_bridge/cv_bridge.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <string>
#include <mutex>
#include <queue>

class pipelineBase {
    private:

        struct baseElements {
            GstElement *pipeline; 
            GstElement *appsrc; 
            GstElement *videoconvert;
            GstElement *encoder;
            GstElement *muxer;
            GstElement *filesink;
        };
        

        bool pipelineRunning;
        std::mutex pipelineMtx;
        
        std::queue<GstBuffer*> frameBuffer;

        int bufferSize;
        unsigned int width, height; 
    
    public:
        pipelineBase(int bufferSize, unsigned int width, unsigned int height);
        ~pipelineBase();

        bool initPipeline(const std::string& sinkPath);
        bool startCapture(void);
        void stopPipeline(void);

        void pushFrame(const cv::Mat &frame);

}