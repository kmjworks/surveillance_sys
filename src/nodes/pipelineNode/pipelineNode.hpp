#pragma once 

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

class HarrierCaptureSrc;
class PipelineInternal;
class PipelineInitialDetection;

namespace internal {
    struct ROSInterface {
        ros::Publisher motionPublisher;
        ros::Publisher errorPublisher;
        ros::Publisher processedFramePublisher;
    };

    struct RuntimeState {
        std::string devicePath;
        std::string outputPath;

        bool nightMode;
        bool showDebugFrames;  
        int frameRate;
        int motionPublishingRate;
        int bufferSize;
    };

    struct OnError {
        int cameraHandleRetryCount;
        int cameraHandleRetryDelay;
    };
}

class PipelineNode {
    public:
        PipelineNode(ros::NodeHandle& nh);
        ~PipelineNode();

        bool initialize();
        void run();

        void publishMotionFrame(const cv::Mat& frame);
        void publishError(const std::string& errorMsg);
        
    private:
        ros::NodeHandle& nodeHandle;
        ros::Timer timer;
        internal::ROSInterface rosInterface;
        internal::RuntimeState state;
        internal::OnError errorHandling;

        std::unique_ptr<HarrierCaptureSrc> captureSrcRaw;
        std::unique_ptr<PipelineInternal> pipelineInternal;
        std::unique_ptr<PipelineInitialDetection> initialMotionDetection;

        std::mutex frameMtx;
        std::thread pipelineProcessingThread;
        std::atomic<bool> pipelineRunning;

        cv::Mat latestFrame;
        cv::Mat latestProcessedFrame;
        cv::Mat previousRawFrame;
        std::mutex frameCacheMutex;
        ros::Time lastCaptureTime;
        int frameCount = 0;
        ros::Time statsStartTime;

        void loadPipelineParams();
        void pipelineProcessingLoop();
        void pipelineCleanup();
        void timerCallback(const ros::TimerEvent &);
        bool isNewFrame(const cv::Mat& currentFrame);

};