#include "pipelineNode.hpp"
#include "components/harrierCaptureSrc.hpp"
#include "components/pipelineInternal.hpp"
#include "components/pipelineInitialDetection.hpp"

PipelineNode::PipelineNode(ros::NodeHandle& nh) : nodeHandle(nh), pipelineRunning(false),
    errorHandling{3, 5} {
    
    rosInterface.motionPublisher = nodeHandle.advertise<sensor_msgs::Image>("pipeline/runtime_potentialMotionEvents", 10);
    rosInterface.errorPublisher = nodeHandle.advertise<std_msgs::String>("pipeline/runtime_videoSrcErrors", 10);
    rosInterface.processedFramePublisher = nodeHandle.advertise<sensor_msgs::Image>("pipeline/runtime_processedFrames", 10);

    loadPipelineParams();

    captureSrcRaw = std::make_unique<HarrierCaptureSrc>(state.devicePath, state.frameRate);
    pipelineInternal = std::make_unique<PipelineInternal>(state.nightMode, state.showDebugFrames);
    initialMotionDetection = std::make_unique<PipelineInitialDetection>(state.motionPublishingRate);    

    timer = nh.createTimer(ros::Duration(1.0 / state.frameRate), &PipelineNode::timerCallback, this);
}

PipelineNode::~PipelineNode() {
    pipelineCleanup();
}

void PipelineNode::loadPipelineParams() {
    nodeHandle.param<std::string>("video_src", state.devicePath, "/dev/video0");
    nodeHandle.param<int>("frame_rate", state.frameRate, 30);
    nodeHandle.param<bool>("show_debug_frames", state.showDebugFrames, false);
    nodeHandle.param<int>("buffer_size", state.bufferSize, 100);
    nodeHandle.param<std::string>("output_path", state.outputPath, "/tmp/sur_sys_recordings");
    nodeHandle.param<bool>("night_mode", state.nightMode, false);
    nodeHandle.param<int>("motion_publish_rate", state.motionPublishingRate, 1);
    nodeHandle.param<int>("camera_retry_count", errorHandling.cameraHandleRetryCount, 3);
    nodeHandle.param<int>("camera_retry_delay", errorHandling.cameraHandleRetryDelay, 5);

    ROS_INFO("Pipeline parameters: ");
    ROS_INFO(" Device path: %s", state.devicePath.c_str());
    ROS_INFO(" Frame rate: %d", state.frameRate);
    ROS_INFO(" Night mode: %s", state.nightMode ? "enabled" : "disabled");
    ROS_INFO(" Motion publishing rate: %d hz", state.motionPublishingRate);
}

bool PipelineNode::initialize() {
    int retryCount = 0;
    bool cameraInitialized = false;

    while (retryCount < errorHandling.cameraHandleRetryCount && !cameraInitialized) {
        ROS_INFO("Initializing camera, attempt %d of %d", retryCount+1, errorHandling.cameraHandleRetryCount);

        cameraInitialized = captureSrcRaw->initialize();

        if (!cameraInitialized) {
            ROS_WARN("Failed to initialize camera, retrying in %d seconds", 
                    errorHandling.cameraHandleRetryDelay);
            ros::Duration(errorHandling.cameraHandleRetryDelay).sleep();
            retryCount++;
        }
    }

    if(!cameraInitialized) {
        std::string errorMsg = "Failed to initialize camera after " + 
                              std::to_string(errorHandling.cameraHandleRetryCount) + " attempts";
        ROS_ERROR("%s", errorMsg.c_str());
        publishError(errorMsg);
        return false;
    }

    if(!pipelineInternal->initialize()) {
        std::string errorMsg = "Failed to initialize pipeline internal components";
        ROS_ERROR("%s", errorMsg.c_str());
        publishError(errorMsg);
        return false;
    }

    if(!initialMotionDetection->initialize()) {
        std::string errorMsg = "Failed to initialize motion detector";
        ROS_ERROR("%s", errorMsg.c_str());
        publishError(errorMsg);
        return false;
    }

    ROS_INFO("Pipeline initialized.");
    return true;
}

void PipelineNode::run() {
    if(pipelineRunning) {
        ROS_WARN("Pipeline already running.");
        return;
    }

    pipelineRunning = true;
    pipelineProcessingThread = std::thread(&PipelineNode::pipelineProcessingLoop, this);
    ROS_INFO("Pipeline started.");
}


void PipelineNode::pipelineProcessingLoop() {
    cv::Mat frame;
    cv::Mat processedFrame;
    bool hasMotion = false;
    

    
    ros::WallTime lastMotionPublishTime = ros::WallTime::now();
    ros::WallTime lastFramePublishTime = ros::WallTime::now();
    ros::WallRate processingRate(120);
    
    // Fixed frame rate for video feed (30 fps if not specified otherwise)
    const double videoPublishRate = state.frameRate > 0 ? state.frameRate : 30.0;
    const double videoPublishInterval = 1.0 / videoPublishRate;
    const double motionPublishInterval = 1.0 / state.motionPublishingRate;
    
    ROS_INFO("Target video publish rate: %.1f fps (%.3f sec interval)", 
        videoPublishRate, videoPublishInterval);

    while(pipelineRunning && ros::ok()) {
        
        /*
        
        */
        bool captureSuccess = captureSrcRaw->captureFrame(frame);
        if(!captureSuccess || frame.empty()) {
            ros::Duration(0.001).sleep();
            continue;
        }

        // Process frame
        {
            std::lock_guard<std::mutex> lock(frameMtx);
            processedFrame = pipelineInternal->processFrame(frame);
        }

        if(processedFrame.empty()) {
            ROS_WARN_THROTTLE(1, "Empty processed frame.");
            continue;
        }

        ros::WallTime currentTime = ros::WallTime::now();
        
        // Publish processed frames at the specified fixed rate (30fps by default)
        double timeSinceLastFrame = (currentTime - lastFramePublishTime).toSec();
        
        if(timeSinceLastFrame >= videoPublishInterval) {
            cv_bridge::CvImage cvImg;
            cvImg.header.stamp = ros::Time::now();
            cvImg.encoding = state.nightMode ? "mono8" : "bgr8";
            cvImg.image = processedFrame;
            rosInterface.processedFramePublisher.publish(cvImg.toImageMsg());
            lastFramePublishTime = currentTime;
            
            ROS_DEBUG("Frame published: interval=%.4fs (target: %.4fs)", 
                     timeSinceLastFrame, videoPublishInterval);

            lastFramePublishTime = currentTime;
        }
        
        // Motion detection and publishing at a controlled rate (1fps by default)
        hasMotion = initialMotionDetection->detectMotion(processedFrame);
        if(hasMotion) {
            double timesinceLastMotion = (currentTime - lastMotionPublishTime).toSec();

            if(timesinceLastMotion >= motionPublishInterval) {
                publishMotionFrame(processedFrame);
                lastMotionPublishTime = currentTime;
            }
        }

        processingRate.sleep();
    }
}

void PipelineNode::pipelineCleanup() {
    pipelineRunning = false;
    
    if (pipelineProcessingThread.joinable()) {
        pipelineProcessingThread.join();
    }
  
    ROS_INFO("Pipeline resources cleaned up");
}

void PipelineNode::publishMotionFrame(const cv::Mat& frame) {
    cv_bridge::CvImage cvImg;
    cvImg.header.stamp = ros::Time::now();
    cvImg.encoding = state.nightMode ? "mono8" : "bgr8";
    cvImg.image = frame;
    rosInterface.motionPublisher.publish(cvImg.toImageMsg());
}

void PipelineNode::publishError(const std::string& errorMsg) {
    std_msgs::String msg;
    msg.data = errorMsg;
    rosInterface.errorPublisher.publish(msg);
}

void PipelineNode::timerCallback(const ros::TimerEvent &) {
    cv::Mat processedFrame, frame;
    cv_bridge::CvImage cvImg;

    {
        std::lock_guard<std::mutex> lock(frameMtx);

        if(!captureSrcRaw->captureFrame(frame)) {
            ROS_INFO("Error capturing frame.");
        }
        processedFrame = pipelineInternal->processFrame(frame);

        cvImg.header.stamp = ros::Time::now();
        cvImg.encoding = state.nightMode ? "mono8" : "bgr8";
        cvImg.image = processedFrame;
    }

    if(processedFrame.empty()) {
        ROS_WARN_THROTTLE(1, "Empty processed frame.");
    }

    rosInterface.processedFramePublisher.publish(cvImg.toImageMsg());
}

