#include "pipelineNode.hpp"
#include "opencv2/videoio.hpp"
#include "sensor_msgs/Image.h"
#include "surveillance_system/motion_event.h"

PipelineNode::PipelineNode(ros::NodeHandle nh) : nh(nh) {
    nh.param("buffer_size", videoWriterBufferSize, 30);
    nh.param<std::string>("output_path", outputPath, "recordings");

    ROSInternalInterface.sub_motionEvents = nh.subscribe("/motion_detection_node/motion_events", 10,
                                                         &PipelineNode::cb_motionEvent, this);

    ROSInternalInterface.sub_rawImageData =
        nh.subscribe("/camera_node/image_raw", 1, &PipelineNode::cb_imageData, this);

    std::string mkdir_cmd = "mkdir -p" + outputPath;
    system(mkdir_cmd.c_str());

    ROS_INFO("Pipeline OK.");
}

void PipelineNode::cb_motionEvent(const surveillance_system::motion_event::ConstPtr& msg) {
    lastMotionEventRegistered = ros::Time::now();
    if (!videoWriterStatus) {
        videoWriterRecording(true, 0);
    }
}

void PipelineNode::cb_imageData(const sensor_msgs::ImageConstPtr& msg) {
    cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

    if (videoWriterStatus) {
        videoWriterForSimulation.write(frame);

        if ((ros::Time::now() - lastMotionEventRegistered).toSec() > 3.0) {
            videoWriterRecording(false, 999);
        }
    } else {
        frameBuffer.push(frame.clone());
        if (frameBuffer.size() > videoWriterBufferSize) {
            frameBuffer.pop();
        }
    }
}

void PipelineNode::videoWriterRecording(bool start, uint32_t motionEventID) {
    if (start) {
        std::string filename = outputPath + "/event_" + std::to_string(motionEventID) + ".avi";

        videoWriterForSimulation.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                                      cv::Size(1920, 1080));
        while (!frameBuffer.empty()) {
            videoWriterForSimulation.write(frameBuffer.front());
            frameBuffer.pop();
        }

        videoWriterStatus = true;
    } else {
        videoWriterForSimulation.release();
        videoWriterStatus = false;
    }
}
