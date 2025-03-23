#include "pipelineInternal.hpp"
#include <ros/ros.h>

namespace pipeline {
    
    PipelineInternal::PipelineInternal(int frameRate, bool nightMode) :
        state{frameRate, nightMode} {}


    PipelineInternal::~PipelineInternal() {}

    cv::Mat PipelineInternal::processFrame(const cv::Mat& frame) {

        if(frame.empty()) {
            return cv::Mat();
        }

        cv::Mat convertedFrame = convertFormat(frame);

        return preprocessFrame(convertedFrame);
    }

    cv::Mat PipelineInternal::convertFormat(const cv::Mat& frame) {
        cv::Mat convertedFrame;

        if ((state.nightMode && frame.channels() == 1) || 
            (!state.nightMode && frame.channels() == 3 && frame.type() == CV_8UC3)) {

            frame.copyTo(convertedFrame);
            return convertedFrame;
        }

        if(state.nightMode && frame.channels() == 3) {
            cv::cvtColor(frame, convertedFrame, cv::COLOR_BGR2GRAY);
        } else if (!state.nightMode && frame.channels() == 1) {
            cv::cvtColor(frame, convertedFrame, cv::COLOR_GRAY2BGR);
        } else if (!state.nightMode && frame.channels() == 3 && frame.type() != CV_8UC3) {
            frame.convertTo(convertedFrame, CV_8UC3);
        } else {
            frame.copyTo(convertedFrame);
        }

        return convertedFrame;
    }

    cv::Mat PipelineInternal::preprocessFrame(const cv::Mat& frame) {
        cv::Mat resultingFrame;

        if(frame.channels() == 1) {
            cv::equalizeHist(frame, resultingFrame);
        } else {
            cv::Mat ycrcb;
            cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> splitChannels;
            cv::split(ycrcb, splitChannels);

            cv::equalizeHist(splitChannels[0], splitChannels[0]);

            cv::merge(splitChannels, ycrcb);

            cv::cvtColor(ycrcb, resultingFrame, cv::COLOR_YCrCb2BGR);
        }

        return resultingFrame;
    }

}
