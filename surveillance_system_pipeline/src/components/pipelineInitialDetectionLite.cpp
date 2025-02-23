#include "pipelineInitialDetectionLite.hpp"
#include <ros/ros.h>

namespace pipeline {

PipelineInitialDetectionLite::PipelineInitialDetectionLite(int minAreaPixels, float downscaleFactor, int history) {
    params.minAreaPx = minAreaPixels;
    params.downScale = downscaleFactor;
    params.historyLength = history;
    
    // Initialize background subtractor
    bgSubtractor = cv::createBackgroundSubtractorMOG2(params.historyLength, 16.0, false);
    
    ROS_INFO("[PipelineInitialDetectionLite] Initialized with minArea=%d, downScale=%.2f, history=%d", 
             minAreaPixels, downscaleFactor, history);
    
    initialized = true;
}

PipelineInitialDetectionLite::~PipelineInitialDetectionLite() {
    ROS_INFO("[PipelineInitialDetectionLite] Shutting down");
}

void PipelineInitialDetectionLite::preprocessForDetection(const cv::Mat& input, cv::Mat& output) {
    // Convert to grayscale if not already
    if (input.channels() > 1) {
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
    } else {
        output = input.clone();
    }
    
    // Downscale for faster processing
    if (params.downScale < 1.0) {
        cv::Size newSize(
            static_cast<int>(output.cols * params.downScale), 
            static_cast<int>(output.rows * params.downScale)
        );
        cv::resize(output, output, newSize);
    }
    
    // Apply slight blur to reduce noise
    cv::GaussianBlur(output, output, cv::Size(5, 5), 0);
}

void PipelineInitialDetectionLite::postprocessMask(cv::Mat& mask) {
    // Apply morphological operations to remove noise
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);
}

bool PipelineInitialDetectionLite::detectMotion(const cv::Mat& frame, cv::Mat& outputMotionMask, cv::Mat& annotatedFrame) {
    if (!initialized || frame.empty()) {
        ROS_WARN_THROTTLE(5.0, "[PipelineInitialDetectionLite] Not initialized or empty frame");
        return false;
    }
    
    cv::Mat processedFrame;
    preprocessForDetection(frame, processedFrame);
    
    // Apply background subtraction
    bgSubtractor->apply(processedFrame, fgMask);
    
    // Post-process the mask
    postprocessMask(fgMask);
    
    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Scale mask back to original size if downscaled
    if (params.downScale < 1.0) {
        cv::resize(fgMask, outputMotionMask, frame.size());
    } else {
        outputMotionMask = fgMask.clone();
    }
    
    // Create annotated frame for visualization
    annotatedFrame = frame.clone();
    
    // Filter contours by size and draw on annotated frame
    bool motionDetected = false;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        
        // Scale area based on downscale factor
        double scaledArea = area / (params.downScale * params.downScale);
        
        if (scaledArea >= params.minAreaPx) {
            motionDetected = true;
            
            // Scale contour points to original image size
            std::vector<cv::Point> scaledContour;
            for (const auto& pt : contour) {
                scaledContour.push_back(cv::Point(
                    static_cast<int>(pt.x / params.downScale),
                    static_cast<int>(pt.y / params.downScale)
                ));
            }
            
            // Draw contour on annotated frame
            cv::drawContours(annotatedFrame, std::vector<std::vector<cv::Point>>{scaledContour}, 
                            0, cv::Scalar(0, 255, 0), 2);
        }
    }
    
    return motionDetected;
}

} // namespace pipeline