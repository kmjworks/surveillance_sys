#include "pipelineInitialDetectionLite.hpp"
#include <opencv2/cudafilters.hpp>
#include <ros/ros.h>


namespace pipeline {
PipelineInitialDetectionLite::PipelineInitialDetectionLite(int minAreaPx,
    float downScale,
    int history,
    int detectInterval)
: minArea(minAreaPx),
scale(downScale),
frameInterval(detectInterval),
frameCounter(0)
{
    bgsub_ = cv::cuda::createBackgroundSubtractorFGD(history, /*nmixtures*/ cv::cuda::BACKGROUND_MODEL_FGD);
    ROS_INFO_STREAM("InitialMotionLite - CUDA bg-sub enabled " << "(history=" << history << ", minArea=" << minAreaPx << ")");
}


cv::Mat PipelineInitialDetectionLite::convertGrayFaster(const cv::Mat& src) {
    cv::Mat gray;
    if (src.channels() == 1)
        gray = src;
    else
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

void PipelineInitialDetectionLite::toOriginalScale(std::vector<cv::Rect>& rois, double scale, int maxW, int maxH) {
    for (auto& r : rois) {
        r.x      = std::max(0, std::min(int(r.x / scale), maxW-1));
        r.y      = std::max(0, std::min(int(r.y / scale), maxH-1));
        r.width  = std::min(int(r.width / scale),  maxW - r.x);
        r.height = std::min(int(r.height/ scale),  maxH - r.y);
    }
}

bool PipelineInitialDetectionLite::detect(const cv::Mat& frame, std::vector<cv::Rect>& outRois) {
    if (frame.empty())
        return false;

    /* down‑scale & grayscale (CPU processed) */
    cv::Mat small;
    cv::resize(convertGrayFaster(frame), small, cv::Size(), scale, scale, cv::INTER_LINEAR);

    /* only process every Nth frame */
    if (++frameCounter_ % frameInterval_ != 0)
        return false;

    /* upload to GPU (zero‑copy on Jetson if CV_8UC1) */
    gpuIn.upload(small);

    /* background subtraction -> foreground mask */
    bgsub->apply(gpuIn, gpuFg);

    /* morphology op - closing to connect blobs (GPU processed) */
    static cv::Ptr<cv::cuda::Filter> morph = cv::cuda::createMorphologyFilter(
        cv::MORPH_CLOSE, gpuFg_.type(), cv::Mat::ones(3, 3, CV_8U));
    morph->apply(gpuFg, gpuFg);

    /* download mask & find contours (CPU processed) */
    cv::Mat fgCpu;
    gpuFg.download(fgCpu);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgCpu, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    outRois.clear();
    for (auto& c : contours) {
        const double area = cv::contourArea(c);
        if (area < minArea)
            continue;
        outRois.emplace_back(cv::boundingRect(c));
    }

    if (!outRois.empty())
        toOriginalScale(outRois, scale, frame.cols, frame.rows);

    return !outRois.empty();
}

} //namespace pipeline