#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <mutex>

namespace pipeline {
    class PipelineInitialDetectionLite {
    public:
        explicit PipelineInitialDetectionLite(int minAreaPx = 800,
            float downScale = 0.5f,
            int history = 120,
            int detectInterval = 2);

        bool detect(const cv::Mat& frame, std::vector<cv::Rect>& outRois);

    private:
        cv::Mat convertGrayFaster(const cv::Mat& in);    
        void toOriginalScale(std::vector<cv::Rect>& rois, double scale, int maxW, int maxH);
        
        int minArea;
        float scale;
        int frameInterval;
        int frameCounter;

        /* state */
        cv::Ptr<cv::cuda::BackgroundSubtractorFGD> bgsub_;   // GPU bg‑subtract
        cv::cuda::GpuMat gpuIn_, gpuFg_;

        std::mutex mtx; 
    };
} // namespace pipeline