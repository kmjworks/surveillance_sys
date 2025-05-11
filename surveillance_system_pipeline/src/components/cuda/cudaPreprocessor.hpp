#pragma once
#include <opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include "memoryPool.hpp"

namespace cuda_components {

    struct EnhancementParameters {
        bool enableCLAHE;
        bool prefilterNoise;
        bool markEnhancedDebug;
        double claheClipLim;
        float denoiseStrength;
        int claheTileSize;


    };

    class CUDAPreprocessor {
        public:
            CUDAPreprocessor(int inputWidth, int inputHeight, int poolSize = 5);
            ~CUDAPreprocessor();

            cv::Mat download(const cv::cuda::GpuMat& gpuMat);
            cv::cuda::GpuMat upload(const cv::Mat& input);
            void release(cv::cuda::GpuMat& mat);

            bool isGrayscale(const cv::Mat& input) const;
            bool isGrayscale(const cv::cuda::GpuMat& input) const;

            cv::cuda::GpuMat processFrame(const cv::cuda::GpuMat& input);
            cv::Mat processFrame(const cv::Mat& input);
        
        private:
            std::unique_ptr<CUDAMemoryPool> colorPool;
            std::unique_ptr<CUDAMemoryPool> grayPool;
            cv::Ptr<cv::cuda::CLAHE> clahe;
            EnhancementParameters config;

            cv::cuda::GpuMat enhanceGrayscale(const cv::cuda::GpuMat& input);
            cv::cuda::GpuMat denoise(const cv::cuda::GpuMat& input, float strength);
            void initCLAHE(double clipLimit, int tileSize);
    };
}
