#pragma once
#include <opencv4/opencv2/core/cuda.hpp>
#include "memoryPool.hpp"

namespace cuda_components {

    class CUDAPreprocessor {
        public:
            CUDAPreprocessor(int inputWidth, int inputHeight, int poolSize = 5);
            ~CUDAPreprocessor();

            cv::cuda::GpuMat process(const cv::cuda::GpuMat& inputGpu, int targetWidth, int targetHeight, bool grayscale = false, bool equalize = false);
            cv::cuda::GpuMat processBGR(const cv::cuda::GpuMat& input, int targetWidth, int targetHeight);
            cv::cuda::GpuMat processGrayscale(const cv::cuda::GpuMat& input, int targetWidth, int targetHeight, bool equalize = false);

            cv::Mat download(const cv::cuda::GpuMat& gpuMat);
            cv::cuda::GpuMat upload(const cv::Mat& input);
        
        private:
            std::unique_ptr<CUDAMemoryPool> colorPool;
            std::unique_ptr<CUDAMemoryPool> grayPool;
            void launchPreprocessKernel(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, bool grayscale, bool equalize);
    };
}
