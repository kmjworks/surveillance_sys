#pragma once 

#include <mutex>
#include <vector>
#include <opencv4/opencv2/core/cuda.hpp>

namespace cuda_components {
    class CUDAMemoryPool {
        public:
            CUDAMemoryPool(int width, int height, int type, int poolSize = 10) : width(width), height(height), type(type) {
                for(int i = 0; i < poolSize; ++i) availableMats.emplace_back(height, width, type);
            }

            cv::cuda::GpuMat acquire() {
                std::lock_guard<std::mutex> lock(memPoolMtx);
                if(availableMats.empty()) availableMats.emplace_back(height, width, type);

                cv::cuda::GpuMat mat = availableMats.back();
                availableMats.pop_back();
                return mat;
            }

            void release(cv::cuda::GpuMat& mat) {
                std::lock_guard<std::mutex> lock(memPoolMtx);
                availableMats.push_back(mat);
            }


        private:
            std::vector<cv::cuda::GpuMat> availableMats;
            std::mutex memPoolMtx;
            int width, height, type;
    };
}
