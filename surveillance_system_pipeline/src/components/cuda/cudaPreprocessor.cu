#include "cudaPreprocessor.hpp"

#include <memory>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>


namespace cuda_components {
    __global__ void preprocessingKernel(const uchar* input, uchar* output, int sourceWidth, int sourceHeight, int sourceStep, int targetWidth, int targetHeight, int targetStep, int channels, bool grayscale, bool equalize) {
    
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if(x >= targetWidth || y >= targetHeight) return;
    
        const float scaleX = static_cast<float>(sourceWidth) / targetWidth;
        const float scaleY = static_cast<float>(sourceHeight) / targetHeight;
    
        const float srcX = x * scaleX;
        const float srcY = y * scaleY;
    
        const int srcX0 = __float2int_rd(srcX);
        const int srcY0 = __float2int_rd(srcY);
        const int srcX1 = min(srcX0 + 1, sourceWidth - 1);
        const int srcY1 = min(srcY0 + 1, sourceHeight - 1);
    
        const float weightX = srcX - srcX0;
        const float weightY = srcY - srcY0;
    
        if(grayscale && channels == 3) {
            uchar r = 0;
    
            for (int c = 0; c < 3; c++) {
                const uchar topLeft     = input[srcY0 * sourceStep + srcX0 * 3 + c];
                const uchar topRight    = input[srcY0 * sourceStep + srcX1 * 3 + c];
                const uchar bottomLeft  = input[srcY1 * sourceStep + srcX0 * 3 + c];
                const uchar bottomRight = input[srcY1 * sourceStep + srcX1 * 3 + c];
    
                float pixel = (1 - weightX) * (1 - weightY) * topLeft + 
                               weightX * (1 - weightY) * topRight +
                               (1 - weightX) * weightY * bottomLeft + 
                               weightX * weightY * bottomRight;
    
                if (c == 0) pixel *= 0.114f;      // ch B
                else if (c == 1) pixel *= 0.587f; // ch G
                else pixel *= 0.299f;             // ch R
                
                r += pixel;
            }
    
            if(equalize) {
                output[y * targetStep + x] = min(255, max(0, r));
            } else {
                output[y * targetStep + x] = r;
            }
        } else if (channels == 3) {
            for (int c = 0; c < 3; c++) {
                const uchar topLeft     = input[srcY0 * sourceStep + srcX0 * 3 + c];
                const uchar topRight    = input[srcY0 * sourceStep + srcX1 * 3 + c];
                const uchar bottomLeft  = input[srcY1 * sourceStep + srcX0 * 3 + c];
                const uchar bottomRight = input[srcY1 * sourceStep + srcX1 * 3 + c];
                
                
                uchar pixel = (1 - weightX) * (1 - weightY) * topLeft + 
                              weightX * (1 - weightY) * topRight +
                              (1 - weightX) * weightY * bottomLeft + 
                              weightX * weightY * bottomRight;
                              
                output[y * targetStep + x * 3 + c] = pixel;
            }
        } else if (channels == 1) {
            const uchar topLeft     = input[srcY0 * sourceStep + srcX0];
            const uchar topRight    = input[srcY0 * sourceStep + srcX1];
            const uchar bottomLeft  = input[srcY1 * sourceStep + srcX0];
            const uchar bottomRight = input[srcY1 * sourceStep + srcX1];
            
            
            uchar pixel = (1 - weightX) * (1 - weightY) * topLeft + 
                          weightX * (1 - weightY) * topRight +
                          (1 - weightX) * weightY * bottomLeft + 
                          weightX * weightY * bottomRight;
    
            if(equalize) {
                output[y * targetStep + x] = min(255, max(0, pixel));
            } else {
                output[y * targetStep + x] = pixel;
            }
        }
    }

    CUDAPreprocessor::CUDAPreprocessor(int inputWidth, int inputHeight, int poolSize) {
        colorPool = std::make_unique<CUDAMemoryPool>(inputWidth, inputHeight, CV_8UC3, poolSize);
        grayPool = std::make_unique<CUDAMemoryPool>(inputWidth, inputHeight, CV_8UC1, poolSize);
    }

    
    CUDAPreprocessor::~CUDAPreprocessor() {
    
    }

    cv::cuda::GpuMat CUDAPreprocessor::upload(const cv::Mat& input) {
        cv::cuda::GpuMat gpuMat(input);
        cv::cuda::GpuMat gpuMatFromPool = colorPool->acquire();
        gpuMat.copyTo(gpuMatFromPool);

        return gpuMat;
    }

    cv::cuda::GpuMat CUDAPreprocessor::process(const cv::cuda::GpuMat& inputGpu, int targetWidth, int targetHeight, bool grayscale, bool equalize) {
        cv::cuda::GpuMat output;
        if(grayscale) {
            output = grayPool->acquire();
        } else {
            output = colorPool->acquire();
        }

        if (output.cols != targetWidth || output.rows != targetHeight) {
            int type = grayscale ? CV_8UC1 : CV_8UC3;
            output.create(targetHeight, targetWidth, type);
        }

        dim3 blockDim(16, 16);
        dim3 gridDim((targetWidth + blockDim.x - 1) / blockDim.x, (targetHeight + blockDim.y - 1) / blockDim.y);

        cuda_components::preprocessingKernel<<<gridDim, blockDim>>>(
            inputGpu.data, output.data,
            inputGpu.cols, inputGpu.rows, inputGpu.step,
            targetWidth, targetHeight, output.step,
            inputGpu.channels(), grayscale, equalize
        );
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) throw std::runtime_error("CUDA kernel err: " + std::string(cudaGetErrorString(err)));
        cudaDeviceSynchronize();

        return output;
    }

    cv::cuda::GpuMat CUDAPreprocessor::processBGR(const cv::cuda::GpuMat& input, int targetWidth, int targetHeight) {
        return process(input, targetWidth, targetHeight, false, false);
    }

    cv::cuda::GpuMat CUDAPreprocessor::processGrayscale(const cv::cuda::GpuMat& input, int targetWidth, int targetHeight, bool equalize) {
        return process(input, targetWidth, targetHeight, true, equalize);
    }

    cv::Mat CUDAPreprocessor::download(const cv::cuda::GpuMat& gpuMat) {
        cv::Mat cpuMat;
        gpuMat.download(cpuMat);
        return cpuMat;
    }

    void CUDAPreprocessor::launchPreprocessKernel(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output, bool grayscale, bool equalize) {
      dim3 blockDim(16, 16);
      dim3 gridDim((output.cols + blockDim.x - 1) / blockDim.x,
                   (output.rows + blockDim.y - 1) / blockDim.y);

      preprocessingKernel<<<gridDim, blockDim>>>(
          input.data, output.data, input.cols, input.rows, input.step,
          output.cols, output.rows, output.step, input.channels(), grayscale,
          equalize);
    }

} //namespace