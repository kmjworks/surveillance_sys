#include "cudaPreprocessor.hpp"

#include <memory>
#include <iostream>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>


namespace cuda_components {
    
    __global__ void preprocessingKernel(const uchar* input, uchar* output, int sourceWidth, int sourceHeight, int sourceStep, int targetWidth, int targetHeight, int targetStep, int channels, bool grayscale, bool equalize, bool denoise, float denoiseStrength) {
    
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

            for(int c = 0; c < 3; ++c) {
                const uchar topLeft = input[srcY0 * sourceStep + srcX0 * 3 + c];
                const uchar topRight = input[srcY0 * sourceStep + srcX1 * 3 + c];
                const uchar bottomLeft = input[srcY1 * sourceStep + srcX0 * 3 + c];
                const uchar bottomRight = input[srcY1 * sourceStep + srcX1 * 3 + c];

                float pixel = (1 - weightX) * (1 - weightY) * topLeft + weightX * (1 - weightY) * topRight + (1 - weightX) * weightY * bottomLeft + weightX * weightY * bottomRight;

                if (c == 0) pixel *= 0.114f;      // ch B 
                else if (c == 1) pixel *= 0.587f; // ch G
                else pixel *= 0.299f;             // ch R

                r += pixel;
            }

            if(equalize) {
                output[y*targetStep+x] = min(255, max(0,r));
            } else {
                output[y*targetStep+x] = r;
            }
        } else if (channels == 3) {
            for (int c = 0; c < 3; c++) {
                const uchar topLeft = input[srcY0 * sourceStep + srcX0 * 3 + c];
                const uchar topRight = input[srcY0 * sourceStep + srcX1 * 3 + c];
                const uchar bottomLeft = input[srcY1 * sourceStep + srcX0 * 3 + c];
                const uchar bottomRight = input[srcY1 * sourceStep + srcX1 * 3 + c];
                
                uchar pixel = (1 - weightX) * (1 - weightY) * topLeft + weightX * (1 - weightY) * topRight + (1 - weightX) * weightY * bottomLeft + weightX * weightY * bottomRight;
                output[y * targetStep + x * 3 + c] = pixel;
            }
        } else if (channels == 1) {
            float pixel = 0.0f; // Grayscale

            const uchar topLeft = input[srcY0 * sourceStep + srcX0];
            const uchar topRight = input[srcY0 * sourceStep + srcX1];
            const uchar bottomLeft = input[srcY1 * sourceStep + srcX0];
            const uchar bottomRight = input[srcY1 * sourceStep + srcX1];

            pixel = (1 - weightX) * (1 - weightY) * topLeft +
                 weightX * (1 - weightY) * topRight +
                (1 - weightX) * weightY * bottomLeft +
                 weightX * weightY * bottomRight;

            if(denoise) {
                /* adaptive bilateral filtering */
                float totalWeight = 0.0f;
                float weightedSum = 0.0f;
                const float spatialSigma = 1.0f * denoiseStrength;
                const float colorSigma = 30.0f * denoiseStrength;

                /* 3x3 neighbouring */
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = min(max(srcX0 + dx, 0), sourceWidth - 1);
                        int ny = min(max(srcY0 + dy, 0), sourceHeight - 1);

                        float neigh = input[ny*sourceStep+nx];
                        float spatialDist = dx*dx+dy*dy;
                        float colorDist = (neigh - pixel) * (neigh - pixel);


                        float spatialWeight = exp(-spatialDist) / (2.0f*powf(spatialSigma, 2));
                        float colorWeight = exp(-colorDist / (2.0f * colorSigma * colorSigma));
                        float weight = spatialWeight * colorWeight;

                        weightedSum += neigh*weight;
                        totalWeight += weight;
                    }
                }

                if(totalWeight > 0.0f) {
                    pixel = weightedSum / totalWeight;
                }
            }

            if(equalize) {
                output[y*targetStep+x] = min(255, max(0, (int)pixel));
            } else {
                output[y*targetStep+x] = (uchar)pixel;
            }
        }
    }

    CUDAPreprocessor::CUDAPreprocessor(int inputWidth, int inputHeight, int poolSize) {
        colorPool = std::make_unique<CUDAMemoryPool>(inputWidth, inputHeight, CV_8UC3, poolSize);
        grayPool = std::make_unique<CUDAMemoryPool>(inputWidth, inputHeight, CV_8UC1, poolSize);

        

        config.enableCLAHE = true;
        config.prefilterNoise = true;
        config.markEnhancedDebug = false;
    
        config.claheClipLim = 2.0;
        config.denoiseStrength = 1.0f;
        config.claheTileSize = 8;

        initCLAHE(config.claheClipLim, config.claheTileSize);
    }

    void CUDAPreprocessor::initCLAHE(double clipLimit, int tileSize) {
        clahe = cv::cuda::createCLAHE(clipLimit, cv::Size(tileSize, tileSize));
    }

    
    CUDAPreprocessor::~CUDAPreprocessor() {
    
    }

    cv::cuda::GpuMat CUDAPreprocessor::upload(const cv::Mat& input) {

        cv::cuda::GpuMat tmp(input);
        
        cv::cuda::GpuMat gpuMat = (input.channels() == 1) ? grayPool->acquire() : colorPool->acquire();
        tmp.copyTo(gpuMat);

        return gpuMat;
    }

    cv::Mat CUDAPreprocessor::download(const cv::cuda::GpuMat& gpuMat) {
        cv::Mat cpuMat;
        gpuMat.download(cpuMat);
        return cpuMat;
    }

    void CUDAPreprocessor::release(cv::cuda::GpuMat& mat) {
        if (mat.channels() == 1) {
            grayPool->release(mat);
        } else {
            colorPool->release(mat);
        }
    }

    bool CUDAPreprocessor::isGrayscale(const cv::Mat& input) const {
        return input.channels() == 1;
    }
    
    bool CUDAPreprocessor::isGrayscale(const cv::cuda::GpuMat& input) const {
        return input.channels() == 1;
    }

    cv::cuda::GpuMat CUDAPreprocessor::processFrame(const cv::cuda::GpuMat& input) {
        if(isGrayscale(input)) {
            return enhanceGrayscale(input);
        } else {
            cv::cuda::GpuMat out;
            input.copyTo(out);

            return out; 
        }
    }

    cv::Mat CUDAPreprocessor::processFrame(const cv::Mat& input) {
        cv::cuda::GpuMat gpuInput = upload(input);
        cv::cuda::GpuMat gpuOut = processFrame(gpuInput);
        cv::Mat output = download(gpuOut);

        return output;
    }

    cv::cuda::GpuMat CUDAPreprocessor::enhanceGrayscale(const cv::cuda::GpuMat& input) {
    if (!config.enableCLAHE) {
        cv::cuda::GpuMat output;
        input.copyTo(output);
        return output;
    }
    
    cv::cuda::GpuMat preprocessed;
    if (config.prefilterNoise) {
        preprocessed = denoise(input, config.denoiseStrength);
    } else {
        preprocessed = input;
    }
    
    if (clahe->getClipLimit() != config.claheClipLim) {
        initCLAHE(config.claheClipLim, config.claheTileSize);
    }
    
    cv::cuda::GpuMat enhanced = grayPool->acquire();
    clahe->apply(preprocessed, enhanced);
    
    return enhanced;
}

cv::cuda::GpuMat CUDAPreprocessor::denoise(const cv::cuda::GpuMat& input, float strength) {
    cv::cuda::GpuMat output = isGrayscale(input) ? grayPool->acquire() : colorPool->acquire();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);
    

    preprocessingKernel<<<gridSize, blockSize>>>(
        input.data, output.data,
        input.cols, input.rows, input.step,
        input.cols, input.rows, output.step,
        input.channels(), 
        isGrayscale(input),    
        false,                 
        true,                  
        strength               
    );
    
    cudaDeviceSynchronize();
    return output;
}


} //namespace