#pragma once

#include <opencv2/opencv.hpp>
#include <mutex>

namespace pipeline {
    
    struct TrackedRegion {
        int identifier;
        cv::Rect rect;
        cv::Point2f centroid;

        int frameAge = 0;
        int consecutiveFrames = 0;
        int framesSinceSeen = 0;
        bool continuityConfirmed = false;

        cv::KalmanFilter kalman;
        cv::Point2f predictedCentroid;
        cv::Point2f velocity;
        bool kalmanState = false;
        TrackedRegion() : kalman(4,2,0, CV_32F) {
            kalmanState = false;
        }
    };

    struct DetectionParameters {
        double adaptiveThresholdBase;
        double minContourArea;
        std::pair<double, double> minMaxAspectRatio;
        
        int samplingRate;
        int maxLostFrames;
        int minConsecutiveFrames;
        int nextTrackIdentifier; 
    };

    class PipelineInitialDetection {
        public:
            PipelineInitialDetection(int samplingRate = 1);
            ~PipelineInitialDetection();

            bool detectedPotentialMotion(const cv::Mat& frame);
            bool detectedPotentialMotion(const cv::Mat& frame, std::vector<cv::Rect>& confirmedMotionRects);

        private:
            DetectionParameters state;
            std::vector<TrackedRegion> trackedRegions;
            std::vector<cv::Rect> regionOfInterestAreas;
            std::mutex frameMtx;

            cv::Mat previousFrame;
            bool useRoi = false;

            cv::Mat getAdaptiveThresholdedDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame);
            cv::Mat prepareFrameForDifferencing(const cv::Mat& frame);
            cv::Rect computeActiveRegion(const cv::Mat& frame);

            double calculateFrameDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame);
            void findMotionRegions(const cv::Mat& thresholdedDifference, std::pair<std::vector<cv::Rect>, std::vector<cv::Point2f>>& motionRectsAndCentroids);
            void updateTrackedRegions(const std::vector<cv::Rect>& currentRects, const std::vector<cv::Point2f>& currentCentroids);
            void updateTracking(cv::Rect newRect, cv::Point2f newCentroid);
            void configure(bool useRoi, const std::vector<cv::Rect>& regionsOfInterest);
            void initializeKalmanFilter(TrackedRegion& region);
            
    };
}