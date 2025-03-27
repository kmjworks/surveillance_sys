#include "pipelineInitialDetection.hpp"
#include <ros/ros.h>
#include <numeric>
#include <limits>

namespace pipeline {

    PipelineInitialDetection::PipelineInitialDetection(int samplingRate) :
        state{25.0, 500.0, std::make_pair(4.0, 0.25), samplingRate, 5, 3, 0} {
            ROS_INFO("Pipeline initial motion detector  initialized.");
        }


    PipelineInitialDetection::~PipelineInitialDetection() {}

    void PipelineInitialDetection::configure(bool useRoi, const std::vector<cv::Rect>& regionsOfInterest) {
        std::lock_guard<std::mutex> lock(frameMtx);
        this->useRoi = useRoi;
        this->regionOfInterestAreas = regionsOfInterest;
    }

    bool PipelineInitialDetection::detectedPotentialMotion(const cv::Mat& frame) {
        std::vector<cv::Rect> unused;
        return detectedPotentialMotion(frame, unused);
    }

    bool PipelineInitialDetection::detectedPotentialMotion(const cv::Mat& frame, std::vector<cv::Rect>& confirmedMotionRects) {
        if(frame.empty()) {
            ROS_WARN_THROTTLE(5.0, "detectPotentialMotion - Received empty frame.");
            return false;
        }

        std::lock_guard<std::mutex> lock(frameMtx);

        cv::Mat currentFrame = prepareFrameForDifferencing(frame);

        if(previousFrame.empty()) {
            currentFrame.copyTo(previousFrame);
            return false;
        }

        cv::Mat binaryDifference = getAdaptiveThresholdedDifference(currentFrame, previousFrame);
        std::pair<std::vector<cv::Rect>, std::vector<cv::Point2f>> potentialRectsAndCentroids;

        findMotionRegions(binaryDifference, potentialRectsAndCentroids);
        updateTrackedRegions(potentialRectsAndCentroids.first, potentialRectsAndCentroids.second);

        confirmedMotionRects.clear();
        for(const auto& region : trackedRegions) {
            if(region.continuityConfirmed) {
                cv::Rect defaultScaleRect = region.rect;
                defaultScaleRect.x *= 2;
                defaultScaleRect.y *= 2;
                defaultScaleRect.width *= 2;
                defaultScaleRect.height *= 2;
                confirmedMotionRects.push_back(defaultScaleRect);
            }
        }

        previousFrame = currentFrame.clone();

        return !confirmedMotionRects.empty();
    }
    
    cv::Mat PipelineInitialDetection::prepareFrameForDifferencing(const cv::Mat& frame) {
        cv::Mat resultingFrame, downsampledFrame;

        cv::resize(frame, downsampledFrame, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

        if(downsampledFrame.channels() > 1) {
            cv::cvtColor(downsampledFrame, resultingFrame, cv::COLOR_BGR2GRAY);
        } else {
            frame.copyTo(resultingFrame);
        }

        cv::GaussianBlur(resultingFrame, resultingFrame, cv::Size(5, 5), 0);

        return resultingFrame;
    }

    double PipelineInitialDetection::calculateFrameDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame) {
        cv::Mat diff;
        cv::absdiff(currentFrame, previousFrame, diff);

        cv::Mat thresholdedDifference;
        cv::threshold(diff, thresholdedDifference, 25, 255, cv::THRESH_BINARY);

        int nonzeroPx = cv::countNonZero(thresholdedDifference);

        double frameSize = currentFrame.rows * currentFrame.cols;

        return static_cast<double>(nonzeroPx) / frameSize;
    }

    cv::Mat PipelineInitialDetection::getAdaptiveThresholdedDifference(const cv::Mat& currentFrame, const cv::Mat& previousFrame) {
        cv::Mat diff, thresholdedDifference;
        cv::absdiff(currentFrame, previousFrame, diff);

        double avgBrightness = cv::mean(currentFrame)[0];
        int adaptiveThresholdValue = static_cast<int>(std::max(10.0, std::min(50.0, state.adaptiveThresholdBase + (avgBrightness - 128.0) * 0.1)));
        cv::threshold(diff, thresholdedDifference, adaptiveThresholdValue, 255, cv::THRESH_BINARY);

        if(useRoi && !regionOfInterestAreas.empty()) {
            cv::Mat mask = cv::Mat::zeros(thresholdedDifference.size(), CV_8UC1);
            for(const auto& roi : regionOfInterestAreas) {
                cv::Rect scaledRoi = roi;
                scaledRoi.x /= 2; scaledRoi.y /= 2;
                scaledRoi.width /=2; scaledRoi.height /=2;
                cv::rectangle(mask, scaledRoi, cv::Scalar(255), -1);
            }

            cv::bitwise_and(thresholdedDifference, mask, thresholdedDifference);
        }
        //cv::Mat kernel3by3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::Mat kernel5by5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
        cv::dilate(thresholdedDifference, thresholdedDifference, kernel5by5);
        cv::morphologyEx(thresholdedDifference, thresholdedDifference, cv::MORPH_CLOSE, kernel5by5);

        return thresholdedDifference;
    }
    
    void PipelineInitialDetection::findMotionRegions(const cv::Mat& thresholdedDifference, std::pair<std::vector<cv::Rect>, std::vector<cv::Point2f>>& motionRectsAndCentroids) {
        motionRectsAndCentroids.first.clear(); motionRectsAndCentroids.second.clear();

        std::vector<std::vector<cv::Point>> contoursForDetection;
        cv::findContours(thresholdedDifference, contoursForDetection, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for(const auto& contour : contoursForDetection) {
            double area = cv::contourArea(contour);
            if(area < state.minContourArea) continue;

            cv::Rect boundingRect = cv::boundingRect(contour);
            if(boundingRect.height == 0) continue;

            double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
            if(aspectRatio < state.minMaxAspectRatio.first || aspectRatio > state.minMaxAspectRatio.second) continue;

            motionRectsAndCentroids.first.push_back(boundingRect);
            cv::Moments mu = cv::moments(contour);
            if(mu.m00 == 0) {
                motionRectsAndCentroids.second.push_back(cv::Point2f(static_cast<float>(boundingRect.x + boundingRect.width / 2.0), 
                                                                    static_cast<float>(boundingRect.y + boundingRect.height / 2.0)));
            } else {
                motionRectsAndCentroids.second.push_back(cv::Point2f(static_cast<float>(mu.m10 / mu.m00), 
                                                                    static_cast<float>(mu.m01 / mu.m00)));
            }
        }
    }

    void PipelineInitialDetection::updateTrackedRegions(const std::vector<cv::Rect>& currentRects, const std::vector<cv::Point2f>& currentCentroids) {
        const double maxCentroidDistance = 30.0;

        for(auto& region : trackedRegions) {
            region.framesSinceSeen++;
        }

        std::vector<bool> currentMatched(currentRects.size(), false);
        std::vector<int> trackIndices(trackedRegions.size());
        std::iota(trackIndices.begin(), trackIndices.end(), 0);

        for (int i = 0; i < trackedRegions.size(); ++i) {
            if(trackedRegions[i].framesSinceSeen > state.maxLostFrames +1) continue;

            double minDistance = std::numeric_limits<double>::max();
            int bestMatchingIdentifier = -1;

            for(int j = 0; j < currentRects.size(); ++j) {
                double dist = cv::norm(trackedRegions[i].centroid - currentCentroids[j]);
                if(dist < maxCentroidDistance && dist < minDistance) {
                    minDistance = dist;
                    bestMatchingIdentifier = j;
                }
            }

            if(bestMatchingIdentifier != -1) {
                trackedRegions[i].rect = currentRects[bestMatchingIdentifier];
                trackedRegions[i].centroid = currentCentroids[bestMatchingIdentifier];
                trackedRegions[i].frameAge++;
                trackedRegions[i].consecutiveFrames++;
                trackedRegions[i].framesSinceSeen = 0;
                currentMatched[bestMatchingIdentifier] = true;

                if(!trackedRegions[i].continuityConfirmed && trackedRegions[i].consecutiveFrames >= state.minConsecutiveFrames) {
                    trackedRegions[i].continuityConfirmed = true;
                } else {
                    trackedRegions[i].consecutiveFrames = 0;
                    trackedRegions[i].continuityConfirmed = false;
                }
            }

            for (int j = 0; j < currentRects.size(); ++j) {
                if (!currentMatched[j]) {
                    TrackedRegion newRegion;
                    newRegion.identifier = state.nextTrackIdentifier++;
                    newRegion.rect = currentRects[j];
                    newRegion.centroid = currentCentroids[j];
                    newRegion.frameAge = 1;
                    newRegion.consecutiveFrames = 1;
                    newRegion.framesSinceSeen = 0;
                    newRegion.continuityConfirmed = (state.minConsecutiveFrames <= 1);
                    trackedRegions.push_back(newRegion);
                }
            }

            trackedRegions.erase(
                std::remove_if(trackedRegions.begin(), trackedRegions.end(),
                    [this](const TrackedRegion& region) {
                        bool remove = region.framesSinceSeen > state.maxLostFrames;
                        return remove;
                    }),
                    trackedRegions.end()
            );
        }
    }
}