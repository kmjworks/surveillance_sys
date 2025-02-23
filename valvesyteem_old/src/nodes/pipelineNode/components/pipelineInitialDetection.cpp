#include "pipelineInitialDetection.hpp"
#include <ros/ros.h>
#include <numeric>
#include <limits>

namespace pipeline {

    PipelineInitialDetection::PipelineInitialDetection(int samplingRate) :
        state{20.0, 300.0, std::make_pair(0.2, 7.0), samplingRate, 7, 7, 0} {
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

        const float scalingFactor = 0.5f;
        const int beforeScalingWidth = frame.cols;
        const int beforeScalingHeight = frame.rows;

        cv::Rect activeRegion = computeActiveRegion(frame);
        cv::Mat roiAdjustedFrame;

        if(activeRegion.width == frame.cols && activeRegion.height == frame.rows) {
            roiAdjustedFrame = prepareFrameForDifferencing(frame);
        } else {
            cv::Mat frameROI = frame(activeRegion);
            roiAdjustedFrame = prepareFrameForDifferencing(frameROI);
        }



        if (previousFrame.empty() || 
            previousFrame.cols != roiAdjustedFrame.cols || 
            previousFrame.rows != roiAdjustedFrame.rows) {
            roiAdjustedFrame.copyTo(previousFrame);
            return false;
        }

        cv::Mat binaryDifference = getAdaptiveThresholdedDifference(roiAdjustedFrame, previousFrame);
        std::pair<std::vector<cv::Rect>, std::vector<cv::Point2f>> potentialRectsAndCentroids;

        findMotionRegions(binaryDifference, potentialRectsAndCentroids);
        
        if (!(activeRegion.width == frame.cols && activeRegion.height == frame.rows)) {
            for (auto& rect : potentialRectsAndCentroids.first) {
                rect.x += activeRegion.x / 2;  
                rect.y += activeRegion.y / 2;
            }
            
            for (auto& point : potentialRectsAndCentroids.second) {
                point.x += activeRegion.x / 2;
                point.y += activeRegion.y / 2;
            }
        }

        updateTrackedRegions(potentialRectsAndCentroids.first, potentialRectsAndCentroids.second);

        confirmedMotionRects.clear();
        for(const auto& region : trackedRegions) {
            if(region.continuityConfirmed) {

                cv::Rect scaledRect;
                scaledRect.x = static_cast<int>(region.rect.x / scalingFactor);
                scaledRect.y = static_cast<int>(region.rect.y / scalingFactor);
                scaledRect.width = static_cast<int>(region.rect.width / scalingFactor);
                scaledRect.height = static_cast<int>(region.rect.height / scalingFactor);

                scaledRect.x = std::max(0, std::min(scaledRect.x, beforeScalingWidth - 1));
                scaledRect.y = std::max(0, std::min(scaledRect.y, beforeScalingHeight - 1));
                scaledRect.width = std::min(scaledRect.width, beforeScalingWidth - scaledRect.x);
                scaledRect.height = std::min(scaledRect.height, beforeScalingHeight - scaledRect.y);

                confirmedMotionRects.push_back(scaledRect);
            }
        }

        roiAdjustedFrame.copyTo(previousFrame);

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
        const double maxCentroidDistance = 50.0;

        for(auto& region : trackedRegions) {
            if(!region.kalmanState) {
                initializeKalmanFilter(region);
            } else {
                cv::Mat prediction = region.kalman.predict();
                region.predictedCentroid.x = prediction.at<float>(0);
                region.predictedCentroid.y = prediction.at<float>(1);
                region.velocity.x = prediction.at<float>(2);
                region.velocity.y = prediction.at<float>(3);
            }
        }

        std::vector<bool> currentMatched(currentRects.size(), false);

        for (int i = 0; i < trackedRegions.size(); ++i) {
            if(trackedRegions[i].framesSinceSeen > state.maxLostFrames +1) continue;

            double minDistance = std::numeric_limits<double>::max();
            int bestMatchingIdentifier = -1;

            for(int j = 0; j < currentRects.size(); ++j) {
                if(currentMatched[j]) continue;

                cv::Point2f comparePoint = trackedRegions[i].kalmanState ? trackedRegions[i].predictedCentroid : trackedRegions[i].centroid;

                double dist = cv::norm(comparePoint - currentCentroids[j]);
                if(dist < maxCentroidDistance && dist < minDistance) {
                    minDistance = dist;
                    bestMatchingIdentifier = j;
                }
            }

            if(bestMatchingIdentifier != -1) {

                cv::Mat measurement = (cv::Mat_<float>(2, 1) << currentCentroids[bestMatchingIdentifier].x, currentCentroids[bestMatchingIdentifier].y);
                
                if(trackedRegions[i].kalmanState) {
                    cv::Mat corrected = trackedRegions[i].kalman.correct(measurement);
                    trackedRegions[i].centroid.x = corrected.at<float>(0);
                    trackedRegions[i].centroid.y = corrected.at<float>(1);

                    float alpha = 0.7f;
                    trackedRegions[i].rect.width = alpha * currentRects[bestMatchingIdentifier].width +  (1-alpha) * trackedRegions[i].rect.width;
                    trackedRegions[i].rect.height = alpha * currentRects[bestMatchingIdentifier].height + (1-alpha) * trackedRegions[i].rect.height;
                } else {
                    trackedRegions[i].centroid = currentCentroids[bestMatchingIdentifier];
                    trackedRegions[i].rect = currentRects[bestMatchingIdentifier];
                }

                
                trackedRegions[i].frameAge++;
                trackedRegions[i].consecutiveFrames++;
                trackedRegions[i].framesSinceSeen = 0;
                currentMatched[bestMatchingIdentifier] = true;

                if(!trackedRegions[i].continuityConfirmed && trackedRegions[i].consecutiveFrames >= state.minConsecutiveFrames) {
                    trackedRegions[i].continuityConfirmed = true;
                    ROS_INFO("Track %d confirmed", trackedRegions[i].identifier);

                } else if(trackedRegions[i].kalmanState && trackedRegions[i].framesSinceSeen <= state.maxLostFrames / 2) {

                    trackedRegions[i].centroid = trackedRegions[i].predictedCentroid;

                    if (trackedRegions[i].framesSinceSeen > 2) {
                        trackedRegions[i].consecutiveFrames = 0;

                        if(trackedRegions[i].continuityConfirmed) {
                            trackedRegions[i].continuityConfirmed = false;
                        }

                    }
                }
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

                initializeKalmanFilter(newRegion);
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


    void PipelineInitialDetection::initializeKalmanFilter(TrackedRegion& region) {
        float dt = 1.0f;

        region.kalman.transitionMatrix = (cv::Mat_<float>(4,4) <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1
        );

        region.kalman.measurementMatrix = (cv::Mat_<float>(2,4) <<
            1, 0, 0, 0,
            0, 1, 0, 0
        );

        float processNoise = 1e-4f;
        cv::setIdentity(region.kalman.processNoiseCov, cv::Scalar::all(processNoise));

        float measurementNoise = 1e-1f;
        cv::setIdentity(region.kalman.measurementNoiseCov, cv::Scalar::all(measurementNoise));

        cv::setIdentity(region.kalman.errorCovPost, cv::Scalar::all(1));

        region.kalman.statePost.at<float>(0) = region.centroid.x;
        region.kalman.statePost.at<float>(1) = region.centroid.y;
        region.kalman.statePost.at<float>(2) = 0;
        region.kalman.statePost.at<float>(3) = 0;

        region.kalmanState = true;
        
    }

    cv::Rect PipelineInitialDetection::computeActiveRegion(const cv::Mat& frame) {
        cv::Rect activeRegion(0, 0, frame.cols, frame.rows);

        bool trackingActive = false;
        int minX = frame.cols, minY = frame.rows;
        int maxX = 0, maxY = 0;

        for(const auto& region : trackedRegions) {
            if (region.framesSinceSeen <= state.maxLostFrames / 2) {
                trackingActive = true;

                cv::Point2f trackPos = region.kalmanState && region.framesSinceSeen > 0 ? region.predictedCentroid : region.centroid;
                
                float vxAbs = region.kalmanState ? std::abs(region.velocity.x) : 0;
                float vyAbs = region.kalmanState ? std::abs(region.velocity.y) : 0;

                int xMargin = static_cast<int>(std::max(region.rect.width / 2.0f, vxAbs * 3.0f));
                int yMargin = static_cast<int>(std::max(region.rect.height / 2.0f, vyAbs * 3.0f));
                
                minX = std::min(minX, static_cast<int>(trackPos.x - xMargin));
                minY = std::min(minY, static_cast<int>(trackPos.y - yMargin));
                maxX = std::max(maxX, static_cast<int>(trackPos.x + xMargin));
                maxY = std::max(maxY, static_cast<int>(trackPos.y + yMargin));
            }
        }

        if(trackingActive) {
            const int fixedMargin = 50;
            minX = std::max(0, minX - fixedMargin);
            minY = std::max(0, minY - fixedMargin);
            maxX = std::min(frame.cols, maxX + fixedMargin);
            maxY = std::min(frame.rows, maxY + fixedMargin);


            if ((maxX - minX) * (maxY - minY) < 0.7 * frame.cols * frame.rows) {
                activeRegion = cv::Rect(minX, minY, maxX - minX, maxY - minY);

                if (activeRegion.width <= 0 || activeRegion.height <= 0) {
                    activeRegion = cv::Rect(0, 0, frame.cols, frame.rows);
                }

                ROS_DEBUG("ROI: %d,%d %dx%d (%.1f%% of frame)", 
                    activeRegion.x, activeRegion.y, 
                    activeRegion.width, activeRegion.height,
                    100.0 * (activeRegion.width * activeRegion.height) / 
                    (frame.cols * frame.rows));
               
            } else {
                ROS_DEBUG("ROI: Region too large, using the full frame instead.");
            }
        } else {
            static int fullFrameCounter = 0;
            if (++fullFrameCounter % 10 == 0) {
                ROS_DEBUG("No active tracks, using full frame (periodic scan)");
            } else {
                int gridSize = 3;
                int regionIndex = (fullFrameCounter % (gridSize * gridSize));
                int gridX = regionIndex % gridSize;
                int gridY = regionIndex / gridSize;
            
                int regionWidth = frame.cols / gridSize;
                int regionHeight = frame.rows / gridSize;
            
                activeRegion = cv::Rect(
                    gridX * regionWidth, 
                    gridY * regionHeight,
                regionWidth, 
                regionHeight
                );
            
                ROS_DEBUG("Grid scanning region %d: %d,%d %dx%d", 
                     regionIndex, activeRegion.x, activeRegion.y, 
                     activeRegion.width, activeRegion.height);
            }
        }

        return activeRegion;
    }
}