#pragma once

#include <vector>

#include "utilities/model.hpp"
#include "utilities/kalmanFilter.hpp"
#include "Track.hpp"


class NearNeighborDisMetric;

class Tracker {
public:
    Tracker(/*NearNeighborDisMetric* metric,*/ float maxCosineDistance, int nnBudget, float maxIOUDistance = 0.7, int maxAge = 200, int nInit = 20);
    ~Tracker();
    std::unique_ptr<NearNeighborDisMetric> metric;
    std::unique_ptr<KalmanFilter> kalman_f;
    std::vector<Track> tracks;
    float maxIOUDistance;
    int maxAge;
    int nInit;
    int nextIndex;
    
    linearassignment::DYNAMICM gatedMatric(std::vector<Track>& tracks, const model_internal::DETECTIONS& dets, const std::vector<int>& track_indices, const std::vector<int>& detection_indices);
    linearassignment::DYNAMICM iouCost(std::vector<Track>& tracks, const model_internal::DETECTIONS& dets, const std::vector<int>& track_indices, const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(tracking::DETECTBOX& bbox, tracking::DETECTBOXSS& candidates);
    void predict();
    void update(const model_internal::DETECTIONS& detections);
    void update(const model_internal::DETECTIONSV2& detectionsv2);
    using GATED_METRIC_FUNC = linearassignment::DYNAMICM (Tracker::*)(std::vector<Track> &, const model_internal::DETECTIONS &, const std::vector<int> &, const std::vector<int> &);

private:
    void match(const model_internal::DETECTIONS& detections, motiontracker::TRACKER_MATCH& res);
    void initTrack(const DETECTION_ROW& detection);
    void initTrack(const DETECTION_ROW& detection, CLSCONF clsConf);
};