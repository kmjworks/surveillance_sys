#pragma once

#include "dataTypes.hpp"
#include "../Tracker.hpp"

#define INFTY_COST 1e5
class Tracker;
//for matching;
class linear_assignment
{
    linear_assignment();
    linear_assignment(const linear_assignment& );
    linear_assignment& operator=(const linear_assignment&);
    static linear_assignment* instance;

public:
    static linear_assignment* getInstance();
    motiontracker::TRACKER_MATCH matchingCascade(
        Tracker* distance_metric, Tracker::GATED_METRIC_FUNC distance_metric_func,
        float max_distance, int cascade_depth, std::vector<Track>& tracks,
        const model_internal::DETECTIONS& detections, std::vector<int>& track_indices,
        std::vector<int> detection_indices = std::vector<int>());

    motiontracker::TRACKER_MATCH minCostMatching(Tracker* distance_metric,
                                                   Tracker::GATED_METRIC_FUNC distance_metric_func,
                                                   float max_distance, std::vector<Track>& tracks,
                                                   const model_internal::DETECTIONS& detections,
                                                   std::vector<int>& track_indices,
                                                   std::vector<int>& detection_indices);
    linearassignment::DYNAMICM gateCostMatrix(
        std::unique_ptr<KalmanFilter>& kf, linearassignment::DYNAMICM& cost_matrix, std::vector<Track>& tracks,
        const model_internal::DETECTIONS& detections, const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices, float gated_cost = INFTY_COST,
        bool only_position = false);
};
