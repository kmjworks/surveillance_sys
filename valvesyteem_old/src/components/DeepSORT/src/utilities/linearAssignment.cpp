
#include <map>

#include "../../incl/utilities/linearAssignment.hpp"
#include "../../incl/utilities/hungarianOpener.hpp"

linear_assignment *linear_assignment::instance = NULL;
linear_assignment::linear_assignment() {}

linear_assignment *linear_assignment::getInstance() {
    if (instance == NULL)
        instance = new linear_assignment();
    return instance;
}

motiontracker::TRACKER_MATCH linear_assignment::matchingCascade(
    Tracker *distance_metric, Tracker::GATED_METRIC_FUNC distance_metric_func, float max_distance,
    int cascade_depth, std::vector<Track> &tracks, const model_internal::DETECTIONS &detections,
    std::vector<int> &track_indices, std::vector<int> detection_indices) {
    motiontracker::TRACKER_MATCH res;
    // std::cout << "distance_metric" << distance_metric << std::endl;
    // std::cout << "max_distance" << max_distance << std::endl;
    // std::cout << "cascade_depth" << cascade_depth << std::endl;
    // std::cout << "tracks [" << std::endl;
    // for (auto i : tracks)
    //     std::cout << i.hits << ", ";
    // std::cout << "]" << endl;
    // std::cout << "detections [" << std::endl;
    // for (auto i : detections)
    //     std::cout << i.confidence << ", ";
    // std::cout << "]" << endl;
    // std::cout << "track_indices [" << std::endl;
    // for (auto i : track_indices)
    //     std::cout << i << ", ";
    // std::cout << "]" << endl;
    // std::cout << "detection_indices [" << std::endl;
    // for (auto i : detection_indices)
    //     std::cout << i << ", ";
    // std::cout << "]" << endl;
    // !!!python diff: track_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }

    //!!!python diff: detection_indices will always be None.
    for (size_t i = 0; i < detections.size(); i++) {
        detection_indices.push_back(int(i));
    }

    std::vector<int> unmatched_detections;
    unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
    res.matches.clear();
    std::vector<int> track_indices_l;

    std::map<int, int> matches_trackid;
    for (int level = 0; level < cascade_depth; level++) {
        if (unmatched_detections.size() == 0)
            break;  // No detections left;

        track_indices_l.clear();
        for (int k : track_indices) {
            if (tracks[k].time_since_update == 1 + level)
                track_indices_l.push_back(k);
        }
        if (track_indices_l.size() == 0)
            continue;  // Nothing to match at this level.

        motiontracker::TRACKER_MATCH tmp =
            minCostMatching(distance_metric, distance_metric_func, max_distance, tracks, detections,
                            track_indices_l, unmatched_detections);
        unmatched_detections.assign(tmp.unmatchedDetections.begin(), tmp.unmatchedDetections.end());
        for (size_t i = 0; i < tmp.matches.size(); i++) {
            motiontracker::MATCH_DATA pa = tmp.matches[i];
            res.matches.push_back(pa);
            matches_trackid.insert(pa);
        }
    }
    res.unmatchedDetections.assign(unmatched_detections.begin(), unmatched_detections.end());
    for (size_t i = 0; i < track_indices.size(); i++) {
        int tid = track_indices[i];
        if (matches_trackid.find(tid) == matches_trackid.end())
            res.unmatchedTracks.push_back(tid);
    }
    return res;
}

motiontracker::TRACKER_MATCH linear_assignment::minCostMatching(
    Tracker *distance_metric, Tracker::GATED_METRIC_FUNC distance_metric_func, float max_distance,
    std::vector<Track> &tracks, const model_internal::DETECTIONS &detections,
    std::vector<int> &track_indices, std::vector<int> &detection_indices) {
    motiontracker::TRACKER_MATCH res;
    //!!!python diff: track_indices && detection_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }
    //    if(detection_indices.empty() == true) {
    //        for(size_t i = 0; i < detections.size(); i++) {
    //            detection_indices.push_back(int(i));
    //        }
    //    }
    if ((detection_indices.size() == 0) || (track_indices.size() == 0)) {
        res.matches.clear();
        res.unmatchedTracks.assign(track_indices.begin(), track_indices.end());
        res.unmatchedDetections.assign(detection_indices.begin(), detection_indices.end());
        return res;
    }
    linearassignment::DYNAMICM cost_matrix = (distance_metric->*(distance_metric_func))(
        tracks, detections, track_indices, detection_indices);
    for (int i = 0; i < cost_matrix.rows(); i++) {
        for (int j = 0; j < cost_matrix.cols(); j++) {
            float tmp = cost_matrix(i, j);
            if (tmp > max_distance)
                cost_matrix(i, j) = max_distance + 1e-5;
        }
    }
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = HungarianOper::Solve(cost_matrix);
    res.matches.clear();
    res.unmatchedTracks.clear();
    res.unmatchedDetections.clear();
    for (size_t col = 0; col < detection_indices.size(); col++) {
        bool flag = false;
        for (int i = 0; i < indices.rows(); i++)
            if (indices(i, 1) == col) {
                flag = true;
                break;
            }
        if (flag == false)
            res.unmatchedDetections.push_back(detection_indices[col]);
    }
    for (size_t row = 0; row < track_indices.size(); row++) {
        bool flag = false;
        for (int i = 0; i < indices.rows(); i++)
            if (indices(i, 0) == row) {
                flag = true;
                break;
            }
        if (flag == false)
            res.unmatchedTracks.push_back(track_indices[row]);
    }
    for (int i = 0; i < indices.rows(); i++) {
        int row = indices(i, 0);
        int col = indices(i, 1);

        int track_idx = track_indices[row];
        int detection_idx = detection_indices[col];
        if (cost_matrix(row, col) > max_distance) {
            res.unmatchedTracks.push_back(track_idx);
            res.unmatchedDetections.push_back(detection_idx);
        } else
            res.matches.push_back(std::make_pair(track_idx, detection_idx));
    }
    return res;
}

linearassignment::DYNAMICM linear_assignment::gateCostMatrix(
    std::unique_ptr<KalmanFilter>& kf, linearassignment::DYNAMICM &cost_matrix, std::vector<Track> &tracks,
    const model_internal::DETECTIONS &detections, const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices, float gated_cost, bool only_position) {
    // std::cout << "input cost matric" << cost_matrix << std::endl;
    int gating_dim = (only_position == true ? 2 : 4);
    double gating_threshold = KalmanFilter::chi2inv95[gating_dim];
    std::vector<tracking::DETECTBOX> measurements;
    for (int i : detection_indices) {
        const DETECTION_ROW &t = detections[i];
        measurements.push_back(t.to_xyah());
    }
    for (size_t i = 0; i < track_indices.size(); i++) {
        Track &track = tracks[track_indices[i]];
        Eigen::Matrix<float, 1, -1> gating_distance =
            kf->getGatingDistance(track.mean, track.covariance, measurements, only_position);
        for (int j = 0; j < gating_distance.cols(); j++) {
            if (gating_distance(0, j) > gating_threshold)
                cost_matrix(i, j) = gated_cost;
        }
    }
    // std::cout << "out cost matrix" << cost_matrix << std::endl;
    return cost_matrix;
}
