#include "../incl/Track.hpp"
#include "../incl/utilities/linearAssignment.hpp"
#include "../incl/utilities/nnMatching.hpp"

Tracker::Tracker(float maxCosineDistance, int nnBudget, float maxIOUDistance, int maxAge,int nInit) :
    maxIOUDistance(maxIOUDistance), maxAge(maxAge), nInit(nInit) {
    metric = std::make_unique<NearNeighborDisMetric>(METRIC_TYPE::cosine, maxCosineDistance, nnBudget);
    kalman_f = std::make_unique<KalmanFilter>();    
    tracks.clear();
    nextIndex = 1;
}

Tracker::~Tracker() {}

void Tracker::predict() {
    for (Track &track : tracks) {
        track.predict(kalman_f);
    }
}

void Tracker::update(const model_internal::DETECTIONS &detections) {
    motiontracker::TRACKER_MATCH res;
    match(detections, res);

    std::vector<motiontracker::MATCH_DATA> &matches = res.matches;
    for (motiontracker::MATCH_DATA &data : matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(kalman_f, detections[detection_idx]);
    }

    std::vector<int> &unmatched_tracks = res.unmatchedTracks;
    for (int &track_idx : unmatched_tracks) {
        this->tracks[track_idx].markMissed();
    }
    std::vector<int> &unmatched_detections = res.unmatchedDetections;
    for (int &detection_idx : unmatched_detections) {
        this->initTrack(detections[detection_idx]);
    }
    std::vector<Track>::iterator it;
    for (it = tracks.begin(); it != tracks.end();) {
        if ((*it).isDeleted())
            it = tracks.erase(it);
        else
            ++it;
    }
    std::vector<int> active_targets;
    std::vector<motiontracker::TRACKER_DATA> tid_features;
    for (Track &track : tracks) {
        if (track.isConfirmed() == false)
            continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        tracking::FEATURESS t = tracking::FEATURESS(0, 256);
        track.features = t;
    }
    this->metric->partialFit(tid_features, active_targets);
}

void Tracker::update(const model_internal::DETECTIONSV2 &detectionsv2) {
    const std::vector<CLSCONF> &clsConf = detectionsv2.first;
    const model_internal::DETECTIONS &detections = detectionsv2.second;
    motiontracker::TRACKER_MATCH res;
    match(detections, res);

    std::vector<motiontracker::MATCH_DATA> &matches = res.matches;
    for (motiontracker::MATCH_DATA &data : matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kalman_f, detections[detection_idx], clsConf[detection_idx]);
    }

    std::vector<int> &unmatched_tracks = res.unmatchedTracks;
    for (int &track_idx : unmatched_tracks) {
        this->tracks[track_idx].markMissed();
    }
    std::vector<int> &unmatched_detections = res.unmatchedDetections;
    for (int &detection_idx : unmatched_detections) {
        this->initTrack(detections[detection_idx], clsConf[detection_idx]);
    }
    std::vector<Track>::iterator it;
    for (it = tracks.begin(); it != tracks.end();) {
        if ((*it).isDeleted())
            it = tracks.erase(it);
        else
            ++it;
    }
    std::vector<int> active_targets;
    std::vector<motiontracker::TRACKER_DATA> tid_features;
    for (Track &track : tracks) {
        if (track.isConfirmed() == false) continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        tracking::FEATURESS t = tracking::FEATURESS(0, 256);
        track.features = t;
    }
    this->metric->partialFit(tid_features, active_targets);
}

void Tracker::match(const model_internal::DETECTIONS &detections,
                    motiontracker::TRACKER_MATCH &res) {
    std::vector<int> confirmed_tracks;
    std::vector<int> unconfirmed_tracks;
    int idx = 0;
    for (Track &t : tracks) {
        if (t.isConfirmed())
            confirmed_tracks.push_back(idx);
        else
            unconfirmed_tracks.push_back(idx);
        idx++;
    }

    motiontracker::TRACKER_MATCH matcha = linear_assignment::getInstance()->matchingCascade(
        this, &Tracker::gatedMatric, this->metric->matingThreshold, this->maxAge, this->tracks,
        detections, confirmed_tracks);
    std::vector<int> iou_track_candidates;
    iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
    std::vector<int>::iterator it;
    for (it = matcha.unmatchedTracks.begin(); it != matcha.unmatchedTracks.end();) {
        int idx = *it;
        if (tracks[idx].time_since_update == 1) {  // push into unconfirmed
            iou_track_candidates.push_back(idx);
            it = matcha.unmatchedTracks.erase(it);
            continue;
        }
        ++it;
    }
    motiontracker::TRACKER_MATCH matchb = linear_assignment::getInstance()->minCostMatching(
        this, &Tracker::iouCost, this->maxIOUDistance, this->tracks, detections,
        iou_track_candidates, matcha.unmatchedDetections);
    // get result:
    res.matches.assign(matcha.matches.begin(), matcha.matches.end());
    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
    // unmatched_tracks;
    res.unmatchedTracks.assign(matcha.unmatchedTracks.begin(), matcha.unmatchedTracks.end());
    res.unmatchedTracks.insert(res.unmatchedTracks.end(), matchb.unmatchedTracks.begin(),
                               matchb.unmatchedTracks.end());
    res.unmatchedDetections.assign(matchb.unmatchedDetections.begin(),
                                   matchb.unmatchedDetections.end());
}

void Tracker::initTrack(const DETECTION_ROW &detection) {
    kalman::KAL_DATA data = kalman_f->initiateFilter(detection.to_xyah());
    kalman::KAL_MEAN mean = data.first;
    kalman::KAL_COVA covariance = data.second;

    this->tracks.push_back(Track(mean, covariance, this->nextIndex, this->nInit, this->maxAge, detection.feature));
    nextIndex += 1;
}
void Tracker::initTrack(const DETECTION_ROW &detection, CLSCONF clsConf) {
    kalman::KAL_DATA data = kalman_f->initiateFilter(detection.to_xyah());
    kalman::KAL_MEAN mean = data.first;
    kalman::KAL_COVA covariance = data.second;

    this->tracks.push_back(Track(mean, covariance, this->nextIndex, this->nInit, this->maxAge,
                                 detection.feature, clsConf.cls, clsConf.conf));
    nextIndex += 1;
}

linearassignment::DYNAMICM Tracker::gatedMatric(std::vector<Track> &tracks,
                                                const model_internal::DETECTIONS &dets,
                                                const std::vector<int> &track_indices,
                                                const std::vector<int> &detection_indices) {
    tracking::FEATURESS features(detection_indices.size(), 256);
    int pos = 0;
    for (int i : detection_indices) {
        features.row(pos++) = dets[i].feature;
    }
    std::vector<int> targets;
    for (int i : track_indices) {
        targets.push_back(tracks[i].track_id);
    }
    linearassignment::DYNAMICM cost_matrix = this->metric->distance(features, targets);
    linearassignment::DYNAMICM res = linear_assignment::getInstance()->gateCostMatrix(
        kalman_f, cost_matrix, tracks, dets, track_indices, detection_indices);
    return res;
}

linearassignment::DYNAMICM Tracker::iouCost(std::vector<Track> &tracks,
                                            const model_internal::DETECTIONS &dets,
                                            const std::vector<int> &track_indices,
                                            const std::vector<int> &detection_indices) {
    //!!!python diff: track_indices && detection_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }
    //    if(detection_indices.empty() == true) {
    //        for(size_t i = 0; i < dets.size(); i++) {
    //            detection_indices.push_back(i);
    //        }
    //    }
    int rows = track_indices.size();
    int cols = detection_indices.size();
    linearassignment::DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for (int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        if (tracks[track_idx].time_since_update > 1) {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
            continue;
        }
        tracking::DETECTBOX bbox = tracks[track_idx].toTlwh();
        int csize = detection_indices.size();
        tracking::DETECTBOXSS candidates(csize, 4);
        for (int k = 0; k < csize; k++)
            candidates.row(k) = dets[detection_indices[k]].tlwh;
        Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
        cost_matrix.row(i) = rowV;
    }
    return cost_matrix;
}

Eigen::VectorXf Tracker::iou(tracking::DETECTBOX &bbox, tracking::DETECTBOXSS &candidates) {
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];

    Eigen::Matrix<float, -1, 2> candidates_tl;
    Eigen::Matrix<float, -1, 2> candidates_br;
    candidates_tl = candidates.leftCols(2);
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    //    Eigen::VectorXf area_intersection(size);
    //    Eigen::VectorXf area_candidates(size);
    Eigen::VectorXf res(size);
    for (int i = 0; i < size; i++) {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1;
        w = (w < 0 ? 0 : w);
        float h = br_2 - tl_2;
        h = (h < 0 ? 0 : h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection / (area_bbox + area_candidates - area_intersection);
    }
    return res;
}
