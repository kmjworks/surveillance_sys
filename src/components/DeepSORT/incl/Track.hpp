#pragma once
#include "utilities/kalmanFilter.hpp"
#include "utilities/dataTypes.hpp"
#include "utilities/model.hpp"


class Track {
    enum TrackState { Tentative = 1, Confirmed, Deleted };

public:
    Track(kalman::KAL_MEAN& mean, kalman::KAL_COVA& covariance, int trackIdentifier, int n_init, int maxAge, const tracking::FEATURE& feature);
    Track(kalman::KAL_MEAN& mean, kalman::KAL_COVA& covariance, int trackIdentifier, int n_init, int maxAge, const tracking::FEATURE& feature, int cls, float conf);
    void predict(std::unique_ptr<KalmanFilter>& kf);
    void update(std::unique_ptr<KalmanFilter>& kf, const DETECTION_ROW& detection);
    void update(std::unique_ptr<KalmanFilter>& kf, const DETECTION_ROW& detection, CLSCONF pair_det);
    void markMissed();
    bool isConfirmed();
    bool isDeleted();
    bool isTentative();
    tracking::DETECTBOX toTlwh();

    int time_since_update;
    int track_id;
    tracking::FEATURESS features;
    kalman::KAL_MEAN mean;
    kalman::KAL_COVA covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;

    int cls;
    float conf;

private:
    void featuresAppendOne(const tracking::FEATURE& f);
};