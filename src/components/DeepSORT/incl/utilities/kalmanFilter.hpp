#pragma once
#include "dataTypes.hpp"

class KalmanFilter {
public:
    static const double chi2inv95[10];
    KalmanFilter();
    KalmanFilterInternal::KAL_DATA initiateFilter(const TrackingInternal::DETECTBOX& measurement);
    void predict(KalmanFilterInternal::KAL_MEAN& mean, KalmanFilterInternal::KAL_COVA& covariance);
    KalmanFilterInternal::KAL_HDATA project(const KalmanFilterInternal::KAL_MEAN& mean, const KalmanFilterInternal::KAL_COVA& covariance);
    KalmanFilterInternal::KAL_DATA update(const KalmanFilterInternal::KAL_MEAN& mean, const KalmanFilterInternal::KAL_COVA& covariance, const TrackingInternal::DETECTBOX& measurement);
    Eigen::Matrix<float, 1, -1> getGatingDistance(const KalmanFilterInternal::KAL_MEAN& mean, const KalmanFilterInternal::KAL_COVA& covariance, const std::vector<TrackingInternal::DETECTBOX>& measurements, bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> motionMatrix;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> updateMatrix;
    float weightPosition;
    float weightVelocity;
};
