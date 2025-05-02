#pragma once
#include "dataTypes.hpp"

class KalmanFilter {
public:
    static constexpr std::array<double, 10> chi2inv95 = {
        0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919
    };
    
    KalmanFilter();
    kalman::KAL_DATA initiateFilter(const tracking::DETECTBOX& measurement);
    void predict(kalman::KAL_MEAN& mean, kalman::KAL_COVA& covariance);
    kalman::KAL_HDATA project(const kalman::KAL_MEAN& mean, const kalman::KAL_COVA& covariance);
    kalman::KAL_DATA update(const kalman::KAL_MEAN& mean, const kalman::KAL_COVA& covariance, const tracking::DETECTBOX& measurement);
    Eigen::Matrix<float, 1, -1> getGatingDistance(const kalman::KAL_MEAN& mean, const kalman::KAL_COVA& covariance, const std::vector<tracking::DETECTBOX>& measurements, bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> motionMatrix;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> updateMatrix;
    float weightPosition;
    float weightVelocity;
};
