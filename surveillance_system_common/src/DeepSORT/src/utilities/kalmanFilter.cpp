#include "../../incl/utilities/kalmanFilter.hpp"
#include <Eigen/Cholesky>
#include <iostream>


KalmanFilter::KalmanFilter() {
    int ndim = 4;
    double dt = 1.;

    motionMatrix = Eigen::MatrixXf::Identity(8, 8);
    for(int i = 0; i < ndim; i++) {
        motionMatrix(i, ndim+i) = dt;
    }
    updateMatrix = Eigen::MatrixXf::Identity(4, 8);

    this->weightPosition = 1. / 20;
    this->weightVelocity = 1. / 160;
}

kalman::KAL_DATA KalmanFilter::initiateFilter(const tracking::DETECTBOX& measurement) {
    tracking::DETECTBOX meanPosition = measurement;
    tracking::DETECTBOX meanVelocity;
    for(int i = 0; i < 4; i++) meanVelocity(i) = 0;

    kalman::KAL_MEAN mean;
    for(int i = 0; i < 8; i++){
        if(i < 4) mean(i) = meanPosition(i);
        else mean(i) = meanVelocity(i - 4);
    }

    kalman::KAL_MEAN std;
    std(0) = 2 * weightPosition * measurement[3];
    std(1) = 2 * weightPosition * measurement[3];
    std(2) = 1e-2;
    std(3) = 2 * weightPosition * measurement[3];
    std(4) = 10 * weightVelocity * measurement[3];
    std(5) = 10 * weightVelocity * measurement[3];
    std(6) = 1e-5;
    std(7) = 10 * weightVelocity * measurement[3];

    kalman::KAL_MEAN tmp = std.array().square();
    kalman::KAL_COVA var = tmp.asDiagonal();
    return std::make_pair(mean, var);
}

void KalmanFilter::predict(kalman::KAL_MEAN &mean, kalman::KAL_COVA &covariance) {
    //revise the data;
    tracking::DETECTBOX std_pos;
    std_pos << weightPosition * mean(3), weightPosition * mean(3), 1e-2, weightPosition * mean(3);
    tracking::DETECTBOX std_vel;
    std_vel << weightVelocity * mean(3), weightVelocity * mean(3),1e-5, weightVelocity * mean(3);
    
    kalman::KAL_MEAN tmp;
    tmp.block<1,4>(0,0) = std_pos;
    tmp.block<1,4>(0,4) = std_vel;
    tmp = tmp.array().square();
    kalman::KAL_COVA motion_cov = tmp.asDiagonal();
    kalman::KAL_MEAN mean1 = this->motionMatrix * mean.transpose();
    kalman::KAL_COVA covariance1 = this->motionMatrix * covariance *(motionMatrix.transpose());
    covariance1 += motion_cov;

    mean = mean1;
    covariance = covariance1;
}

kalman::KAL_HDATA KalmanFilter::project(const kalman::KAL_MEAN &mean, const kalman::KAL_COVA &covariance) {
    tracking::DETECTBOX std;
    std << weightPosition * mean(3), weightPosition * mean(3), 1e-1, weightPosition * mean(3);
    kalman::KAL_HMEAN mean1 = updateMatrix * mean.transpose();
    kalman::KAL_HCOVA covariance1 = updateMatrix * covariance * (updateMatrix.transpose());
    Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
    diag = diag.array().square().matrix();
    covariance1 += diag;
//    covariance1.diagonal() << diag;
    return std::make_pair(mean1, covariance1);
}

kalman::KAL_DATA KalmanFilter::update(const kalman::KAL_MEAN &mean, const kalman::KAL_COVA &covariance, const tracking::DETECTBOX &measurement) {
    kalman::KAL_HDATA pa = project(mean, covariance);
    kalman::KAL_HMEAN projected_mean = pa.first;
    kalman::KAL_HCOVA projected_cov = pa.second;

    //chol_factor, lower =
    //scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
    //kalmain_gain =
    //scipy.linalg.cho_solve((cho_factor, lower),
    //np.dot(covariance, self._upadte_mat.T).T,
    //check_finite=False).T
    Eigen::Matrix<float, 4, 8> B = (covariance * (updateMatrix.transpose())).transpose();
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
    Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
    auto tmp = innovation*(kalman_gain.transpose());
    kalman::KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
    kalman::KAL_COVA new_covariance = covariance - kalman_gain*projected_cov*(kalman_gain.transpose());
    return std::make_pair(new_mean, new_covariance);
}

Eigen::Matrix<float, 1, -1>
KalmanFilter::getGatingDistance(const kalman::KAL_MEAN &mean, const kalman::KAL_COVA &covariance, const std::vector<tracking::DETECTBOX> &measurements, bool only_position) {
    kalman::KAL_HDATA pa = this->project(mean, covariance);
    if(only_position) {
        printf("not implement!");
        exit(0);
    }
    kalman::KAL_HMEAN mean1 = pa.first;
    kalman::KAL_HCOVA covariance1 = pa.second;

//    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
    tracking::DETECTBOXSS d(measurements.size(), 4);
    int pos = 0;
    for(tracking::DETECTBOX box:measurements) {        
        d.row(pos++) = box - mean1;
    }
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
    Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
    auto zz = ((z.array())*(z.array())).matrix();
    auto square_maha = zz.colwise().sum();
    return square_maha;
}
