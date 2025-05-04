#include "../incl/Track.hpp"

Track::Track(kalman::KAL_MEAN& mean, kalman::KAL_COVA & covariance, int trackIdentifier, int n_init, int maxAge, const tracking::FEATURE & feature) {
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = trackIdentifier;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = tracking::FEATURESS(1, 256);
    features.row(0) = feature;  //features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = maxAge;
}

Track::Track(kalman::KAL_MEAN& mean, kalman::KAL_COVA& covariance, int trackIdentifier, int n_init, int maxAge, const tracking::FEATURE& feature, int cls, float conf) {
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = trackIdentifier;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = tracking::FEATURESS(1, 256);
    features.row(0) = feature;  //features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = maxAge;

    this->cls = cls;
    this->conf = conf; 
}

void Track::predict(std::unique_ptr<KalmanFilter>& kf)
{
    kf->predict(this->mean, this->covariance);

    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(std::unique_ptr<KalmanFilter>& kf, const DETECTION_ROW & detection)
{
    kalman::KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;

    featuresAppendOne(detection.feature);
    //    this->features.row(features.rows()) = detection.feature;
    this->hits += 1;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
}

void Track::update(std::unique_ptr<KalmanFilter>& kf, const DETECTION_ROW & detection, CLSCONF pair_det)
{
    kalman::KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;

    featuresAppendOne(detection.feature);
    //    this->features.row(features.rows()) = detection.feature;
    this->hits += 1;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
    this->cls = pair_det.cls;
    this->conf = pair_det.conf;
}

void Track::markMissed()
{
    if (this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if (this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}

bool Track::isConfirmed()
{
    return this->state == TrackState::Confirmed;
}

bool Track::isDeleted()
{
    return this->state == TrackState::Deleted;
}

bool Track::isTentative()
{
    return this->state == TrackState::Tentative;
}

tracking::DETECTBOX Track::toTlwh()
{
    tracking::DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2) / 2);
    return ret;
}

void Track::featuresAppendOne(const tracking::FEATURE & f)
{
    int size = this->features.rows();
    tracking::FEATURESS newfeatures = tracking::FEATURESS(size + 1, 256);
    newfeatures.block(0, 0, size, 256) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}