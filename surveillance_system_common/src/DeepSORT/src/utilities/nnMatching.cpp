#include "../../incl/utilities/nnMatching.hpp"
#include <iostream>
#include "ros/console.h"

using namespace Eigen;

NearNeighborDisMetric::NearNeighborDisMetric(
    METRIC_TYPE metric, 
    float matching_threshold, int budget)
{
    if (metric == euclidean) {
        metricInternal = &NearNeighborDisMetric::nneuclideanDistance;
    } else if (metric == cosine) {
        metricInternal = &NearNeighborDisMetric::nncosineDistance;
    }

    this->matingThreshold = matching_threshold;
    this->budget = budget;
    this->samples.clear();
}

linearassignment::DYNAMICM 
NearNeighborDisMetric::distance(
    const tracking::FEATURESS & features,
    const std::vector < int >&targets)
{
    linearassignment::DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
    int idx = 0;
  for (int target:targets) {
        cost_matrix.row(idx) = (this->*metricInternal) (this->samples[target], features);
        idx++;
    }
    return cost_matrix;
}

void
 NearNeighborDisMetric::partialFit(
     std::vector <motiontracker::TRACKER_DATA> &tid_feats,
    std::vector < int >&active_targets)
{
    /*python code:
     * let feature(target_id) append to samples;
     * && delete not comfirmed target_id from samples.
     * update samples;
     */
  for (motiontracker::TRACKER_DATA & data:tid_feats) {
        int track_id = data.first;
        tracking::FEATURESS newFeatOne = data.second;

        if (newFeatOne.rows() == 0 || newFeatOne.cols() != 256) {
            ROS_WARN("[NearNeighborDisMetric] Skipping invalid feature matrix for track %d", track_id);
            continue;
        }

        if (samples.find(track_id) != samples.end()) {  //append
            int oldSize = samples[track_id].rows();
            int addSize = newFeatOne.rows();
            int newSize = oldSize + addSize;

            if (newSize <= this->budget) {
                tracking::FEATURESS newSampleFeatures(newSize, 256);
                newSampleFeatures.block(0, 0, oldSize, 256) = samples[track_id];
                newSampleFeatures.block(oldSize, 0, addSize, 256) = newFeatOne;
                samples[track_id] = newSampleFeatures;
            } else {
                if (oldSize < this->budget) {   //original space is not enough;
                    tracking::FEATURESS newSampleFeatures(this->budget, 256);
                    if (addSize >= this->budget) {
                        newSampleFeatures = newFeatOne.block(0, 0, this->budget, 256);
                    } else {
                        newSampleFeatures.block(0, 0, this->budget - addSize, 256) =
                            samples[track_id].block(addSize - 1, 0, this->budget - addSize, 256).eval();
                        newSampleFeatures.block(this->budget - addSize, 0, addSize,256) = newFeatOne;
                    }
                    samples[track_id] = newSampleFeatures;
                } else {        //original space is ok;
                    if (addSize >= this->budget) {
                        samples[track_id] = newFeatOne.block(0, 0, this->budget, 256);
                    } else {
                        samples[track_id].block(0, 0, this->budget - addSize, 256) =
                            samples[track_id].block(addSize - 1, 0, this->budget - addSize, 256).eval();
                        samples[track_id].block(this->budget - addSize, 0, addSize, 256) = newFeatOne;
                    }
                }
            }
        } else {                //not exit, create new one;
            samples[track_id] = newFeatOne;
        }
    }                           //add features;

    //erase the samples which not in active_targets;
    for (std::map < int, tracking::FEATURESS >::iterator i = samples.begin(); i != samples.end();) {
        bool flag = false; 
        for (int j:active_targets) if (j == i->first) { flag = true; break; }
        if (flag == false)samples.erase(i++);
        else i++;
    }
}

Eigen::VectorXf
    NearNeighborDisMetric::nncosineDistance(
        const tracking::FEATURESS & x, const tracking::FEATURESS & y)
{
    MatrixXf distances = cosineDistance(x, y);
    VectorXf res = distances.colwise().minCoeff().transpose();
    return res;
}

Eigen::VectorXf
    NearNeighborDisMetric::nneuclideanDistance(
        const tracking::FEATURESS & x, const tracking::FEATURESS & y)
{
    MatrixXf distances = pdist(x, y);
    VectorXf res = distances.colwise().maxCoeff().transpose();
    res = res.array().max(VectorXf::Zero(res.rows()).array());
    return res;
}

Eigen::MatrixXf
    NearNeighborDisMetric::pdist(const tracking::FEATURESS & x, const tracking::FEATURESS & y)
{
    int len1 = x.rows(), len2 = y.rows();
    if (len1 == 0 || len2 == 0) {
        return Eigen::MatrixXf::Zero(len1, len2);
    }
    MatrixXf res = -2.0 * x * y.transpose();
    res = res.colwise() + x.rowwise().squaredNorm();
    res = res.rowwise() + y.rowwise().squaredNorm().transpose();
    res = res.array().max(MatrixXf::Zero(res.rows(), res.cols()).array());
    return res;
}

Eigen::MatrixXf
    NearNeighborDisMetric::cosineDistance(
        const tracking::FEATURESS & a, const tracking::FEATURESS & b, bool data_is_normalized)
{
    tracking::FEATURESS aa = a;
    tracking::FEATURESS bb = b;
    if (!data_is_normalized) {
        //undo:
        for (int i = 0; i < a.rows(); ++i) {
            aa.row(i) =  a.row(i) / sqrt(a.row(i).squaredNorm());
        }
        for (int i = 0; i < b.rows(); ++i) {
            bb.row(i) =  b.row(i) / sqrt(b.row(i).squaredNorm());
        }        
    }
    MatrixXf res = 1. - (aa * bb.transpose()).array();
    return res;
}