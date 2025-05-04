#pragma once 

#include <map>
#include "dataTypes.hpp"

enum METRIC_TYPE : uint8_t {euclidean=1, cosine};

class NearNeighborDisMetric {
    public:
        
        NearNeighborDisMetric(METRIC_TYPE metric, float matchingThreshold, int budget);
        linearassignment::DYNAMICM distance (const tracking::FEATURESS& features, const std::vector<int> &targets);
        void partialFit(std::vector<motiontracker::TRACKER_DATA>& tidFeats, std::vector<int>& activeTargets);
        float matingThreshold;

    private:
        using PTRFUN = Eigen::VectorXf (NearNeighborDisMetric::*)(const tracking::FEATURESS &, const tracking::FEATURESS &);
        Eigen::VectorXf nncosineDistance(const tracking::FEATURESS& x, const tracking::FEATURESS& y);
        Eigen::VectorXf nneuclideanDistance(const tracking::FEATURESS& x, const tracking::FEATURESS& y);
        Eigen::MatrixXf cosineDistance(const tracking::FEATURESS& a, const tracking::FEATURESS& b, bool dataIsNormalized = false);
        Eigen::MatrixXf pdist(const tracking::FEATURESS& x, const tracking::FEATURESS& y);

        PTRFUN metricInternal;
        int budget;
        std::map<int, tracking::FEATURESS> samples;
};
