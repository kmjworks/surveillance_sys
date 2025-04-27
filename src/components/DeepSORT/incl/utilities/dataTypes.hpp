#pragma once 

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

using DetectionBox = struct DetectionBox {

    DetectionBox(float x1=0, float y1=0, float x2=0, float y2=0, float confidence=0, float classIdentifier=-1, float trackIdentifier=-1) {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->confidence = confidence;
        this->classIdentifier = classIdentifier;
        this->trackIdentifier = trackIdentifier;
    }

    float x1, y1, x2, y2;
    float confidence;
    float classIdentifier;
    float trackIdentifier;
};

using CLSCONF = struct CLSCONF {
    CLSCONF() {
        this->cls = -1;
        this->conf = -1; 
    }

    CLSCONF(int cls, float conf) {
        this->cls = cls;
        this->conf = conf;  
    }

    int cls;
    float conf;
};

namespace TrackingInternal {
    using DETECTBOX = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
    using DETECTBOXSS = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;
    using FEATURE = Eigen::Matrix<float, 1, 256, Eigen::RowMajor>;
    using FEATURESS = Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor>;
}




namespace KalmanFilterInternal {
    using KAL_MEAN = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
    using KAL_COVA = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
    using KAL_HMEAN = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
    using KAL_HCOVA = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
    using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
    using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;
}



namespace MotionTracker {
    using RESULT_DATA = std::pair<int, TrackingInternal::DETECTBOX>;
    using TRACKER_DATA = std::pair<int, TrackingInternal::FEATURESS>;
    using MATCH_DATA = std::pair<int, int>;
    using TRACKER_MATCH = struct tracks {
        std::vector<MATCH_DATA> matches;
        std::vector<int> unmatchedTracks;
        std::vector<int> unmatchedDetections;
    };
}


namespace LinearAssignment {
    using DYNAMICM = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
}