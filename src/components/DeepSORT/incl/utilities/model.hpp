#pragma once 

#include <algorithm>
#include "dataTypes.hpp"

const float kRatio = 0.5; 
enum DETECTIONBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H };

class DETECTION_ROW {
    public:
        tracking::DETECTBOX tlwh;
        tracking::FEATURE feature;
        float confidence;
        tracking::DETECTBOX to_xyah() const {
            tracking::DETECTBOX ret = tlwh;
            ret(0, IDX_X) += (ret(0, IDX_W)*kRatio);
            ret(0, IDX_Y) += (ret(0, IDX_H)*kRatio);
            ret(0, IDX_W) /= ret(0, IDX_H);
            return ret; 
        }

        tracking::DETECTBOX to_tlbr() const {
            tracking::DETECTBOX ret = tlwh;
            ret(0, IDX_X) += ret(0, IDX_W);
            ret(0, IDX_Y) += ret(0, IDX_H);
            return ret;
        }
};

namespace model_internal {
    using DETECTIONS = std::vector<DETECTION_ROW>;
    using DETECTIONSV2 = std::pair<std::vector<CLSCONF>, DETECTIONS>;
}
