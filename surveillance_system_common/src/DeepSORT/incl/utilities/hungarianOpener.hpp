#pragma once 

#include "munkres.hpp"
#include "dataTypes.hpp"


class HungarianOper {
public:
    static Eigen::Matrix<float, -1, 2, Eigen::RowMajor> Solve(const linearassignment::DYNAMICM &cost_matrix);
};