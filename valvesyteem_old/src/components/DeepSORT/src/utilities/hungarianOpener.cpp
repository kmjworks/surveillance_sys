#include "../../incl/utilities/hungarianOpener.hpp"

Eigen::Matrix<float, -1, 2, Eigen::RowMajor> HungarianOper::Solve(const linearassignment::DYNAMICM &cost_matrix) {
    int rows = cost_matrix.rows();
    int cols = cost_matrix.cols();
    Matrix<double> matrix(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            matrix(row, col) = cost_matrix(row, col);
        }
    }
    Munkres<double> m;
    m.solve(matrix);

    std::vector<std::pair<int, int>> pairs;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int tmp = (int)matrix(row, col);
            if (tmp == 0) pairs.emplace_back(row, col);
        }
    }

    int count = pairs.size();
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> re(count, 2);
    for (int i = 0; i < count; i++) {
        re(i, 0) = pairs[i].first;
        re(i, 1) = pairs[i].second;
    }
    return re;
}