#pragma once

#include "matrix.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <list>
#include <utility>




template <typename Data>
class Munkres {
    static constexpr int NORMAL = 0;
    static constexpr int STAR = 1;
    static constexpr int PRIME = 2;

private:
    struct Masks {
        Matrix<int> matrixMask;
        bool *rowMask;
        bool *colMask;
    };    

public:
    /*
     *
     * Linear assignment problem solution
     * [modifies matrix in-place.]
     * matrix(row,col): row major format assumed.
     *
     * Assignments are remaining 0 values
     * (extra 0 values are replaced with -1)
     *
     */
    void solve(Matrix<Data> &m) {
        const size_t rows = m.rows(), columns = m.columns(), size = XYZMAX(rows, columns);
        this->matrix = m;

        if (rows != columns) {
            matrix.resize(size, size, matrix.mmax());
        }

        masks.matrixMask.resize(size, size);

        masks.rowMask = new bool[size];
        masks.colMask = new bool[size];
        for (size_t i = 0; i < size; i++) {
            masks.rowMask[i] = false;
        }

        for (size_t i = 0; i < size; i++) {
            masks.colMask[i] = false;
        }

        // Prepare the matrix values...

        // If there were any infinities, replace them with a value greater
        // than the maximum value in the matrix.
        replaceInfinites(matrix);

        minimizeAlongDirection(matrix, rows >= columns);
        minimizeAlongDirection(matrix, rows < columns);

        // Follow the steps
        int step = 1;
        while (step) {
            switch (step) {
                case 1:
                    step = munkres1();
                    // step is always 2
                    break;
                case 2:
                    step = munkres2();
                    // step is always either 0 or 3
                    break;
                case 3:
                    step = munkres3();
                    // step in [3, 4, 5]
                    break;
                case 4:
                    step = munkres4();
                    // step is always 2
                    break;
                case 5:
                    step = munkres5();
                    // step is always 3
                    break;
            }
        }

        // Store results
        for (size_t row = 0; row < size; row++) {
            for (size_t col = 0; col < size; col++) {
                if (masks.matrixMask(row, col) == STAR) {
                    matrix(row, col) = 0;
                } else {
                    matrix(row, col) = -1;
                }
            }
        }

        // Remove the excess rows or columns that we added to fit the
        // input to a square matrix.
        matrix.resize(rows, columns);

        m = matrix;

        delete[] masks.rowMask;
        delete[] masks.colMask;
    }

    static void replaceInfinites(Matrix<Data> &matrix) {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        // assert( rows > 0 && columns > 0 );
        double max = matrix(0, 0);
        constexpr auto infinity = std::numeric_limits<double>::infinity();

        // Find the greatest value in the matrix that isn't infinity.
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < columns; col++) {
                if (matrix(row, col) != infinity) {
                    if (max == infinity) {
                        max = matrix(row, col);
                    } else {
                        max = XYZMAX(max, matrix(row, col));
                    }
                }
            }
        }

        // a value higher than the maximum value present in the matrix.
        if (max == infinity) {
            // This case only occurs when all values are infinite.
            max = 0;
        } else {
            max++;
        }

        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < columns; col++) {
                if (matrix(row, col) == infinity) {
                    matrix(row, col) = max;
                }
            }
        }
    }
    static void minimizeAlongDirection(Matrix<Data> &matrix, const bool over_columns) {
        const size_t outer_size = over_columns ? matrix.columns() : matrix.rows(),
                     inner_size = over_columns ? matrix.rows() : matrix.columns();

        // Look for a minimum value to subtract from all values along
        // the "outer" direction.
        for (size_t i = 0; i < outer_size; i++) {
            double min = over_columns ? matrix(0, i) : matrix(i, 0);

            // As long as the current minimum is greater than zero,
            // keep looking for the minimum.
            // Start at one because we already have the 0th value in min.
            for (size_t j = 1; j < inner_size && min > 0; j++) {
                min = XYZMIN(min, over_columns ? matrix(j, i) : matrix(i, j));
            }

            if (min > 0) {
                for (size_t j = 0; j < inner_size; j++) {
                    if (over_columns) {
                        matrix(j, i) -= min;
                    } else {
                        matrix(i, j) -= min;
                    }
                }
            }
        }
    }

private:
    inline bool findUncoveredInMatrix(const double item, size_t &row, size_t &col) const {
        const size_t rows = matrix.rows(), columns = matrix.columns();

        for (row = 0; row < rows; row++) {
            if (!masks.rowMask[row]) {
                for (col = 0; col < columns; col++) {
                    if (!masks.colMask[col]) {
                        if (matrix(row, col) == item) {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    bool pairInList(const std::pair<size_t, size_t> &needle,
                      const std::list<std::pair<size_t, size_t> > &haystack) {
        for (std::list<std::pair<size_t, size_t> >::const_iterator i = haystack.begin();
             i != haystack.end(); i++) {
            if (needle == *i) {
                return true;
            }
        }

        return false;
    }

    int munkres1() {
        const size_t rows = matrix.rows(), columns = matrix.columns();

        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < columns; col++) {
                if (0 == matrix(row, col)) {
                    for (size_t nrow = 0; nrow < row; nrow++)
                        if (STAR == masks.matrixMask(nrow, col))
                            goto next_column;

                    masks.matrixMask(row, col) = STAR;
                    goto next_row;
                }
            next_column:;
            }
        next_row:;
        }

        return 2;
    }

    int munkres2() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        size_t covercount = 0;

        for (size_t row = 0; row < rows; row++)
            for (size_t col = 0; col < columns; col++)
                if (STAR == masks.matrixMask(row, col)) {
                    masks.colMask[col] = true;
                    covercount++;
                }

        if (covercount >= matrix.minsize()) {
            return 0;
        }

        return 3;
    }

    int munkres3() {
        /*
        Main Zero Search

         1. Find an uncovered Z in the distance matrix and prime it. If no such zero exists, go to
        Step 5
         2. If No Z* exists in the row of the Z', go to Step 4.
         3. If a Z* exists, cover this row and uncover the column of the Z*. Return to Step 3.1 to
        find a new Z
        */
        if (findUncoveredInMatrix(0, saveRow, saveCol)) {
            masks.matrixMask(saveRow, saveCol) = PRIME;  // prime it.
        } else {
            return 5;
        }

        for (size_t ncol = 0; ncol < matrix.columns(); ncol++) {
            if (masks.matrixMask(saveRow, ncol) == STAR) {
                masks.rowMask[saveRow] = true;  // cover this row and
                masks.colMask[ncol] = false;    // uncover the column containing the starred zero
                return 3;                  // repeat
            }
        }

        return 4;  // no starred zero in the row containing this primed zero
    }

    int munkres4() {
        const size_t rows = matrix.rows(), columns = matrix.columns();

        // seq contains pairs of row/column values where we have found
        // either a star or a prime that is part of the ``alternating sequence``.
        std::list<std::pair<size_t, size_t> > seq;
        // use saverow, savecol from step 3.
        std::pair<size_t, size_t> z0(saveRow, saveCol);
        seq.insert(seq.end(), z0);

        // We have to find these two pairs:
        std::pair<size_t, size_t> z1(-1, -1);
        std::pair<size_t, size_t> z2n(-1, -1);

        size_t row, col = saveCol;
        /*
        Increment Set of Starred Zeros

         1. Construct the ``alternating sequence'' of primed and starred zeros:

               Z0 : Unpaired Z' from Step 4.2
               Z1 : The Z* in the column of Z0
               Z[2N] : The Z' in the row of Z[2N-1], if such a zero exists
               Z[2N+1] : The Z* in the column of Z[2N]

            The sequence eventually terminates with an unpaired Z' = Z[2N] for some N.
        */
        bool madepair;
        do {
            madepair = false;
            for (row = 0; row < rows; row++) {
                if (masks.matrixMask(row, col) == STAR) {
                    z1.first = row;
                    z1.second = col;
                    if (pairInList(z1, seq)) {
                        continue;
                    }

                    madepair = true;
                    seq.insert(seq.end(), z1);
                    break;
                }
            }

            if (!madepair)
                break;

            madepair = false;

            for (col = 0; col < columns; col++) {
                if (masks.matrixMask(row, col) == PRIME) {
                    z2n.first = row;
                    z2n.second = col;
                    if (pairInList(z2n, seq)) {
                        continue;
                    }
                    madepair = true;
                    seq.insert(seq.end(), z2n);
                    break;
                }
            }
        } while (madepair);

        for (std::list<std::pair<size_t, size_t> >::iterator i = seq.begin(); i != seq.end(); i++) {
            // 2. Unstar each starred zero of the sequence.
            if (masks.matrixMask(i->first, i->second) == STAR)
                masks.matrixMask(i->first, i->second) = NORMAL;

            // 3. Star each primed zero of the sequence,
            // thus increasing the number of starred zeros by one.
            if (masks.matrixMask(i->first, i->second) == PRIME)
                masks.matrixMask(i->first, i->second) = STAR;
        }

        // 4. Erase all primes, uncover all columns and rows,
        for (size_t row = 0; row < masks.matrixMask.rows(); row++) {
            for (size_t col = 0; col < masks.matrixMask.columns(); col++) {
                if (masks.matrixMask(row, col) == PRIME) {
                    masks.matrixMask(row, col) = NORMAL;
                }
            }
        }

        for (size_t i = 0; i < rows; i++) {
            masks.rowMask[i] = false;
        }

        for (size_t i = 0; i < columns; i++) {
            masks.colMask[i] = false;
        }

        // and return to Step 2.
        return 2;
    }

    int munkres5() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        /*
        New Zero Manufactures

         1. Let h be the smallest uncovered entry in the (modified) distance matrix.
         2. Add h to all covered rows.
         3. Subtract h from all uncovered columns
         4. Return to Step 3, without altering stars, primes, or covers.
        */
        double h = 100000;  // xyzoylz std::numeric_limits<double>::max();
        for (size_t row = 0; row < rows; row++) {
            if (!masks.rowMask[row]) {
                for (size_t col = 0; col < columns; col++) {
                    if (!masks.colMask[col]) {
                        if (h > matrix(row, col) && matrix(row, col) != 0) {
                            h = matrix(row, col);
                        }
                    }
                }
            }
        }

        for (size_t row = 0; row < rows; row++) {
            if (masks.rowMask[row]) {
                for (size_t col = 0; col < columns; col++) {
                    matrix(row, col) += h;
                }
            }
        }

        for (size_t col = 0; col < columns; col++) {
            if (!masks.colMask[col]) {
                for (size_t row = 0; row < rows; row++) {
                    matrix(row, col) -= h;
                }
            }
        }

        return 3;
    }

    Masks masks;
    Matrix<Data> matrix;
    size_t saveRow = 0;
    size_t saveCol = 0;
};