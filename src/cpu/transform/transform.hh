#pragma once

#include <math.h>
#include <tuple>
#include <vector>

#include "cpu/parser/parser.hh"

namespace transform
{
    using point_t = parser::point_t;
    using points_t = parser::points_t;

    unsigned int get_fit_transform(const points_t& A, const points_t& B;

    void get_centroid(const points_t& set_point, points_t& result);

    void subtract_by_centroid(const points_t& set_point,
                             const points_t& centroid,
                             points_t& result);

    void matrix_transpose(const points_t& matrix1, points_t& matrix2);

    void matrix_by_matrix(const points_t& matrix1,
                          const points_t& matrix2,
                          points_t& matrix3);

    double get_determinant(const points_t& set_point, int dimension);

    // ------------------------------------------

    void svd(std::vector<std::vector<double>> matrix,
             std::vector<std::vector<double>>& s,
             std::vector<std::vector<double>>& u,
             std::vector<std::vector<double>>& v);
} // namespace transform