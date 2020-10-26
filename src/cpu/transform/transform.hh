#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include "cpu/parser/parser.hh"

namespace transform
{
    using point_t = parser::point_t;
    using points_t = parser::points_t;

    void get_fit_transform(const points_t& A, const points_t& B, points_t& T, points_t& R, points_t& t);

    void get_centroid(const points_t& set_point, points_t& result);

    void subtract_by_centroid(const points_t& set_point,
                             const points_t& centroid,
                             points_t& result);

    void matrix_transpose(const points_t& matrix1, points_t& matrix2);

    void matrix_by_matrix(const points_t& matrix1,
                          const points_t& matrix2,
                          points_t& matrix3);

    double get_determinant(const points_t& set_point, int dimension);

    void svd(const points_t& matrix, points_t& s, points_t& u, points_t& v);

    void compute_evd(const points_t& matrix, point_t& eigenvalues, points_t& eigenvectors, std::size_t eig_count);

    void get_hermitian_matrix(point_t eigenvector, points_t& h_matrix);
    void get_hermitian_matrix_inverse(point_t eigenvector, points_t& ih_matrix);

    void jordan_gaussian_transform(points_t& matrix, point_t& eigenvector);

    void get_inverse_diagonal_matrix(const points_t& matrix, points_t& inv_matrix);

    void get_reduced_matrix(const points_t& matrix, points_t& r_matrix, std::size_t new_size);
} // namespace transform