#pragma once

#include <vector>

#include "cpu/parser/parser.hh"
#include "cpu/utils/utils.hh"

namespace utils
{
    matrix_t gen_matrix(std::size_t rows, std::size_t cols, value_t value = 0.0);
    void gen_matrix(std::size_t rows, std::size_t cols, matrix_t& result, value_t value = 0.0);

    inline std::size_t matrix_row_count(const matrix_t& matrix)
    {
        return matrix.size();
    }
    inline std::size_t matrix_col_count(const matrix_t& matrix)
    {
        if (matrix.empty())
        {
            return 0;
        }

        return matrix[0].size();
    }

    void sub_matrix(const matrix_t& matrix,
                    std::size_t starting_row,
                    std::size_t starting_col,
                    std::size_t row_count,
                    std::size_t col_count,
                    matrix_t& result,
                    bool init_matrix = true);

    void matrix_transpose(const matrix_t& matrix, matrix_t& result, bool init_matrix = true);
    void matrix_dot_product(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix = true);
    void
    matrix_subtract_vector(const matrix_t& matrix, const matrix_t& vector, matrix_t& result, bool init_matrix = true);
    void matrix_inverse_diagonal(const matrix_t& matrix, matrix_t& result, bool init_matrix = true);
    void matrix_reduced(const matrix_t& matrix, std::size_t new_size, matrix_t& result, bool init_matrix = true);

    void matrix_centroid(const matrix_t& matrix, matrix_t& result, bool init_matrix = true);
    double matrix_determinant(const matrix_t& matrix);

    void hermitian_matrix(const vector_t& eigen_vector, matrix_t& result, bool init_matrix = true);
    void hermitian_matrix_inverse(const vector_t& eigen_vector, matrix_t& result, bool init_matrix = true);
    void jordan_gaussian_transform(const matrix_t& matrix, vector_t& result, bool init_matrix = true);
} // namespace utils