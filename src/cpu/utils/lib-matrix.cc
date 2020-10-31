#include "lib-matrix.hh"

#include <cmath>
#include <iomanip>

namespace utils
{
    void matrix_dot_product(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = lhs.get_rows();
        std::size_t col_count = rhs.get_cols();

        std::size_t common_dim = lhs.get_cols();

        if (init_matrix)
        {
            result.matrix_fill(row_count, col_count, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                for (std::size_t k = 0; k < common_dim; k++)
                {
                    result.at(row, col) += lhs.at(row, k) * rhs.at(k, col);
                }
            }
        }
    }

    void
    vector_element_wise_multiplication(const vector_t& lhs, const vector_t& rhs, vector_t& result, bool init_vector)
    {
        for (std::size_t i = 0; i < lhs.size(); i++)
        {
            if (init_vector)
            {
                result.push_back(lhs[i] * rhs[i]);
            }
            else
            {
                result[i] = lhs[i] * rhs[i];
            }
        }
    }

    double vector_sum(const vector_t& vector)
    {
        double sum = 0.0;

        for (const auto& element : vector)
        {
            sum += element;
        }

        return sum;
    }

    void matrix_subtract(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = lhs.get_rows();
        std::size_t col_count = lhs.get_cols();

        if (init_matrix)
        {
            result.matrix_fill(row_count, col_count, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.at(row, col) = lhs.at(row, col) - rhs.at(row, col);
            }
        }
    }
} // namespace utils