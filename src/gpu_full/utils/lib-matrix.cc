#include "lib-matrix.hh"

#include <cmath>
#include <iomanip>

namespace utils
{
    void matrix_dot_product(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        std::size_t row_count = lhs.get_rows();
        std::size_t col_count = rhs.get_cols();

        std::size_t common_dim = lhs.get_cols();

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = 0;

                for (std::size_t k = 0; k < common_dim; k++)
                {
                    result.data_[row][col] += lhs.at(row, k) * rhs.at(k, col);
                }
            }
        }
    }

    void vector_element_wise_multiplication(const vector_device_t& lhs,
                                            const vector_device_t& rhs,
                                            vector_device_t& result,
                                            std::size_t vector_size)
    {
        for (std::size_t i = 0; i < vector_size; i++)
        {
            result[i] = lhs[i] * rhs[i];
        }
    }

    double vector_sum(const vector_device_t& vector, std::size_t vector_size)
    {
        double sum = 0.0;

        for (std::size_t col = 0; col < vector_size; col++)
        {
            sum += vector[col];
        }

        return sum;
    }

    void matrix_subtract(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        std::size_t row_count = lhs.get_rows();
        std::size_t col_count = lhs.get_cols();

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = lhs.at(row, col) - rhs.at(row, col);
            }
        }
    }
} // namespace utils