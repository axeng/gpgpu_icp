/*
__global__ void gen_matrix(int rows, int cols, thrust::device_vector<thrust::device_vector<double>>& result,
                double value)
    {
        int i = threadId.x;
        if (i >= rows)
        {
            return;
        }
        thrust::device_vector<double> D(cols, value);
        result[i] = D;
    }

    */

#include <cmath>

#include "lib-matrix.hh"

namespace utils
{
    matrix_t gen_matrix(std::size_t rows, std::size_t cols, value_t value)
    {
        vector_t row(cols, value);
        return matrix_t(rows, row);
    }

    void gen_matrix(std::size_t rows, std::size_t cols, matrix_t& result, value_t value)
    {
        result.resize(rows);
        for (std::size_t i = 0; i < rows; i++)
        {
            result[i].resize(cols, value);
        }
    }

    void sub_matrix(const matrix_t& matrix,
                    std::size_t starting_row,
                    std::size_t starting_col,
                    std::size_t row_count,
                    std::size_t col_count,
                    matrix_t& result,
                    bool init_matrix)
    {
        std::size_t origin_row_count = matrix_row_count(matrix);
        std::size_t origin_col_count = matrix_col_count(matrix);

        if (starting_row + row_count > origin_row_count || starting_col + col_count > origin_col_count)
        {
            return;
        }

        if (init_matrix)
        {
            utils::gen_matrix(row_count, col_count, result, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result[row][col] = matrix[row + starting_row][col + starting_col];
            }
        }
    }

    void matrix_transpose(const matrix_t& matrix, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(matrix);
        std::size_t col_count = matrix_col_count(matrix);

        if (init_matrix)
        {
            gen_matrix(col_count, row_count, result);
        }

        for (std::size_t row = 0; row < col_count; row++)
        {
            for (std::size_t col = 0; col < row_count; col++)
            {
                result[row][col] = matrix[col][row];
            }
        }
    }

    void matrix_dot_product(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(lhs);
        std::size_t col_count = matrix_col_count(rhs);

        std::size_t common_dim = matrix_col_count(lhs);

        if (init_matrix)
        {
            gen_matrix(row_count, col_count, result, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                for (std::size_t k = 0; k < common_dim; k++)
                {
                    result[row][col] += lhs[row][k] * rhs[k][col];
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

    double matrix_norm_2(const matrix_t& matrix)
    {
        double sum = 0.0;

        for (const auto& row : matrix)
        {
            for (const auto& element : row)
            {
                sum += pow(element, 2);
            }
        }

        return sqrt(sum);
    }

    void matrix_subtract(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(lhs);
        std::size_t col_count = matrix_col_count(lhs);

        if (init_matrix)
        {
            gen_matrix(row_count, col_count, result, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result[row][col] = lhs[row][col] - rhs[row][col];
            }
        }
    }

    void matrix_subtract_vector(const matrix_t& matrix, const matrix_t& vector, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(matrix);
        std::size_t col_count = matrix_col_count(matrix);

        if (init_matrix)
        {
            gen_matrix(row_count, col_count, result, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                // considering the vector as a line vector (as returned by the centroid)
                result[row][col] = matrix[row][col] - vector[0][col];
            }
        }
    }

    void matrix_add_vector(const matrix_t& matrix, const matrix_t& vector, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(matrix);
        std::size_t col_count = matrix_col_count(matrix);

        if (init_matrix)
        {
            gen_matrix(row_count, col_count, result, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result[row][col] = matrix[row][col] + vector[0][col];
            }
        }
    }

    void matrix_centroid(const matrix_t& matrix, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(matrix);
        std::size_t col_count = matrix_col_count(matrix);

        if (init_matrix)
        {
            gen_matrix(1, col_count, result, 0.0);
        }

        for (const auto& row : matrix)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result[0][col] += row[col];
            }
        }

        result[0][0] /= row_count;
        result[0][1] /= row_count;
        result[0][2] /= row_count;
    }

    void multiply_by_scalar(const matrix_t& matrix, double val, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(matrix);
        std::size_t col_count = matrix_col_count(matrix);

        if (init_matrix)
        {
            utils::gen_matrix(row_count, col_count, result, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result[row][col] = matrix[row][col] * val;
            }
        }
    }

} // namespace utils