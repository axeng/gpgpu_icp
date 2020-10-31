#include <cmath>
#include <iomanip>

#include "lib-matrix.hh"
#include "matrix.hh"

namespace utils
{
    __global__ void sub_matrix_cuda(const matrix_device_t& matrix,
                                    std::size_t starting_row,
                                    std::size_t starting_col,
                                    std::size_t row_count,
                                    std::size_t col_count,
                                    matrix_device_t& result)
    {
        if (starting_row + row_count > matrix.rows_ || starting_col + col_count > matrix.cols_)
        {
            return;
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = matrix.data_[row + starting_row][col + starting_col];
            }
        }
    }

    __global__ void matrix_transpose_cuda(const matrix_device_t& matrix, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < matrix.cols_; row++)
        {
            for (std::size_t col = 0; col < matrix.rows_; col++)
            {
                result.data_[row][col] = matrix.data_[col][row];
            }
        }
    }

    __global__ void
    matrix_subtract_vector_cuda(const matrix_device_t& matrix, const matrix_device_t& vector, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < matrix.rows_; row++)
        {
            for (std::size_t col = 0; col < matrix.cols_; col++)
            {
                // considering the vector as a line vector (as returned by the centroid)
                result.data_[row][col] = matrix.data_[row][col] - vector.data_[0][col];
            }
        }
    }

    __global__ void
    matrix_add_vector_cuda(const matrix_device_t& matrix, const matrix_device_t& vector, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < matrix.rows_; row++)
        {
            for (std::size_t col = 0; col < matrix.cols_; col++)
            {
                result.data_[row][col] = matrix.data_[row][col] + vector.data_[0][col];
            }
        }
    }

    __global__ void multiply_by_scalar_cuda(const matrix_device_t& matrix, double val, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < matrix.rows_; row++)
        {
            for (std::size_t col = 0; col < matrix.cols_; col++)
            {
                result.data_[row][col] = matrix.data_[row][col] * val;
            }
        }
    }

    __global__ void matrix_dot_product_cuda(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        std::size_t row_count = lhs.rows_;
        std::size_t col_count = rhs.cols_;

        std::size_t common_dim = lhs.cols_;

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = 0;

                for (std::size_t k = 0; k < common_dim; k++)
                {
                    result.data_[row][col] += lhs.data_[row][k] * rhs.data_[k][col];
                }
            }
        }
    }

    __global__ void vector_element_wise_multiplication_cuda(const vector_device_t& lhs,
                                                            const vector_device_t& rhs,
                                                            vector_device_t& result,
                                                            std::size_t vector_size)
    {
        for (std::size_t i = 0; i < vector_size; i++)
        {
            result[i] = lhs[i] * rhs[i];
        }
    }

    __global__ void matrix_subtract_cuda(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < lhs.rows_; row++)
        {
            for (std::size_t col = 0; col < lhs.cols_; col++)
            {
                result.data_[row][col] = lhs.data_[row][col] - rhs.data_[row][col];
            }
        }
    }

    void matrix_dot_product(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        matrix_dot_product_cuda<<<1, 1>>>(lhs, rhs, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void vector_element_wise_multiplication(const vector_device_t& lhs,
                                            const vector_device_t& rhs,
                                            vector_device_t& result,
                                            std::size_t vector_size)
    {
        vector_element_wise_multiplication_cuda<<<1, 1>>>(lhs, rhs, result, vector_size);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
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
        matrix_subtract_cuda<<<1, 1>>>(lhs, rhs, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }
} // namespace utils
