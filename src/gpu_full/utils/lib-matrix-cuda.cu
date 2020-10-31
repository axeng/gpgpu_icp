#include "lib-matrix.hh"

namespace utils
{
    [[gnu::noinline]]
    void _abortError(const char* msg, const char* fname, int line)
    {
        cudaError_t err = cudaGetLastError();
        spdlog::error("{} ({}, line: {})", msg, fname, line);
        spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
        std::exit(1);
    }

    __global__ void sub_matrix_cuda(const matrix_device_t& matrix,
                               std::size_t starting_row,
                               std::size_t starting_col,
                               std::size_t row_count,
                               std::size_t col_count,
                               matrix_device_t& result)
    {
        if (starting_row + row_count > this->rows_ || starting_col + col_count > this->cols_)
        {
            return;
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = this->data_[row + starting_row][col + starting_col];
            }
        }
    }

    __global__ void matrix_transpose_cuda(const matrix_device_t& matrix, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < this->cols_; row++)
        {
            for (std::size_t col = 0; col < this->rows_; col++)
            {
                result.data_[row][col] = this->data_[col][row];
            }
        }
    }

    __global__ void
    matrix_subtract_vector_cuda(const matrix_device_t& matrix, const matrix_device_t& vector, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < this->rows_; row++)
        {
            for (std::size_t col = 0; col < this->cols_; col++)
            {
                // considering the vector as a line vector (as returned by the centroid)
                result.data_[row][col] = this->data_[row][col] - vector.data_[0][col];
            }
        }
    }

    __global__ void
    matrix_add_vector_cuda(const matrix_device_t& matrix, const matrix_device_t& vector, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < this->rows_; row++)
        {
            for (std::size_t col = 0; col < this->cols_; col++)
            {
                result.data_[row][col] = this->data_[row][col] + vector.data_[0][col];
            }
        }
    }

    __global__ void multiply_by_scalar_cuda(const matrix_device_t& matrix, double val, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < this->rows_; row++)
        {
            for (std::size_t col = 0; col < this->cols_; col++)
            {
                result.data_[row][col] = this->data_[row][col] * val;
            }
        }
    }

    __global__ void matrix_dot_product_cuda(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
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
        for (std::size_t row = 0; row < lhs.get_rows(); row++)
        {
            for (std::size_t col = 0; col < lhs.get_cols(); col++)
            {
                result.data_[row][col] = lhs.at(row, col) - rhs.at(row, col);
            }
        }
    }
} // namespace utils