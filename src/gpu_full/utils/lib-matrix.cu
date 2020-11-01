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
                value_t *result_ptr;
                value_t *matrix_ptr;
                result.get_val_ptr_cuda(row, col, &result_ptr);
                matrix.get_val_ptr_cuda(row + starting_row, col + starting_col, &matrix_ptr);

                *result_ptr = *matrix_ptr;
            }
        }
    }

    __global__ void matrix_transpose_cuda(const matrix_device_t& matrix, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < matrix.cols_; row++)
        {
            for (std::size_t col = 0; col < matrix.rows_; col++)
            {
                value_t *result_ptr;
                value_t *matrix_ptr;
                result.get_val_ptr_cuda(row, col, &result_ptr);
                matrix.get_val_ptr_cuda(col, row, &matrix_ptr);

                *result_ptr = *matrix_ptr;
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
                value_t *result_ptr;
                value_t *matrix_ptr;
                value_t *vector_ptr;
                result.get_val_ptr_cuda(row, col, &result_ptr);
                matrix.get_val_ptr_cuda(row, col, &matrix_ptr);
                vector.get_val_ptr_cuda(0, col, &vector_ptr);

                *result_ptr = *matrix_ptr - *vector_ptr;
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
                value_t *result_ptr;
                value_t *matrix_ptr;
                value_t *vector_ptr;
                result.get_val_ptr_cuda(row, col, &result_ptr);
                matrix.get_val_ptr_cuda(row, col, &matrix_ptr);
                vector.get_val_ptr_cuda(0, col, &vector_ptr);

                *result_ptr = *matrix_ptr + *vector_ptr;
            }
        }
    }

    __global__ void multiply_by_scalar_cuda(const matrix_device_t& matrix, double val, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < matrix.rows_; row++)
        {
            for (std::size_t col = 0; col < matrix.cols_; col++)
            {
                value_t *result_ptr;
                value_t *matrix_ptr;
                result.get_val_ptr_cuda(row, col, &result_ptr);
                matrix.get_val_ptr_cuda(row, col, &matrix_ptr);

                *result_ptr = *matrix_ptr + val;
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
                value_t *result_ptr;
                result.get_val_ptr_cuda(row, col, &result_ptr);

                *result_ptr = 0;

                for (std::size_t k = 0; k < common_dim; k++)
                {
                    value_t *lhs_ptr;
                    value_t *rhs_ptr;
                    lhs.get_val_ptr_cuda(row, k, &lhs_ptr);
                    rhs.get_val_ptr_cuda(k, col, &rhs_ptr);

                    *result_ptr += *lhs_ptr * *rhs_ptr;
                }
            }
        }
    }

    __global__ void vector_element_wise_multiplication_cuda(const matrix_device_t& lhs,
                                                            std::size_t lhs_row,
                                                            const matrix_device_t& rhs,
                                                            std::size_t rhs_row,
                                                            matrix_device_t& result)
    {
        for (std::size_t i = 0; i < lhs.cols_; i++)
        {
            value_t *result_ptr;
            value_t *lhs_ptr;
            value_t *rhs_ptr;
            result.get_val_ptr_cuda(0, i, &result_ptr);
            lhs.get_val_ptr_cuda(0, i, &lhs_ptr);
            rhs.get_val_ptr_cuda(0, i, &rhs_ptr);

            *result_ptr = *lhs_ptr * *rhs_ptr;
        }
    }

    __global__ void matrix_subtract_cuda(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        for (std::size_t row = 0; row < lhs.rows_; row++)
        {
            for (std::size_t col = 0; col < lhs.cols_; col++)
            {
                value_t *result_ptr;
                value_t *lhs_ptr;
                value_t *rhs_ptr;
                result.get_val_ptr_cuda(row, col, &result_ptr);
                lhs.get_val_ptr_cuda(row, col, &lhs_ptr);
                rhs.get_val_ptr_cuda(row, col, &rhs_ptr);

                *result_ptr = *lhs_ptr - *rhs_ptr;
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

    void vector_element_wise_multiplication(const matrix_device_t& lhs,
                                            std::size_t lhs_row,
                                            const matrix_device_t& rhs,
                                            std::size_t rhs_row,
                                            matrix_device_t& result)
    {
        vector_element_wise_multiplication_cuda<<<1, 1>>>(lhs, lhs_row, rhs, rhs_row, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    double vector_sum(const matrix_device_t& vector)
    {
        double sum = 0.0;

        for (std::size_t col = 0; col < vector.get_rows(); col++)
        {
            sum += vector.at(0, col);
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

    __device__ void Matrix::get_val_ptr_cuda(std::size_t row, std::size_t col, const value_t** val) const
    {
        *val = (value_t*)((this->data_ + row * this->pitch_) + col * sizeof(value_t));
    }

    __device__ void Matrix::get_val_ptr_cuda(std::size_t row, std::size_t col, value_t** val)
    {
        *val = (value_t*)((this->data_ + row * this->pitch_) + col * sizeof(value_t));
    }

    value_t* Matrix::get_val_ptr(std::size_t row, std::size_t col)
    {
        return (value_t*)((this->data_ + row * this->pitch_) + col * sizeof(value_t));
    }

    const value_t* Matrix::get_val_ptr(std::size_t row, std::size_t col) const
    {
        return (value_t*)((this->data_ + row * this->pitch_) + col * sizeof(value_t));
    }
} // namespace utils
