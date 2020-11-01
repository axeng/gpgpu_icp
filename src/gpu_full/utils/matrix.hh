#pragma once

#include <memory>
#include <vector>

#include "gpu_full/parser/parser.hh"

namespace utils
{
    class Matrix
    {
    public:
        using value_t = double;
        using matrix_device_t = Matrix;

        Matrix(std::size_t rows, std::size_t cols, value_t value = 0);
        ~Matrix();

        void sub_matrix(std::size_t starting_row,
                        std::size_t starting_col,
                        std::size_t row_count,
                        std::size_t col_count,
                        matrix_device_t& result) const;
        void matrix_transpose(matrix_device_t& result) const;

        void matrix_norm_2(double& norm) const;
        void matrix_subtract_vector(const matrix_device_t& vector, matrix_device_t& result) const;
        void matrix_add_vector(const matrix_device_t& vector, matrix_device_t& result) const;

        void matrix_centroid(matrix_device_t& result) const;

        void multiply_by_scalar(double val, matrix_device_t& result) const;

        void copy_line(const matrix_device_t& line, std::size_t line_row, std::size_t row);
        void copy_line(const parser::vector_host_t& line, std::size_t row);

        inline std::size_t get_rows() const
        {
            return this->rows_;
        }

        inline std::size_t get_cols() const
        {
            return this->cols_;
        }

        inline value_t at(size_t row, size_t col)
        {
            return *this->get_val_ptr(row, col);
        }

        inline value_t at(size_t row, size_t col) const
        {
            return *this->get_val_ptr(row, col);
        }

        // Kernels
        friend __global__ void sub_matrix_cuda(const matrix_device_t& matrix,
                                               std::size_t starting_row,
                                               std::size_t starting_col,
                                               std::size_t row_count,
                                               std::size_t col_count,
                                               matrix_device_t& result);
        friend __global__ void matrix_transpose_cuda(const matrix_device_t& matrix, matrix_device_t& result);

        friend __global__ void matrix_subtract_vector_cuda(const matrix_device_t& matrix,
                                                           const matrix_device_t& vector,
                                                           matrix_device_t& result);
        friend __global__ void
        matrix_add_vector_cuda(const matrix_device_t& matrix, const matrix_device_t& vector, matrix_device_t& result);

        friend __global__ void
        multiply_by_scalar_cuda(const matrix_device_t& matrix, double val, matrix_device_t& result);

        friend __global__ void
        copy_line_cuda(const matrix_device_t& matrix, const matrix_device_t& line, std::size_t row);

        friend __global__ void vector_element_wise_multiplication_cuda(const matrix_device_t& lhs,
                                                                       std::size_t lhs_row,
                                                                       const matrix_device_t& rhs,
                                                                       std::size_t rhs_row,
                                                                       matrix_device_t& result);

        friend __global__ void
        matrix_dot_product_cuda(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);
        friend __global__ void
        matrix_subtract_cuda(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);

    private:
        const std::size_t rows_;
        const std::size_t cols_;
        std::size_t pitch_;

        value_t* data_;

        __device__ void get_val_ptr(std::size_t row, std::size_t col, value_t** val);
        value_t* get_val_ptr(std::size_t row, std::size_t col);
        value_t* get_val_ptr(std::size_t row, std::size_t col) const;
    };

} // namespace utils
