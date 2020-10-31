#pragma once

#include <memory>
#include <vector>

#include "uniform-random.hh"
#include "gpu_full/parser/parser.hh"

namespace utils
{
    class Matrix
    {
    public:
        using value_t = double;
        using vector_device_t = value_t*;
        using matrix_device_t = Matrix;

        Matrix(std::size_t rows, std::size_t cols, value_t value = 0);
        ~Matrix();

        void sub_matrix(std::size_t starting_row,
                        std::size_t starting_col,
                        std::size_t row_count,
                        std::size_t col_count,
                        matrix_device_t& result) const;
        void matrix_transpose(matrix_device_t& result) const;

        double matrix_norm_2() const;
        void
        matrix_subtract_vector(const matrix_device_t& vector, matrix_device_t& result) const;
        void matrix_add_vector(const matrix_device_t& vector, matrix_device_t& result) const;

        void matrix_centroid(matrix_device_t& result) const;

        void multiply_by_scalar(double val, matrix_device_t& result) const;

        void copy_line(const vector_device_t& line, std::size_t row);
        void copy_line(const parser::vector_host_t& line, std::size_t row);
        void copy_data(const value_t& data, std::size_t row, std::size_t col);

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
            return this->data_[row][col];
        }

        inline value_t at(size_t row, size_t col) const
        {
            return this->data_[row][col];
        }

        inline const auto& get_data() const
        {
            return this->data_;
        }

        friend void matrix_dot_product(const matrix_device_t& lhs,
                                       const matrix_device_t& rhs,
                                       matrix_device_t& result);
        friend void matrix_subtract(const matrix_device_t& lhs,
                                    const matrix_device_t& rhs,
                                    matrix_device_t& result);

    private:
        const std::size_t rows_;
        const std::size_t cols_;

        vector_device_t* data_;
    };

} // namespace utils
