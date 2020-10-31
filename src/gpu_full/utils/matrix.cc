#include "matrix.hh"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace utils
{
    Matrix::Matrix(std::size_t rows, std::size_t cols, value_t value)
        : rows_(rows)
        , cols_(cols)
    {
        this->data_ = static_cast<vector_device_t*>(malloc(sizeof(vector_device_t) * rows));
        for (std::size_t i = 0; i < rows; i++)
        {
            this->data_[i] = static_cast<vector_device_t>(malloc(sizeof(value_t) * cols));
            for (std::size_t j = 0; j < cols; j++)
            {
                this->data_[i][j] = value;
            }
        }
    }

    Matrix::~Matrix()
    {
        for (std::size_t i = 0; i < this->rows_; i++)
        {
            free(this->data_[i]);
        }
        free(this->data_);
    }

    void Matrix::sub_matrix(std::size_t starting_row,
                            std::size_t starting_col,
                            std::size_t row_count,
                            std::size_t col_count,
                            matrix_device_t& result) const
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

    void Matrix::matrix_transpose(matrix_device_t& result) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        for (std::size_t row = 0; row < col_count; row++)
        {
            for (std::size_t col = 0; col < row_count; col++)
            {
                result.data_[row][col] = this->data_[col][row];
            }
        }
    }

    double Matrix::matrix_norm_2() const
    {
        double sum = 0.0;

        for (std::size_t row = 0; row < this->rows_; row++)
        {
            for (std::size_t col = 0; col < this->cols_; col++)
            {
                sum += pow(this->data_[row][col], 2);
            }
        }

        return sqrt(sum);
    }

    void Matrix::matrix_subtract_vector(const matrix_device_t& vector, matrix_device_t& result) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                // considering the vector as a line vector (as returned by the centroid)
                result.data_[row][col] = this->data_[row][col] - vector.data_[0][col];
            }
        }
    }

    void Matrix::matrix_add_vector(const matrix_device_t& vector, matrix_device_t& result) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = this->data_[row][col] + vector.data_[0][col];
            }
        }
    }

    void Matrix::matrix_centroid(matrix_device_t& result) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[0][col] += this->data_[row][col];
            }
        }

        result.data_[0][0] /= row_count;
        result.data_[0][1] /= row_count;
        result.data_[0][2] /= row_count;
    }

    void Matrix::multiply_by_scalar(double val, matrix_device_t& result) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = this->data_[row][col] * val;
            }
        }
    }

    void Matrix::copy_line(const vector_device_t& line, std::size_t row)
    {
        for (std::size_t col = 0; col < this->cols_; col++)
            this->data_[row][col] = line[col];
    }

    void Matrix::copy_line(const parser::vector_host_t& line, std::size_t row)
    {
        for (std::size_t col = 0; col < this->cols_; col++)
            this->data_[row][col] = line[col];
    }

    void Matrix::copy_data(const value_t& data, std::size_t row, std::size_t col)
    {
        this->data_[row][col] = data;
    }
} // namespace utils