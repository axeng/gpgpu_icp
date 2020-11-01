#include "matrix.hh"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace cpu::utils
{
    Matrix::Matrix(std::size_t rows, std::size_t cols, value_t value)
    {
        matrix_fill(rows, cols, value);
    }

    void Matrix::matrix_fill(std::size_t rows, std::size_t cols, value_t value)
    {
        rows_ = rows;
        cols_ = cols;

        data_.resize(rows);
        for (std::size_t i = 0; i < rows; i++)
        {
            data_[i].resize(cols, value);
        }
    }

    void Matrix::sub_matrix(std::size_t starting_row,
                    std::size_t starting_col,
                    std::size_t row_count,
                    std::size_t col_count,
                    matrix_t& result,
                    bool init_matrix) const
    {
        if (starting_row + row_count > this->rows_ || starting_col + col_count > this->cols_)
        {
            return;
        }

        if (init_matrix)
        {
            result.matrix_fill(row_count, col_count, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = this->data_[row + starting_row][col + starting_col];
            }
        }
    }

    void Matrix::matrix_transpose(matrix_t& result, bool init_matrix) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        if (init_matrix)
        {
            result.matrix_fill(col_count, row_count, 0.0);
        }

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

        for (const auto& row : this->data_)
        {
            for (const auto& element : row)
            {
                sum += pow(element, 2);
            }
        }

        return sqrt(sum);
    }

    void
    Matrix::matrix_subtract_vector(const matrix_t& vector, matrix_t& result, bool init_matrix) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        if (init_matrix)
        {
            result.matrix_fill(row_count, col_count, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                // considering the vector as a line vector (as returned by the centroid)
                result.data_[row][col] = this->data_[row][col] - vector.data_[0][col];
            }
        }
    }

    void Matrix::matrix_add_vector(const matrix_t& vector, matrix_t& result, bool init_matrix) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        if (init_matrix)
        {
            result.matrix_fill(row_count, col_count, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = this->data_[row][col] + vector.data_[0][col];
            }
        }
    }

    void Matrix::matrix_centroid(matrix_t& result, bool init_matrix) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        if (init_matrix)
        {
            result.matrix_fill(1, col_count, 0.0);
        }

        for (const auto& row : this->data_)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[0][col] += row[col];
            }
        }

        result.data_[0][0] /= row_count;
        result.data_[0][1] /= row_count;
        result.data_[0][2] /= row_count;
    }

    void Matrix::multiply_by_scalar(double val, matrix_t& result, bool init_matrix) const
    {
        std::size_t row_count = this->rows_;
        std::size_t col_count = this->cols_;

        if (init_matrix)
        {
            result.matrix_fill(row_count, col_count, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result.data_[row][col] = this->data_[row][col] * val;
            }
        }
    }

    void Matrix::print_matrix() const
    {
        for (const auto& row : this->data_)
        {
            std::cout << "| ";

            for (const auto& element : row)
            {
                std::cout << std::fixed << std::setw(7) << std::setprecision(4) << element << " ";
            }

            std::cout << "|" << std::endl;
        }
    }

    void Matrix::matrix_to_csv(const std::string& path) const
    {
        std::ofstream file;
        file.open(path);

        file << "x,y,z" << std::endl;

        for (const auto& row : this->data_)
        {
            if (row.empty())
                break;

            file << row[0];
            for (std::size_t i = 1; i < row.size(); i++)
            {
                file << ',' << row[i];
            }

            file << std::endl;
        }

        file.close();
    }

} // namespace utils