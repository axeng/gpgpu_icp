#pragma once

#include <vector>

#include "uniform-random.hh"

namespace utils
{
    class Matrix
    {
    public:
        using value_t = double;
        using vector_t = std::vector<value_t>;
        using matrix_t = Matrix;

        Matrix() = default;
        Matrix(size_t rows, size_t cols, value_t value = 0);

        void matrix_fill(size_t rows, size_t cols, value_t value = 0);

        void sub_matrix(std::size_t starting_row,
                        std::size_t starting_col,
                        std::size_t row_count,
                        std::size_t col_count,
                        matrix_t& result,
                        bool init_matrix = true) const;
        void matrix_transpose(matrix_t& result, bool init_matrix = true) const;

        double matrix_norm_2() const;
        void
        matrix_subtract_vector(const matrix_t& vector, matrix_t& result, bool init_matrix = true) const;
        void matrix_add_vector(const matrix_t& vector, matrix_t& result, bool init_matrix = true) const;

        void matrix_centroid(matrix_t& result, bool init_matrix = true) const;

        void multiply_by_scalar(double val, matrix_t& result, bool init_matrix = true) const;

        void print_matrix() const;
        void matrix_to_csv(const std::string& path) const;

        inline std::size_t get_rows() const
        {
            return this->rows_;
        }

        inline std::size_t get_cols() const
        {
            return this->cols_;
        }

        inline value_t& at(size_t row, size_t col)
        {
            return this->data_[row][col];
        }

        inline value_t at(size_t row, size_t col) const
        {
            return this->data_[row][col];
        }

        inline const std::vector<vector_t>& get_data() const
        {
            return this->data_;
        }

        inline void push_line(const vector_t& values)
        {
            this->data_.push_back(values);
            this->rows_++;
            this->cols_ = values.size();
        }

        template<class... Args>
        inline void emplace_line(Args&&... values)
        {
            this->data_.emplace_back(values...);
            this->rows_++;
            this->cols_ = this->data_[this->data_.size() - 1].size();
        }

        inline bool empty() const
        {
            return data_.empty();
        }

    private:
        std::size_t rows_ = 0;
        std::size_t cols_ = 0;

        std::vector<vector_t> data_;
    };

} // namespace utils
