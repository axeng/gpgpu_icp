#include "lib-matrix.hh"

#include <cmath>

namespace utils
{
    matrix_t gen_matrix(std::size_t rows, std::size_t cols, value_t value)
    {
        vector_t row(cols, value);
        return matrix_t(rows, row);
    }

    void gen_matrix(std::size_t rows, std::size_t cols, matrix_t& result, value_t value)
    {
        vector_t row(cols, value);
        for (std::size_t i = 0; i < rows; i++)
        {
            result.push_back(row);
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

        for (std::size_t row = starting_row; row < row_count; row++)
        {
            for (std::size_t col = starting_col; col < col_count; col++)
            {
                result[row - starting_row][col - starting_col] = matrix[row][col];
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

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                result[row][col] = matrix[col][row];
            }
        }
    }

    void matrix_dot_product(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix)
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
                for (std::size_t k = 0; k < col_count; k++)
                {
                    result[row][col] += lhs[row][k] * rhs[k][col];
                }
            }
        }
    }

    void
    matrix_element_wise_multiplication(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix)
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
                result[row][col] = lhs[row][col] * rhs[row][col];
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

    double matrix_sum(const matrix_t& matrix)
    {
        double sum = 0.0;

        for (const auto& row : matrix)
        {
            for (const auto& element : row)
            {
                sum += element;
            }
        }

        return sum;
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

    double vector_norm_2(const vector_t& vector)
    {
        double sum = 0.0;

        for (const auto& element : vector)
        {
            sum += pow(element, 2);
        }

        return sqrt(sum);
    }

    void matrix_dot_product_copy_rhs(const matrix_t& lhs, matrix_t rhs, matrix_t& result, bool init_matrix)
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
                for (std::size_t k = 0; k < col_count; k++)
                {
                    result[row][col] += lhs[row][k] * rhs[k][col];
                }
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

    void matrix_inverse_diagonal(const matrix_t& matrix, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(matrix);
        std::size_t col_count = matrix_col_count(matrix);

        if (init_matrix)
        {
            gen_matrix(row_count, col_count, result, 0.0);
        }

        for (std::size_t row = 0; row < row_count; row++)
        {
            result[row][row] = 1.0 / matrix[row][row];
        }
    }

    void matrix_reduced(const matrix_t& matrix, std::size_t new_size, matrix_t& result, bool init_matrix)
    {
        std::size_t row_count = matrix_row_count(matrix);

        if (init_matrix)
        {
            gen_matrix(row_count, new_size, result, 0.0);
        }

        // FIXME The original code was very weird, it looks like they assumed that the matrix was square
        std::size_t starting_index = row_count - new_size;

        for (std::size_t row = starting_index, new_row = 0; row < row_count; row++, new_row++)
        {
            for (std::size_t col = starting_index, new_col = 0; col < row_count; col++, new_col++)
            {
                result[new_row][new_col] = matrix[row][col];
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

    static double matrix_determinant_rec(const matrix_t& matrix, std::size_t dimension)
    {
        if (dimension == 2)
        {
            return (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1]);
        }

        matrix_t sub_matrix = gen_matrix(matrix_row_count(matrix), matrix_col_count(matrix), 0.0);

        double det = 0.0;

        for (std::size_t x = 0; x < dimension; x++)
        {
            for (std::size_t i = 1; i < dimension; i++)
            {
                for (std::size_t j = 0, sub_j = 0; j < dimension; j++)
                {
                    if (j == x)
                        continue;

                    sub_matrix[i][sub_j] = matrix[i][j];
                    // FIXME in the case where j == x we don't update sub_j, I'm not sure this is normal
                    sub_j++;
                }
            }

            det += (pow(-1, x) * matrix[0][x] * matrix_determinant_rec(sub_matrix, dimension - 1));
        }
        return det;
    }

    double matrix_determinant(const matrix_t& matrix)
    {
        return matrix_determinant_rec(matrix, matrix_row_count(matrix));
    }

    void hermitian_matrix(const vector_t& eigen_vector, matrix_t& result, bool init_matrix)
    {
        std::size_t value_count = eigen_vector.size();

        if (init_matrix)
        {
            gen_matrix(value_count, value_count, result, 0.0);
        }

        result[0][0] = 1 / eigen_vector[0];
        for (std::size_t row = 1; row < value_count; row++)
        {
            result[row][0] = -eigen_vector[row] / eigen_vector[0];
        }

        for (std::size_t row = 1; row < value_count; row++)
        {
            result[row][row] = 1;
        }
    }

    void hermitian_matrix_inverse(const vector_t& eigen_vector, matrix_t& result, bool init_matrix)
    {
        std::size_t value_count = eigen_vector.size();

        if (init_matrix)
        {
            gen_matrix(value_count, value_count, result, 0.0);
        }

        result[0][0] = eigen_vector[0];
        for (std::size_t row = 1; row < value_count; row++)
        {
            result[row][0] = -eigen_vector[row];
        }

        for (std::size_t row = 1; row < value_count; row++)
        {
            result[row][row] = 1;
        }
    }

    void jordan_gaussian_transform(matrix_t matrix, vector_t& result)
    {
        const double eps = 0.000001;
        bool eigen_value_found = false;

        for (std::size_t row = 0; row < matrix.size() - 1 && !eigen_value_found; row++)
        {
            std::size_t col = row;
            double alpha = matrix[row][row];

            while (col < matrix[row].size() && alpha != 0 && alpha != 1)
            {
                matrix[row][col++] /= alpha;
            }

            for (std::size_t col_sec = row; col_sec < matrix[row].size() && alpha == 0; col_sec++)
            {
                std::swap(matrix[row][col_sec], matrix[row + 1][col_sec]);
            }

            for (std::size_t row_sec = 0; row_sec < matrix.size(); row_sec++)
            {
                double gamma = matrix[row_sec][row];

                for (std::size_t col_sec = row; col_sec < matrix[row_sec].size() && row_sec != row; col_sec++)
                {
                    matrix[row_sec][col_sec] = matrix[row_sec][col_sec] - matrix[row][col_sec] * gamma;
                }
            }

            std::size_t row_sec = 0;
            while (row_sec < matrix.size() && (row == matrix.size() - 1 || std::fabs(matrix[row + 1][row + 1]) < eps))
            {
                result.push_back(-matrix[row_sec++][row + 1]);
            }

            if (result.size() == matrix.size())
            {
                eigen_value_found = true;
                result[row + 1] = 1;
                for (std::size_t index = row + 1; index < result.size(); index++)
                {
                    result[index] = (std::fabs(result[index]) >= eps) ? result[index] : 0;
                }
            }
        }
    }

    void multiply_by_scalar(const matrix_t& matrix, double val, matrix_t& result, bool init_matrix = true)
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

    void print_matrix(const matrix_t& matrix)
    {
        for (const auto& row : matrix)
        {
            for (const auto& col : row)
            {
                std::cout << col << " ";
            }
            std::cout << std::endl;
        }
    }

} // namespace utils