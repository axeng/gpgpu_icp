#include "matrix.hh"

#include <fstream>
#include <iomanip>
#include <iostream>

#include "lib-matrix.hh"

namespace utils
{
    Matrix::Matrix(std::size_t rows, std::size_t cols, value_t value)
        : rows_(rows)
        , cols_(cols)
    {
        cudaError_t rc = cudaSuccess;
        std::size_t pitch;

        rc = cudaMallocPitch(this->data_, &pitch, cols, rows);
        if (rc)
        {
            abortError("Fail buffer allocation");
        }
    }

    Matrix::~Matrix()
    {
        cudaError_t rc = cudaSuccess;
        rc = cudaFree(this->data_);
        if (rc)
        {
            abortError("Unable to free memory");
        }
    }

    void Matrix::sub_matrix(std::size_t starting_row,
                            std::size_t starting_col,
                            std::size_t row_count,
                            std::size_t col_count,
                            matrix_device_t& result) const
    {
        sub_matrix_cuda<<<1, 1>>>(this, starting_row, starting_col, row_count, col_count, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::matrix_transpose(matrix_device_t& result) const
    {
        matrix_transpose_cuda<<<1, 1>>>(this, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::matrix_norm_2(double& norm) const
    {
        double sum = 0.0;

        for (std::size_t row = 0; row < this->rows_; row++)
        {
            for (std::size_t col = 0; col < this->cols_; col++)
            {
                sum += pow(this->data_[row][col], 2);
            }
        }

        norm = sqrt(sum);
    }

    void Matrix::matrix_subtract_vector(const matrix_device_t& vector, matrix_device_t& result) const
    {
        matrix_subtract_vector_cuda<<<1, 1>>>(this, vector, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::matrix_add_vector(const matrix_device_t& vector, matrix_device_t& result) const
    {
        matrix_add_vector_cuda<<<1, 1>>>(this, vector, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
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
        multiply_by_scalar_cuda<<<1, 1>>>(this, val, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::copy_line(const vector_device_t& line, std::size_t row)
    {
        cudaMemcpy(this->data_[row], line, this->cols_, cudaMemcpyDeviceToDevice);
    }

    void Matrix::copy_line(const parser::vector_host_t& line, std::size_t row)
    {
        // FIXME PAS OUF
        value_t *line_ptr = static_cast<vector_device_t>(malloc(sizeof(value_t) * this->cols_));

        for (std::size_t col = 0; col < this->cols_; col++)
        {
            line_ptr[col] = line[col];
        }

        cudaMemcpy(this->data_[row], line, this->cols_, cudaMemcpyHostToDevice);

        free(line_ptr);
    }

    void Matrix::copy_data(const value_t& data, std::size_t row, std::size_t col)
    {
        cudaMemcpy(this->data_[row][col], &data, sizeof(value_t), cudaMemcpyHostToDevice);
    }
} // namespace utils