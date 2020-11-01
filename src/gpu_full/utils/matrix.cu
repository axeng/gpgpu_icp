#include <fstream>
#include <iomanip>
#include <iostream>

#include "matrix.hh"
#include "utils.hh"

namespace utils
{
    Matrix::Matrix(std::size_t rows, std::size_t cols, value_t value)
        : rows_(rows)
        , cols_(cols)
        , pitch_(0)
        , data_(nullptr)
    {
        cudaError_t rc = cudaSuccess;
        rc = cudaMallocPitch(&this->data_, &this->pitch_, cols * sizeof(value_t), rows);
        if (rc)
        {
            abortError("Fail buffer allocation");
        }
        // FIXME fill with value
    }

    Matrix::~Matrix()
    {
        /*
        cudaError_t rc = cudaSuccess;
        rc = cudaFree(this->data_);
        if (rc)
        {
            abortError("Unable to free memory");
        }
        */
    }

    void Matrix::sub_matrix(std::size_t starting_row,
                            std::size_t starting_col,
                            std::size_t row_count,
                            std::size_t col_count,
                            matrix_device_t& result) const
    {
        sub_matrix_cuda<<<1, 1>>>(
            this->data_, this->pitch_, starting_row, starting_col, row_count, col_count, result.data_, result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::matrix_transpose(matrix_device_t& result) const
    {
        matrix_transpose_cuda<<<1, 1>>>(
            this->data_, this->pitch_, this->rows_, this->cols_, result.data_, result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    double Matrix::matrix_norm_2() const
    {
        double norm = 0.0;
        matrix_norm_2_cuda<<<1, 1>>>(this->data_, this->pitch_, this->rows_, this->cols_, &norm);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
        return norm;
    }

    void Matrix::matrix_subtract_vector(const matrix_device_t& vector, matrix_device_t& result) const
    {
        matrix_subtract_vector_cuda<<<1, 1>>>(this->data_,
                                              this->pitch_,
                                              this->rows_,
                                              this->cols_,
                                              vector.data_,
                                              vector.pitch_,
                                              result.data_,
                                              result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::matrix_add_vector(const matrix_device_t& vector, matrix_device_t& result) const
    {
        matrix_add_vector_cuda<<<1, 1>>>(this->data_,
                                         this->pitch_,
                                         this->rows_,
                                         this->cols_,
                                         vector.data_,
                                         vector.pitch_,
                                         result.data_,
                                         result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::matrix_centroid(matrix_device_t& result) const
    {
        matrix_centroid_cuda<<<1, 1>>>(
            this->data_, this->pitch_, this->rows_, this->cols_, result.data_, result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::multiply_by_scalar(double val, matrix_device_t& result) const
    {
        multiply_by_scalar_cuda<<<1, 1>>>(
            this->data_, this->pitch_, this->rows_, this->cols_, val, result.data_, result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    double Matrix::matrix_diag_sum() const
    {
        double sum = 0;
        matrix_diag_sum_cuda<<<1, 1>>>(this->data_, this->pitch_, this->rows_, &sum);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
        return sum;
    }

    void Matrix::set_val(std::size_t row, std::size_t col, value_t val)
    {
        set_val_cuda<<<1, 1>>>(this->data_, this->pitch_, row, col, val);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void Matrix::set_val_ptr(std::size_t row, std::size_t col, value_t* val)
    {
        set_val_ptr_cuda<<<1, 1>>>(this->data_, this->pitch_, row, col, val);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    value_t Matrix::get_val(std::size_t row, std::size_t col) const
    {
        value_t val;
        get_val_cuda<<<1, 1>>>(this->data_, this->pitch_, row, col, &val);
        return val;
    }
} // namespace utils
