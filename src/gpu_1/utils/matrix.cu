#include <fstream>
#include <iomanip>
#include <iostream>

#include "matrix.hh"
#include "utils.hh"

namespace gpu_1::utils
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
        double *norm_device;
        cudaError_t rc = cudaSuccess;
        rc = cudaMalloc(&norm_device, sizeof(double));
        if (rc)
        {
            abortError("Fail buffer allocation");
        }

        matrix_norm_2_cuda<<<1, 1>>>(this->data_, this->pitch_, this->rows_, this->cols_, norm_device);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }

        double norm_host;
        rc = cudaMemcpy(&norm_host, norm_device, sizeof(double), cudaMemcpyDeviceToHost);
        if (rc)
        {
            abortError("Fail buffer copy");
        }

        rc = cudaFree(norm_device);
        if (rc)
        {
            abortError("Fail buffer free");
        }

        return norm_host;
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
        double *sum_device;
        cudaError_t rc = cudaSuccess;
        rc = cudaMalloc(&sum_device, sizeof(double));
        if (rc)
        {
            abortError("Fail buffer allocation");
        }

        matrix_diag_sum_cuda<<<1, 1>>>(this->data_, this->pitch_, this->rows_, sum_device);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }

        double sum_host;
        rc = cudaMemcpy(&sum_host, sum_device, sizeof(double), cudaMemcpyDeviceToHost);
        if (rc)
        {
            abortError("Fail buffer copy");
        }

        rc = cudaFree(sum_device);
        if (rc)
        {
            abortError("Fail buffer free");
        }

        return sum_host;
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
        value_t *val_device;
        cudaError_t rc = cudaSuccess;
        rc = cudaMalloc(&val_device, sizeof(value_t));
        if (rc)
        {
            abortError("Fail buffer allocation");
        }

        get_val_cuda<<<1, 1>>>(this->data_, this->pitch_, row, col, val_device);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }

        double val_host;
        rc = cudaMemcpy(&val_host, val_device, sizeof(value_t), cudaMemcpyDeviceToHost);
        if (rc)
        {
            abortError("Fail buffer copy");
        }

        rc = cudaFree(val_device);
        if (rc)
        {
            abortError("Fail buffer free");
        }

        return val_host;
    }

    void Matrix::print_matrix() const
    {
        std::cout << "rows: " << this->rows_ << " cols: " << this->cols_ << std::endl;
        print_matrix_cuda<<<1, 1>>>(this->data_, this->pitch_, this->rows_, this->cols_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }
} // namespace utils