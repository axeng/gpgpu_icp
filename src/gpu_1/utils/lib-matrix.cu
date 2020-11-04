#include <cmath>
#include <iomanip>

#include "matrix.hh"
#include "utils.hh"

namespace gpu_1::utils
{
    __global__ void sub_matrix_cuda(const char* matrix_data,
                                    std::size_t matrix_pitch,
                                    std::size_t starting_row,
                                    std::size_t starting_col,
                                    std::size_t row_count,
                                    std::size_t col_count,
                                    char* result_data,
                                    std::size_t result_pitch)
    {
        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                value_t* result_ptr;
                const value_t* matrix_ptr;
                get_val_ptr_cuda(result_data, result_pitch, row, col, &result_ptr);
                get_val_ptr_const_cuda(matrix_data, matrix_pitch, row + starting_row, col + starting_col, &matrix_ptr);

                *result_ptr = *matrix_ptr;
            }
        }
    }

    __global__ void matrix_transpose_cuda(const char* matrix_data,
                                          std::size_t matrix_pitch,
                                          std::size_t matrix_rows,
                                          std::size_t matrix_cols,
                                          char* result_data,
                                          std::size_t result_pitch)
    {
        for (std::size_t row = 0; row < matrix_cols; row++)
        {
            for (std::size_t col = 0; col < matrix_rows; col++)
            {
                value_t* result_ptr;
                const value_t* matrix_ptr;
                get_val_ptr_cuda(result_data, result_pitch, row, col, &result_ptr);
                get_val_ptr_const_cuda(matrix_data, matrix_pitch, col, row, &matrix_ptr);

                *result_ptr = *matrix_ptr;
            }
        }
    }

    __global__ void matrix_subtract_vector_cuda(const char* matrix_data,
                                                std::size_t matrix_pitch,
                                                std::size_t matrix_rows,
                                                std::size_t matrix_cols,
                                                const char* vector_data,
                                                std::size_t vector_pitch,
                                                char* result_data,
                                                std::size_t result_pitch)
    {
        for (std::size_t row = 0; row < matrix_rows; row++)
        {
            for (std::size_t col = 0; col < matrix_cols; col++)
            {
                value_t* result_ptr;
                const value_t* matrix_ptr;
                const value_t* vector_ptr;
                get_val_ptr_cuda(result_data, result_pitch, row, col, &result_ptr);
                get_val_ptr_const_cuda(matrix_data, matrix_pitch, row, col, &matrix_ptr);
                get_val_ptr_const_cuda(vector_data, vector_pitch, 0, col, &vector_ptr);

                *result_ptr = *matrix_ptr - *vector_ptr;
            }
        }
    }

    __global__ void matrix_add_vector_cuda(const char* matrix_data,
                                           std::size_t matrix_pitch,
                                           std::size_t matrix_rows,
                                           std::size_t matrix_cols,
                                           const char* vector_data,
                                           std::size_t vector_pitch,
                                           char* result_data,
                                           std::size_t result_pitch)
    {
        for (std::size_t row = 0; row < matrix_rows; row++)
        {
            for (std::size_t col = 0; col < matrix_cols; col++)
            {
                value_t* result_ptr;
                const value_t* matrix_ptr;
                const value_t* vector_ptr;
                get_val_ptr_cuda(result_data, result_pitch, row, col, &result_ptr);
                get_val_ptr_const_cuda(matrix_data, matrix_pitch, row, col, &matrix_ptr);
                get_val_ptr_const_cuda(vector_data, vector_pitch, 0, col, &vector_ptr);

                *result_ptr = *matrix_ptr + *vector_ptr;
            }
        }
    }

    __global__ void multiply_by_scalar_cuda(const char* matrix_data,
                                            std::size_t matrix_pitch,
                                            std::size_t matrix_rows,
                                            std::size_t matrix_cols,
                                            double val,
                                            char* result_data,
                                            std::size_t result_pitch)
    {
        for (std::size_t row = 0; row < matrix_rows; row++)
        {
            for (std::size_t col = 0; col < matrix_cols; col++)
            {
                value_t* result_ptr;
                const value_t* matrix_ptr;
                get_val_ptr_cuda(result_data, result_pitch, row, col, &result_ptr);
                get_val_ptr_const_cuda(matrix_data, matrix_pitch, row, col, &matrix_ptr);

                *result_ptr = *matrix_ptr * val;
            }
        }
    }

    __global__ void matrix_dot_product_cuda(const char* lhs_data,
                                            std::size_t lhs_pitch,
                                            std::size_t lhs_rows,
                                            std::size_t lhs_cols,
                                            const char* rhs_data,
                                            std::size_t rhs_pitch,
                                            std::size_t rhs_rows,
                                            std::size_t rhs_cols,
                                            char* result_data,
                                            std::size_t result_pitch)
    {
        std::size_t row_count = lhs_rows;
        std::size_t col_count = rhs_cols;

        std::size_t common_dim = lhs_cols;

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                value_t* result_ptr;
                get_val_ptr_cuda(result_data, result_pitch, row, col, &result_ptr);

                *result_ptr = 0;

                for (std::size_t k = 0; k < common_dim; k++)
                {
                    const value_t* lhs_ptr;
                    const value_t* rhs_ptr;
                    get_val_ptr_const_cuda(lhs_data, lhs_pitch, row, k, &lhs_ptr);
                    get_val_ptr_const_cuda(rhs_data, rhs_pitch, k, col, &rhs_ptr);

                    *result_ptr += *lhs_ptr * *rhs_ptr;
                }
            }
        }
    }

    __global__ void vector_element_wise_multiplication_cuda(const char* lhs_data,
                                                            std::size_t lhs_pitch,
                                                            std::size_t lhs_cols,
                                                            std::size_t lhs_row,
                                                            const char* rhs_data,
                                                            std::size_t rhs_pitch,
                                                            std::size_t rhs_cols,
                                                            std::size_t rhs_row,
                                                            char* result_data,
                                                            std::size_t result_pitch)
    {
        for (std::size_t i = 0; i < lhs_cols; i++)
        {
            value_t* result_ptr;
            const value_t* lhs_ptr;
            const value_t* rhs_ptr;
            get_val_ptr_cuda(result_data, result_pitch, 0, i, &result_ptr);
            get_val_ptr_const_cuda(lhs_data, lhs_pitch, lhs_row, i, &lhs_ptr);
            get_val_ptr_const_cuda(rhs_data, rhs_pitch, rhs_row, i, &rhs_ptr);

            *result_ptr = *lhs_ptr * *rhs_ptr;
        }
    }

    __global__ void matrix_subtract_cuda(const char* lhs_data,
                                         std::size_t lhs_pitch,
                                         std::size_t lhs_rows,
                                         std::size_t lhs_cols,
                                         const char* rhs_data,
                                         std::size_t rhs_pitch,
                                         std::size_t rhs_rows,
                                         std::size_t rhs_cols,
                                         char* result_data,
                                         std::size_t result_pitch)
    {
        for (std::size_t row = 0; row < lhs_rows; row++)
        {
            for (std::size_t col = 0; col < lhs_cols; col++)
            {
                value_t* result_ptr;
                const value_t* lhs_ptr;
                const value_t* rhs_ptr;
                get_val_ptr_cuda(result_data, result_pitch, row, col, &result_ptr);
                get_val_ptr_const_cuda(lhs_data, lhs_pitch, row, col, &lhs_ptr);
                get_val_ptr_const_cuda(rhs_data, rhs_pitch, row, col, &rhs_ptr);

                *result_ptr = *lhs_ptr - *rhs_ptr;
            }
        }
    }

    __global__ void
    vector_sum_cuda(const char* vector_data, std::size_t vector_pitch, std::size_t vector_cols, double* sum)
    {
        *sum = 0.0;

        for (std::size_t col = 0; col < vector_cols; col++)
        {
            const value_t* data;
            get_val_ptr_const_cuda(vector_data, vector_pitch, 0, col, &data);

            *sum += *data;
        }
    }

    __global__ void matrix_norm_2_cuda(const char* matrix_data,
                                       std::size_t matrix_pitch,
                                       std::size_t matrix_rows,
                                       std::size_t matrix_cols,
                                       double* norm)
    {
        double sum = 0.0;

        for (std::size_t row = 0; row < matrix_rows; row++)
        {
            const value_t* line;
            get_val_ptr_const_cuda(matrix_data, matrix_pitch, row, 0, &line);

            for (std::size_t col = 0; col < matrix_cols; col++)
            {
                sum += pow(line[col], 2);
            }
        }

        *norm = sqrt(sum);
    }

    __global__ void matrix_centroid_cuda(const char* matrix_data,
                                         std::size_t matrix_pitch,
                                         std::size_t matrix_rows,
                                         std::size_t matrix_cols,
                                         char* result_data,
                                         std::size_t result_pitch)
    {
        std::size_t row_count = matrix_rows;
        std::size_t col_count = matrix_cols;

        for (std::size_t row = 0; row < row_count; row++)
        {
            for (std::size_t col = 0; col < col_count; col++)
            {
                value_t* result_ptr;
                const value_t* matrix_ptr;
                get_val_ptr_cuda(result_data, result_pitch, 0, col, &result_ptr);
                get_val_ptr_const_cuda(matrix_data, matrix_pitch, row, col, &matrix_ptr);

                if (row == 0)
                {
                    *result_ptr = *matrix_ptr;
                }
                else
                {
                    *result_ptr += *matrix_ptr;
                }
            }
        }

        value_t* result_ptr;
        get_val_ptr_cuda(result_data, result_pitch, 0, 0, &result_ptr);

        result_ptr[0] /= row_count;
        result_ptr[1] /= row_count;
        result_ptr[2] /= row_count;
    }

    __device__ void compute_distance_cuda(const char* p_data,
                                          std::size_t p_pitch,
                                          std::size_t p_row,
                                          const char* q_data,
                                          std::size_t q_pitch,
                                          std::size_t q_row,
                                          double* distance)
    {
        const double* X1;
        const double* Y1;
        const double* Z1;
        const double* X2;
        const double* Y2;
        const double* Z2;

        get_val_ptr_const_cuda(p_data, p_pitch, p_row, 0, &X1);
        get_val_ptr_const_cuda(p_data, p_pitch, p_row, 1, &Y1);
        get_val_ptr_const_cuda(p_data, p_pitch, p_row, 2, &Z1);
        get_val_ptr_const_cuda(q_data, q_pitch, q_row, 0, &X2);
        get_val_ptr_const_cuda(q_data, q_pitch, q_row, 1, &Y2);
        get_val_ptr_const_cuda(q_data, q_pitch, q_row, 2, &Z2);

        *distance = sqrt(pow(*X2 - *X1, 2) + pow(*Y2 - *Y1, 2) + pow(*Z2 - *Z1, 2) * 1.0);
    }

    __global__ void get_nearest_neighbors_cuda(const char* P_data,
                                               std::size_t P_pitch,
                                               std::size_t P_rows,
                                               const char* Q_data,
                                               std::size_t Q_pitch,
                                               std::size_t Q_rows,
                                               char* res_data,
                                               std::size_t res_pitch)
    {
        std::size_t p_row = blockDim.x * blockIdx.x + threadIdx.x;

        if (p_row >= P_rows)
            return;

        float min_dist = MAXFLOAT;
        std::size_t choosen_row = 0;

        for (std::size_t q_row = 0; q_row < Q_rows; q_row++)
        {
            double dist;
            compute_distance_cuda(P_data, P_pitch, p_row, Q_data, Q_pitch, q_row, &dist);
            if (dist < min_dist)
            {
                min_dist = dist;
                choosen_row = q_row;
            }
        }

        const double* Q_line;
        double* res_line;
        get_val_ptr_const_cuda(Q_data, Q_pitch, choosen_row, 0, &Q_line);
        get_val_ptr_cuda(res_data, res_pitch, p_row, 0, &res_line);

        for (std::size_t i = 0; i < 3; i++)
        {
            res_line[i] = Q_line[i];
        }
    }

    __global__ void
    matrix_diag_sum_cuda(const char* matrix_data, std::size_t matrix_pitch, std::size_t matrix_rows, double* sum)
    {
        *sum = 0.0;

        for (std::size_t row = 0; row < matrix_rows; row++)
        {
            const value_t* matrix_ptr;
            get_val_ptr_const_cuda(matrix_data, matrix_pitch, row, row, &matrix_ptr);

            *sum += *matrix_ptr;
        }
    }

    __device__ void
    get_val_ptr_const_cuda(const char* data, std::size_t pitch, std::size_t row, std::size_t col, const value_t** val)
    {
        *val = (value_t*)((data + row * pitch) + col * sizeof(value_t));
    }

    __device__ void get_val_ptr_cuda(char* data, std::size_t pitch, std::size_t row, std::size_t col, value_t** val)
    {
        *val = (value_t*)((data + row * pitch) + col * sizeof(value_t));
    }

    __global__ void
    set_val_cuda(char* matrix_data, std::size_t matrix_pitch, std::size_t row, std::size_t col, value_t val)
    {
        value_t* val_ptr;
        get_val_ptr_cuda(matrix_data, matrix_pitch, row, col, &val_ptr);
        *val_ptr = val;
    }

    __global__ void
    set_val_ptr_cuda(char* matrix_data, std::size_t matrix_pitch, std::size_t row, std::size_t col, value_t* val)
    {
        value_t* val_ptr;
        get_val_ptr_cuda(matrix_data, matrix_pitch, row, col, &val_ptr);
        *val_ptr = *val;
    }

    __global__ void compute_rotation_matrix_cuda(const char* q_data,
                                                 std::size_t q_pitch,
                                                 char* QBar_T_data,
                                                 std::size_t QBar_T_pitch,
                                                 char* Q_data,
                                                 std::size_t Q_pitch)
    {
        const value_t* q0_ptr;
        const value_t* q1_ptr;
        const value_t* q2_ptr;
        const value_t* q3_ptr;
        get_val_ptr_const_cuda(q_data, q_pitch, 0, 0, &q0_ptr);
        get_val_ptr_const_cuda(q_data, q_pitch, 1, 0, &q1_ptr);
        get_val_ptr_const_cuda(q_data, q_pitch, 2, 0, &q2_ptr);
        get_val_ptr_const_cuda(q_data, q_pitch, 3, 0, &q3_ptr);

        value_t* QBar_T_0_ptr;
        value_t* QBar_T_1_ptr;
        value_t* QBar_T_2_ptr;
        value_t* QBar_T_3_ptr;
        get_val_ptr_cuda(QBar_T_data, QBar_T_pitch, 0, 0, &QBar_T_0_ptr);
        get_val_ptr_cuda(QBar_T_data, QBar_T_pitch, 1, 0, &QBar_T_1_ptr);
        get_val_ptr_cuda(QBar_T_data, QBar_T_pitch, 2, 0, &QBar_T_2_ptr);
        get_val_ptr_cuda(QBar_T_data, QBar_T_pitch, 3, 0, &QBar_T_3_ptr);

        QBar_T_0_ptr[0] = *q0_ptr;
        QBar_T_0_ptr[1] = *q1_ptr;
        QBar_T_0_ptr[2] = *q2_ptr;
        QBar_T_0_ptr[3] = *q3_ptr;
        QBar_T_1_ptr[0] = -*q1_ptr;
        QBar_T_1_ptr[1] = *q0_ptr;
        QBar_T_1_ptr[2] = *q3_ptr;
        QBar_T_1_ptr[3] = -*q2_ptr;
        QBar_T_2_ptr[0] = -*q2_ptr;
        QBar_T_2_ptr[1] = -*q3_ptr;
        QBar_T_2_ptr[2] = *q0_ptr;
        QBar_T_2_ptr[3] = *q1_ptr;
        QBar_T_3_ptr[0] = -*q3_ptr;
        QBar_T_3_ptr[1] = *q2_ptr;
        QBar_T_3_ptr[2] = -*q1_ptr;
        QBar_T_3_ptr[3] = *q0_ptr;

        value_t* Q_0_ptr;
        value_t* Q_1_ptr;
        value_t* Q_2_ptr;
        value_t* Q_3_ptr;
        get_val_ptr_cuda(Q_data, Q_pitch, 0, 0, &Q_0_ptr);
        get_val_ptr_cuda(Q_data, Q_pitch, 1, 0, &Q_1_ptr);
        get_val_ptr_cuda(Q_data, Q_pitch, 2, 0, &Q_2_ptr);
        get_val_ptr_cuda(Q_data, Q_pitch, 3, 0, &Q_3_ptr);

        Q_0_ptr[0] = *q0_ptr;
        Q_0_ptr[1] = -*q1_ptr;
        Q_0_ptr[2] = -*q2_ptr;
        Q_0_ptr[3] = -*q3_ptr;
        Q_1_ptr[0] = *q1_ptr;
        Q_1_ptr[1] = *q0_ptr;
        Q_1_ptr[2] = *q3_ptr;
        Q_1_ptr[3] = -*q2_ptr;
        Q_2_ptr[0] = *q2_ptr;
        Q_2_ptr[1] = -*q3_ptr;
        Q_2_ptr[2] = *q0_ptr;
        Q_2_ptr[3] = *q1_ptr;
        Q_3_ptr[0] = *q3_ptr;
        Q_3_ptr[1] = *q2_ptr;
        Q_3_ptr[2] = -*q1_ptr;
        Q_3_ptr[3] = *q0_ptr;
    }

    __global__ void
    get_val_cuda(const char* matrix_data, std::size_t matrix_pitch, std::size_t row, std::size_t col, value_t* val)
    {
        const value_t* val_ptr;
        get_val_ptr_const_cuda(matrix_data, matrix_pitch, row, col, &val_ptr);
        *val = *val_ptr;
    }

    __global__ void print_matrix_cuda(const char* matrix, std::size_t pitch, std::size_t rows, std::size_t cols)
    {
        for (std::size_t row = 0; row < rows; row++)
        {
            printf("| ");
            for (std::size_t col = 0; col < cols; col++)
            {
                const value_t* val;
                get_val_ptr_const_cuda(matrix, pitch, row, col, &val);

                printf("%f ", *val);
            }
            printf("|\n");
        }
    }

    void matrix_dot_product(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        matrix_dot_product_cuda<<<1, 1>>>(lhs.data_,
                                          lhs.pitch_,
                                          lhs.rows_,
                                          lhs.cols_,
                                          rhs.data_,
                                          rhs.pitch_,
                                          rhs.rows_,
                                          rhs.cols_,
                                          result.data_,
                                          result.pitch_);
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
        vector_element_wise_multiplication_cuda<<<1, 1>>>(lhs.data_,
                                                          lhs.pitch_,
                                                          lhs.cols_,
                                                          lhs_row,
                                                          rhs.data_,
                                                          rhs.pitch_,
                                                          rhs.cols_,
                                                          rhs_row,
                                                          result.data_,
                                                          result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    double vector_sum(const matrix_device_t& vector)
    {
        double* sum_device;
        cudaError_t rc = cudaSuccess;
        rc = cudaMalloc(&sum_device, sizeof(double));
        if (rc)
        {
            abortError("Fail buffer allocation");
        }

        vector_sum_cuda<<<1, 1>>>(vector.data_, vector.pitch_, vector.cols_, sum_device);
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

    void matrix_subtract(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        matrix_subtract_cuda<<<1, 1>>>(lhs.data_,
                                       lhs.pitch_,
                                       lhs.rows_,
                                       lhs.cols_,
                                       rhs.data_,
                                       rhs.pitch_,
                                       rhs.rows_,
                                       rhs.cols_,
                                       result.data_,
                                       result.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void compute_rotation_matrix(const matrix_device_t& q, matrix_device_t& QBar_T, matrix_device_t& Q)
    {
        compute_rotation_matrix_cuda<<<1, 1>>>(q.data_, q.pitch_, QBar_T.data_, QBar_T.pitch_, Q.data_, Q.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    value_t* get_val_ptr(char* data, std::size_t pitch, std::size_t row, std::size_t col)
    {
        return (value_t*)((data + row * pitch) + col * sizeof(value_t));
    }

} // namespace utils