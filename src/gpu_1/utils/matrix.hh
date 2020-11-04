#pragma once

#include <memory>
#include <vector>

#include "gpu_1/parser/parser.hh"

namespace gpu_1::utils
{
    class Matrix
    {
    public:
        using value_t = float;
        using matrix_device_t = Matrix;

        Matrix(std::size_t rows, std::size_t cols, value_t value = 0);
        ~Matrix();

        void sub_matrix(std::size_t starting_row,
                        std::size_t starting_col,
                        std::size_t row_count,
                        std::size_t col_count,
                        matrix_device_t& result) const;
        void matrix_transpose(matrix_device_t& result) const;

        float matrix_norm_2() const;
        void matrix_subtract_vector(const matrix_device_t& vector, matrix_device_t& result) const;
        void matrix_add_vector(const matrix_device_t& vector, matrix_device_t& result) const;

        void matrix_centroid(matrix_device_t& result) const;

        void multiply_by_scalar(float val, matrix_device_t& result) const;

        float matrix_diag_sum() const;

        void set_val(std::size_t row, std::size_t col, value_t val);
        void set_val_ptr(std::size_t row, std::size_t col, value_t* val);

        value_t get_val(std::size_t row, std::size_t col) const;

        void print_matrix() const;

        const std::size_t rows_;
        const std::size_t cols_;
        std::size_t pitch_;

        char* data_;
    };

    float vector_sum(const Matrix::matrix_device_t& vector);

    void matrix_dot_product(const Matrix::matrix_device_t& lhs,
                            const Matrix::matrix_device_t& rhs,
                            Matrix::matrix_device_t& result);
    void vector_element_wise_multiplication(const Matrix::matrix_device_t& lhs,
                                            std::size_t lhs_row,
                                            const Matrix::matrix_device_t& rhs,
                                            std::size_t rhs_row,
                                            Matrix::matrix_device_t& result);
    void matrix_subtract(const Matrix::matrix_device_t& lhs,
                         const Matrix::matrix_device_t& rhs,
                         Matrix::matrix_device_t& result);

    void compute_rotation_matrix(const Matrix::matrix_device_t& q,
                                 Matrix::matrix_device_t& QBar_T,
                                 Matrix::matrix_device_t& Q);

    Matrix::value_t* get_val_ptr(char* data, std::size_t pitch, std::size_t row, std::size_t col);

    // Kernels
    __global__ void sub_matrix_cuda(const char* matrix_data,
                                    std::size_t matrix_pitch,
                                    std::size_t starting_row,
                                    std::size_t starting_col,
                                    std::size_t row_count,
                                    std::size_t col_count,
                                    char* result_data,
                                    std::size_t result_pitch);
    __global__ void matrix_transpose_cuda(const char* matrix_data,
                                          std::size_t matrix_pitch,
                                          std::size_t matrix_rows,
                                          std::size_t matrix_cols,
                                          char* result_data,
                                          std::size_t result_pitch);

    __global__ void matrix_subtract_vector_cuda(const char* matrix_data,
                                                std::size_t matrix_pitch,
                                                std::size_t matrix_rows,
                                                std::size_t matrix_cols,
                                                const char* vector_data,
                                                std::size_t vector_pitch,
                                                char* result_data,
                                                std::size_t result_pitch);
    __global__ void matrix_add_vector_cuda(const char* matrix_data,
                                           std::size_t matrix_pitch,
                                           std::size_t matrix_rows,
                                           std::size_t matrix_cols,
                                           const char* vector_data,
                                           std::size_t vector_pitch,
                                           char* result_data,
                                           std::size_t result_pitch);

    __global__ void multiply_by_scalar_cuda(const char* matrix_data,
                                            std::size_t matrix_pitch,
                                            std::size_t matrix_rows,
                                            std::size_t matrix_cols,
                                            float val,
                                            char* result_data,
                                            std::size_t result_pitch);

    __global__ void vector_element_wise_multiplication_cuda(const char* lhs_data,
                                                            std::size_t lhs_pitch,
                                                            std::size_t lhs_cols,
                                                            std::size_t lhs_row,
                                                            const char* rhs_data,
                                                            std::size_t rhs_pitch,
                                                            std::size_t rhs_cols,
                                                            std::size_t rhs_row,
                                                            char* result_data,
                                                            std::size_t result_pitch);

    __global__ void matrix_subtract_cuda(const char* lhs_data,
                                         std::size_t lhs_pitch,
                                         std::size_t lhs_rows,
                                         std::size_t lhs_cols,
                                         const char* rhs_data,
                                         std::size_t rhs_pitch,
                                         std::size_t rhs_rows,
                                         std::size_t rhs_cols,
                                         char* result_data,
                                         std::size_t result_pitch);
    __global__ void matrix_dot_product_cuda(const char* lhs_data,
                                            std::size_t lhs_pitch,
                                            std::size_t lhs_rows,
                                            std::size_t lhs_cols,
                                            const char* rhs_data,
                                            std::size_t rhs_pitch,
                                            std::size_t rhs_rows,
                                            std::size_t rhs_cols,
                                            char* result_data,
                                            std::size_t result_pitch);

    __device__ void get_val_ptr_const_cuda(const char* data,
                                           std::size_t pitch,
                                           std::size_t row,
                                           std::size_t col,
                                           const Matrix::value_t** val);

    __device__ void
    get_val_ptr_cuda(char* data, std::size_t pitch, std::size_t row, std::size_t col, Matrix::value_t** val);

    __global__ void
    vector_sum_cuda(const char* vector_data, std::size_t vector_pitch, std::size_t vector_cols, float* sum);

    __global__ void matrix_norm_2_cuda(const char* matrix_data,
                                       std::size_t matrix_pitch,
                                       std::size_t matrix_rows,
                                       std::size_t matrix_cols,
                                       float* norm);

    __global__ void matrix_centroid_cuda(const char* matrix_data,
                                         std::size_t matrix_pitch,
                                         std::size_t matrix_rows,
                                         std::size_t matrix_cols,
                                         char* result_data,
                                         std::size_t result_pitch);

    __device__ void compute_distance_cuda(const char* p,
                                          std::size_t p_pitch,
                                          std::size_t p_row,
                                          const char* q,
                                          std::size_t q_pitch,
                                          std::size_t q_row);
    __global__ void get_nearest_neighbors_cuda(const char* P_data,
                                               std::size_t P_pitch,
                                               std::size_t P_rows,
                                               const char* Q_data,
                                               std::size_t Q_pitch,
                                               std::size_t Q_rows,
                                               char* res_data,
                                               std::size_t res_pitch);

    __global__ void
    matrix_diag_sum_cuda(const char* matrix_data, std::size_t matrix_pitch, std::size_t matrix_rows, float* sum);

    __global__ void
    set_val_cuda(char* matrix_data, std::size_t matrix_pitch, std::size_t row, std::size_t col, Matrix::value_t val);
    __global__ void set_val_ptr_cuda(char* matrix_data,
                                     std::size_t matrix_pitch,
                                     std::size_t row,
                                     std::size_t col,
                                     Matrix::value_t* val);

    __global__ void compute_rotation_matrix_cuda(const char* q_data,
                                                 std::size_t q_pitch,
                                                 char* QBar_T_data,
                                                 std::size_t QBar_T_pitch,
                                                 char* Q_data,
                                                 std::size_t Q_pitch);

    __global__ void get_val_cuda(const char* matrix_data,
                                 std::size_t matrix_pitch,
                                 std::size_t row,
                                 std::size_t col,
                                 Matrix::value_t* val);
    __global__ void print_matrix_cuda(const char* matrix, std::size_t pitch, std::size_t rows, std::size_t cols);

} // namespace utils