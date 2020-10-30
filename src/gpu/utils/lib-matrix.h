#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void gen_matrix(int rows, int cols, thrust::device_vector<thrust::device_vector<double>>& result,
                double value);

thrust::host_vector<thrust::host_vector<double>> gen_matrix(std::size_t rows, std::size_t cols, value_t value = 0.0);
    void gen_matrix(std::size_t rows, std::size_t cols, thrust::host_vector<thrust::host_vector<double>>& result, value_t value = 0.0);

    int matrix_row_count(const thrust::host_vector<thrust::host_vector<double>>& matrix);
    int matrix_col_count(const thrust::host_vector<thrust::host_vector<double>>& matrix);

    void sub_matrix(const thrust::host_vector<thrust::host_vector<double>>& matrix,
                    std::size_t starting_row,
                    std::size_t starting_col,
                    std::size_t row_count,
                    std::size_t col_count,
                    thrust::host_vector<thrust::host_vector<double>>& result,
                    bool init_matrix = true);

    void matrix_transpose(const thrust::host_vector<thrust::host_vector<double>>& matrix, thrust::host_vector<thrust::host_vector<double>>& result, bool init_matrix = true);
    void matrix_dot_product(const thrust::host_vector<thrust::host_vector<double>>& lhs, const thrust::host_vector<thrust::host_vector<double>>& rhs, thrust::host_vector<thrust::host_vector<double>>& result, bool init_matrix = true);
    void vector_element_wise_multiplication(const vector_t& lhs,
                                            const vector_t& rhs,
                                            vector_t& result,
                                            bool init_vector = true);
    double vector_sum(const vector_t& vector);
    double matrix_norm_2(const thrust::host_vector<thrust::host_vector<double>>& matrix);
    void matrix_subtract(const thrust::host_vector<thrust::host_vector<double>>& lhs, const thrust::host_vector<thrust::host_vector<double>>& rhs, thrust::host_vector<thrust::host_vector<double>>& result, bool init_matrix = true);
    void
    matrix_subtract_vector(const thrust::host_vector<thrust::host_vector<double>>& matrix, const thrust::host_vector<thrust::host_vector<double>>& vector, thrust::host_vector<thrust::host_vector<double>>& result, bool init_matrix = true);
    void matrix_add_vector(const thrust::host_vector<thrust::host_vector<double>>& matrix, const thrust::host_vector<thrust::host_vector<double>>& vector, thrust::host_vector<thrust::host_vector<double>>& result, bool init_matrix = true);

    void matrix_centroid(const thrust::host_vector<thrust::host_vector<double>>& matrix, thrust::host_vector<thrust::host_vector<double>>& result, bool init_matrix = true);

    void multiply_by_scalar(const thrust::host_vector<thrust::host_vector<double>>& matrix, double val, thrust::host_vector<thrust::host_vector<double>>& result, bool init_matrix = true);

    void print_matrix(const thrust::host_vector<thrust::host_vector<double>>& matrix);
    void thrust::host_vector<thrust::host_vector<double>>o_csv(const thrust::host_vector<thrust::host_vector<double>>& matrix, const std::string& path);
