#pragma once

#include <vector>

#include "gpu_full/parser/parser.hh"
#include "gpu_full/utils/utils.hh"

namespace utils
{
    void _abortError(const char* msg, const char* fname, int line);
#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

    // Kernels
    __global__ void sub_matrix_cuda(const matrix_device_t& matrix,
                                    std::size_t starting_row,
                                    std::size_t starting_col,
                                    std::size_t row_count,
                                    std::size_t col_count,
                                    matrix_device_t& result);
    __global__ void matrix_transpose_cuda(const matrix_device_t& matrix, matrix_device_t& result);

    __global__ void
    matrix_subtract_vector_cuda(const matrix_device_t& matrix, const matrix_device_t& vector, matrix_device_t& result);
    __global__ void
    matrix_add_vector_cuda(const matrix_device_t& matrix, const matrix_device_t& vector, matrix_device_t& result);

    __global__ void multiply_by_scalar_cuda(const matrix_device_t& matrix, double val, matrix_device_t& result);

    __global__ void copy_line_cuda(const matrix_device_t& matrix, const vector_device_t& line, std::size_t row);

    __global__ void vector_element_wise_multiplication_cuda(const vector_device_t& lhs,
                                                            const vector_device_t& rhs,
                                                            vector_device_t& result,
                                                            std::size_t vector_size);

    // Others
    void matrix_dot_product(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);
    void vector_element_wise_multiplication(const vector_device_t& lhs,
                                            const vector_device_t& rhs,
                                            vector_device_t& result,
                                            std::size_t vector_size);
    void matrix_subtract(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);

    double vector_sum(const vector_device_t& vector, std::size_t vector_size);
} // namespace utils