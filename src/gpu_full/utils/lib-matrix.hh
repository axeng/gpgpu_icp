#pragma once

#include <vector>

#include "gpu_full/utils/utils.hh"

namespace utils
{
    void matrix_dot_product(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);
    void vector_element_wise_multiplication(const matrix_device_t& lhs,
                                            std::size_t lhs_row,
                                            const matrix_device_t& rhs,
                                            std::size_t rhs_row,
                                            matrix_device_t& result);
    void matrix_subtract(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);

    double vector_sum(const matrix_device_t& vector);
} // namespace utils