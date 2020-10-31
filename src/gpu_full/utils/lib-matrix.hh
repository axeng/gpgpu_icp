#pragma once

#include <vector>

#include "gpu_full/parser/parser.hh"
#include "gpu_full/utils/utils.hh"

namespace utils
{
    void matrix_dot_product(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);
    void vector_element_wise_multiplication(const vector_device_t& lhs,
                                            const vector_device_t& rhs,
                                            vector_device_t& result,
                                            std::size_t vector_size);
    void matrix_subtract(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result);

    double vector_sum(const vector_device_t& vector, std::size_t vector_size);
} // namespace utils