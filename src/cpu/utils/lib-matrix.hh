#pragma once

#include <vector>

#include "cpu/parser/parser.hh"
#include "cpu/utils/utils.hh"

namespace cpu::utils
{
    void matrix_dot_product(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix = true);
    void vector_element_wise_multiplication(const vector_t& lhs,
                                            const vector_t& rhs,
                                            vector_t& result,
                                            bool init_vector = true);
    void matrix_subtract(const matrix_t& lhs, const matrix_t& rhs, matrix_t& result, bool init_matrix = true);

    double vector_sum(const vector_t& vector);
} // namespace utils