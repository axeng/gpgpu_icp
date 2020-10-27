#pragma once

#include "cpu/parser/parser.hh"

namespace icp
{
    using value_t = parser::value_t;
    using vector_t = parser::vector_t;
    using matrix_t = parser::matrix_t;

    void icp(const matrix_t& A, const matrix_t& B, std::size_t max_iterations = 20, double tolerance = 0.001);
} // namespace icp