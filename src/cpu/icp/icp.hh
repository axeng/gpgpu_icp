#pragma once

#include "cpu/parser/parser.hh"

namespace icp
{
    using value_t = parser::value_t;
    using vector_t = parser::vector_t;
    using matrix_t = parser::matrix_t;

    void icp(const matrix_t& M,
             const matrix_t& P,
             double& s,
             matrix_t& R,
             matrix_t& t,
             matrix_t& newP,
             std::size_t max_iterations = 200,
             double threshold = 1e-5);

    bool find_alignment(const matrix_t& P, const matrix_t& Y, double& s, matrix_t& R, matrix_t& t, double& error);
    void power_iteration(const matrix_t& A, matrix_t& eigen_vector, std::size_t num_simulations=10);
} // namespace icp