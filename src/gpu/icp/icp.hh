#pragma once

#include <thrust/host_vector.h>

#include "gpu/parser/parser.hh"

namespace icp
{
    using value_t = double;
    using vector_t = thrust::device_vector<value_t>;
    using matrix_t = thrust::device_vector<vector_t>;

    std::size_t icp_gpu(const parser::matrix_t& M_host,
                        const parser::matrix_t& P_host,
                        parser::matrix_t& newP_host,
                        double& err,
                        bool verbose = false,
                        std::size_t max_iterations = 200,
                        double threshold = 1e-5,
                        std::size_t power_iteration_simulations = 100);

    bool find_alignment(const matrix_t& P,
                        const matrix_t& Y,
                        double& s,
                        matrix_t& R,
                        matrix_t& t,
                        std::size_t power_iteration_simulations);
    void power_iteration(const matrix_t& A, matrix_t& eigen_vector, std::size_t num_simulations = 100);
    void apply_alignment(const matrix_t& P, double s, const matrix_t& R, const matrix_t& t, matrix_t& newP);
} // namespace icp