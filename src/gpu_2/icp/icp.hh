#pragma once

#include "gpu_2/parser/parser.hh"
#include "gpu_2/utils/utils.hh"

namespace gpu_2::icp
{
    using value_t = utils::Matrix::value_t;

    using vector_host_t = parser::vector_host_t;
    using matrix_host_t = parser::matrix_host_t;

    using matrix_device_t = utils::Matrix::matrix_device_t;

    std::size_t icp_gpu(const matrix_host_t& M,
                        const matrix_host_t& P,
                        matrix_host_t& newP,
                        float& err,
                        bool verbose = false,
                        std::size_t max_iterations = 20,
                        float threshold = 1e-5,
                        std::size_t power_iteration_simulations = 1000);

    bool find_alignment(const matrix_device_t& P,
                        const matrix_device_t& Y,
                        float& s,
                        matrix_device_t& R,
                        matrix_device_t& t,
                        std::size_t power_iteration_simulations);
    void power_iteration(const matrix_device_t& A, matrix_device_t& eigen_vector, std::size_t num_simulations = 100);
    void apply_alignment(const matrix_device_t& P,
                         float

                             s,
                         const matrix_device_t& R,
                         const matrix_device_t& t,
                         matrix_device_t& newP);
} // namespace gpu_full::icp
