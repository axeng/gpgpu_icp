#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_3/utils/matrix.hh"
#include "gpu_3/parser/parser.hh"

#define MAX_CUDA_THREADS 1024
#define MAX_CUDA_THREADS_X 32
#define MAX_CUDA_THREADS_Y 32

namespace gpu_3::utils
{
#define abortError(msg) utils::_abortError(msg, __FUNCTION__, __LINE__)
    void _abortError(const char* msg, const char* fname, int line);

    using value_t = Matrix::value_t;

    using vector_host_t = parser::vector_host_t;
    using matrix_host_t = parser::matrix_host_t;

    using matrix_device_t = Matrix::matrix_device_t;

    void get_nearest_neighbors_cuda(const matrix_device_t& P, const matrix_device_t& Q, matrix_device_t& res);
    float compute_distance(const value_t* p, const value_t* q);
    void get_nearest_neighbors(const matrix_device_t& P, const matrix_device_t& Q, matrix_device_t& res);
    unsigned int get_line_count(const std::string& path);

    void save_result(std::size_t iteration, float error);

    void string_split(std::string str, const std::string& delimiter, std::vector<std::string>& words);

    value_t* host_matrix_to_ptr(const matrix_host_t& host_matrix);
} // namespace utils