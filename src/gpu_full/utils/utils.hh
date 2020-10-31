#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_full/parser/parser.hh"
#include "gpu_full/utils/matrix.hh"

namespace utils
{
    using value_t = Matrix::value_t;

    using vector_host_t = parser::vector_host_t;
    using matrix_host_t = parser::matrix_host_t;

    using vector_device_t = Matrix::vector_device_t;
    using matrix_device_t = Matrix::matrix_device_t;

    double compute_distance(const vector_device_t& p, const vector_device_t& q);
    void get_nearest_neighbors(const matrix_device_t& P,
                               const matrix_device_t& Q,
                               matrix_device_t& res,
                               std::vector<double>& distances);
    unsigned int get_line_count(const std::string& path);

    void save_result(std::size_t iteration, double error);

    void string_split(std::string str, const std::string& delimiter, std::vector<std::string>& words);
} // namespace utils