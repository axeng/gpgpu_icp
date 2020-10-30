#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include "gpu/icp/icp.hh"

namespace utils
{
    using value_t = icp::value_t;
    using vector_t = icp::vector_t;
    using matrix_t = icp::matrix_t;

    double compute_distance(const vector_t& p, const vector_t& q);
    void get_nearest_neighbors(const matrix_t& P, const matrix_t& Q, matrix_t& res, std::vector<double>& distances);
    unsigned int get_line_count(const std::string& path);
} // namespace utils