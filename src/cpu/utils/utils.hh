#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "cpu/parser/parser.hh"

namespace utils
{
    using value_t = parser::value_t;
    using vector_t = parser::vector_t;
    using matrix_t = parser::matrix_t;

    double compute_distance(const vector_t& p, const vector_t& q);
    void get_nearest_neighbors(const matrix_t& P, const matrix_t& Q, std::vector<std::tuple<vector_t, vector_t>>& NN, std::vector<double>& distances);
    unsigned int get_line_count(const std::string& path);

    double get_mean_vector(const std::vector<double>& values);
} // namespace utils