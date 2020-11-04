#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "cpu/parser/parser.hh"
#include "cpu/utils/matrix.hh"

namespace cpu::utils
{
    using value_t = Matrix::value_t;
    using vector_t = Matrix::vector_t;
    using matrix_t = Matrix::matrix_t;

    float compute_distance(const vector_t& p, const vector_t& q);
    void get_nearest_neighbors(const matrix_t& P, const matrix_t& Q, matrix_t& res, std::vector<float>& distances);
    unsigned int get_line_count(const std::string& path);

    void save_result(std::size_t iteration, float error);

    void string_split(std::string str, const std::string& delimiter, std::vector<std::string>& words);
} // namespace utils