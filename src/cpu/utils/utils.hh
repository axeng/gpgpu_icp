#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "cpu/parser/parser.hh"

namespace utils
{
    using point_t = parser::point_t;
    using points_t = parser::points_t;;

    float compute_distance(point_t p, point_t q);
    void get_nearest_neighbors(points_t P, points_t Q, std::vector<std::tuple<point_t, point_t>>& NN);
    unsigned int get_line_count(const std::string& path);
} // namespace utils