#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <math.h>

namespace utils
{
    using point = std::tuple<double,double,double>;
    using points = std::vector<point>;

    float compute_distance(point p, point q);
    void get_nearest_neighbors(points P, points Q, std::vector<std::tuple<point,point>>& NN);
    unsigned int get_line_count(const std::string& path);
} // namespace utils