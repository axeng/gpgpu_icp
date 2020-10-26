#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

namespace parser
{
    using point_t = std::vector<double>;
    using points_t = std::vector<point_t>;

    bool parse_file(const std::string& path, points_t& point_list);
} // namespace parser