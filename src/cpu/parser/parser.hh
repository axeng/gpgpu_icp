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
    bool parse_file(const std::string& path, std::vector<std::tuple<double, double, double>>& point_list);
} // namespace parser