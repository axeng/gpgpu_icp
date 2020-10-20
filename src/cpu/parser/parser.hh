#pragma once

#include <string>
#include <vector>

namespace parser
{
    bool parse_file(const std::string& path, std::vector<std::tuple<double, double, double>>& point_list);
} // namespace parser