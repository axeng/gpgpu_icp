#pragma once

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace parser
{
    using value_t = double;
    using vector_t = std::vector<value_t>;
    using matrix_t = std::vector<vector_t>;

    bool parse_file(const std::string& path, matrix_t& point_list);
} // namespace parser