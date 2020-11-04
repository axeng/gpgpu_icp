#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace gpu_1::parser
{
    using value_t = float;
    using vector_host_t = std::vector<value_t>;
    using matrix_host_t = std::vector<vector_host_t>;

    bool parse_file(const std::string& path, matrix_host_t& point_list);
} // namespace parser