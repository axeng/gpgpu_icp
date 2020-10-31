#pragma once

#include <thrust/host_vector.h>

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace parser
{
    using value_t = double;
    using vector_t = thrust::host_vector<value_t>;
    using matrix_t = thrust::host_vectorr<vector_t>;

    bool parse_file(const std::string& path, matrix_t& point_list);
} // namespace parser
