#pragma once


#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "gpu_test/utils/utils.hh"

namespace parser
{
    using value_t = utils::value_t;
    using vector_t = utils::vector_t;
    using matrix_t = utils::matrix_t;

    bool parse_file(const std::string& path, matrix_t& point_list);
} // namespace parser
