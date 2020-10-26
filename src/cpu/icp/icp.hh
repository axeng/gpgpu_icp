#pragma once

#include "cpu/parser/parser.hh"

namespace icp
{
    using point_t = parser::point_t;
    using points_t = parser::points_t;

    void icp(const points_t& A, const points_t& B, std::size_t max_iterations=20, double tolerance=0.001);

}// namespace icp