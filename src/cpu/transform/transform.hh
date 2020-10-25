#pragma once

#include <tuple>
#include <vector>

namespace transform
{
    unsigned int getFitTransform(std::vector<std::tuple<double, double, double>> A, std::vector<std::tuple<double, double, double>> B);

    void getCendroid(const &std::vector<std::tuple<double,double,double>> setPoint,
        const &std::tuple<double,double,double> result);
} // namespace transform