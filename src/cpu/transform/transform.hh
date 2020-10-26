#pragma once

#include <tuple>
#include <vector>
#include <math.h>

namespace transform
{
    unsigned int getFitTransform(std::vector<std::tuple<double, double, double>> A, std::vector<std::tuple<double, double, double>> B);

    std::vector<std::vector<int>> transformToMarix(const std::vector<std::tuple<double,double,double>>& setPoint);


    void getCendroid(const std::vector<std::tuple<double,double,double>>& setPoint,
        const std::tuple<double,double,double>& result);

    void getCentroid(const std::vector<std::vector<double>>& setPoint,
        std::vector<double>& result);

    

    void substractByCentroid(const std::vector<std::tuple<double,double,double>>& setPoint,
        const std::tuple<double,double,double>& centroid, 
        std::vector<std::tuple<double,double,double>>& setPoint result);

    void substractByCentroid(const std::vector<std::vector<double>>& setPoint,
        const std::vector<double>& centroid, 
        std::vector<std::vector<double>>& result);


    void dotProduct(const std::vector<std::tuple<double,double,double>>& firstSet,
        const std::vector<std::tuple<double,double,double>>& secondSet,
        std::vector<std::tuple<double,double,double>>& result);


    void transpose(const std::vector<std::tuple<double,double,double>>& firstSet,
        std::vector<std::tuple<double,double,double>>& result);

    double getDeterminant(const std::vector<std::vector<double>>& setPoint,
        int dimension);

    void svd(std::vector<std::vector<double>> matrix, std::vector<std::vector<double>>& s,
	std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v);

} // namespace transform