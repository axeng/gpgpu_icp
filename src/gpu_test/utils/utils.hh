#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include "gpu_test/icp/icp.hh"
#include "gpu_test/utils/matrix.hh"

namespace utils
{
    using value_t = MatrixGPU::value_t;
    using vector_t = MatrixGPU::vector_t;
    using matrix_t = MatrixGPU::matrix_t;

    #define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

    __device__ double compute_distance(const vector_t& p, const vector_t& q);
    __global__ void get_nearest_neighbors(matrix_t& P, matrix_t& Q, matrix_t& res, int P_rows, int Q_rows);
    unsigned int get_line_count(const std::string& path);

    void string_split(std::string str, const std::string& delimiter, std::vector<std::string>& words);
} // namespace utils
