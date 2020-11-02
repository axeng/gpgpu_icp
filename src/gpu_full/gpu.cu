#include <iostream>

#include "icp/icp.hh"
#include "parser/parser.hh"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "[ERROR] takes an argument\n";
        return 1;
    }


    gpu_full::parser::matrix_host_t A;
    bool ret = gpu_full::parser::parse_file(argv[1], A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    gpu_full::parser::matrix_host_t B;
    ret = gpu_full::parser::parse_file(argv[2], B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    gpu_full::parser::matrix_host_t newP;
    float error = 0;
    gpu_full::icp::icp_gpu(A, B, newP, error, true);

    return 0;
}