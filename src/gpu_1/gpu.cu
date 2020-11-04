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


    gpu_1::parser::matrix_host_t A;
    bool ret = gpu_1::parser::parse_file(argv[1], A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    gpu_1::parser::matrix_host_t B;
    ret = gpu_1::parser::parse_file(argv[2], B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    gpu_1::parser::matrix_host_t newP;
    double error = 0;
    gpu_1::icp::icp_gpu(A, B, newP, error, true);

    return 0;
}