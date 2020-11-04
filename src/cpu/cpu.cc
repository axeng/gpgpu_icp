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


    cpu::parser::matrix_t A;
    bool ret = cpu::parser::parse_file(argv[1], A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    cpu::parser::matrix_t B;
    ret = cpu::parser::parse_file(argv[2], B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    cpu::parser::matrix_t newP;
    float error = 0;
    cpu::icp::icp_cpu(A, B, newP, error, true);

    return 0;
}