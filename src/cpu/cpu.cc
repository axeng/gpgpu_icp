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


    parser::matrix_t A;
    bool ret = parser::parse_file(argv[1], A);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    parser::matrix_t B;
    ret = parser::parse_file(argv[2], B);
    if (!ret)
    {
        std::cerr << "[ERROR] Parse A" << std::endl;
        return 1;
    }

    /*parser::vector_t Aone({0, 0, 0});
    parser::vector_t Atwo({0.5, 0.5, 0.5});
    parser::vector_t Athree({1, 1, 1});
    parser::vector_t Afour({1.5, 1.5, 1.5});

    parser::matrix_t A;
    A.push_back(Aone);
    A.push_back(Atwo);
    A.push_back(Athree);
    A.push_back(Afour);

    parser::vector_t Bone({0.5, 0.5, 0.5});
    parser::vector_t Btwo({1, 1, 1});
    parser::vector_t Bthree({1.5, 1.5, 1.5});
    parser::vector_t Bfour({2, 2, 2});

    parser::matrix_t B;
    B.push_back(Bone);
    B.push_back(Btwo);
    B.push_back(Bthree);
    B.push_back(Bfour);

    parser::matrix_t T;
*/
    parser::matrix_t newP;
    std::size_t max_iterations = 200;   // default value set to 200
    double threshold = 0.00001;         // default value set to 1 x 10^-5

    icp::icp(A, B, newP, max_iterations, threshold);
/*
    for (const auto& row : T)
    {
        for (const auto& col: row)
            std::cout << col << " ";
        std::cout << std::endl;
    }*/



    return 0;
}