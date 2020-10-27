#include <iostream>

#include "icp/icp.hh"
#include "parser/parser.hh"

int main(/*int argc, char* argv[]*/ void)
{
    /* if (argc < 2)
    {
        std::cerr << "[ERROR] takes an argument\n";
        return 1;
    }
    */
    parser::matrix_t vect_point;
    // bool ret = parser::parse_file(argv[1], vect_point);

    parser::vector_t Aone({0, 0, 0});
    parser::vector_t Atwo({0.5, 0.5, 0.5});
    parser::vector_t Athree({1, 1, 1});

    parser::matrix_t A;
    A.push_back(Aone);
    A.push_back(Atwo);
    A.push_back(Athree);

    parser::vector_t Bone({0.5, 0.5, 0.5});
    parser::vector_t Btwo({1, 1, 1});
    parser::vector_t Bthree({1.5, 1.5, 1.5});

    parser::matrix_t B;
    B.push_back(Bone);
    B.push_back(Btwo);
    B.push_back(Bthree);

    parser::matrix_t T;

    icp::icp(A, B, T);

    for (const auto& row : T)
    {
        for (const auto& col: row)
            std::cout << col << " ";
        std::cout << std::endl;
    }

    /*
    if (!ret)
    {
        std::cerr << "[ERROR] Parser" << std::endl;
        return 1;
    }
    */
    return 0;
}