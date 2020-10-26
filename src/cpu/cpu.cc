#include "parser/parser.hh"
#include "icp/icp.hh"

#include <iostream>

int main(int argc, char *argv[])
{
    /* if (argc < 2)
    {
        std::cerr << "[ERROR] takes an argument\n";
        return 1;
    }
    */
    parser::points_t vect_point;
    //bool ret = parser::parse_file(argv[1], vect_point);

    parser::point_t Aone = {0, 0, 0};
    parser::point_t Atwo = {0.5, 0.5, 0.5};
    parser::point_t Athree = {1, 1, 1};

    parser::points_t A;
    A.push_back(Aone);
    A.push_back(Atwo);
    A.push_back(Athree);

    parser::point_t Bone = {0.5, 0.5, 0.5};
    parser::point_t Btwo = {1, 1, 1};
    parser::point_t Bthree = {1.5, 1.5, 1.5};

    parser::points_t B;
    B.push_back(Bone);
    B.push_back(Btwo);
    B.push_back(Bthree);

    icp::icp(A, B);

    /*
    if (!ret)
    {
        std::cerr << "[ERROR] Parser" << std::endl;
        return 1;
    }
    */
    return 0;

}