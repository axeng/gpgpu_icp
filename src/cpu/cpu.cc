#include "parser/parser.hh"

#include <iostream>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "[ERROR] takes an argument\n";
        return 1;
    }
    parser::points_t vect_point;
    bool ret = parser::parse_file(argv[1], vect_point); 
    if (!ret)
    {
        std::cerr << "[ERROR] Parser" << std::endl;
        return 1;
    }
    return 0;

}