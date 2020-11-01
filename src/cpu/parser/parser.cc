#include "parser.hh"

#include "cpu/utils/utils.hh"

namespace cpu::parser
{
    bool parse_file(const std::string& path, matrix_t& point_list)
    {
        std::string line;
        std::ifstream file;

        file.open(path, std::ifstream::in);

        if (file.good())
        {
            std::getline(file, line);
            while (std::getline(file, line))
            {
                std::vector<std::string> words;
                utils::string_split(line, ",", words);

                point_list.emplace_line(
                    std::initializer_list<double>{std::stod(words[0]), std::stod(words[1]), std::stod(words[2])});
            }
            return true;
        }
        return false;
    }
} // namespace parser