#include "parser.hh"

#include "gpu_test/utils/utils.hh"

namespace parser
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

                vector_t point = {std::stod(words[0]), std::stod(words[1]), std::stod(words[2])};
                point_list.push_line(point);
            }
            return true;
        }
        return false;
    }
} // namespace parser