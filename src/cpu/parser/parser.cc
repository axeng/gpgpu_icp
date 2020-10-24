#include "parser.hh"

namespace parser
{
    bool parse_file(const std::string& path, std::vector<std::tuple<double, double, double>>& point_list)
    {
        std::string line;
        std::ifstream file;

        file.open(path, std::ifstream::in);
        bool first_line = false;

        if (file.good())
        {
            while (std::getline(file, line))
            {
                if (!first_line)
                {
                    first_line = true;
                    continue;
                }
                std::vector<std::string> words;
                boost::split(words, line, boost::is_any_of(","));
                
                double first = std::stod(words[0]);
                double second = std::stod(words[1]);
                double third = std::stod(words[2]);

                auto tuple = std::make_tuple(first, second, third);
                point_list.push_back(tuple);
            }
            return true;
        }
        return false;
    }  
} 