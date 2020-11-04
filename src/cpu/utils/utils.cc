#include "utils.hh"

namespace cpu::utils
{
    float compute_distance(const vector_t& p, const vector_t& q)
    {
        float X1 = p[0];
        float Y1 = p[1];
        float Z1 = p[2];
        float X2 = q[0];
        float Y2 = q[1];
        float Z2 = q[2];

        return sqrt(pow(X2 - X1, 2) + pow(Y2 - Y1, 2) + pow(Z2 - Z1, 2) * 1.0);
    }

    void get_nearest_neighbors(const matrix_t& P, const matrix_t& Q, matrix_t& res, std::vector<float>& distances)
    {
        res.clear();
        distances.clear();

        for (const auto& p_point : P.get_data())
        {
            float min_dist = MAXFLOAT;

            vector_t chosen;
            for (const auto& q_point : Q.get_data())
            {
                auto dist = compute_distance(p_point, q_point);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    chosen = q_point;
                }
            }
            distances.emplace_back(min_dist);
            res.emplace_line(chosen);
        }
    }

    void save_result(std::size_t iteration, float error)
    {
        std::ofstream file;
        // FIXME append mode
        file.open("results.csv");

        file << iteration << ',' << error;

        file.close();
    }

    void string_split(std::string str, const std::string& delimiter, std::vector<std::string>& words)
    {
        std::size_t position = 0;
        std::string word;

        while ((position = str.find(delimiter)) != std::string::npos)
        {
            word = str.substr(0, position);
            words.push_back(word);
            str.erase(0, position + delimiter.length());
        }

        word = str.substr(0, position);
        words.push_back(word);
    }
} // namespace utils
