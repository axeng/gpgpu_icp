#include "utils.hh"

namespace utils
{
    double compute_distance(const vector_t& p, const vector_t& q)
    {
        double X1 = p[0];
        double Y1 = p[1];
        double Z1 = p[2];
        double X2 = q[0];
        double Y2 = q[1];
        double Z2 = q[2];

        return sqrt(pow(X2 - X1, 2) + pow(Y2 - Y1, 2) + pow(Z2 - Z1, 2) * 1.0);
    }

    __device__ void get_nearest_neighbors(const matrix_t& P, const matrix_t& Q, matrix_t& res)
    {
        /**
        GPU
        */
        int i = threadIdx.x;
        if (i >= P.get_rows())
            return;

        float min_dist = MAXFLOAT;
        vector_t chosen;
        for (size_t ind = 0; ind < Q.get_rows(); ind++)
        {
            auto q_point = Q.get_row(ind)
            auto dist = compute_distance(P.get_row(i), q_point);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    chosen = q_point;
                }
        }
        res.set_data(i, chosen);



        /**
        CPU
        */
        /*
        for (const auto& p_point : P)
        {
            float min_dist = MAXFLOAT;

            vector_t chosen;
            for (const auto& q_point : Q)
            {
                auto dist = compute_distance(p_point, q_point);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    chosen = q_point;
                }
            }
            res.emplace_back(chosen);
        }
        */
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
