#include "utils.hh"

namespace utils
{
    [[gnu::noinline]]
    void _abortError(const char* msg, const char* fname, int line)
    {
        cudaError_t err = cudaGetLastError();
        spdlog::error("{} ({}, line: {})", msg, fname, line);
        spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
        std::exit(1);
    }

    double compute_distance(const vector_device_t& p, const vector_device_t& q)
    {
        double X1 = p[0];
        double Y1 = p[1];
        double Z1 = p[2];
        double X2 = q[0];
        double Y2 = q[1];
        double Z2 = q[2];

        return sqrt(pow(X2 - X1, 2) + pow(Y2 - Y1, 2) + pow(Z2 - Z1, 2) * 1.0);
    }

    void get_nearest_neighbors(const matrix_device_t& P,
                               const matrix_device_t& Q,
                               matrix_device_t& res,
                               std::vector<double>& distances)
    {
        distances.clear();

        for (std::size_t p_row = 0; p_row < P.get_rows(); p_row++)
        {
            float min_dist = MAXFLOAT;

            vector_device_t chosen = NULL;
            for (std::size_t q_row = 0; q_row < Q.get_rows(); q_row++)
            {
                auto dist = compute_distance(P.get_data()[p_row], Q.get_data()[q_row]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    chosen = Q.get_data()[q_row];
                }
            }
            distances.emplace_back(min_dist);
            res.copy_line(chosen, p_row);
        }
    }

    void save_result(std::size_t iteration, double error)
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
