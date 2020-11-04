#include <iostream>

#include "matrix.hh"
#include "utils.hh"

namespace gpu_3::utils
{
    [[gnu::noinline]] void _abortError(const char* msg, const char* fname, int line)
    {
        cudaError_t err = cudaGetLastError();
        std::cerr << msg << "(" << fname << ", line: " << line << ")" << std::endl;
        std::cerr << "Error " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }

    void get_nearest_neighbors_cuda(const matrix_device_t& P, const matrix_device_t& Q, matrix_device_t& res)
    {
        int xThreads = MAX_CUDA_THREADS;
        dim3 dim_block(xThreads);

        int xBlocks = (int)ceil((float)P.rows_ / xThreads);
        dim3 dim_grid(xBlocks);

        get_nearest_neighbors_cuda<<<dim_grid, dim_block>>>(
            P.data_, P.pitch_, P.rows_, Q.data_, Q.pitch_, Q.rows_, res.data_, res.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    float compute_distance(const value_t* p, const value_t* q)
    {
        float X1 = p[0];
        float Y1 = p[1];
        float Z1 = p[2];
        float X2 = q[0];
        float Y2 = q[1];
        float Z2 = q[2];

        return sqrt(pow(X2 - X1, 2) + pow(Y2 - Y1, 2) + pow(Z2 - Z1, 2) * 1.0);
    }

    void get_nearest_neighbors(const matrix_device_t& P, const matrix_device_t& Q, matrix_device_t& res)
    {
        value_t* P_host = (value_t*)(malloc(sizeof(value_t) * P.rows_ * P.cols_));
        cudaMemcpy2D(P_host,
                     sizeof(value_t) * P.cols_,
                     P.data_,
                     P.pitch_,
                     sizeof(value_t) * P.cols_,
                     P.rows_,
                     cudaMemcpyDeviceToHost);

        value_t* Q_host = (value_t*)(malloc(sizeof(value_t) * Q.rows_ * Q.cols_));
        cudaMemcpy2D(Q_host,
                     sizeof(value_t) * Q.cols_,
                     Q.data_,
                     Q.pitch_,
                     sizeof(value_t) * Q.cols_,
                     Q.rows_,
                     cudaMemcpyDeviceToHost);

        value_t* res_host = (value_t*)(malloc(sizeof(value_t) * P.rows_ * P.cols_));

        for (std::size_t p_row = 0; p_row < P.rows_; p_row++)
        {
            float min_dist = MAXFLOAT;
            std::size_t choosen_row = 0;

            for (std::size_t q_row = 0; q_row < Q.rows_; q_row++)
            {
                float dist = compute_distance(P_host + (p_row * P.cols_), Q_host + (q_row * Q.cols_));
                if (dist < min_dist)
                {
                    min_dist = dist;
                    choosen_row = q_row;
                }
            }

            for (std::size_t i = 0; i < 3; i++)
            {
                res_host[p_row * P.cols_ + i] = Q_host[choosen_row * Q.cols_ + i];
            }
        }

        cudaMemcpy2D(res.data_,
                     res.pitch_,
                     res_host,
                     sizeof(value_t) * P.cols_,
                     sizeof(value_t) * P.cols_,
                     P.rows_,
                     cudaMemcpyHostToDevice);

        free(P_host);
        free(Q_host);
        free(res_host);
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

    value_t* host_matrix_to_ptr(const matrix_host_t& host_matrix)
    {
        value_t* ptr = (value_t*)malloc(sizeof(value_t) * host_matrix.size() * host_matrix[0].size());

        for (std::size_t row = 0; row < host_matrix.size(); row++)
        {
            for (std::size_t col = 0; col < host_matrix[row].size(); col++)
            {
                ptr[row * host_matrix[row].size() + col] = host_matrix[row][col];
            }
        }

        return ptr;
    }
} // namespace gpu_full::utils