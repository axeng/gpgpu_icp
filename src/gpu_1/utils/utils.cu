#include <iostream>

#include "matrix.hh"
#include "utils.hh"

namespace gpu_1::utils
{
    [[gnu::noinline]] void _abortError(const char* msg, const char* fname, int line)
    {
        cudaError_t err = cudaGetLastError();
        std::cerr << msg << "(" << fname << ", line: " << line << ")" << std::endl;
        std::cerr << "Error " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }

    void get_nearest_neighbors(const matrix_device_t& P, const matrix_device_t& Q, matrix_device_t& res)
    {
        auto deviceProp = utils::get_cuda_properties();

        int xThreads = deviceProp.maxThreadsDim[0];
        dim3 dim_block(xThreads);

        int xBlocks = (int) ceil((float)P.rows_ / xThreads);
        dim3 dim_grid(xBlocks);

        get_nearest_neighbors_cuda<<<dim_grid, dim_block>>>(
            P.data_, P.pitch_, P.rows_, Q.data_, Q.pitch_, Q.rows_, res.data_, res.pitch_);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
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

    cudaDeviceProp get_cuda_properties(int dev_id)
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev_id);
        return device_prop;
    }
} // namespace utils