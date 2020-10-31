#include <cmath>
#include <iomanip>

#include "lib-matrix.hh"
#include "matrix.hh"

namespace utils
{
    void matrix_dot_product(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        matrix_dot_product_cuda<<<1, 1>>>(lhs, rhs, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    void vector_element_wise_multiplication(const vector_device_t& lhs,
                                            const vector_device_t& rhs,
                                            vector_device_t& result,
                                            std::size_t vector_size)
    {
        vector_element_wise_multiplication_cuda<<<1, 1>>>(lhs, rhs, result, vector_size);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }

    double vector_sum(const vector_device_t& vector, std::size_t vector_size)
    {
        double sum = 0.0;

        for (std::size_t col = 0; col < vector_size; col++)
        {
            sum += vector[col];
        }

        return sum;
    }

    void matrix_subtract(const matrix_device_t& lhs, const matrix_device_t& rhs, matrix_device_t& result)
    {
        matrix_subtract_cuda<<<1, 1>>>(lhs, rhs, result);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError())
        {
            abortError("Computation Error");
        }
    }
} // namespace utils
