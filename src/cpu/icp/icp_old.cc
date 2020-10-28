#include "icp.hh"

#include "cpu/transform/transform.hh"
#include "cpu/utils/lib-matrix.hh"
#include "cpu/utils/utils.hh"

namespace icp
{
    void icp(const matrix_t& A, const matrix_t& B, matrix_t& result, std::size_t max_iterations, double tolerance)
    {
        std::cout << "- Start ICP" << std::endl;

        if (A.empty() || B.empty() || (A[0].size() != B[0].size()))
        {
            return;
        }

        auto m = A[0].size();

        matrix_t src = utils::gen_matrix(m + 1, A.size(), 1.0);
        matrix_t dst = utils::gen_matrix(m + 1, A.size(), 1.0);

        matrix_t A_T;
        utils::matrix_transpose(A, A_T);
        // TODO use submatrix
        for (std::size_t row = 0; row < m; row++)
        {
            for (std::size_t col = 0; col < src[row].size(); col++)
            {
                src[row][col] = A_T[row][col];
            }
        }

        matrix_t B_T;
        utils::matrix_transpose(B, B_T);
        // TODO use submatrix
        for (std::size_t row = 0; row < m; row++)
        {
            for (std::size_t col = 0; col < dst[row].size(); col++)
            {
                dst[row][col] = B_T[row][col];
            }
        }

        // Take a look at homogeneous transformation
        double prev_error = 0.0;
        for (std::size_t i = 0; i < max_iterations; i++)
        {
            std::cout << "Iteration: " << i << std::endl;

            matrix_t sub_src;
            utils::sub_matrix(src, 0, 0, m, src[0].size(), sub_src);
            matrix_t sub_src_T;
            utils::matrix_transpose(sub_src, sub_src_T);

            matrix_t sub_dst;
            utils::sub_matrix(dst, 0, 0, m, dst[0].size(), sub_dst);
            matrix_t sub_dst_T;
            utils::matrix_transpose(sub_dst, sub_dst_T);

            matrix_t nearest_neighbors;
            std::vector<double> distances;
            utils::get_nearest_neighbors(sub_src_T, sub_dst_T, nearest_neighbors, distances);

            matrix_t T;
            transform::get_fit_transform(sub_src_T, nearest_neighbors, T);

            utils::matrix_dot_product_copy_rhs(T, src, src, false);

            auto mean_error = utils::get_mean_vector(distances);
            if (abs(prev_error - mean_error) < tolerance)
            {
                break;
            }
            prev_error = mean_error;
        }

        matrix_t sub_src;
        utils::sub_matrix(src, 0, 0, m, src[0].size(), sub_src);
        matrix_t sub_src_T;
        utils::matrix_transpose(sub_src, sub_src_T);

        transform::get_fit_transform(A, sub_src_T, result);
    }
} // namespace icp