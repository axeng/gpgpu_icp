#include "icp.hh"

#include "cpu/transform/transform.hh"
#include "cpu/utils/lib-matrix.hh"
#include "cpu/utils/utils.hh"

namespace icp
{
    void icp(const matrix_t& A, const matrix_t& B, std::size_t max_iterations, double tolerance)
    {
        if (A.empty() || B.empty() || (A[0].size() != B[0].size()))
        {
            return;
        }

        auto m = A[0].size();

        matrix_t src = utils::gen_matrix(m + 1, A.size(), 1.0);
        matrix_t dst = utils::gen_matrix(m + 1, A.size(), 1.0);

        matrix_t A_T;
        utils::matrix_transpose(A, A_T);
        for (std::size_t row = 0; row < m; row++)
        {
            for (std::size_t col = 0; col < src[row].size(); col++)
            {
                src[row][col] = A_T[row][col];
            }
        }

        matrix_t B_T;
        utils::matrix_transpose(B, B_T);
        for (std::size_t row = 0; row < m; row++)
        {
            for (std::size_t col = 0; col < dst[row].size(); col++)
            {
                dst[row][col] = B_T[row][col];
            }
        }

        // Take a look at homogeneous transformation

        for (std::size_t i = 0; i < max_iterations; i++)
        {
            matrix_t sub_src;
            utils::sub_matrix(src, 0, 0, m, src[0].size(), sub_src);
            matrix_t sub_src_T;
            utils::matrix_transpose(sub_src, sub_src_T);

            matrix_t sub_dst;
            utils::sub_matrix(dst, 0, 0, m, dst[0].size(), sub_dst);
            matrix_t sub_dst_T;
            utils::matrix_transpose(sub_dst, sub_dst_T);

            std::vector<std::tuple<vector_t, vector_t>> nearest_neighbors;
            utils::get_nearest_neighbors(sub_src_T, sub_dst_T, nearest_neighbors);

            // FIXME wtf is he doing with indices in python's code
            matrix_t T;
            transform::get_fit_transform(sub_src_T, sub_dst_T, T);

            // FIXME not sure this actually works
            utils::matrix_dot_product(T, src, src, false);

            // FIXME error (need to see how can we get 'distances', maybe compute them after
        }

        // FIXME run best_fit (get_fit_transform) one last time
    }
} // namespace icp