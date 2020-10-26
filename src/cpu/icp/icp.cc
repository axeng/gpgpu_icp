#include "icp.hh"

#include "cpu/transform/transform.hh"

namespace icp
{
    void icp(const points_t& A, const points_t& B, std::size_t max_iterations=20, double tolerance=0.001)
    {
        if (A.empty() || B.empty() || (A[0].size() != B[0].size()))
        {
            return;
        }

        auto m = A[0].size();

        points_t src;
        for (std::size_t i = 0; i < m + 1; i++)
        {
            point_t sub;
            for (std::size_t j = 0; j < A.size(); j++)
            {
                sub.push_back(1);
            }
            src.push_back(sub);
        }

        points_t dst;
        for (std::size_t i = 0; i < m + 1; i++)
        {
            point_t sub;
            for (std::size_t j = 0; j < A.size(); j++)
            {
                sub.push_back(1);
            }
            src.push_back(sub);
        }

        points_t A_T;
        transform::matrix_transpose(A, A_T);
        for (std::size_t i = 0; i < m; i++)
        {
            for (std::size_t j = 0; j < src[i].size(); j++)
            {
                src[i][j] = A_T[i][j];
            }
        }

        points_t B_T;
        transform::matrix_transpose(B, B_T);
        for (std::size_t i = 0; i < m; i++)
        {
            for (std::size_t j = 0; j < dst[i].size(); j++)
            {
                dst[i][j] = B_T[i][j];
            }
        }

        // Take a look at homogeneous transformation

        for (std::size_t i = 0; i < max_iterations; i++)
        {

            //get_nearest_neighbors(points_t P, points_t Q, std::vector<std::tuple<point_t, point_t>>& NN);
        }
    }
} // namespace icp