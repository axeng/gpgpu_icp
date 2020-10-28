#include "icp.hh"

#include "cpu/transform/transform.hh"
#include "cpu/utils/lib-matrix.hh"
#include "cpu/utils/utils.hh"
#include "cpu/utils/uniform-random.hh"

namespace icp
{
    void icp(const matrix_t& M,
             const matrix_t& P,
             double& s,
             matrix_t& R,
             matrix_t& t,
             matrix_t& newP,
             std::size_t max_iterations,
             double threshold)
    {
        if (M.empty() || P.empty() || (M[0].size() != P[0].size()))
        {
            return;
        }

        // ----------------------------------------
        // Initialization
        s = 1.0;

        // R = eye(size(M, 1))
        utils::gen_matrix(M[0].size(), M[0].size(), R);
        for (std::size_t row = 0; row < M[0].size(); row++)
        {
            R[row][row] = 1;
        }

        // t = zeros(size(M, 1), 1)
        utils::gen_matrix(M[0].size(), 1, t, 0);

        // newP = P
        utils::sub_matrix(P, 0, 0, P.size(), P[0].size(), newP);

        auto Np = P.size();
        auto Nm = M.size();
        auto dim = P[0].size();

        // ----------------------------------------
        // Find Correspondences
        for (std::size_t iteration = 1; iteration < max_iterations; iteration++)
        {
            matrix_t Y;
            std::vector<double> distances;
            utils::get_nearest_neighbors(M, P, Y, distances);


            // ----------------------------------------
            // Find Alignment

            // ----------------------------------------
            // Apply Alignment

            // ----------------------------------------
            // Compute Residual Error

        }
    }

    bool find_alignment(const matrix_t& P, const matrix_t& Y, double& s, matrix_t& R, matrix_t& t, double& error)
    {
        auto Np = P.size();
        auto dim_p = Np > 0 ? P[0].size() : 0;

        auto Ny = Y.size();
        auto dim_y = Ny > 0 ? Y[0].size() : 0;

        if (Np != Ny)
        {
            return false;
        }

        if (dim_p != 3 || dim_y != 3)
        {
            return false;
        }

        if (Np < 4)
        {
            return false;
        }

        std::size_t N = Np;

        // ----------------------------------------
        // Zero mean point sets
        matrix_t Mu_p;
        utils::matrix_centroid(P, Mu_p);

        matrix_t Mu_y;
        utils::matrix_centroid(Y, Mu_y);

        matrix_t Pprime;
        utils::matrix_subtract_vector(P, Mu_p, Pprime);

        matrix_t Yprime;
        utils::matrix_subtract_vector(Y, Mu_y, Yprime);


        // ----------------------------------------
        // Quaternion computation
        matrix_t Pprime_T;
        matrix_t Yprime_T;
        utils::matrix_transpose(Pprime, Pprime_T);
        utils::matrix_transpose(Yprime, Yprime_T);

        auto Px = Pprime_T[0];
        auto Py = Pprime_T[1];
        auto Pz = Pprime_T[2];

        auto Yx = Yprime_T[0];
        auto Yy = Yprime_T[1];
        auto Yz = Yprime_T[2];

        vector_t xx;
        utils::vector_element_wise_multiplication(Px, Yx, xx);
        double Sxx = utils::vector_sum(xx);
        vector_t xy;
        utils::vector_element_wise_multiplication(Px, Yy, xy);
        double Sxy = utils::vector_sum(xy);
        vector_t xz;
        utils::vector_element_wise_multiplication(Px, Yz, xz);
        double Sxz = utils::vector_sum(xz);

        vector_t yx;
        utils::vector_element_wise_multiplication(Py, Yx, yx);
        double Syx = utils::vector_sum(yx);
        vector_t yy;
        utils::vector_element_wise_multiplication(Py, Yy, yy);
        double Syy = utils::vector_sum(yy);
        vector_t yz;
        utils::vector_element_wise_multiplication(Py, Yz, yz);
        double Syz = utils::vector_sum(yz);

        vector_t zx;
        utils::vector_element_wise_multiplication(Pz, Yx, zx);
        double Szx = utils::vector_sum(zx);
        vector_t zy;
        utils::vector_element_wise_multiplication(Pz, Yy, zy);
        double Szy = utils::vector_sum(zy);
        vector_t zz;
        utils::vector_element_wise_multiplication(Pz, Yz, zz);
        double Szz = utils::vector_sum(zz);

        matrix_t Nmatrix;
        Nmatrix.emplace_back(vector_t({  Sxx + Syy + Szz,    -Szy + Syz,         Szx - Sxz,          -Syx + Sxy      }));
        Nmatrix.emplace_back(vector_t({  Syz - Szy,          Sxx - Szz - Syy,    Syx + Sxy,          Szx + Sxz       }));
        Nmatrix.emplace_back(vector_t({  -Sxz + Szx,         Sxy + Syx,          Syy - Szz - Sxx,    Szy + Syz       }));
        Nmatrix.emplace_back(vector_t({  Sxy - Syx,          Sxz + Szx,          Syz + Szy,          Szz - Syy - Sxx }));

        matrix_t q;
        power_iteration(Nmatrix, q);

        // ----------------------------------------
        // Rotation matrix computation

        // ----------------------------------------
        // Scaling factor computation

        // ----------------------------------------
        // Translational offset computation

        // ----------------------------------------
        // Residual error computation
    }

    void power_iteration(const matrix_t& A, matrix_t& eigen_vector, std::size_t num_simulations)
    {
        vector_t vector(A[0].size());
        std::generate_n(vector.begin(), A[0].size(), utils::UniformRandom<double>(0.0, 1.1));
        eigen_vector.push_back(vector);

        for (std::size_t simulation = 0; simulation < num_simulations; simulation++)
        {
            matrix_t b_k1;
            utils::matrix_dot_product(A, eigen_vector, b_k1);

            double b_k1_norm = utils::vector_norm_2(b_k1[0]);

            for (std::size_t i = 0; i < eigen_vector[0].size(); i++)
            {
                eigen_vector[0][i] = b_k1[0][i] / b_k1_norm;
            }
        }
    }
} // namespace icp