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
        for (std::size_t iteration = 0; iteration < max_iterations; iteration++)
        {
            matrix_t Y;
            std::vector<double> distances;
            utils::get_nearest_neighbors(M, P, Y, distances);

            // ----------------------------------------
            // Find Alignment
            double scaling_factor = 0.0;
            matrix_t rotation_matrix;
            matrix_t translation_matrix;
            find_alignment(P, Y, scaling_factor, rotation_matrix, translation_matrix);

            // ----------------------------------------
            // Apply Alignment and Compute Residual Error
            matrix_t translated_P;
            apply_alignment(P, scaling_factor, rotation_matrix, translation_matrix, translated_P);

            // ----------------------------------------
            // Compute Residual Error

        }
    }

    bool find_alignment(const matrix_t& P, const matrix_t& Y, double& s, matrix_t& R, matrix_t& t)
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
        auto q0 = q[0][0];
        auto q1 = q[0][1];
        auto q2 = q[0][2];
        auto q3 = q[0][3];

        matrix_t Qbar;
        Qbar.emplace_back(vector_t({q0, -q1, -q2, -q3}));
        Qbar.emplace_back(vector_t({q1, q0, q3, -q2}));
        Qbar.emplace_back(vector_t({q2, -q3, q0, q1}));
        Qbar.emplace_back(vector_t({q3, q2, -q1, q0}));

        matrix_t Q;
        Q.emplace_back(vector_t({q0, q1, q2, q3}));
        Q.emplace_back(vector_t({-q1, q0, q3, -q2}));
        Q.emplace_back(vector_t({-q2, -q3, q0, q1}));
        Q.emplace_back(vector_t({-q3, q2, -q1, q0}));

        matrix_t R_full;
        utils::matrix_dot_product(Qbar, Q, R_full);

        utils::sub_matrix(R_full, 1, 1, 3, 3, R);

        // ----------------------------------------
        // Scaling factor computation
        double Sp = 0.0;
        double D = 0.0;

        matrix_t dot_product = utils::gen_matrix(1, 1);

        for (std::size_t i = 0; i < N; i++)
        {
            // D = D + Yprime(:,i)' * Yprime(:,i)
            matrix_t Yprime_i;
            Yprime_i.push_back(Yprime[i]);

            matrix_t Yprime_i_T;
            utils::matrix_transpose(Yprime_i, Yprime_i_T);

            utils::matrix_dot_product(Yprime_i, Yprime_i_T, dot_product, false);

            D += dot_product[0][0];


            // Sp = Sp + Pprime(:,i)' * Pprime(:,i)
            matrix_t Pprime_i;
            Pprime_i.push_back(Pprime[i]);

            matrix_t Pprime_i_T;
            utils::matrix_transpose(Pprime_i, Pprime_i_T);

            utils::matrix_dot_product(Pprime_i, Pprime_i_T, dot_product, false);

            Sp += dot_product[0][0];
        }

        s = sqrt(D/Sp);

        // ----------------------------------------
        // Translational offset computation
        matrix_t s_time_R;
        utils::multiply_by_scalar(R, s, s_time_R);
        matrix_t R_dot_Mu_p;
        utils::matrix_dot_product(s_time_R, Mu_p, R_dot_Mu_p);

        utils::matrix_subtract(Mu_y, R_dot_Mu_p, t);
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

    void apply_alignment(const matrix_t& P, double s, const matrix_t& R, const matrix_t& t, matrix_t& newP)
    {
        matrix_t s_time_R;
        utils::multiply_by_scalar(R, s, s_time_R);

        matrix_t s_time_R_T;
        utils::matrix_transpose(s_time_R, s_time_R_T);

        matrix_t P_time_R;
        utils::matrix_dot_product(P, s_time_R_T, P_time_R);

        utils::matrix_add_vector(P_time_R, t, newP);
    }
} // namespace icp