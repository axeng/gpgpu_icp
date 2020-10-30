#include "icp.hh"

#include <sstream>

#include "cpu/utils/lib-matrix.hh"
#include "cpu/utils/uniform-random.hh"
#include "cpu/utils/utils.hh"

namespace icp
{
    std::size_t icp(const matrix_t& M /*dst*/,
                    const matrix_t& P /*src*/,
                    matrix_t& newP,
                    double& err,
                    bool verbose,
                    bool save_results,
                    std::size_t max_iterations,
                    double threshold,
                    std::size_t power_iteration_simulations)
    {
        if (M.empty() || P.empty() || (M[0].size() != P[0].size()))
        {
            return 0;
        }

        // ----------------------------------------
        // Initialization
        // newP = P
        utils::sub_matrix(P, 0, 0, P.size(), P[0].size(), newP);

        auto Np = P.size();
        // auto Nm = M.size();     // FIXME : Unused ?
        // auto dim = P[0].size();  // FIXME : Unused ?

        if (save_results)
        {
            utils::matrix_to_csv(newP, "../res.csv/res.csv.0");
        }

        // ----------------------------------------
        // Find Correspondences
        std::size_t iteration = 0;
        for (; iteration < max_iterations; iteration++)
        {
            if (verbose)
            {
                std::cout << "----------------------------------------" << std::endl
                          << "Iteration: " << iteration << std::endl;
            }

            matrix_t Y;
            std::vector<double> distances;
            utils::get_nearest_neighbors(newP, M, Y, distances);

            // ----------------------------------------
            // Find Alignment
            double scaling_factor = 0.0; // s
            matrix_t rotation_matrix; // R
            matrix_t translation_matrix; // t
            find_alignment(newP, Y, scaling_factor, rotation_matrix, translation_matrix, power_iteration_simulations);

            // ----------------------------------------
            // Apply Alignment
            apply_alignment(newP, scaling_factor, rotation_matrix, translation_matrix, newP);

            // ----------------------------------------
            // Compute Residual Error
            matrix_t d;
            utils::matrix_subtract(Y, newP, d, true);

            matrix_t d_T;
            utils::matrix_transpose(d, d_T);

            matrix_t d_dot_d_T;
            utils::matrix_dot_product(d_T, d, d_dot_d_T);

            err = 0;

            for (std::size_t i = 0; i < d_dot_d_T.size(); i++)
            {
                err += d_dot_d_T[i][i];
            }

            err /= Np;

            if (verbose)
            {
                std::cout << "error: " << err << std::endl;
            }

            if (save_results)
            {
                std::stringstream filename;
                filename << "../res.csv/res.csv." << (iteration + 1);
                utils::matrix_to_csv(newP, filename.str());


            }

            if (err < threshold)
            {
                break;
            }
        }

        return iteration;
    }

    bool find_alignment(const matrix_t& P,
                        const matrix_t& Y,
                        double& s,
                        matrix_t& R,
                        matrix_t& t,
                        std::size_t power_iteration_simulations)
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
        /*Nmatrix.emplace_back(std::initializer_list<double>{ Sxx + Syy + Szz,    Syz - Szy,          -Sxz + Szx, Sxy -
        Syx}); Nmatrix.emplace_back(std::initializer_list<double>{ -Szy + Syz,         Sxx - Szz - Syy,    Sxy + Syx,
        Sxz + Szx}); Nmatrix.emplace_back(std::initializer_list<double>{ Szx - Sxz,          Syx + Sxy,          Syy -
        Szz - Sxx,    Syz + Szy}); Nmatrix.emplace_back(std::initializer_list<double>{ -Syx + Sxy,         Szx + Sxz,
        Szy + Syz,          Szz - Syy - Sxx});*/

        Nmatrix.emplace_back(std::initializer_list<double>{Sxx + Syy + Szz, -Szy + Syz, Szx - Sxz, -Syx + Sxy});
        Nmatrix.emplace_back(std::initializer_list<double>{Syz - Szy, Sxx - Szz - Syy, Syx + Sxy, Szx + Sxz});
        Nmatrix.emplace_back(std::initializer_list<double>{-Sxz + Szx, Sxy + Syx, Syy - Szz - Sxx, Szy + Syz});
        Nmatrix.emplace_back(std::initializer_list<double>{Sxy - Syx, Sxz + Szx, Syz + Szy, Szz - Syy - Sxx});

        matrix_t q;
        power_iteration(Nmatrix, q, power_iteration_simulations);

        // ----------------------------------------
        // Rotation matrix computation
        double q0 = q[0][0];
        double q1 = q[1][0];
        double q2 = q[2][0];
        double q3 = q[3][0];

        matrix_t Qbar;
        Qbar.emplace_back(std::initializer_list<double>{q0, q1, q2, q3});
        Qbar.emplace_back(std::initializer_list<double>{-q1, q0, q3, -q2});
        Qbar.emplace_back(std::initializer_list<double>{-q2, -q3, q0, q1});
        Qbar.emplace_back(std::initializer_list<double>{-q3, q2, -q1, q0});

        matrix_t Q;
        Q.emplace_back(std::initializer_list<double>{q0, -q1, -q2, -q3});
        Q.emplace_back(std::initializer_list<double>{q1, q0, q3, -q2});
        Q.emplace_back(std::initializer_list<double>{q2, -q3, q0, q1});
        Q.emplace_back(std::initializer_list<double>{q3, q2, -q1, q0});

        matrix_t R_full;
        utils::matrix_dot_product(Qbar, Q, R_full);

        matrix_t R_full_T;
        utils::matrix_transpose(R_full, R_full_T);

        utils::sub_matrix(R_full_T, 1, 1, 3, 3, R);

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

        s = sqrt(D / Sp);

        // ----------------------------------------
        // Translational offset computation
        matrix_t s_time_R;
        utils::multiply_by_scalar(R, s, s_time_R);

        matrix_t Mu_p_T;
        utils::matrix_transpose(Mu_p, Mu_p_T);

        matrix_t R_dot_Mu_p;
        utils::matrix_dot_product(s_time_R, Mu_p_T, R_dot_Mu_p);

        matrix_t R_dot_Mu_p_T;
        utils::matrix_transpose(R_dot_Mu_p, R_dot_Mu_p_T);

        utils::matrix_subtract(Mu_y, R_dot_Mu_p_T, t);

        return true;
    }

    void power_iteration(const matrix_t& A, matrix_t& eigen_vector, std::size_t num_simulations)
    {
        vector_t vector(A[0].size());
        std::generate_n(vector.begin(), A[0].size(), utils::UniformRandom<double>(0.0, 1.1));
        for (std::size_t i = 0; i < A[0].size(); i++)
        {
            eigen_vector.emplace_back(std::initializer_list<double>{vector[i]});
        }

        for (std::size_t simulation = 0; simulation < num_simulations; simulation++)
        {
            matrix_t b_k1;
            utils::matrix_dot_product(A, eigen_vector, b_k1);

            double b_k1_norm = utils::matrix_norm_2(b_k1);

            for (std::size_t i = 0; i < eigen_vector.size(); i++)
            {
                eigen_vector[i][0] = b_k1[i][0] / b_k1_norm;
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

        utils::matrix_add_vector(P_time_R, t, newP, false);
    }
} // namespace icp