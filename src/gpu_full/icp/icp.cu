#include "icp.hh"

#include <algorithm>
#include <sstream>

#include "gpu_full/utils/lib-matrix.hh"
#include "gpu_full/utils/uniform-random.hh"
#include "gpu_full/utils/utils.hh"

namespace icp
{
    std::size_t icp_gpu(const matrix_host_t& M_host /*dst*/,
                        const matrix_host_t& P_host /*src*/,
                        matrix_host_t&,
                        double& err,
                        bool verbose,
                        std::size_t max_iterations,
                        double threshold,
                        std::size_t power_iteration_simulations)
    {
        // TODO create device copy
        matrix_device_t M(M_host.size(), M_host[0].size());
        matrix_device_t P(P_host.size(), P_host[0].size());
        matrix_device_t newP(P_host.size(), P_host[0].size());

        for (std::size_t i = 0; i < M_host.size(); i++)
        {
            M.copy_line(M_host[i], i);
        }

        for (std::size_t i = 0; i < P_host.size(); i++)
        {
            P.copy_line(P_host[i], i);
        }

        if (M.get_cols() == 0 || M.get_cols() != P.get_cols())
        {
            return 0;
        }

        // ----------------------------------------
        // Initialization
        // newP = P
        P.sub_matrix(0, 0, P.get_rows(), P.get_cols(), newP);

        auto Np = P.get_rows();
        // auto Nm = M.size();     // FIXME : Unused ?
        // auto dim = P[0].size();  // FIXME : Unused ?

        // ----------------------------------------
        // Find Correspondences
        matrix_device_t Y(newP.get_rows(), newP.get_cols());
        std::vector<double> distances;

        double scaling_factor = 0.0; // s
        matrix_device_t rotation_matrix(3, 3); // R
        matrix_device_t translation_matrix(1, 3); // t

        matrix_device_t d(newP.get_rows(), newP.get_cols());
        matrix_device_t d_T(newP.get_cols(), newP.get_rows());

        matrix_device_t d_dot_d_T(newP.get_cols(), newP.get_cols());

        std::size_t iteration = 0;
        for (; iteration < max_iterations; iteration++)
        {
            if (verbose)
            {
                std::cout << "----------------------------------------" << std::endl
                          << "Iteration: " << iteration << std::endl;
            }

            utils::get_nearest_neighbors(newP, M, Y, distances);

            // ----------------------------------------
            // Find Alignment
            find_alignment(newP, Y, scaling_factor, rotation_matrix, translation_matrix, power_iteration_simulations);

            // ----------------------------------------
            // Apply Alignment
            apply_alignment(newP, scaling_factor, rotation_matrix, translation_matrix, newP);

            // ----------------------------------------
            // Compute Residual Error
            utils::matrix_subtract(Y, newP, d);
            d.matrix_transpose(d_T);
            utils::matrix_dot_product(d_T, d, d_dot_d_T);

            err = 0;

            for (std::size_t i = 0; i < d_dot_d_T.get_rows(); i++)
            {
                err += d_dot_d_T.at(i, i);
            }

            err /= Np;

            if (verbose)
            {
                std::cout << "error: " << err << std::endl;
            }

            if (err < threshold)
            {
                break;
            }
        }

        return iteration;
    }

    bool find_alignment(const matrix_device_t& P,
                        const matrix_device_t& Y,
                        double& s,
                        matrix_device_t& R,
                        matrix_device_t& t,
                        std::size_t power_iteration_simulations)
    {
        s = 0.0;

        auto Np = P.get_rows();
        auto dim_p = Np > 0 ? P.get_cols() : 0;

        auto Ny = Y.get_rows();
        auto dim_y = Ny > 0 ? Y.get_cols() : 0;

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
        matrix_device_t Mu_p(1, P.get_cols());
        P.matrix_centroid(Mu_p);

        matrix_device_t Mu_y(1, Y.get_cols());
        Y.matrix_centroid(Mu_y);

        matrix_device_t Pprime(P.get_rows(), P.get_cols());
        P.matrix_subtract_vector(Mu_p, Pprime);

        matrix_device_t Yprime(Y.get_rows(), Y.get_cols());
        Y.matrix_subtract_vector(Mu_y, Yprime);

        // ----------------------------------------
        // Quaternion computation
        matrix_device_t Pprime_T(Pprime.get_cols(), Pprime.get_rows());
        matrix_device_t Yprime_T(Yprime.get_cols(), Yprime.get_rows());
        Pprime.matrix_transpose(Pprime_T);
        Yprime.matrix_transpose(Yprime_T);

        matrix_device_t xx(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 0, Yprime_T, 0, xx);
        double Sxx = utils::vector_sum(xx);
        matrix_device_t xy(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 0, Yprime_T, 1, xy);
        double Sxy = utils::vector_sum(xy);
        matrix_device_t xz(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 0, Yprime_T, 2, xz);
        double Sxz = utils::vector_sum(xz);

        matrix_device_t yx(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 1, Yprime_T, 0, yx);
        double Syx = utils::vector_sum(yx);
        matrix_device_t yy(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 1, Yprime_T, 1, yy);
        double Syy = utils::vector_sum(yy);
        matrix_device_t yz(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 1, Yprime_T, 2, yz);
        double Syz = utils::vector_sum(yz);

        matrix_device_t zx(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 2, Yprime_T, 0, zx);
        double Szx = utils::vector_sum(zx);
        matrix_device_t zy(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 2, Yprime_T, 1, zy);
        double Szy = utils::vector_sum(zy);
        matrix_device_t zz(1, Pprime_T.get_cols());
        utils::vector_element_wise_multiplication(Pprime_T, 2, Yprime_T, 2, zz);
        double Szz = utils::vector_sum(zz);

        matrix_device_t Nmatrix(4, 4);
        /*Nmatrix.emplace_back(std::initializer_list<double>{ Sxx + Syy + Szz,    Syz - Szy,          -Sxz + Szx, Sxy -
        Syx}); Nmatrix.emplace_back(std::initializer_list<double>{ -Szy + Syz,         Sxx - Szz - Syy,    Sxy + Syx,
        Sxz + Szx}); Nmatrix.emplace_back(std::initializer_list<double>{ Szx - Sxz,          Syx + Sxy,          Syy -
        Szz - Sxx,    Syz + Szy}); Nmatrix.emplace_back(std::initializer_list<double>{ -Syx + Sxy,         Szx + Sxz,
        Szy + Syz,          Szz - Syy - Sxx});*/

        Nmatrix.copy_line(std::initializer_list<double>{Sxx + Syy + Szz, -Szy + Syz, Szx - Sxz, -Syx + Sxy}, 0);
        Nmatrix.copy_line(std::initializer_list<double>{Syz - Szy, Sxx - Szz - Syy, Syx + Sxy, Szx + Sxz}, 1);
        Nmatrix.copy_line(std::initializer_list<double>{-Sxz + Szx, Sxy + Syx, Syy - Szz - Sxx, Szy + Syz}, 2);
        Nmatrix.copy_line(std::initializer_list<double>{Sxy - Syx, Sxz + Szx, Syz + Szy, Szz - Syy - Sxx}, 3);

        matrix_device_t q(Nmatrix.get_cols(), 1);
        power_iteration(Nmatrix, q, power_iteration_simulations);

        // ----------------------------------------
        // Rotation matrix computation
        double q0 = q.at(0, 0);
        double q1 = q.at(1, 0);
        double q2 = q.at(2, 0);
        double q3 = q.at(3, 0);

        matrix_device_t Qbar(4, 4);
        Qbar.copy_line(std::initializer_list<double>{q0, q1, q2, q3}, 0);
        Qbar.copy_line(std::initializer_list<double>{-q1, q0, q3, -q2}, 1);
        Qbar.copy_line(std::initializer_list<double>{-q2, -q3, q0, q1}, 2);
        Qbar.copy_line(std::initializer_list<double>{-q3, q2, -q1, q0}, 3);

        matrix_device_t Q(4, 4);
        Q.copy_line(std::initializer_list<double>{q0, -q1, -q2, -q3}, 0);
        Q.copy_line(std::initializer_list<double>{q1, q0, q3, -q2}, 1);
        Q.copy_line(std::initializer_list<double>{q2, -q3, q0, q1}, 2);
        Q.copy_line(std::initializer_list<double>{q3, q2, -q1, q0}, 3);

        matrix_device_t R_full(Qbar.get_rows(), Q.get_cols());
        utils::matrix_dot_product(Qbar, Q, R_full);

        matrix_device_t R_full_T(R_full.get_cols(), R_full.get_rows());
        R_full.matrix_transpose(R_full_T);

        R_full_T.sub_matrix(1, 1, 3, 3, R);

        // ----------------------------------------
        // Scaling factor computation
        double Sp = 0.0;
        double D = 0.0;

        matrix_device_t dot_product(1, 1);

        for (std::size_t i = 0; i < N; i++)
        {
            // D = D + Yprime(:,i)' * Yprime(:,i)
            matrix_device_t Yprime_i(1, Yprime.get_cols());
            Yprime_i.copy_line(Yprime, i, 0);

            matrix_device_t Yprime_i_T(Yprime_i.get_cols(), Yprime_i.get_rows());
            Yprime_i.matrix_transpose(Yprime_i_T);

            utils::matrix_dot_product(Yprime_i, Yprime_i_T, dot_product);

            D += dot_product.at(0, 0);;

            // Sp = Sp + Pprime(:,i)' * Pprime(:,i)
            matrix_device_t Pprime_i(1, Pprime.get_cols());
            Pprime_i.copy_line(Pprime, i, 0);

            matrix_device_t Pprime_i_T(Pprime_i.get_cols(), Pprime_i.get_rows());
            Pprime_i.matrix_transpose(Pprime_i_T);

            utils::matrix_dot_product(Pprime_i, Pprime_i_T, dot_product);

            Sp += dot_product.at(0, 0);
        }

        s = sqrt(D / Sp);

        // ----------------------------------------
        // Translational offset computation
        matrix_device_t s_time_R(R.get_rows(), R.get_cols());
        R.multiply_by_scalar(s, s_time_R);

        matrix_device_t Mu_p_T(Mu_p.get_cols(), Mu_p.get_rows());
        Mu_p.matrix_transpose(Mu_p_T);

        matrix_device_t R_dot_Mu_p(s_time_R.get_rows(), Mu_p_T.get_cols());
        utils::matrix_dot_product(s_time_R, Mu_p_T, R_dot_Mu_p);

        matrix_device_t R_dot_Mu_p_T(R_dot_Mu_p.get_cols(), R_dot_Mu_p.get_rows());
        R_dot_Mu_p.matrix_transpose(R_dot_Mu_p_T);

        utils::matrix_subtract(Mu_y, R_dot_Mu_p_T, t);

        return true;
    }

    void power_iteration(const matrix_device_t& A, matrix_device_t& eigen_vector, std::size_t num_simulations)
    {
        vector_host_t vector(A.get_cols());
        std::generate_n(vector.begin(), A.get_cols(), utils::UniformRandom<double>(0.0, 1.1));
        for (std::size_t i = 0; i < A.get_cols(); i++)
        {
            eigen_vector.copy_line(std::initializer_list<double>{vector[i]}, i);
        }

        matrix_device_t b_k1(A.get_rows(), eigen_vector.get_cols(), 0.0);

        for (std::size_t simulation = 0; simulation < num_simulations; simulation++)
        {
            utils::matrix_dot_product(A, eigen_vector, b_k1);

            double b_k1_norm = 0.0;
            b_k1.matrix_norm_2(b_k1_norm);

            for (std::size_t i = 0; i < eigen_vector.get_rows(); i++)
            {
                *eigen_vector.get_val_ptr(i, 0) = b_k1.at(i, 0) / b_k1_norm;
            }
        }
    }

    void apply_alignment(const matrix_device_t& P,
                         double s,
                         const matrix_device_t& R,
                         const matrix_device_t& t,
                         matrix_device_t& newP)
    {
        matrix_device_t s_time_R(R.get_rows(), R.get_cols());
        R.multiply_by_scalar(s, s_time_R);

        matrix_device_t s_time_R_T(s_time_R.get_cols(), s_time_R.get_rows());
        s_time_R.matrix_transpose(s_time_R_T);

        matrix_device_t P_time_R(P.get_rows(), s_time_R_T.get_cols());
        utils::matrix_dot_product(P, s_time_R_T, P_time_R);

        P_time_R.matrix_add_vector(t, newP);
    }
} // namespace icp
