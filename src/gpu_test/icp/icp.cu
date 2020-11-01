#include "icp.hh"

#include <algorithm>
#include <sstream>

#include "gpu_test/utils/lib-matrix.hh"
#include "gpu_test/utils/uniform-random.hh"
#include "gpu_test/utils/utils.hh"

namespace icp
{
    std::size_t icp_gpu(const matrix_t& M,
                        const matrix_t& P,
                        matrix_t& newP,
                        double& err,
                        bool verbose,
                        bool save_results,
                        std::size_t max_iterations,
                        double threshold,
                        std::size_t power_iteration_simulations)
    {
        if (M.empty() || P.empty() || (M.get_cols() != P.get_cols()))
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

        if (save_results)
        {
            newP.matrix_to_csv("../res.csv/res.csv.0");
        }

        // ----------------------------------------
        // Find Correspondences
        matrix_t Y;

        double scaling_factor = 0.0; // s
        matrix_t rotation_matrix; // R
        matrix_t translation_matrix; // t

        matrix_t d(newP.get_rows(), newP.get_cols());
        matrix_t d_T(newP.get_cols(), newP.get_rows());

        matrix_t d_dot_d_T(newP.get_cols(), newP.get_cols());


        std::size_t iteration = 0;
        for (; iteration < max_iterations; iteration++)
        {
            if (verbose)
            {
                std::cout << "----------------------------------------" << std::endl
                          << "Iteration: " << iteration << std::endl;
            }
            
            matrix_t *d_newP, *d_M, *d_Y;
            cudaMalloc((&d_newP, sizeof(double) * newP.get_cols() * newP.get_rows());
            cudaMalloc(&d_M, sizeof(double) * M.get_cols() * M.get_rows());
            cudaMalloc(&d_Y, sizeof(double) * Y.get_cols() * Y.get_rows());

            cudaMemcpy(d_newP, newP, sizeof(double) * newP.get_cols() * newP.get_rows(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_M, M, sizeof(double) * M.get_cols() * M.get_rows(), cudaMemcpyHostToDevice);


            utils::get_nearest_neighbors<<<1, d_newP->get_rows()>>>(d_newP, d_M, d_Y);
            cudaDeviceSynchronize();
            cudaCheckError();

            cudaMemcpy(Y, d_Y, sizeof(double) * Y.get_cols() * Y.get_rows(), cudaMemcpyDeviceToHost);
            cudaThreadSynchronize();

            // ----------------------------------------
            // Find Alignment
            find_alignment(newP, Y, scaling_factor, rotation_matrix, translation_matrix, power_iteration_simulations);

            // ----------------------------------------
            // Apply Alignment
            apply_alignment(newP, scaling_factor, rotation_matrix, translation_matrix, newP);

            // ----------------------------------------
            // Compute Residual Error
            utils::matrix_subtract(Y, newP, d, false);
            d.matrix_transpose(d_T, false);
            utils::matrix_dot_product(d_T, d, d_dot_d_T, false);

            err = 0;

            for (std::size_t i = 0; i < d_dot_d_T.get_rows(); i++)
            {
                err += d_dot_d_T.get_data()[i][i];
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
                newP.matrix_to_csv(filename.str());
            }

            cudaFree(d_newP);
            cudaFree(d_M);
            cuda_Free(d_Y);

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
        s = 0.0;
        R.clear();
        t.clear();

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
        matrix_t Mu_p;
        P.matrix_centroid(Mu_p);

        matrix_t Mu_y;
        Y.matrix_centroid(Mu_y);

        matrix_t Pprime;
        P.matrix_subtract_vector(Mu_p, Pprime);

        matrix_t Yprime;
        Y.matrix_subtract_vector(Mu_y, Yprime);

        // ----------------------------------------
        // Quaternion computation
        matrix_t Pprime_T;
        matrix_t Yprime_T;
        Pprime.matrix_transpose(Pprime_T);
        Yprime.matrix_transpose(Yprime_T);

        const auto& Px = Pprime_T.get_data()[0];
        const auto& Py = Pprime_T.get_data()[1];
        const auto& Pz = Pprime_T.get_data()[2];

        const auto& Yx = Yprime_T.get_data()[0];
        const auto& Yy = Yprime_T.get_data()[1];
        const auto& Yz = Yprime_T.get_data()[2];

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

        Nmatrix.emplace_line(std::initializer_list<double>{Sxx + Syy + Szz, -Szy + Syz, Szx - Sxz, -Syx + Sxy});
        Nmatrix.emplace_line(std::initializer_list<double>{Syz - Szy, Sxx - Szz - Syy, Syx + Sxy, Szx + Sxz});
        Nmatrix.emplace_line(std::initializer_list<double>{-Sxz + Szx, Sxy + Syx, Syy - Szz - Sxx, Szy + Syz});
        Nmatrix.emplace_line(std::initializer_list<double>{Sxy - Syx, Sxz + Szx, Syz + Szy, Szz - Syy - Sxx});

        matrix_t q;
        power_iteration(Nmatrix, q, power_iteration_simulations);

        // ----------------------------------------
        // Rotation matrix computation
        double q0 = q.at(0, 0);
        double q1 = q.at(1, 0);
        double q2 = q.at(2, 0);
        double q3 = q.at(3, 0);

        matrix_t Qbar;
        Qbar.emplace_line(std::initializer_list<double>{q0, q1, q2, q3});
        Qbar.emplace_line(std::initializer_list<double>{-q1, q0, q3, -q2});
        Qbar.emplace_line(std::initializer_list<double>{-q2, -q3, q0, q1});
        Qbar.emplace_line(std::initializer_list<double>{-q3, q2, -q1, q0});

        matrix_t Q;
        Q.emplace_line(std::initializer_list<double>{q0, -q1, -q2, -q3});
        Q.emplace_line(std::initializer_list<double>{q1, q0, q3, -q2});
        Q.emplace_line(std::initializer_list<double>{q2, -q3, q0, q1});
        Q.emplace_line(std::initializer_list<double>{q3, q2, -q1, q0});

        matrix_t R_full;
        utils::matrix_dot_product(Qbar, Q, R_full);

        matrix_t R_full_T;
        R_full.matrix_transpose(R_full_T);

        R_full_T.sub_matrix(1, 1, 3, 3, R);

        // ----------------------------------------
        // Scaling factor computation
        double Sp = 0.0;
        double D = 0.0;

        matrix_t dot_product = matrix_t(1, 1);

        for (std::size_t i = 0; i < N; i++)
        {
            // D = D + Yprime(:,i)' * Yprime(:,i)
            matrix_t Yprime_i;
            Yprime_i.push_line(Yprime.get_data()[i]);

            matrix_t Yprime_i_T;
            Yprime_i.matrix_transpose(Yprime_i_T);

            utils::matrix_dot_product(Yprime_i, Yprime_i_T, dot_product, false);

            D += dot_product.at(0, 0);

            // Sp = Sp + Pprime(:,i)' * Pprime(:,i)
            matrix_t Pprime_i;
            Pprime_i.push_line(Pprime.get_data()[i]);

            matrix_t Pprime_i_T;
            Pprime_i.matrix_transpose(Pprime_i_T);

            utils::matrix_dot_product(Pprime_i, Pprime_i_T, dot_product, false);

            Sp += dot_product.at(0, 0);
        }

        s = sqrt(D / Sp);

        // ----------------------------------------
        // Translational offset computation
        matrix_t s_time_R;
        R.multiply_by_scalar(s, s_time_R);

        matrix_t Mu_p_T;
        Mu_p.matrix_transpose(Mu_p_T);

        matrix_t R_dot_Mu_p;
        utils::matrix_dot_product(s_time_R, Mu_p_T, R_dot_Mu_p);

        matrix_t R_dot_Mu_p_T;
        R_dot_Mu_p.matrix_transpose(R_dot_Mu_p_T);

        utils::matrix_subtract(Mu_y, R_dot_Mu_p_T, t);

        return true;
    }

    void power_iteration(const matrix_t& A, matrix_t& eigen_vector, std::size_t num_simulations)
    {
        vector_t vector(A.get_cols());
        std::generate_n(vector.begin(), A.get_cols(), utils::UniformRandomGPU<double>(0.0, 1.1));
        for (std::size_t i = 0; i < A.get_cols(); i++)
        {
            eigen_vector.emplace_line(std::initializer_list<double>{vector[i]});
        }

        matrix_t b_k1(A.get_rows(), eigen_vector.get_cols(), 0.0);

        for (std::size_t simulation = 0; simulation < num_simulations; simulation++)
        {
            utils::matrix_dot_product(A, eigen_vector, b_k1, false);

            double b_k1_norm = b_k1.matrix_norm_2();

            for (std::size_t i = 0; i < eigen_vector.get_rows(); i++)
            {
                eigen_vector.at(i, 0) = b_k1.at(i, 0) / b_k1_norm;
            }
        }
    }

    void apply_alignment(const matrix_t& P, double s, const matrix_t& R, const matrix_t& t, matrix_t& newP)
    {
        matrix_t s_time_R;
        R.multiply_by_scalar(s, s_time_R);

        matrix_t s_time_R_T;
        s_time_R.matrix_transpose(s_time_R_T);

        matrix_t P_time_R;
        utils::matrix_dot_product(P, s_time_R_T, P_time_R);

        P_time_R.matrix_add_vector(t, newP, false);
    }
} // namespace icp