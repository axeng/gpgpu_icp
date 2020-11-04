#include <algorithm>
#include <sstream>

#include "gpu_1/icp/icp.hh"
#include "gpu_1/utils/uniform-random.hh"

namespace gpu_1::icp
{
    std::size_t icp_gpu(const matrix_host_t& M_host /*dst*/,
                        const matrix_host_t& P_host /*src*/,
                        matrix_host_t& newP_host,
                        float& err,
                        bool verbose,
                        std::size_t max_iterations,
                        float threshold,
                        std::size_t power_iteration_simulations)
    {
        matrix_device_t M(M_host.size(), M_host[0].size());
        matrix_device_t P(P_host.size(), P_host[0].size());
        matrix_device_t newP(P_host.size(), P_host[0].size());

        auto M_host_ptr = utils::host_matrix_to_ptr(M_host);
        cudaError_t rc = cudaSuccess;
        rc = cudaMemcpy2D(M.data_,
                          M.pitch_,
                          M_host_ptr,
                          sizeof(value_t) * M_host[0].size(),
                          sizeof(value_t) * M_host[0].size(),
                          M_host.size(),
                          cudaMemcpyHostToDevice);
        if (rc)
        {
            abortError("Fail buffer copy");
        }
        free(M_host_ptr);

        auto P_host_ptr = utils::host_matrix_to_ptr(P_host);
        rc = cudaMemcpy2D(P.data_,
                          P.pitch_,
                          P_host_ptr,
                          sizeof(value_t) * P_host[0].size(),
                          sizeof(value_t) * P_host[0].size(),
                          P_host.size(),
                          cudaMemcpyHostToDevice);
        if (rc)
        {
            abortError("Fail buffer copy");
        }
        free(P_host_ptr);

        if (M.cols_ == 0 || M.cols_ != P.cols_)
        {
            return 0;
        }

        // ----------------------------------------
        // Initialization
        // newP = P
        P.sub_matrix(0, 0, P.rows_, P.cols_, newP);

        auto Np = P.rows_;

        // ----------------------------------------
        // Find Correspondences
        matrix_device_t Y(newP.rows_, newP.cols_);

        float scaling_factor = 0.0; // s
        matrix_device_t rotation_matrix(3, 3); // R
        matrix_device_t translation_matrix(1, 3); // t

        matrix_device_t d(newP.rows_, newP.cols_);
        matrix_device_t d_T(newP.cols_, newP.rows_);

        matrix_device_t d_dot_d_T(newP.cols_, newP.cols_);

        std::size_t iteration = 0;
        for (; iteration < max_iterations; iteration++)
        {
            if (verbose)
            {
                std::cout << "----------------------------------------" << std::endl
                          << "Iteration: " << iteration << std::endl;
            }

            utils::get_nearest_neighbors(newP, M, Y);

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

            err = d_dot_d_T.matrix_diag_sum() / Np;

            if (verbose)
            {
                std::cout << "error: " << err << std::endl;
            }

            if (err < threshold)
            {
                break;
            }
        }

        value_t* newP_host_ptr = (value_t*)malloc(sizeof(value_t) * P.cols_ * P.rows_);
        rc = cudaMemcpy2D(newP_host_ptr,
                          sizeof(value_t) * P.cols_,
                          newP.data_,
                          newP.pitch_,
                          sizeof(value_t) * newP.cols_,
                          newP.rows_,
                          cudaMemcpyDeviceToHost);
        if (rc)
        {
            abortError("Fail buffer copy");
        }

        newP_host.resize(P.rows_);
        for (std::size_t row = 0; row < P.rows_; row++)
        {
            newP_host[row].resize(P.cols_);
            for (std::size_t col = 0; col < P.cols_; col++)
            {
                newP_host[row][col] = newP_host_ptr[row * P.cols_ + col];
            }
        }

        free(newP_host_ptr);
        return iteration;
    }

    bool find_alignment(const matrix_device_t& P,
                        const matrix_device_t& Y,
                        float& s,
                        matrix_device_t& R,
                        matrix_device_t& t,
                        std::size_t power_iteration_simulations)
    {
        s = 0.0;

        auto Np = P.rows_;
        auto dim_p = Np > 0 ? P.cols_ : 0;

        auto Ny = Y.rows_;
        auto dim_y = Ny > 0 ? Y.cols_ : 0;

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
        matrix_device_t Mu_p(1, P.cols_);
        P.matrix_centroid(Mu_p);

        matrix_device_t Mu_y(1, Y.cols_);
        Y.matrix_centroid(Mu_y);

        matrix_device_t Pprime(P.rows_, P.cols_);
        P.matrix_subtract_vector(Mu_p, Pprime);

        matrix_device_t Yprime(Y.rows_, Y.cols_);
        Y.matrix_subtract_vector(Mu_y, Yprime);

        // ----------------------------------------
        // Quaternion computation
        matrix_device_t Pprime_T(Pprime.cols_, Pprime.rows_);
        matrix_device_t Yprime_T(Yprime.cols_, Yprime.rows_);
        Pprime.matrix_transpose(Pprime_T);
        Yprime.matrix_transpose(Yprime_T);

        matrix_device_t xx(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 0, Yprime_T, 0, xx);
        float Sxx = utils::vector_sum(xx);
        matrix_device_t xy(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 0, Yprime_T, 1, xy);
        float Sxy = utils::vector_sum(xy);
        matrix_device_t xz(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 0, Yprime_T, 2, xz);
        float Sxz = utils::vector_sum(xz);

        matrix_device_t yx(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 1, Yprime_T, 0, yx);
        float Syx = utils::vector_sum(yx);
        matrix_device_t yy(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 1, Yprime_T, 1, yy);
        float Syy = utils::vector_sum(yy);
        matrix_device_t yz(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 1, Yprime_T, 2, yz);
        float Syz = utils::vector_sum(yz);

        matrix_device_t zx(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 2, Yprime_T, 0, zx);
        float Szx = utils::vector_sum(zx);
        matrix_device_t zy(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 2, Yprime_T, 1, zy);
        float Szy = utils::vector_sum(zy);
        matrix_device_t zz(1, Pprime_T.cols_);
        utils::vector_element_wise_multiplication(Pprime_T, 2, Yprime_T, 2, zz);
        float Szz = utils::vector_sum(zz);

        matrix_device_t Nmatrix(4, 4);
        Nmatrix.set_val(0, 0, Sxx + Syy + Szz);
        Nmatrix.set_val(0, 1, -Szy + Syz);
        Nmatrix.set_val(0, 2, Szx - Sxz);
        Nmatrix.set_val(0, 3, -Syx + Sxy);
        Nmatrix.set_val(1, 0, Syz - Szy);
        Nmatrix.set_val(1, 1, Sxx - Szz - Syy);
        Nmatrix.set_val(1, 2, Syx + Sxy);
        Nmatrix.set_val(1, 3, Szx + Sxz);
        Nmatrix.set_val(2, 0, -Sxz + Szx);
        Nmatrix.set_val(2, 1, Sxy + Syx);
        Nmatrix.set_val(2, 2, Syy - Szz - Sxx);
        Nmatrix.set_val(2, 3, Szy + Syz);
        Nmatrix.set_val(3, 0, Sxy - Syx);
        Nmatrix.set_val(3, 1, Sxz + Szx);
        Nmatrix.set_val(3, 2, Syz + Szy);
        Nmatrix.set_val(3, 3, Szz - Syy - Sxx);

        matrix_device_t q(Nmatrix.cols_, 1);
        power_iteration(Nmatrix, q, power_iteration_simulations);

        // ----------------------------------------
        // Rotation matrix computation
        matrix_device_t Qbar_T(4, 4);
        matrix_device_t Q(4, 4);
        utils::compute_rotation_matrix(q, Qbar_T, Q);

        matrix_device_t R_full(Qbar_T.rows_, Q.cols_);
        utils::matrix_dot_product(Qbar_T, Q, R_full);

        matrix_device_t R_full_T(R_full.cols_, R_full.rows_);
        R_full.matrix_transpose(R_full_T);

        R_full_T.sub_matrix(1, 1, 3, 3, R);

        // ----------------------------------------
        // Scaling factor computation
        float Sp = 0.0;
        float D = 0.0;

        matrix_device_t dot_product(1, 1);

        for (std::size_t i = 0; i < N; i++)
        {
            // D = D + Yprime(:,i)' * Yprime(:,i)
            matrix_device_t Yprime_i(1, Yprime.cols_);
            for (std::size_t col = 0; col < Yprime.cols_; col++)
            {
                Yprime_i.set_val_ptr(0, col, utils::get_val_ptr(Yprime.data_, Yprime.pitch_, i, col));
            }

            matrix_device_t Yprime_i_T(Yprime_i.cols_, Yprime_i.rows_);
            Yprime_i.matrix_transpose(Yprime_i_T);

            utils::matrix_dot_product(Yprime_i, Yprime_i_T, dot_product);

            D += dot_product.get_val(0, 0);

            // Sp = Sp + Pprime(:,i)' * Pprime(:,i)
            matrix_device_t Pprime_i(1, Pprime.cols_);
            for (std::size_t col = 0; col < Yprime.cols_; col++)
            {
                Pprime_i.set_val_ptr(0, col, utils::get_val_ptr(Pprime.data_, Pprime.pitch_, i, col));
            }

            matrix_device_t Pprime_i_T(Pprime_i.cols_, Pprime_i.rows_);
            Pprime_i.matrix_transpose(Pprime_i_T);

            utils::matrix_dot_product(Pprime_i, Pprime_i_T, dot_product);

            Sp += dot_product.get_val(0, 0);
        }

        s = sqrt(D / Sp);

        // ----------------------------------------
        // Translational offset computation
        matrix_device_t s_time_R(R.rows_, R.cols_);
        R.multiply_by_scalar(s, s_time_R);

        matrix_device_t Mu_p_T(Mu_p.cols_, Mu_p.rows_);
        Mu_p.matrix_transpose(Mu_p_T);

        matrix_device_t R_dot_Mu_p(s_time_R.rows_, Mu_p_T.cols_);
        utils::matrix_dot_product(s_time_R, Mu_p_T, R_dot_Mu_p);

        matrix_device_t R_dot_Mu_p_T(R_dot_Mu_p.cols_, R_dot_Mu_p.rows_);
        R_dot_Mu_p.matrix_transpose(R_dot_Mu_p_T);

        utils::matrix_subtract(Mu_y, R_dot_Mu_p_T, t);

        return true;
    }

    void power_iteration(const matrix_device_t& A, matrix_device_t& eigen_vector, std::size_t num_simulations)
    {
        vector_host_t vector(A.cols_);
        std::generate_n(vector.begin(), A.cols_, utils::UniformRandom<float>(0.0, 1.1));
        for (std::size_t i = 0; i < A.cols_; i++)
        {
            eigen_vector.set_val(i, 0, vector[i]);
        }

        matrix_device_t b_k1(A.rows_, eigen_vector.cols_, 0.0);

        for (std::size_t simulation = 0; simulation < num_simulations; simulation++)
        {
            utils::matrix_dot_product(A, eigen_vector, b_k1);

            float b_k1_norm = b_k1.matrix_norm_2();

            for (std::size_t i = 0; i < eigen_vector.rows_; i++)
            {
                eigen_vector.set_val(i, 0, b_k1.get_val(i, 0) / b_k1_norm);
            }
        }
    }

    void apply_alignment(const matrix_device_t& P,
                         float s,
                         const matrix_device_t& R,
                         const matrix_device_t& t,
                         matrix_device_t& newP)
    {
        matrix_device_t s_time_R(R.rows_, R.cols_);
        R.multiply_by_scalar(s, s_time_R);

        matrix_device_t s_time_R_T(s_time_R.cols_, s_time_R.rows_);
        s_time_R.matrix_transpose(s_time_R_T);

        matrix_device_t P_time_R(P.rows_, s_time_R_T.cols_);
        utils::matrix_dot_product(P, s_time_R_T, P_time_R);

        P_time_R.matrix_add_vector(t, newP);
    }
} // namespace icp