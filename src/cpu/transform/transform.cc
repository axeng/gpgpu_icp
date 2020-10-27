#include "transform.hh"

#include "cpu/utils/lib-matrix.hh"

namespace transform
{
    // Input : array containing tuple of 3 elements
    void get_fit_transform(const matrix_t& A, const matrix_t& B, matrix_t& T, bool init_matrix)
    {
        /*
        def best_fit_transform(A, B):
            '''
            Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial
        dimensions Input: A: Nxm numpy array of corresponding points B: Nxm numpy array of corresponding points Returns:
              T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
              R: mxm rotation matrix
              t: mx1 translation vector
            '''

            assert A.shape == B.shape

            # get number of dimensions
            m = A.shape[1]

            # translate points to their centroids
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            AA = A - centroid_A
            BB = B - centroid_B

            # rotation matrix
            H = np.dot(AA.T, BB)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # special reflection case
            if np.linalg.det(R) < 0:
               Vt[m-1,:] *= -1
               R = np.dot(Vt.T, U.T)

            # translation
            t = centroid_B.T - np.dot(R,centroid_A.T)

            # homogeneous transformation
            T = np.identity(m+1)
            T[:m, :m] = R
            T[:m, m] = t

            return T, R, t*/

        // Check shape for each set
        if (A.empty() || B.empty() || (A[0].size() != B[0].size()))
        {
            return; // FIXME
        }

        // Get the dimension of stored points
        std::size_t m = 3;

        // Get the centroids : mean point values
        matrix_t centroid_A;
        matrix_t centroid_B;
        utils::matrix_centroid(A, centroid_A);
        utils::matrix_centroid(B, centroid_B);

        // Translate points to their centroids
        matrix_t AA;
        matrix_t BB;
        utils::matrix_subtract_vector(A, centroid_A, AA);
        utils::matrix_subtract_vector(B, centroid_B, BB);

        matrix_t AA_T;
        utils::matrix_transpose(AA, AA_T);

        // Rotation Matrix
        matrix_t H;
        utils::matrix_dot_product(AA_T, BB, H);

        // FIXME
        matrix_t s;
        matrix_t u;
        matrix_t Vt;
        svd(H, s, u, Vt);

        matrix_t Vt_T;
        matrix_t u_T;
        utils::matrix_transpose(Vt, Vt_T);
        utils::matrix_transpose(u, u_T);

        auto R = utils::gen_matrix(m, m, 0.0);
        utils::matrix_dot_product(Vt_T, u_T, R);

        if (utils::matrix_determinant(R) < 0)
        {
            // Setup second dimension to *= -1
            // Vt[m-1,:] *= -1

            for (std::size_t col = 0; col < Vt[m - 1].size(); col++)
            {
                Vt[m - 1][col] *= 1;
            }

            utils::matrix_dot_product(Vt_T, u_T, R);
        }

        // translation
        // t = centroid_B.T - np.dot(R,centroid_A.T);
        matrix_t centroid_A_T;
        matrix_t centroid_B_T;
        matrix_t tmp;
        utils::matrix_transpose(centroid_A, centroid_A_T);
        utils::matrix_transpose(centroid_B, centroid_B_T);
        utils::matrix_dot_product(R, centroid_A_T, tmp);

        auto t = utils::gen_matrix(m, 1, 0.0);

        utils::matrix_subtract_vector(centroid_B_T, tmp, t);

        // T = np.identity(m+1)
        if (init_matrix)
        {
            utils::gen_matrix(m + 1, m + 1, T, 0.0);
        }

        for (std::size_t i = 0; i < m + 1; i++)
        {
            T[i][i] = 1.0;
        }

        // T[:m, :m] = R
        for (std::size_t row = 0; row < m; row++)
        {
            for (std::size_t col = 0; col < m; col++)
            {
                T[row][col] = R[row][col];
            }
        }

        // T[:m, m] = t
        for (std::size_t col = 0; col < m; col++)
        {
            T[col][m] = t[col][0];
        }
    }

    /**
     * Return the SVD computation
     */
    void svd(const matrix_t& matrix, matrix_t& s, matrix_t& u, matrix_t& v, bool init_matrix)
    {
        matrix_t matrix_T;
        utils::matrix_transpose(matrix, matrix_T);

        matrix_t matrix_product_1;
        utils::matrix_dot_product(matrix, matrix_T, matrix_product_1);

        matrix_t matrix_product_2;
        utils::matrix_dot_product(matrix_T, matrix, matrix_product_2);

        matrix_t u_1;
        matrix_t v_1;
        vector_t eigen_values;

        compute_evd(matrix_product_2, eigen_values, v_1, 0);

        utils::matrix_transpose(v_1, v, init_matrix);

        if (init_matrix)
        {
            utils::gen_matrix(matrix.size(), eigen_values.size(), s, 0.0);
        }

        for (std::size_t i = 0; i < eigen_values.size(); i++)
        {
            s[i][i] = eigen_values[i];
        }

        matrix_t s_inverse;
        utils::matrix_inverse_diagonal(s, s_inverse);

        matrix_t av_matrix;
        utils::matrix_dot_product(matrix, v, av_matrix);
        utils::matrix_dot_product(av_matrix, s_inverse, u, init_matrix);
    }

    void compute_evd(const matrix_t& matrix, vector_t& eigen_values, matrix_t& eigen_vectors, std::size_t eig_count)
    {
        std::size_t m_size = matrix.size();

        vector_t vec(m_size, 1.0);

        // TODO take a look at this -> static std::vector<std::vector<double>> matrix_i;
        matrix_t matrix_i;

        if (eigen_values.empty() && eigen_vectors.empty())
        {
            eigen_values.resize(m_size);
            eigen_vectors.resize(m_size);

            // FIXME weird 'copy' here
            matrix_i = matrix;
        }

        matrix_t m = utils::gen_matrix(m_size, 100, 0.0);

        double lambda_old = 0;

        std::uint32_t index_eval = 0;
        bool is_eval = false;

        while (!is_eval)
        {
            for (std::size_t row = 0; row < m_size && (index_eval % 100) == 0; row++)
            {
                m[row].resize(m[row].size() + 100);
            }

            for (std::size_t row = 0; row < m_size; row++)
            {
                m[row][index_eval] = 0;

                for (std::size_t col = 0; col < m_size; col++)
                {
                    m[row][index_eval] += matrix[row][col] * vec[col];
                }
            }

            for (std::size_t col = 0; col < m_size; col++)
            {
                vec[col] = m[col][index_eval];
            }

            if (index_eval > 0)
            {
                double lambda = 0.0;
                if (m[0][index_eval - 1] != 0)
                {
                    lambda = m[0][index_eval] / m[0][index_eval - 1];
                }
                else
                {
                    lambda = m[0][index_eval];
                }

                is_eval = std::fabs(lambda - lambda_old) < 0.0000000001;

                if (std::fabs(lambda) < 10e-6)
                {
                    lambda = 0;
                }

                eigen_values[eig_count] = lambda;
                lambda_old = lambda;
            }

            index_eval++;
        }

        matrix_t matrix_new;

        if (m_size > 1)
        {
            // FIXME change this variable's name
            matrix_t matrix_tdoubleet = utils::gen_matrix(m_size, m_size);

            for (std::size_t row = 0; row < m_size; row++)
            {
                for (std::size_t col = 0; col < m_size; col++)
                {
                    if (row == col)
                    {
                        matrix_tdoubleet[row][col] = matrix[row][col] - eigen_values[eig_count];
                    }
                    else
                    {
                        matrix_tdoubleet[row][col] = matrix[row][col];
                    }
                }
            }

            vector_t eigen_vector;
            utils::jordan_gaussian_transform(matrix_tdoubleet, eigen_vector);

            matrix_t hermitian_matrix;
            utils::hermitian_matrix(eigen_vector, hermitian_matrix);

            matrix_t ha_matrix_product;
            utils::matrix_dot_product(hermitian_matrix, matrix, ha_matrix_product);

            matrix_t inverse_hermitian_matrix;
            utils::hermitian_matrix(eigen_vector, inverse_hermitian_matrix);

            matrix_t iha_matrix_product;
            utils::matrix_dot_product(ha_matrix_product, inverse_hermitian_matrix, iha_matrix_product);

            utils::matrix_reduced(iha_matrix_product, m_size - 1, matrix_new);
        }

        if (m_size <= 1)
        {
            for (std::size_t index = 0; index < eigen_values.size(); index++)
            {
                double lambda = eigen_values[index];
                matrix_t matrix_tdoubleet;
                matrix_tdoubleet.resize(matrix_i.size());

                for (std::size_t row = 0; row < matrix_i.size(); row++)
                {
                    matrix_tdoubleet[row].resize(matrix_i.size());

                    for (std::size_t col = 0; col < matrix_i.size(); col++)
                    {
                        if (row == col)
                        {
                            matrix_tdoubleet[row][col] = matrix_i[row][col] - lambda;
                        }
                        else
                        {
                            matrix_tdoubleet[row][col] = matrix_i[row][col];
                        }
                    }
                }

                eigen_vectors.resize(matrix_i.size());
                utils::jordan_gaussian_transform(matrix_tdoubleet, eigen_vectors[index]);

                double eigsum_sq = 0;
                for (std::size_t v = 0; v < eigen_vectors[index].size(); v++)
                {
                    eigsum_sq += std::pow(eigen_vectors[index][v], 2.0);
                }

                for (std::size_t v = 0; v < eigen_vectors[index].size(); v++)
                {
                    eigen_vectors[index][v] /= sqrt(eigsum_sq);
                }

                eigen_values[index] = std::sqrt(eigen_values[index]);
            }

            return;
        }

        compute_evd(matrix_new, eigen_values, eigen_vectors, eig_count + 1);
    }

    // ----------------------------------------------------
    // -----------------------------------------------------
    //                      LEGACY
    // -----------------------------------------------------
    // ----------------------------------------------------

    /**
     * Return the mean point of the given setPoint : the centroid
     */
    /*
    void get_centroid(const points_t& set_point, points_t& result)
    {
        double x = 0;
        double y = 0;
        double z = 0;

        int nb_elements = set_point.size();

        for (const auto& point : set_point)
        {
            x += point[0];
            y += point[1];
            z += point[2];
        }

        x /= nb_elements;
        y /= nb_elements;
        z /= nb_elements;

        point_t points = {x, y, z};
        result.push_back(points);
    }
     */

    /**
     *
     * Return the determinant of the given set of Points
     */
    /*
    double get_determinant(const points_t& set_point, int dimension)
    {
        double det = 0;

        points_t sub_set;

        for (const auto& e : set_point)
        {
            point_t sub_list;
            for (std::size_t i = 0; i < e.size(); i++)
            {
                sub_list.push_back(0);
            }

            sub_set.push_back(sub_list);
        }

        if (dimension == 2)
        {
            return ((set_point[0][0] * set_point[1][1]) - (set_point[1][0] * set_point[0][1]));
        }
        else
        {
            for (int x = 0; x < dimension; x++)
            {
                int subi = 0;
                for (int i = 1; i < dimension; i++)
                {
                    int subj = 0;
                    for (int j = 0; j < dimension; j++)
                    {
                        if (j == x)
                            continue;
                        sub_set[subi][subj] = set_point[i][j];
                        subj++;
                    }
                    subi++;
                }
                det = det + (pow(-1, x) * set_point[0][x] * get_determinant(sub_set, dimension - 1));
            }
        }
        return det;
    }
     */

    /**
     * Return the set obtained by substracting the given point to each
     *      point of the given vector to the given point.
     */
    /*
    void subtract_by_centroid(const points_t& set_point, const points_t& centroid, points_t& result)
    {
        // init
        result.assign(set_point.begin(), set_point.end());

        int nb_columns = 3;
        const size_t l = set_point.size();

        for (size_t i = 0; i < l; i++)
        {
            for (int j = 0; j < nb_columns; j++)
            {
                result[i][j] = set_point[i][j] - centroid[0][j];
            }
        }
    }
     */

    /*
    void get_hermitian_matrix(point_t eigenvector, points_t& h_matrix)
    {
        h_matrix.resize(eigenvector.size());
        for (std::uint32_t row = 0; row < eigenvector.size(); row++)
        {
            h_matrix[row].resize(eigenvector.size());
        }

        h_matrix[0][0] = 1 / eigenvector[0];
        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
        {
            h_matrix[row][0] = -eigenvector[row] / eigenvector[0];
        }

        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
        {
            h_matrix[row][row] = 1;
        }
    }
     */

    /*
    void get_hermitian_matrix_inverse(point_t eigenvector, points_t& ih_matrix)
    {
        ih_matrix.resize(eigenvector.size());
        for (std::uint32_t row = 0; row < eigenvector.size(); row++)
        {
            ih_matrix[row].resize(eigenvector.size());
        }

        ih_matrix[0][0] = eigenvector[0];
        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
        {
            ih_matrix[row][0] = -eigenvector[row];
        }

        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
        {
            ih_matrix[row][row] = 1;
        }
    }
     */

    /*
    void get_inverse_diagonal_matrix(const points_t& matrix, points_t& inv_matrix)
    {
        inv_matrix.resize(matrix.size());
        for (std::uint32_t index = 0; index < matrix.size(); index++)
        {
            inv_matrix[index].resize(matrix[index].size());
            inv_matrix[index][index] = 1.0 / matrix[index][index];
        }
    }
     */

    /*
    void get_reduced_matrix(const points_t& matrix, points_t& r_matrix, std::size_t new_size)
    {
        r_matrix.resize(new_size);
        std::size_t index_d = matrix.size() - new_size;
        std::uint32_t row = index_d, row_n = 0;
        while (row < matrix.size())
        {
            r_matrix[row_n].resize(new_size);

            std::uint32_t col = index_d, col_n = 0;
            while (col < matrix.size())
            {
                r_matrix[row_n][col_n++] = matrix[row][col++];
            }

            row++;
            row_n++;
        }
    }
     */

    /*
    void matrix_by_matrix(const points_t& matrix1, const points_t& matrix2, points_t& matrix3)
    {
        matrix3.resize(matrix1.size());

        for (std::uint32_t row = 0; row < matrix1.size(); row++)
        {
            matrix3[row].resize(matrix1[row].size());

            for (std::uint32_t col = 0; col < matrix1[row].size(); col++)
            {
                matrix3[row][col] = 0.0;

                for (std::uint32_t k = 0; k < matrix1[row].size(); k++)
                {
                    matrix3[row][col] += matrix1[row][k] * matrix2[k][col];
                }
            }
        }
    }
     */

    /*
    void matrix_transpose(const points_t& matrix1, points_t& matrix2)
    {
        matrix2.resize(matrix1.size());

        for (std::uint32_t row = 0; row < matrix1.size(); row++)
        {
            matrix2[row].resize(matrix1[row].size());

            for (std::uint32_t col = 0; col < matrix1[row].size(); col++)
            {
                matrix2[row][col] = matrix1[col][row];
            }
        }
    }
     */

    /*
    void substract_matrix(const std::vector<std::vector<double>>& first,
                          const std::vector<std::vector<double>>& second,
                          std::vector<std::vector<double>>& result)
    {
        matrix2.resize(matrix1.size());
        for (std::uint32_t row = 0; row < matrix1.size(); row++)
        {
            matrix2[row].resize(matrix1[row].size());
            for (std::uint32_t col = 0; col < matrix1[row].size(); col++)
                matrix2[row][col] = matrix1[col][row];
        }
    }
     */

    /*
    void jordan_gaussian_transform(points_t matrix, point_t& eigenvector)
    {
        const double eps = 0.000001;
        bool eigenv_found = false;

        for (std::uint32_t s = 0; s < matrix.size() - 1 && !eigenv_found; s++)
        {
            std::uint32_t col = s;
            double alpha = matrix[s][s];

            while (col < matrix[s].size() && alpha != 0 && alpha != 1)
            {
                matrix[s][col++] /= alpha;
            }

            for (std::uint32_t col_sec = s; col < matrix[s].size() && !alpha; col_sec++)
            {
                std::swap(matrix[s][col_sec], matrix[s + 1][col_sec]);
            }

            for (std::uint32_t row = 0; row < matrix.size(); row++)
            {
                double gamma = matrix[row][s];

                for (std::uint32_t col_sec = s; col < matrix[row].size() && row != s; col_sec++)
                {
                    matrix[row][col_sec] = matrix[row][col_sec] - matrix[s][col_sec] * gamma;
                }
            }

            std::uint32_t row = 0;
            while (row < matrix.size() && (s == matrix.size() - 1 || std::fabs(matrix[s + 1][s + 1]) < eps))
            {
                eigenvector.push_back(-matrix[row++][s + 1]);
            }

            if (eigenvector.size() == matrix.size())
            {
                eigenv_found = true;
                eigenvector[s + 1] = 1;
                for (std::uint32_t index = s + 1; index < eigenvector.size(); index++)
                {
                    eigenvector[index] = (std::fabs(eigenvector[index]) >= eps) ? eigenvector[index] : 0;
                }
            }
        }
    }
     */

} // namespace transform