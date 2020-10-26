#include "transform.hh"

namespace transform
{
    // Input : array containing tuple of 3 elements
    void get_fit_transform(const points_t& A, const points_t& B, points_t& T, points_t& R, points_t& t)
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
        std::size_t dim = 3;

        // Get the centroids : mean point values
        points_t centroid_A;
        points_t centroid_B;

        get_centroid(A, centroid_A);
        get_centroid(B, centroid_B);

        points_t AA;
        points_t BB;

        // Translate points to their centroids
        subtract_by_centroid(A, centroid_A, AA);
        subtract_by_centroid(B, centroid_B, BB);

        points_t AA_T;
        matrix_transpose(AA, AA_T);

        // Rotation Matrix
        points_t H;
        matrix_by_matrix(AA_T, BB, H);

        points_t s;
        points_t u;
        points_t Vt;

        svd(H, s, u, Vt);

        points_t Vt_T;
        points_t u_T;

        matrix_transpose(Vt, Vt_T);
        matrix_transpose(u, u_T);

        matrix_by_matrix(Vt_T, u_T, R);

        if (get_determinant(R, R.size()) < 0)
        {
            // Setup second dimension to *= -1
            // Vt[m-1,:] *= -1

            size_t l = Vt[dim - 1].size();
            for (size_t i = 0; i < l; i++)
            {
                Vt[dim - 1][i] *= 1;
            }

            matrix_by_matrix(Vt_T, u_T, R);
        }

        // translation
        // t = centroid_B.T - np.dot(R,centroid_A.T);

        points_t centroid_A_T;
        points_t centroid_B_T;
        points_t tmp;

        matrix_transpose(centroid_A, centroid_A_T);
        matrix_transpose(centroid_B, centroid_B_T);
        matrix_by_matrix(R, centroid_A_T, tmp);

        subtract_by_centroid(centroid_B_T, tmp, t);

        // T = np.identity(m+1)
        for (std::size_t i = 0; i < dim + 1; i++)
        {
            point_t sub_T;
            for (std::size_t j = 0; j < dim + 1; j++)
            {
                if (i == j)
                {
                    sub_T.push_back(1);
                }
                else
                {
                    sub_T.push_back(0);
                }
            }

            T.push_back(sub_T);
        }

        // T[:m, :m] = R
        for (std::size_t i = 0; i < dim; i++)
        {
            for (std::size_t j = 0; j < dim; j++)
            {
                T[i][j] = R[i][j];
            }
        }

        // T[:m, m] = t
        for (std::size_t i = 0; i < dim; i++)
        {
            T[i][dim] = t[i][0];
        }
    }

    /**
     * Return the mean point of the given setPoint : the centroid
     */
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

    /**
     * Return the set obtained by substracting the given point to each
     *      point of the given vector to the given point.
     */
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

    /**
     *
     * Return the determinant of the given set of Points
     */
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

    /**
     * Return the SVD computation
     */
    void svd(const points_t& matrix, points_t& s, points_t& u, points_t& v)
    {
        points_t matrix_t;
        matrix_transpose(matrix, matrix_t);

        points_t matrix_product1;
        matrix_by_matrix(matrix, matrix_t, matrix_product1);

        points_t matrix_product2;
        matrix_by_matrix(matrix_t, matrix, matrix_product2);

        points_t u_1;
        points_t v_1;

        point_t eigenvalues;
        compute_evd(matrix_product2, eigenvalues, v_1, 0);

        matrix_transpose(v_1, v);

        s.resize(matrix.size());
        for (std::uint32_t index = 0; index < eigenvalues.size(); index++)
        {
            s[index].resize(eigenvalues.size());
            s[index][index] = eigenvalues[index];
        }

        points_t s_inverse;
        get_inverse_diagonal_matrix(s, s_inverse);

        points_t av_matrix;
        matrix_by_matrix(matrix, v, av_matrix);
        matrix_by_matrix(av_matrix, s_inverse, u);
    }

    void compute_evd(const points_t& matrix, point_t& eigenvalues, points_t& eigenvectors, std::size_t eig_count)
    {
        std::size_t m_size = matrix.size();

        point_t vec;
        vec.resize(m_size);
        std::fill_n(vec.begin(), m_size, 1);

        // TODO take a look at this -> static std::vector<std::vector<double>> matrix_i;
        points_t matrix_i;

        if (eigenvalues.empty() && eigenvectors.empty())
        {
            eigenvalues.resize(m_size);
            eigenvectors.resize(eigenvalues.size());
            matrix_i = matrix;
        }

        points_t m;
        m.resize(m_size);

        for (std::uint32_t row = 0; row < m_size; row++)
        {
            m[row].resize(100);
        }

        double lambda_old = 0;

        std::uint32_t index_eval = 0;
        bool is_eval = false;

        while (!is_eval)
        {
            for (std::uint32_t row = 0; row < m_size && (index_eval % 100) == 0; row++)
            {
                m[row].resize(m[row].size() + 100);
            }

            for (std::uint32_t row = 0; row < m_size; row++)
            {
                m[row][index_eval] = 0;

                for (std::uint32_t col = 0; col < m_size; col++)
                {
                    m[row][index_eval] += matrix[row][col] * vec[col];
                }
            }

            for (std::uint32_t col = 0; col < m_size; col++)
            {
                vec[col] = m[col][index_eval];
            }

            if (index_eval > 0)
            {
                // TODO clean that shit
                double lambda =
                    (m[0][index_eval - 1] != 0) ? (m[0][index_eval] / m[0][index_eval - 1]) : m[0][index_eval];
                is_eval = std::fabs(lambda - lambda_old) < 0.0000000001;

                lambda = (std::fabs(lambda) >= 10e-6) ? lambda : 0;
                eigenvalues[eig_count] = lambda;
                lambda_old = lambda;
            }

            index_eval++;
        }

        points_t matrix_new;

        if (m_size > 1)
        {
            points_t matrix_tdoubleet;
            matrix_tdoubleet.resize(m_size);

            for (std::uint32_t row = 0; row < m_size; row++)
            {
                matrix_tdoubleet[row].resize(m_size);
                for (std::uint32_t col = 0; col < m_size; col++)
                {
                    matrix_tdoubleet[row][col] =
                        (row == col) ? (matrix[row][col] - eigenvalues[eig_count]) : matrix[row][col];
                }
            }

            point_t eigenvector;
            jordan_gaussian_transform(matrix_tdoubleet, eigenvector);

            points_t hermitian_matrix;
            get_hermitian_matrix(eigenvector, hermitian_matrix);

            points_t ha_matrix_product;
            matrix_by_matrix(hermitian_matrix, matrix, ha_matrix_product);

            points_t inverse_hermitian_matrix;
            get_hermitian_matrix_inverse(eigenvector, inverse_hermitian_matrix);

            points_t iha_matrix_product;
            matrix_by_matrix(ha_matrix_product, inverse_hermitian_matrix, iha_matrix_product);

            get_reduced_matrix(iha_matrix_product, matrix_new, m_size - 1);
        }

        if (m_size <= 1)
        {
            for (std::uint32_t index = 0; index < eigenvalues.size(); index++)
            {
                double lambda = eigenvalues[index];
                points_t matrix_tdoubleet;
                matrix_tdoubleet.resize(matrix_i.size());

                for (std::uint32_t row = 0; row < matrix_i.size(); row++)
                {
                    matrix_tdoubleet[row].resize(matrix_i.size());

                    for (std::uint32_t col = 0; col < matrix_i.size(); col++)
                    {
                        matrix_tdoubleet[row][col] = (row == col) ? (matrix_i[row][col] - lambda) : matrix_i[row][col];
                    }
                }

                eigenvectors.resize(matrix_i.size());
                jordan_gaussian_transform(matrix_tdoubleet, eigenvectors[index]);

                double eigsum_sq = 0;
                for (std::uint32_t v = 0; v < eigenvectors[index].size(); v++)
                {
                    eigsum_sq += std::pow(eigenvectors[index][v], 2.0);
                }

                for (std::uint32_t v = 0; v < eigenvectors[index].size(); v++)
                {
                    eigenvectors[index][v] /= sqrt(eigsum_sq);
                }

                eigenvalues[index] = std::sqrt(eigenvalues[index]);
            }

            return;
        }

        compute_evd(matrix_new, eigenvalues, eigenvectors, eig_count + 1);
    }

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

    void jordan_gaussian_transform(points_t& matrix, point_t& eigenvector)
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

    void get_inverse_diagonal_matrix(const points_t& matrix, points_t& inv_matrix)
    {
        inv_matrix.resize(matrix.size());
        for (std::uint32_t index = 0; index < matrix.size(); index++)
        {
            inv_matrix[index].resize(matrix[index].size());
            inv_matrix[index][index] = 1.0 / matrix[index][index];
        }
    }

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

} // namespace transform