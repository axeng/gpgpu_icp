#include "transform.hh"


namespace transform
{
    // Input : array containing tuple of 3 elements
    unsigned int getFitTransform(std::vector<std::tuple<double, double, double>>& first, std::vector<std::tuple<double, double, double>>& second)
    {
/*
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
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

        unsigned int result = 0;

        // Check shape for each set
        // FIXME : Missing the dimension check (== 3)
        if (first.size() != second.size())
        {
            return nullptr;
        }

        std::vector<std::vector<double>> A = transformToMarix(first);
        std::vector<std::vector<double>> B = transformToMarix(second);

        // Get the dimension of stored points
        int dim = 3;

        // Get the centroids : mean point values
        std::vector<double> centroid_A;
        std::vector<double> centroid_B;

        getCendroid(A, centroid_A);
        getCendroid(B, centroid_B);

        std::vector<std::vector<double>> AA;
        std::vector<std::vector<double>> BB;


        // Translate points to their centroids
        substractByCentroid(A, centroid_A, AA);
        substractByCentroid(B, centroid_B, BB);

        std::vector<std::vector<double>> AA_T;
        matrix_transpose(AA, AA_T);

        // Rotation Matrix
        std::vector<std::vector<double>> H;
        matrix_by_matrix(AA_T, BB, H);

        std::vector<std::vector<double>> s;
        std::vector<std::vector<double>> u;
        std::vector<std::vector<double>> v;

        svd(H, s, u, v);

        // Special reflection case
        //double det = getDeterminant(s) --> For SVD computation result




        return result;
    }


    /**
     * 
     * Return the given set of points transformed as a matrix
     */
    std::vector<std::vector<int>> transformToMarix(const std::vector<std::tuple<double,double,double>>& setPoint)
    {
        int numberColumns = 3;
        size_t nbPoints = setPoint.size();
        std::vector<std::vector<double>> result(nbPoints, numberColumns); // All values defined 0

        for (size_t i = 0; i < nbPoints; i++)
        {
            for (int j = 0; j < numberColumns; j++)
            {
                result[i][j] = std::get<j>(setPoint[i]);
            }
        }

        return result;
    }

    /**
     * Return the mean point of the given setPoint : the centroid
     */
    void getCendroid(const std::vector<std::tuple<double,double,double>>& setPoint,
        std::tuple<double,double,double>& result)
    {
        double x = 0;
        double y = 0;
        double z = 0;

        int nbElements = setPoint.size();

        for (const &std::tuple<double,double,double> point : setPoint)
        {
            x += std::get<0>(point);
            y += std::get<1>(point);
            z += std::get<2>(point);
        }

        x /= nbElements;
        y /= nbElements;
        z /= nbElements;

        result = std::make_tuple(x, y, z);
    }

    void getCentroid(const std::vector<std::vector<double>>& setPoint,
        std::vector<double>& result)
    {
        double x = 0;
        double y = 0;
        double z = 0;

        int nbElements = setPoint.size();

        for (const &std::vector<double> point : setPoint)
        {
            x += point[0];
            y += point[1];
            z += point[2];
        }

        x /= nbElements;
        y /= nbElements;
        z /= nbElements;

        result = {x, y, z};
    }


    /**
     * Return the set obtained by substracting the given point to each 
     *      point of the given vector to the given point.
     */
    void substractByCentroid(const std::vector<std::tuple<double,double,double>>& setPoint,
        const std::tuple<double,double,double>& centroid, 
        std::vector<std::tuple<double,double,double>>& setPoint result)
    {
        result = setPoint;

        const size_t l = setPoint.size();
        for (const size_t i = 0; i < l; i++)
        {
            std::tuple<double, double, double> computedTuple(
                std::get<0>(setPoint[i]) - std::get<0>(centroid),
                std::get<1>(setPoint[i]) - std::get<1>(centroid),
                std::get<2>(setPoint[i]) - std::get<2>(centroid),
            );

            result[i] = computedTuple;
        }
    }

    void substractByCentroid(const std::vector<std::vector<double>>& setPoint,
        const std::vector<double>& centroid, 
        std::vector<std::vector<double>>& result)
    {
        // init
        result = setPoint;

        //std::vector<std::vector<double> > result(setPoint[0].size(),
        //                                 std::vector<int>(setPoint.size()));

        int nbColumns = 3;
        const size_t l = setPoint.size();

        for (const size_t i = 0; i < l; i++)
        {
            for (int j = 0; j < nbColumns; j++)
            {
                result[i][j] = setPoint[i][j] - centroid[j]
            }
        }

    }

    /**
     * 
     * Return the determinant of the given set of Points
     */
    double getDeterminant(const std::vector<std::vector<double>>& setPoint,
        int dimension)
    {
        double det = 0;
        double subSet = setPoint;
        std::fill(subSet.begin(), subSet.end(), 0);

        if (dimension == 2)
        {
            return ((setPoint[0][0] * setPoint[1][1]) - (setPoint[1][0] * setPoint[0][1]));
        }
        else 
        {
            for (int x = 0; x < n; x++) 
            {
                int subi = 0;
                for (int i = 1; i < n; i++)
                {
                    int subj = 0;
                    for (int j = 0; j < n; j++) 
                    {
                        if (j == x)
                            continue;
                        subSet[subi][subj] = setPoint[i][j];
                        subj++;
                    }
                    subi++;
                }
                det = det + (pow(-1, x) * setPoint[0][x] * determinant(subSet, n - 1 ));
            }
        }
        return det;
    }

    /**
     * Return the SVD computation
     */
    void svd(std::vector<std::vector<double>> matrix, std::vector<std::vector<double>>& s,
	std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v)
    {
        std::vector<std::vector<double>> matrix_t;
        matrix_transpose(matrix, matrix_t);

        std::vector<std::vector<double>> matrix_product1;
        matrix_by_matrix(matrix, matrix_t, matrix_product1);

        std::vector<std::vector<double>> matrix_product2;
        matrix_by_matrix(matrix_t, matrix, matrix_product2);

        std::vector<std::vector<double>> u_1;
        std::vector<std::vector<double>> v_1;

        std::vector<double> eigenvalues;
        compute_evd(matrix_product2, eigenvalues, v_1, 0);

        matrix_transpose(v_1, v);

        s.resize(matrix.size());
        for (std::uint32_t index = 0; index < eigenvalues.size(); index++)
        {
            s[index].resize(eigenvalues.size());
            s[index][index] = eigenvalues[index];
        }

        std::vector<std::vector<double>> s_inverse;
        get_inverse_diagonal_matrix(s, s_inverse);

        std::vector<std::vector<double>> av_matrix;
        matrix_by_matrix(matrix, v, av_matrix);
        matrix_by_matrix(av_matrix, s_inverse, u);
    }


    void compute_evd(std::vector<std::vector<arg>> matrix,
	std::vector<arg>& eigenvalues, std::vector<std::vector<arg>>& eigenvectors, std::size_t eig_count)
    {
        std::size_t m_size = matrix.size();
        std::vector<arg> vec; vec.resize(m_size);
        std::fill_n(vec.begin(), m_size, 1);

        static std::vector<std::vector<arg>> matrix_i;

        if (eigenvalues.size() == 0 && eigenvectors.size() == 0)
        {
            eigenvalues.resize(m_size);
            eigenvectors.resize(eigenvalues.size());
            matrix_i = matrix;
        }

        std::vector<std::vector<arg>> m; m.resize(m_size);
        for (std::uint32_t row = 0; row < m_size; row++)
            m[row].resize(100);

        Arg lambda_old = 0;

        std::uint32_t index = 0; bool is_eval = false;
        while (is_eval == false)
        {
            for (std::uint32_t row = 0; row < m_size && (index % 100) == 0; row++)
                m[row].resize(m[row].size() + 100);

            for (std::uint32_t row = 0; row < m_size; row++)
            {
                m[row][index] = 0;
                for (std::uint32_t col = 0; col < m_size; col++)
                    m[row][index] += matrix[row][col] * vec[col];
            }

            for (std::uint32_t col = 0; col < m_size; col++)
                vec[col] = m[col][index];

            if (index > 0)
            {
                Arg lambda = (m[0][index - 1] != 0) ? \
                    (m[0][index] / m[0][index - 1]) : m[0][index];
                is_eval = (std::fabs(lambda - lambda_old) < 0.0000000001) ? true : false;

                lambda = (std::fabs(lambda) >= 10e-6) ? lambda : 0;
                eigenvalues[eig_count] = lambda; lambda_old = lambda;
            }

            index++;
        }

        std::vector<std::vector<arg>> matrix_new;

        if (m_size > 1)
        {
            std::vector<std::vector<arg>> matrix_target;
            matrix_target.resize(m_size);

            for (std::uint32_t row = 0; row < m_size; row++)
            {
                matrix_target[row].resize(m_size);
                for (std::uint32_t col = 0; col < m_size; col++)
                    matrix_target[row][col] = (row == col) ? \
                    (matrix[row][col] - eigenvalues[eig_count]) : matrix[row][col];
            }

            std::vector<arg> eigenvector;
            jordan_gaussian_transform(matrix_target, eigenvector);

            std::vector<std::vector<arg>> hermitian_matrix;
            get_hermitian_matrix(eigenvector, hermitian_matrix);

            std::vector<std::vector<arg>> ha_matrix_product;
            matrix_by_matrix(hermitian_matrix, matrix, ha_matrix_product);

            std::vector<std::vector<arg>> inverse_hermitian_matrix;
            get_hermitian_matrix_inverse(eigenvector, inverse_hermitian_matrix);

            std::vector<std::vector<arg>> iha_matrix_product;
            matrix_by_matrix(ha_matrix_product, inverse_hermitian_matrix, iha_matrix_product);

            get_reduced_matrix(iha_matrix_product, matrix_new, m_size - 1);
        }

        if (m_size <= 1)
        {
            for (std::uint32_t index = 0; index < eigenvalues.size(); index++)
            {
                Arg lambda = eigenvalues[index];
                std::vector<std::vector<arg>> matrix_target;
                matrix_target.resize(matrix_i.size());

                for (std::uint32_t row = 0; row < matrix_i.size(); row++)
                {
                    matrix_target[row].resize(matrix_i.size());
                    for (std::uint32_t col = 0; col < matrix_i.size(); col++)
                        matrix_target[row][col] = (row == col) ? \
                        (matrix_i[row][col] - lambda) : matrix_i[row][col];
                }

                eigenvectors.resize(matrix_i.size());
                jordan_gaussian_transform(matrix_target, eigenvectors[index]);

                Arg eigsum_sq = 0;
                for (std::uint32_t v = 0; v < eigenvectors[index].size(); v++)
                    eigsum_sq += std::pow(eigenvectors[index][v], 2.0);

                for (std::uint32_t v = 0; v < eigenvectors[index].size(); v++)
                    eigenvectors[index][v] /= sqrt(eigsum_sq);

                eigenvalues[index] = std::sqrt(eigenvalues[index]);
            }

            return;
        }

        compute_evd(matrix_new, eigenvalues, eigenvectors, eig_count + 1);

        return;
    }


    void get_hermitian_matrix(std::vector<arg> eigenvector,
	std::vector<std::vector<arg>>& h_matrix)
    {
        h_matrix.resize(eigenvector.size());
        for (std::uint32_t row = 0; row < eigenvector.size(); row++)
            h_matrix[row].resize(eigenvector.size());

        h_matrix[0][0] = 1 / eigenvector[0];
        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
            h_matrix[row][0] = -eigenvector[row] / eigenvector[0];

        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
            h_matrix[row][row] = 1;
    }


    void get_hermitian_matrix_inverse(std::vector<arg> eigenvector,
	std::vector<std::vector<arg>>& ih_matrix)
    {
        ih_matrix.resize(eigenvector.size());
        for (std::uint32_t row = 0; row < eigenvector.size(); row++)
            ih_matrix[row].resize(eigenvector.size());

        ih_matrix[0][0] = eigenvector[0];
        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
            ih_matrix[row][0] = -eigenvector[row];

        for (std::uint32_t row = 1; row < eigenvector.size(); row++)
            ih_matrix[row][row] = 1;
    }

    void jordan_gaussian_transform(
	std::vector<std::vector<arg>> matrix, std::vector<arg>& eigenvector)
    {
        const Arg eps = 0.000001; bool eigenv_found = false;
        for (std::uint32_t s = 0; s < matrix.size() - 1 && !eigenv_found; s++)
        {
            std::uint32_t col = s; Arg alpha = matrix[s][s];
            while (col < matrix[s].size() && alpha != 0 && alpha != 1)
                matrix[s][col++] /= alpha;

            for (std::uint32_t col = s; col < matrix[s].size() && !alpha; col++)
                std::swap(matrix[s][col], matrix[s + 1][col]);

            for (std::uint32_t row = 0; row < matrix.size(); row++)
            {
                Arg gamma = matrix[row][s];
                for (std::uint32_t col = s; col < matrix[row].size() && row != s; col++)
                    matrix[row][col] = matrix[row][col] - matrix[s][col] * gamma;
            }

            std::uint32_t row = 0;
            while (row < matrix.size() &&
                (s == matrix.size() - 1 || std::fabs(matrix[s + 1][s + 1]) < eps))
                eigenvector.push_back(-matrix[row++][s + 1]);

            if (eigenvector.size() == matrix.size())
            {
                eigenv_found = true; eigenvector[s + 1] = 1;
                for (std::uint32_t index = s + 1; index < eigenvector.size(); index++)
                    eigenvector[index] = (std::fabs(eigenvector[index]) >= eps) ? eigenvector[index] : 0;
            }
        }
    }

    void get_inverse_diagonal_matrix(std::vector<std::vector<arg>> matrix,
	std::vector<std::vector<arg>>& inv_matrix)
    {
        inv_matrix.resize(matrix.size());
        for (std::uint32_t index = 0; index < matrix.size(); index++)
        {
            inv_matrix[index].resize(matrix[index].size());
            inv_matrix[index][index] = 1.0 / matrix[index][index];
        }
    }

    void get_reduced_matrix(std::vector<std::vector<arg>> matrix,
	std::vector<std::vector<arg>>& r_matrix, std::size_t new_size)
    {
        r_matrix.resize(new_size);
        std::size_t index_d = matrix.size() - new_size;
        std::uint32_t row = index_d, row_n = 0;
        while (row < matrix.size())
        {
            r_matrix[row_n].resize(new_size);
            std::uint32_t col = index_d, col_n = 0;
            while (col < matrix.size())
                r_matrix[row_n][col_n++] = matrix[row][col++];

            row++; row_n++;
        }
    }

    void matrix_by_matrix(std::vector<std::vector<arg>> matrix1,
	std::vector<std::vector<arg>>& matrix2, std::vector<std::vector<arg>>& matrix3)
    {
        matrix3.resize(matrix1.size());
        for (std::uint32_t row = 0; row < matrix1.size(); row++)
        {
            matrix3[row].resize(matrix1[row].size());
            for (std::uint32_t col = 0; col < matrix1[row].size(); col++)
            {
                matrix3[row][col] = 0.00;
                for (std::uint32_t k = 0; k < matrix1[row].size(); k++)
                    matrix3[row][col] += matrix1[row][k] * matrix2[k][col];
            }
        }
    }

    void matrix_transpose(std::vector<std::vector<arg>> matrix1,
	std::vector<std::vector<arg>>& matrix2)
    {
        matrix2.resize(matrix1.size());
        for (std::uint32_t row = 0; row < matrix1.size(); row++)
        {
            matrix2[row].resize(matrix1[row].size());
            for (std::uint32_t col = 0; col < matrix1[row].size(); col++)
                matrix2[row][col] = matrix1[col][row];
        }
    }

    void generate_matrix(std::vector<std::vector<long double="">>& \
	matrix, std::size_t rows, std::size_t cols)
    {
        std::srand((unsigned int)std::time(nullptr)); matrix.resize(rows);
        for (std::size_t row = 0; row < matrix.size(); row++)
        {
            matrix[row].resize(cols);
            for (std::size_t col = 0; col < matrix[row].size(); col++)
                matrix[row][col] = std::rand() % 20 - std::rand() % 20;
        }
    }

    void print_matrix(std::vector<std::vector<long double="">>	matrix)
    {
        for (std::size_t row = 0; row < matrix.size(); row++)
        {
            for (std::size_t col = 0; col < matrix[row].size(); col++)
                std::cout << std::setprecision(5) << std::fixed << matrix[row][col] << " ";

            std::cout << "\n";
        }

        std::cout << "\n";
    }



} // namespace transform