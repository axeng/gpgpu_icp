#include "transform.hh"


namespace transform
{
    // Input : array containing tuple of 3 elements
    unsigned int getFitTransform(std::vector<std::tuple<double, double, double>>& first, std::vector<std::tuple<double, double, double>>& second)
    {
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
        transpose(AA, AA_T);

        // Rotation Matrix
        std::vector<std::vector<double>> H;
        dotProduct(AA_T, BB, H);

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
     * Return the dot product of the two set
     */
    void dotProduct(const std::vector<std::tuple<double,double,double>>& firstSet,
        const std::vector<std::tuple<double,double,double>>& secondSet,
        std::vector<std::tuple<double,double,double>>& result)
    {
        // init
        result = firstSet;
        std::fill(result.begin(), result.end(), 0); // reset every value

        size_t nbFirst = firstSet.size();
        size_t nbSecond = secondSet.size();

        for(int i = 0; i < nbFirst; i++)
        {
            for(j = 0; j < nbSecond; j++)
            {
                mul[i][j]=0;
                for(k = 0; k < nbSecond; k++)
                {
                    result[i][j] += firstSet[i][k] * secondSet[k][j];
                }
            }
        }
    }

    /**
     * Return the transpose of the given set
     */
    void transpose(const std::vector<std::vector<double>>& firstSet,
        std::vector<std::vector<double>>& result)
    {
        result = std::vector()

        for(std::vector<double>::size_type i = 0; i < firstSet[0].size(); i++)
        {
            for (std::vector<int>::size_type j = 0; j < firstSet.size(); j++)
            {
                result[i][j] = firstSet[j][i];
            }
        }
        return result;
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
    void SVD(const std::vector<std::vector<double>>& setPoint)
    {

    }

} // namespace transform