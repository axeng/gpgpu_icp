#include "transform.hh"


namespace transform
{
    // Input : array containing tuple of 3 elements
    unsigned int getFitTransform(std::vector<std::tuple<double, double, double>>& A, std::vector<std::tuple<double, double, double>>& B)
    {
        unsigned int result = 0;

        // Check shape for each set
        if (A.size() != B.size())
        {
            return nullptr;
        }

        // Get the dimension of stored points
        int dim = 3;

        // Get the centrois : mean point values
        std::tuple<double, double,double> centroid_A;
        std::tuple<double, double,double> centroid_B;

        getCendroid(A, centroid_A);
        getCendroid(B, centroid_B);

        

        return result;
    }


    /**
     * 
     * Return the mean point of the given setPoint : the centroid
     */
    void getCendroid(const &std::vector<std::tuple<double,double,double>> setPoint,
        const &std::tuple<double,double,double> result)
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

} // namespace transform