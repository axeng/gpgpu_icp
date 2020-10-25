#include "utils.hh"

namespace utils
{
    float compute_distance(point p, point q)
    {
        double X1 = std::get<0>(p);
        double Y1 = std::get<1>(p);
        double Z1 = std::get<2>(p);
        double X2 = std::get<0>(q);
        double Y2 = std::get<1>(q);
        double Z2 = std::get<2>(q);

        return sqrt(pow(X2 - X1, 2) + pow(Y2 - Y1, 2) + pow(Z2 - Z1, 2) * 1.0);
    }

    void get_nearest_neighbors(points P, points Q, std::vector<std::tuple<point,point>>& NN)
    {
        for (points::iterator it = P.begin(); it != P.end(); ++it)
        {
            point p_point = *it;
            float min_dist = MAXFLOAT;
            point chosen;
            for (points::iterator it2 = Q.begin(); it2 != Q.end(); ++it2)
            {
                point q_point = *it2;       
                float dist = compute_distance(p_point, q_point);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    chosen = q_point;
                }
            }
            NN.push_back(std::make_tuple(p_point, chosen));
        }
    }

} // namespace utils
