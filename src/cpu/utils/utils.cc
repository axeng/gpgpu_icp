#include "utils.hh"

namespace utils
{
    float compute_distance(point_t p, point_t q)
    {
        double X1 = p[0];
        double Y1 = p[1];
        double Z1 = p[2];
        double X2 = q[0];
        double Y2 = q[1];
        double Z2 = q[2];

        return sqrt(pow(X2 - X1, 2) + pow(Y2 - Y1, 2) + pow(Z2 - Z1, 2) * 1.0);
    }

    void get_nearest_neighbors(points_t P, points_t Q, std::vector<std::tuple<point_t, point_t>>& NN)
    {
        for (points_t::iterator it = P.begin(); it != P.end(); ++it)
        {
            point_t p_point = *it;
            float min_dist = MAXFLOAT;
            point_t chosen;

            for (points_t::iterator it2 = Q.begin(); it2 != Q.end(); ++it2)
            {
                point_t q_point = *it2;
                float dist = compute_distance(p_point, q_point);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    chosen = q_point;
                }
            }

            NN.emplace_back(p_point, chosen);
        }
    }

} // namespace utils
