/**
 ** @file misc/random/uniform-random.hh
 ** @brief Declaration of misc::random::UniformRandom
 */

#pragma once

#include <random>

namespace utils
{
    /// Uniform Distribution Random real number generator
    template <typename T>
    class UniformRandom
    {
    public:
        /** @name Constructors
         ** \{ */

        /** @brief Instantiate a new random generator
         ** @param minimum The minimum value that the generator can generate, included
         ** @param width The maximum value that the generator can generate, excluded
         */
        UniformRandom(T minimum, T maximum);

        /** \} */

        /** @name Operators
         ** \{ */

        /** @brief Generate a new random number
         ** @return The random number
         */
        T operator()();

        /** \} */

    private:
        /// The random engine
        std::mt19937 random_engine_;
        /// The distribution
        std::uniform_real_distribution<T> distribution_;
    };
} // namespace utils

#include "gpu_test/utils/uniform-random.hxx"