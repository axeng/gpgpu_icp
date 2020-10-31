/**
 ** @file misc/random/uniform-random.hxx
 ** @brief Inline methods of misc::random::UniformRandom
 */

#pragma once

#include "gpu_test/utils/uniform-random.hh"

namespace utils
{
    template <typename T>
    UniformRandomGPU<T>::UniformRandomGPU(T minimum, T maximum)
        : random_engine_(std::random_device()())
        , distribution_(minimum, maximum)
    {}

    template <typename T>
    T UniformRandomGPU<T>::operator()()
    {
        return this->distribution_(this->random_engine_);
    }
} // namespace utils