//
// Created by Dean Calver on 11/04/2016.
//

#pragma once

#include <limits>
#include <cstdlib>
#include <cmath>

namespace Core {
    using real = float;

    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp) {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        if (std::fabs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp) {
            return true;
        }
        else {
            // denormal case check... maybe make it optional for speed in flush to 0 modes?
            return std::fabs(x - y) < std::numeric_limits<T>::min();
        }
    }
}