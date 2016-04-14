//
// Created by Dean Calver on 12/04/2016.
//

#include <cmath>
#include "core/core.h"
#include "realfunc.h"

Core::real RealFunc::operator()(const Core::real x) {
    return sine(x);
}

Core::real RealFunc::sine(const Core::real x) {
    return std::sin(x);
}