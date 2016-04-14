//
// Created by Dean Calver on 11/04/2016.
//
#include "core/core.h"
#include <iostream>
#include <boost/log/trivial.hpp>
#include <boost/range/irange.hpp>
#include "core/vectoralu.h"
#include "realfunc.h"

int main() {
    using namespace Core;
    BOOST_LOG_TRIVIAL(trace) << "funcapprox main starting...";

    std::shared_ptr<VectorALU> alu = VectorALUFactory();
    RealFunc f;
    for (auto x : boost::irange(-5000, 5000, 1)) {
        real y = f(static_cast<real>(x) * real(0.1));
    }

    BOOST_LOG_TRIVIAL(trace) << "funcapprox main ending";
    return 0;
}