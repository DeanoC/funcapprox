//
// Created by Dean Calver on 11/04/2016.
//
#include "core/core.h"
#include <iostream>
#include <boost/log/trivial.hpp>
#include <boost/range/irange.hpp>
#include <array>
#include "core/vectoralu.h"
#include "realfunc.h"
#include "machinelearning/machinelearning.h"
#include "machinelearning/ANNetwork.h"
#include "machinelearning/InputLayer.h"

int main() {
    using namespace Core;
    BOOST_LOG_TRIVIAL(trace) << "funcapprox main starting...";

    std::shared_ptr<VectorALU> alu = VectorALUFactory();
    RealFunc f;
    for (auto x : boost::irange(-5000, 5000, 1)) {
        real y = f(static_cast<real>(x) * real(0.1));
    }

    using namespace MachineLearning;
    ANNetwork           nn;
    std::array<real, 1> input{ real( 0.0 ) };
    nn.addLayer( std::make_shared<InputLayer>( 1, input.data( ) ) );

    BOOST_LOG_TRIVIAL(trace) << "funcapprox main ending";
    return 0;
}