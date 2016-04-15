//
// Created by Dean Calver on 25/03/2016.
//

#pragma once

#include "core/core.h"
#include <ctime>
#include <boost/random.hpp>

namespace Core {

    // thin wrapper around boost random, just a singleton
    // with a easy place to change and seed
    struct Random {
        typedef boost::random::mt19937                                                           generator_type;
        typedef boost::random::uniform_real_distribution<>                                       ur_distribution_type;
        typedef boost::random::variate_generator<Random::generator_type &, ur_distribution_type> uniform_real_gen_type;

        static void seed( unsigned int seed = static_cast<unsigned int>(std::time( 0 )) ) {
            generator.seed( seed );
        }

        static generator_type generator;


    };
}