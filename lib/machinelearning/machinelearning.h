//
// Created by Dean Calver on 12/04/2016.
//

#pragma once

#include <cmath>
#include "core/core.h"
#include "core/vectoralu.h"

namespace MachineLearning {

    static Core::real SumOfSquare(const size_t numItems, Core::VectorALU::const_real_array_ptr &perfect,
                                  Core::VectorALU::const_real_array_ptr &actual) {
        auto alu  = Core::VectorALUFactory();
        // TODO reduce memory allocation for tmp
        auto tmp  = alu->newRealVector(numItems);
        auto tmp2 = alu->newRealVector( numItems );
        alu->sub(numItems, perfect, actual, tmp);
        alu->mul( numItems, tmp, tmp, tmp2 );
        Core::real result = alu->horizSum( numItems, tmp2 );
        alu->deleteRealVector(tmp);
        alu->deleteRealVector( tmp2 );
        return result;
    }

    static Core::real MeanSquare(const size_t numItems, Core::VectorALU::const_real_array_ptr &perfect,
                                 Core::VectorALU::const_real_array_ptr &actual) {
        return SumOfSquare(numItems, perfect, actual) / Core::real(numItems);
    }

    static Core::real RootMeanSquare(const size_t numItems, Core::VectorALU::const_real_array_ptr &perfect,
                                     Core::VectorALU::const_real_array_ptr &actual) {
        return sqrt(MeanSquare(numItems, perfect, actual));
    }

}

