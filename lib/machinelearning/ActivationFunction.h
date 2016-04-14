//
// Created by Dean Calver on 12/04/2016.
//

#pragma once

#include "layer.h"

namespace MachineLearning {

    enum class ActivationFunctionType : uint8_t {
        Linear,
        Step,
        Sigmoid,
        HyperbolicTangent,
        ReLU
    };

    class ActivationFunction {
    public:

        ActivationFunction(ActivationFunctionType activationLayerType) : activationFunctionType(activationLayerType) {

        }

        bool hasDerivative() const;

        void activate(const size_t numItems, Core::VectorALU::const_real_array_ptr &begin,
                      Core::VectorALU::real_array_ptr &output) const;

        void differentiate(const size_t numItems, Core::VectorALU::const_real_array_ptr &begin,
                           Core::VectorALU::real_array_ptr &output) const;

        const ActivationFunctionType activationFunctionType;

    protected:
        // TODO tidy up these parameter mess
        // for ReLU 0 = low threshold, 1 = low replacment value
        Core::real param0 = Core::real(0.0);
        Core::real param1 = Core::real(0.0);
    };
}
