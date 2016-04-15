//
// Created by Dean Calver on 12/04/2016.
//

#include <cassert>
#include "core/core.h"
#include "ActivationFunction.h"

namespace MachineLearning {
    void ActivationFunction::activate( const size_t numItems, Core::VectorALU::const_real_array_ptr &begin,
                                       Core::VectorALU::real_array_ptr output ) const {
        auto alu = Core::VectorALUFactory();

        switch (activationFunctionType) {
            case ActivationFunctionType::Linear:
                alu->copy(numItems, begin, output);
                break;
            case ActivationFunctionType::Step:
                alu->step(numItems, begin, 0.5, output);
                break;
            case ActivationFunctionType::Sigmoid:
                alu->sigmoid(numItems, begin, output);
                break;
            case ActivationFunctionType::HyperbolicTangent:
                alu->hyperbolicTangent(numItems, begin, output);
                break;
            case ActivationFunctionType::ReLU:
                alu->relu( numItems, begin, param0, output );
                break;
        }
    }

    void ActivationFunction::differentiate(const size_t numItems, Core::VectorALU::const_real_array_ptr &begin,
                                           Core::VectorALU::real_array_ptr &output) const {

        auto alu = Core::VectorALUFactory();

        switch (activationFunctionType) {
            case ActivationFunctionType::Linear:
                // TODO Can we alias rather than copy here?
                alu->copy(numItems, begin, output);
                break;
            case ActivationFunctionType::Step:
                alu->step(numItems, begin, 0.5, output);
                break;
            case ActivationFunctionType::Sigmoid: {
                // todo remove allocations
                auto tmp = alu->newRealVector(numItems);
                auto tmp2 = alu->newRealVector(numItems);
                alu->sigmoid(numItems, begin, tmp);
                alu->set(numItems, Core::real(1.0), tmp2);
                alu->sub(numItems, tmp2, tmp, tmp2);
                alu->mul(numItems, tmp, tmp2, output);
                alu->deleteRealVector(tmp);
                alu->deleteRealVector(tmp2);
            }
                break;
            case ActivationFunctionType::HyperbolicTangent:
                alu->hyperbolicTangent(numItems, begin, output);
                break;
            case ActivationFunctionType::ReLU:
                alu->step(numItems, begin, param0, output);
                break;
        }
    }

    bool ActivationFunction::hasDerivative() const {
        switch (activationFunctionType) {
            case ActivationFunctionType::Linear:
            case ActivationFunctionType::Step:
            case ActivationFunctionType::Sigmoid:
            case ActivationFunctionType::HyperbolicTangent:
            case ActivationFunctionType::ReLU:
                return true;
            default:
                return false;
        }
    }

}