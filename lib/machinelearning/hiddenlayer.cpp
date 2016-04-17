//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "machinelearning/hiddenlayer.h"

namespace MachineLearning {

    static ActivationFunction sActFunc( ActivationFunctionType::Sigmoid );

    HiddenLayer::HiddenLayer( const size_t _neuronCount ) :
            Layer( LayerType::HiddenLayer, _neuronCount, sActFunc, true ) {
    }
}