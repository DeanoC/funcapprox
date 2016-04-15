//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "machinelearning/hiddenlayer.h"

namespace MachineLearning {

    HiddenLayer::HiddenLayer( const size_t _neuronCount ) :
            Layer( LayerType::HiddenLayer, _neuronCount, false ) {
    }
}