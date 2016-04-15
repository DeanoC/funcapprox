//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "outputlayer.h"


namespace MachineLearning {

    OutputLayer::OutputLayer( const size_t _neuronCount ) :
            Layer( LayerType::OutputLayer, _neuronCount, false ) {
    }

}