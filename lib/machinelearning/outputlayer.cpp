//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "outputlayer.h"


namespace MachineLearning {

    static ActivationFunction sActFunc( ActivationFunctionType::Sigmoid );

    OutputLayer::OutputLayer( const size_t _neuronCount ) :
            Layer( LayerType::OutputLayer, _neuronCount, sActFunc, false ) {
    }

}