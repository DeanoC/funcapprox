//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "inputlayer.h"

namespace MachineLearning {

    static ActivationFunction linearAF( ActivationFunctionType::Linear );

    InputLayer::InputLayer( const size_t _neuronCount ) :
            Layer( LayerType::InputLayer, _neuronCount, linearAF, true ),
            inputData( neuronCount, 0 ) {
    }

    InputLayer::InputLayer( const size_t _neuronCount, const Core::real *_inputData ) :
            Layer( LayerType::InputLayer, _neuronCount, linearAF, true ),
            inputData( _inputData, _inputData + neuronCount ) {
    }

}