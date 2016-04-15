//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "inputlayer.h"

namespace MachineLearning {

    InputLayer::InputLayer( const size_t _neuronCount ) :
            Layer( LayerType::InputLayer, _neuronCount, true ),
            inputData( neuronCount, 0 ) {
        // add bias neuron to input array
        inputData.push_back( Core::real( 1.0 ) );
    }

    InputLayer::InputLayer( const size_t _neuronCount, const Core::real *_inputData ) :
            Layer( LayerType::InputLayer, _neuronCount, true ),
            inputData( _inputData, _inputData + neuronCount ) {
        // add bias neuron to input array
        inputData.push_back( Core::real( 1.0 ) );
    }

}