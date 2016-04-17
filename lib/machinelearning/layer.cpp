//
// Created by Dean Calver on 12/04/2016.
//

#include "core/core.h"
#include "machinelearning/layer.h"

namespace MachineLearning {

    Layer::Layer( const LayerType _layerType, const size_t _neuronCount, const ActivationFunction &af,
                  const bool _biased ) :
            layerType( _layerType ),
            neuronCount( _neuronCount ),
            activationFunc( af ),
            biased( _biased ) {
    }

/*
    bool FullLayer::forwardPass() const {
        auto alu = Core::VectorALUFactory();
        // TODO cache layer values if layer hasn't changed at all

        if( previousLayer ) {
            alu->mul( previousLayer->getCurrentResults(), weights, currentResults );
        }
        return nextLayer ? nextLayer->forwardPass() : false;
    }
    bool FullLayer::backwardsPass() const {
        auto alu = Core::VectorALUFactory();
        // TODO cache layer values if layer hasn't changed at all

        if( nextLayer ) {
            alu->mul( nextLayer->getCurrentResults(), weights, currentResults );
        }

        return previousLayer ? previousLayer->backwardsPass() : false;
    }*/

}