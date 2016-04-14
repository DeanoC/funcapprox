//
// Created by Dean Calver on 12/04/2016.
//

#include "core/core.h"
#include "core/vectoralu.h"
#include "machinelearning/layer.h"

namespace MachineLearning {

    void Layer::finalise() {
        totalSize = 1;
        for (auto size : sizeOfDims) {
            totalSize *= size;
        }
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