//
// Created by Dean Calver on 14/04/2016.
//

#pragma once

#include "machinelearning/layer.h"

namespace MachineLearning {

    class HiddenLayer : public Layer {
        friend class ANNetwork;

    public:
        HiddenLayer( const size_t _neuronCount );
    };
}