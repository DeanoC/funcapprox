//
// Created by Dean Calver on 14/04/2016.
//

#pragma once

#include "machinelearning/layer.h"

namespace MachineLearning {

    class OutputLayer : public Layer {
    public:
        friend class ANNetwork;

        OutputLayer( const size_t _neuronCount );
    };
}