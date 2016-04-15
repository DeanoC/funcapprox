//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "machinelearning/layer.h"

#pragma once
namespace MachineLearning {
    /*
     * Between 2 layers there are connections, each connection has a weight
     */
    class Connections {
    public:
        friend class ANNetwork;

        using shared_ptr = std::shared_ptr<Connections>;

        // construct a full connected layer between 2 layers
        Connections( const Layer::shared_ptr &_from, const Layer::shared_ptr &_to );

        // todo partially connected ctor

    public:
        const size_t getWeightCount() const { return weightCount; }

        const size_t getNeuronConnectionCount() const { return neuronConnectionCount; }

    private:
        const Layer::shared_ptr from;

        const Layer::shared_ptr to;

        const size_t weightCount; // how many weights in this layer

        const size_t neuronConnectionCount; // How many connections (aka weights) per neuron in this layer

        mutable size_t weightIndex; // where does the weights for depth N to N + 1 start in the shared array
    };

}