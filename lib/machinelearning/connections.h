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

        // -1 (default) for edgesPerNeuron is shortcut for fully connectioned
        Connections( const Layer::shared_ptr _from, const Layer::shared_ptr _to, int _fromEdgesPerNeuron = -1 );

        const size_t getWeightCount() const { return weightCount; }

    private:
        const Layer::shared_ptr from;

        const Layer::shared_ptr to;

        const size_t weightCount; // how many weights in this layer

        const size_t srcNeuronConnectionCount; // How many connections (aka weights) per neuron relative to the src
        const size_t dstNeuronConnectionCount; // How many connections (aka weights) per neuron relative to the dest

        mutable size_t weightIndex; // where does the weights for depth N to N + 1 start in the shared array
    };

}