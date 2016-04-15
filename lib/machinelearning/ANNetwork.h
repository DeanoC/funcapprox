//
// Created by Dean Calver on 12/04/2016.
//

#pragma once

#include <vector>
#include "core/core.h"
#include "machinelearning/machinelearning.h"
#include "machinelearning/layer.h"
#include "machinelearning/connections.h"

namespace MachineLearning {
    class ANNetwork {
        FRIEND_TEST( MachineLearningTests, ANNetworkStructureInOut );

    public:

        ANNetwork();

        ~ANNetwork();

        void addLayer( const Layer::shared_ptr layer );

        void connectLayers( const Connections::shared_ptr connector );

        std::vector<Layer::shared_ptr>::iterator begin() { return layers.begin( ); }

        std::vector<Layer::shared_ptr>::const_iterator cbegin() const { return layers.cbegin( ); }

        std::vector<Layer::shared_ptr>::iterator end() { return layers.end( ); }

        std::vector<Layer::shared_ptr>::const_iterator cend() const { return layers.cend( ); }

        size_t getLayerCount() const { return layers.size( ); }

        void setRandomWeights();

        /// call this before using the network, if you will be training pass willTrain = true
        void finalise( bool willTrain = false );

        // given input produce the approximate answer output
        void evaluate( Core::VectorALU::const_real_array_ptr &input );

        void computeGradients( Core::VectorALU::const_real_array_ptr &perfect );

        void updateWeights();

        // given known input and output, update the layer weights
        void supervisedTrain( Core::VectorALU::const_real_array_ptr &input,
                              Core::VectorALU::const_real_array_ptr &output );

        Core::real getLearningRate() const { return etalearningRate; }

        void setLearningRate( Core::real _learningRate ) { etalearningRate = _learningRate; }

        Core::real getMomentum() const { return alphaMomentum; }

        void setMomentum( Core::real _momentum ) { alphaMomentum = _momentum; }

        size_t getTotalNeuronCount() const { return totalNeuronCount; }

    private:
        size_t totalNeuronCount; // how many neurons across the whole network
        size_t totalWeightCount; // how many weights across the whole network

        Core::VectorALU::real_array_ptr scratchPad0; // scratch pad 0 used as a temporary, totalWeightCount in size
        Core::VectorALU::real_array_ptr scratchPad1; // scratch pad 1 used as a temporary, totalWeightCount in size
        Core::VectorALU::real_array_ptr sums;       // the summed per activation value of each neuron
        Core::VectorALU::real_array_ptr outputs;    // the output post activation per neuron
        Core::VectorALU::real_array_ptr weights;    // the weight value of each neuron to neuron interconnect

        // training only arrays
        Core::VectorALU::real_array_ptr deltaWeights;
        Core::VectorALU::real_array_ptr nodeDeltas;
        Core::VectorALU::real_array_ptr gradients;

        Core::real etalearningRate;
        Core::real alphaMomentum;

        std::vector<Layer::shared_ptr>       layers;
        std::vector<Connections::shared_ptr> connections;
    };
}


