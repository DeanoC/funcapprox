//
// Created by Dean Calver on 12/04/2016.
//

#pragma once

#include <vector>
#include "core/core.h"
#include "machinelearning/machinelearning.h"
#include "layer.h"

namespace MachineLearning {
    class ANNetwork {
    public:

        ANNetwork();

        ~ANNetwork();

        void push_back(Layer::shared_ptr layer);

        std::vector<Layer::shared_ptr>::iterator begin() { return layers.begin(); }

        std::vector<Layer::shared_ptr>::iterator end() { return layers.end(); }

        std::vector<Layer::shared_ptr>::const_iterator cbegin() const { return layers.cbegin(); }

        std::vector<Layer::shared_ptr>::const_iterator cend() const { return layers.cend(); }

        /// call this before using the network, if you will be training pass willTrain = true
        void finalise(bool willTrain = false);

        // given input produce the approximate answer output
        void evaluate(Core::VectorALU::const_real_array_ptr &input);

        void computeLayerDeltas(Core::VectorALU::const_real_array_ptr &perfect);

        void updateWeights();

        void updateWeights();

        // given known input and output, update the layer weights
        void supervisedTrain(Core::VectorALU::const_real_array_ptr &input, Core::VectorALU::real_array_ptr &output);

        Core::real getLearningRate() const { return learningRate; }

        void setLearningRate(Core::real _learningRate) { learningRate = _learningRate; }

        Core::real getMomentum() const { return momentum; }

        void setMomentum(Core::real _momentum) { momentum = _momentum; }

    private:

        struct ANNWorkspace *workspace;
        struct ANNTrainWorkspace *trainingWorkspace;

        Core::real learningRate;
        Core::real momentum;

        size_t numLayers = 0;
        std::vector<Layer::shared_ptr> layers;
    };
}


