//
// Created by Dean Calver on 12/04/2016.
//

#include <cassert>
#include "core/core.h"
#include "ANNetwork.h"
#include "machinelearning/ActivationFunction.h"

namespace MachineLearning {


    // Example of a simple ANN
    //   L0        I        I      I
    //    W01.x  /   \    /  \    / \
    //          /     \  /    \  /   \
    //   L1    H       H       H      H
    //    W12.x \     / \     / \    /
    //           \  /    \  /    \  /
    //   L2       H       H       H
    //     W23.x  |       |       |
    //   L3       O       O       O
    //
    // 4 layers identified as type layer.index
    //   so I0.0 is Input 0th node at level 0,
    // W01.0 = I0.0 -> H1.0 weight are on I0.0
    // W01.1 = I0.0 -> H1.1 weight are on I0.0
    // W12.0 = H1.0 -> H2.0 weight are on H1.0
    // W12.1 = H1.1 -> H2.0 weight are on H1.1
    // W23.0 = H2.0 -> O3.0 weight are on H2.0


    /// weights are on the connections between layers, to avoid modeling them explicitly they are
    /// stored on the from layer (lowest index)
    struct ANNWorkspace {
        struct LayerInfo {
            size_t neuronIndex;
            size_t neuronCount;

            size_t toWeightIndex;
            size_t neuronConnectionCount;
            size_t totalToWeightCount; // if layer == 2, weights here are W23.x
        };
        size_t totalNeuronCount;
        size_t totalWeightCount;

        Core::VectorALU::real_array_ptr scratchPad;
        Core::VectorALU::real_array_ptr sums;
        Core::VectorALU::real_array_ptr outputs;
        Core::VectorALU::real_array_ptr weights;

        std::vector<LayerInfo> layers;

    };
    struct ANNTrainWorkspace {
        Core::VectorALU::real_array_ptr deltaWeights;
        Core::VectorALU::real_array_ptr nodeDelta;
        Core::VectorALU::real_array_ptr gradients;
    };

    ANNetwork::ANNetwork() :
            workspace(nullptr),
            trainingWorkspace(nullptr) {

    }

    ANNetwork::~ANNetwork() {
        if (workspace) {
            delete workspace;
            workspace = nullptr;
        }
        if (trainingWorkspace) {
            delete trainingWorkspace;
            trainingWorkspace = nullptr;
        }
    }

    void ANNetwork::push_back(Layer::shared_ptr layer) {
        layers.push_back(layer);
    }

    void ANNetwork::finalise(bool willTrain) {
        if (workspace != nullptr) {
            return;
        }

        size_t index = 0;
        size_t weightIndex = 0;

        // prepare workspace for runs
        workspace = new ANNWorkspace();
        trainingWorkspace = willTrain ? new ANNTrainWorkspace() : nullptr;
        auto alu = Core::VectorALUFactory();

        for (int i = 0; i < layers.size(); ++i) {
            const auto &prevLayer = layers[i - 1];
            const auto &layer = layers[i];

            layer->finalise();

            workspace->layers[i].neuronIndex = index;
            index += layer->size() + 1; // each layer has a bias

            // we be replaced except for output layer
            workspace->layers[i].neuronConnectionCount = 1;
            workspace->layers[i].totalToWeightCount = 0;
            if (i > 0) {
                // todo partially connected layers
                workspace->layers[i - 1].neuronConnectionCount = layer->size();
                workspace->layers[i - 1].toWeightIndex = weightIndex;
                workspace->layers[i - 1].totalToWeightCount =
                        workspace->layers[i].neuronConnectionCount * prevLayer->size();
                weightIndex += workspace->layers[i - 1].totalToWeightCount;
            }
        }
        workspace->totalNeuronCount = index;
        workspace->totalWeightCount = weightIndex;

        // a temp buffer reused in several places through an epoch, enough for all weights
        workspace->scratchPad = alu->newRealVector(workspace->totalWeightCount);

        workspace->sums = alu->newRealVector(workspace->totalNeuronCount);
        workspace->outputs = alu->newRealVector(workspace->totalNeuronCount);
        workspace->weights = alu->newRealVector(workspace->totalWeightCount);

        alu->set(workspace->totalNeuronCount, Core::real(0), workspace->sums);
        alu->set(workspace->totalNeuronCount, Core::real(0), workspace->outputs);
        alu->set(workspace->totalWeightCount, Core::real(0), workspace->weights);

        if (willTrain) {
            trainingWorkspace->deltaWeights = alu->newRealVector(workspace->totalWeightCount);
            trainingWorkspace->gradients = alu->newRealVector(workspace->totalWeightCount);
            trainingWorkspace->nodeDelta = alu->newRealVector(workspace->totalNeuronCount);

            alu->set(workspace->totalWeightCount, Core::real(0), trainingWorkspace->deltaWeights);
            alu->set(workspace->totalWeightCount, Core::real(0), trainingWorkspace->gradients);
            alu->set(workspace->totalNeuronCount, Core::real(0), trainingWorkspace->nodeDelta);

        }
    }

    void ANNetwork::evaluate(Core::VectorALU::const_real_array_ptr &input) {
        using namespace Core;
        assert(workspace);

        auto alu = Core::VectorALUFactory();

        for (int i = 1; i < layers.size(); ++i) {
            const auto &fromLayer = layers[i - 1];
            const auto &toLayer = layers[i];

            // weights store on from layer
            const auto fromNeuronCount = workspace->layers[i - 1].neuronCount;
            const auto neuronConnectionCount = workspace->layers[i - 1].neuronConnectionCount;
            const auto fromOutputs = workspace->outputs + workspace->layers[i - 1].neuronIndex;

            auto scratch = workspace->scratchPad + workspace->layers[i - 1].toWeightIndex;
            alu->replicateItems(fromNeuronCount, neuronConnectionCount, fromOutputs, scratch);
        }

        // weight the connections (all at once)
        alu->mul(workspace->totalWeightCount, workspace->weights, workspace->scratchPad, workspace->scratchPad);

        // sum the result per layer
        for (int i = 1; i < layers.size(); ++i) {
            const auto &toLayer = layers[i];

            // weights store on from layer
            const auto toNeuronCount = workspace->layers[i].neuronCount;
            const auto toNeuronIndex = workspace->layers[i].neuronIndex;
            const auto neuronConnectionCount = workspace->layers[i - 1].neuronConnectionCount;
            const auto fromNeuronIndex = workspace->layers[i - 1].neuronIndex;

            auto sums = workspace->sums + toNeuronIndex;
            auto outputs = workspace->outputs + toNeuronIndex;

            for (auto j = 0; j < toNeuronCount; ++j) {
                const auto scratch =
                        workspace->scratchPad + workspace->layers[i - 1].toWeightIndex + (j * neuronConnectionCount);
                *(sums + j) = alu->horizSum(neuronConnectionCount, scratch);
            }

            // activate each neuron in this layer
            toLayer->activationFunc->activate(toNeuronCount, sums, outputs);
        }
    }

    void ANNetwork::computeLayerDeltas(Core::VectorALU::const_real_array_ptr &perfect) {
        using namespace Core;
        assert(workspace);

        auto alu = Core::VectorALUFactory();

        // output layer is a special case (compare with perfect)
        {
            const auto &toLayer = layers[layers.size() - 1];
            const auto toNeuronIndex = workspace->layers.back().neuronIndex;
            const auto toNeuronCount = workspace->layers.back().neuronCount;

            alu->sub(toNeuronCount,
                     perfect,
                     workspace->outputs + toNeuronIndex,
                     workspace->scratchPad);

            auto deltas = (trainingWorkspace->nodeDelta + toNeuronIndex);

            toLayer->activationFunc->differentiate(toNeuronCount, workspace->scratchPad, deltas);
            alu->negate(toNeuronCount, deltas, deltas);
        }

        // note: we are back propagating so last hidden layer to first hidden layer
        for (int i = layers.size() - 2; i >= 0; i--) {
            const auto &fromLayer = layers[i + 1];
            const auto &toLayer = layers[i];


            const auto fromWeight = workspace->weights + workspace->layers[i + 1].toWeightIndex;
            const auto fromWeightCount = workspace->layers[i + 1].totalToWeightCount;
            const auto fromDelta = trainingWorkspace->nodeDelta + workspace->layers[i + 1].neuronIndex;

            const auto toNeuronCount = workspace->layers[i].neuronCount;
            auto toDeltas = trainingWorkspace->nodeDelta + workspace->layers[i].neuronIndex;

            const auto numConnectionsPerNeuron = fromWeightCount / toNeuronCount;

            for (int j = 0; j < toNeuronCount; ++j) {
                alu->mul(numConnectionsPerNeuron,
                         fromWeight + (j * numConnectionsPerNeuron),
                         *(fromDelta + j),
                         workspace->scratchPad);

                *(toDeltas + j) = alu->horizSum(numConnectionsPerNeuron, workspace->scratchPad);
            }

            toLayer->activationFunc->differentiate(toNeuronCount, toDeltas, toDeltas);
        }
    }

    void ANNetwork::updateWeights() {

        auto alu = Core::VectorALUFactory();

        for (auto i = layers.size() - 1; i > 0; i--) {
            const auto &layer = layers[i];

            const auto outputs = workspace->outputs + workspace->layers[i].neuronIndex;
            const auto weightCount = workspace->layers[i].totalToWeightCount;
            const auto nodeDeltas = trainingWorkspace->nodeDelta + workspace->layers[i].neuronIndex;

            const auto toNeuronCount = workspace->layers[i].neuronCount;
            auto toGradients = trainingWorkspace->gradients + workspace->layers[i].toWeightIndex;

            const auto numConnectionsPerNeuron = fromWeightCount / toNeuronCount;

            for (int j = 0; j < fromWeightCount; ++j) {
                alu->mul(numConnectionsPerNeuron, ,
                         nodeDeltas,
                         toGradients + (j * numConnectionsPerNeuron));
            }
        }

    }

/*
    void ANNetwork::updateWeights() {
        auto alu = Core::VectorALUFactory();

        // dW(t) = (learning * gradient) + (momentum *  dW(t-1))
        alu->mul( workspace->totalWeightCount, workspace->weights, momentum, workspace->scratchPad );
        alu->fmad( workspace->totalWeightCount, trainingWorkspace->gradients, learningRate, workspace->scratchPad, workspace->weights );
    }
*/

    void ANNetwork::supervisedTrain(Core::VectorALU::const_real_array_ptr &input,
                                    Core::VectorALU::real_array_ptr &output) {
    }

}
