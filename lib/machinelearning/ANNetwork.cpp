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


    ANNetwork::ANNetwork() { }

    ANNetwork::~ANNetwork() { }

    void ANNetwork::addLayer( const Layer::shared_ptr layer ) {
        if( layers.empty( ) ) {
            assert( layer->getLayerType( ) == LayerType::InputLayer );
            layers.push_back( layer );
        } else {
            assert( layer->getLayerType( ) != LayerType::InputLayer );
            layers.push_back( layer );
        }
    }

    void ANNetwork::connectLayers( const Connections::shared_ptr connector ) {
        connections.push_back( connector );
    }

    void ANNetwork::finalise(bool willTrain) {

        assert( layers.back( )->getLayerType( ) == LayerType::OutputLayer );
        assert( connections.size( ) == ( layers.size( ) - 1 ) );

        auto alu = Core::VectorALUFactory();

        size_t   neuronIndex = 0;
        for( int i           = 0; i < layers.size( ); ++i ) {
            layers[ i ]->neuronIndex = neuronIndex;
            neuronIndex += layers[ i ]->countOfNeurons( );
        }

        size_t   weightIndex = 0;
        for( int j           = 0; j < connections.size( ); ++j ) {
            connections[ j ]->weightIndex = weightIndex;
            weightIndex += connections[ j ]->weightCount;
        }

        totalNeuronCount = neuronIndex;
        totalWeightCount = weightIndex;

        // a temp buffer reused in several places through an epoch, enough for all weights
        scratchPad = alu->newRealVector( totalWeightCount );

        sums    = alu->newRealVector( totalNeuronCount );
        outputs = alu->newRealVector( totalNeuronCount );
        weights = alu->newRealVector( totalWeightCount );

        alu->set( totalNeuronCount, Core::real( 0 ), sums );
        alu->set( totalNeuronCount, Core::real( 0 ), outputs );
        alu->set( totalWeightCount, Core::real( 0 ), weights );

        if (willTrain) {
            deltaWeights = alu->newRealVector( totalWeightCount );
            gradients    = alu->newRealVector( totalWeightCount );
            nodeDelta    = alu->newRealVector( totalNeuronCount );

            alu->set( totalWeightCount, Core::real( 0 ), deltaWeights );
            alu->set( totalWeightCount, Core::real( 0 ), gradients );
            alu->set( totalNeuronCount, Core::real( 0 ), nodeDelta );
        }
    }

    void ANNetwork::evaluate(Core::VectorALU::const_real_array_ptr &input) {
        using namespace Core;

        auto alu = Core::VectorALUFactory();

        for( int i = 0; i < connections.size( ); ++i ) {
            const auto &fromLayer = connections[ i ]->from;

            const auto fromNeuronCount = fromLayer->neuronCount;
            const auto fromOutputs     = outputs + fromLayer->neuronIndex;

            const auto neuronConnectionCount = connections[ i ]->neuronConnectionCount;
            auto       scratch               = scratchPad + connections[ i ]->weightIndex;

            alu->replicateItems(fromNeuronCount, neuronConnectionCount, fromOutputs, scratch);
        }

        // weight the connections (all at once)
        alu->mul( totalWeightCount, weights, scratchPad, scratchPad );

        // sum the result per layer
        for( int i = 0; i < connections.size( ); ++i ) {
            const auto &toLayer = connections[ i ]->to;

            // weights store on from layer
            const auto toNeuronCount = toLayer->neuronCount;
            const auto toNeuronIndex = toLayer->neuronIndex;

            const auto neuronConnectionCount = connections[ i ]->neuronConnectionCount;
            auto       layerSums             = sums + toNeuronIndex;
            auto       layerOutputs          = outputs + toNeuronIndex;

            for (auto j = 0; j < toNeuronCount; ++j) {
                const auto scratch = scratchPad + connections[ i ]->weightIndex + ( j * neuronConnectionCount );
                *( layerSums + j ) = alu->horizSum( neuronConnectionCount, scratch );
            }

            // activate each neuron in this layer
            toLayer->activationFunc->activate( toNeuronCount, sums, layerOutputs );
        }
    }

    void ANNetwork::computeLayerDeltas(Core::VectorALU::const_real_array_ptr &perfect) {
        using namespace Core;

        auto alu = Core::VectorALUFactory();

        // output layer is a special case (compare with perfect)
        {
            const auto &toLayer      = layers.back( );
            const auto toNeuronIndex = toLayer->neuronIndex;
            const auto toNeuronCount = toLayer->neuronCount;

            alu->sub( toNeuronCount,
                      perfect,
                      outputs + toNeuronIndex,
                      scratchPad );

            auto deltas = ( nodeDelta + toNeuronIndex );

            toLayer->activationFunc->differentiate( toNeuronCount, scratchPad, deltas );
            alu->negate(toNeuronCount, deltas, deltas);
        }

        // note: we are back propagating so last hidden layer to first hidden layer
        for( int i = connections.size( ) - 1; i >= 0; i-- ) {
            const auto &fromLayer = connections[ i ]->from;
            const auto &toLayer   = connections[ i ]->to;


            const auto fromWeight      = weights + connections[ i ]->weightIndex;
            const auto fromWeightCount = connections[ i ]->weightCount;

            const auto fromDelta     = nodeDelta + fromLayer->neuronIndex;
            const auto toNeuronCount = toLayer->neuronCount;
            auto       toDeltas      = nodeDelta + toLayer->neuronIndex;

            const auto numConnectionsPerNeuron = connections[ i ]->neuronConnectionCount;

            for (int j = 0; j < toNeuronCount; ++j) {
                alu->mul( numConnectionsPerNeuron,
                         fromWeight + (j * numConnectionsPerNeuron),
                          *( fromDelta + j ),
                          scratchPad );

                *( toDeltas + j ) = alu->horizSum( numConnectionsPerNeuron, scratchPad );
            }

            toLayer->activationFunc->differentiate(toNeuronCount, toDeltas, toDeltas);
        }
    }

    void ANNetwork::updateWeights() {

        auto alu = Core::VectorALUFactory();

        for( int i = connections.size( ) - 1; i >= 0; i-- ) {
            const auto &layer = connections[ i ]->to;

            const auto layerOutputs = outputs + layer->neuronIndex;
            const auto nodeDeltas   = nodeDelta + layer->neuronIndex;

            const auto toNeuronCount = layer->neuronCount;
            const auto weightCount   = connections[ i ]->weightCount;
            auto       toGradients   = gradients + connections[ i ]->weightIndex;

/*            for (int j = 0; j < fromWeightCount; ++j) {
                alu->mul(numConnectionsPerNeuron, ,
                         nodeDeltas,
                         toGradients + (j * numConnectionsPerNeuron));
            }*/
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
