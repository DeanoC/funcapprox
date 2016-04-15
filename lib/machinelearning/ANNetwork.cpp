//
// Created by Dean Calver on 12/04/2016.
//

#include <cassert>
#include "core/core.h"
#include "ANNetwork.h"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include "core/random.h"

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


    ANNetwork::ANNetwork() :
            totalNeuronCount( 0 ),
            totalWeightCount( 0 ),
            scratchPad0( nullptr ),
            scratchPad1( nullptr ),
            sums( nullptr ),
            outputs( nullptr ),
            weights( nullptr ),
            deltaWeights( nullptr ),
            nodeDeltas( nullptr ),
            gradients( nullptr ),
            etalearningRate( 0.4 ),
            alphaMomentum( 0.1 ) {
    }

    ANNetwork::~ANNetwork() {
        auto alu = Core::VectorALUFactory( );

        if( scratchPad0 != nullptr ) { alu->deleteRealVector( scratchPad0 ); }
        if( scratchPad1 != nullptr ) { alu->deleteRealVector( scratchPad1 ); }
        if( sums != nullptr ) { alu->deleteRealVector( sums ); }
        if( outputs != nullptr ) { alu->deleteRealVector( outputs ); }
        if( weights != nullptr ) { alu->deleteRealVector( weights ); }
        if( deltaWeights != nullptr ) { alu->deleteRealVector( deltaWeights ); }
        if( nodeDeltas != nullptr ) { alu->deleteRealVector( nodeDeltas ); }
        if( gradients != nullptr ) { alu->deleteRealVector( gradients ); }

    }

    void ANNetwork::addLayer( const Layer::shared_ptr layer ) {
        if( layers.empty( ) ) {
            assert( layer->getLayerType( ) == LayerType::InputLayer );
            layer->activationFunc = std::make_unique<ActivationFunction>( ActivationFunctionType::Linear );
            layers.push_back( layer );
        } else {
            assert( layer->getLayerType( ) != LayerType::InputLayer );
            layers.push_back( layer );
            layer->activationFunc = std::make_unique<ActivationFunction>( ActivationFunctionType::ReLU );
        }
    }

    void ANNetwork::connectLayers( const Connections::shared_ptr connector ) {
        connections.push_back( connector );
    }

    void ANNetwork::setRandomWeights() {
        assert( weights != nullptr );
        using namespace Core;
        using namespace boost::random;

        // todo random weights range to be user specified
        Random::uniform_real_gen_type                            kRandGen( Random::generator,
                                                                           Random::ur_distribution_type( -10.0,
                                                                                                         10.0 ) );
        boost::generator_iterator<Random::uniform_real_gen_type> kIter( &kRandGen );
        for( int                                                 i = 0; i < totalWeightCount; ++i ) {
            *(weights + i) = *kIter++;
        }

    }

    void ANNetwork::finalise( bool willTrain ) {

        assert( layers.back( )->getLayerType( ) == LayerType::OutputLayer );
        assert( connections.size( ) == (layers.size( ) - 1) );

        auto alu = Core::VectorALUFactory( );

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
        scratchPad0 = alu->newRealVector( totalWeightCount );
        scratchPad1 = alu->newRealVector( totalWeightCount );

        sums    = alu->newRealVector( totalNeuronCount );
        outputs = alu->newRealVector( totalNeuronCount );
        weights = alu->newRealVector( totalWeightCount );

        alu->set( totalNeuronCount, Core::real( 0 ), sums );
        alu->set( totalNeuronCount, Core::real( 0 ), outputs );
        alu->set( totalWeightCount, Core::real( 0 ), weights );

        if( willTrain ) {
            deltaWeights = alu->newRealVector( totalWeightCount );
            gradients    = alu->newRealVector( totalWeightCount );
            nodeDeltas   = alu->newRealVector( totalNeuronCount );

            alu->set( totalWeightCount, Core::real( 0 ), deltaWeights );
            alu->set( totalWeightCount, Core::real( 0 ), gradients );
            alu->set( totalNeuronCount, Core::real( 0 ), nodeDeltas );
        }
    }

    void ANNetwork::evaluate( Core::VectorALU::const_real_array_ptr &input ) {
        using namespace Core;

        auto     alu = Core::VectorALUFactory( );

        {
            const auto &iLayer  = connections[ 0 ]->from;
            auto       iOutputs = outputs + iLayer->neuronIndex;
            alu->copy( iLayer->neuronCount, input, iOutputs ); // neuronCount doesn't have bias in
            iOutputs[ iLayer->neuronCount ] = Core::real( 1.0 ); // bias neuron
        }
        for( int i   = 0; i < connections.size( ); ++i ) {
            const auto &srcLayer = connections[ i ]->from;

            const auto srcNeuronCount = srcLayer->countOfNeurons( );
            const auto srcOutputs     = outputs + srcLayer->neuronIndex;

            const auto neuronConnectionCount = connections[ i ]->neuronConnectionCount;
            auto       scratch               = scratchPad0 + connections[ i ]->weightIndex;

            alu->replicateItems( srcNeuronCount, neuronConnectionCount, srcOutputs, scratch );
        }

        // weight the connections (all at once)
        alu->mul( totalWeightCount, weights, scratchPad0, scratchPad0 );

        // sum the result per layer
        for( int i = 0; i < connections.size( ); ++i ) {
            const auto &toLayer = connections[ i ]->to;

            // weights store on from layer
            const auto toNeuronCount = toLayer->countOfNeurons( );
            const auto toNeuronIndex = toLayer->neuronIndex;

            const auto neuronConnectionCount = connections[ i ]->neuronConnectionCount;
            auto       layerSums             = sums + toNeuronIndex;
            auto       layerOutputs          = outputs + toNeuronIndex;

            for( auto j = 0; j < toNeuronCount; ++j ) {
                const auto scratch = scratchPad0 + connections[ i ]->weightIndex + (j * neuronConnectionCount);
                *(layerSums + j) = alu->horizSum( neuronConnectionCount, scratch );
            }

            // activate each neuron in this layer
            toLayer->activationFunc->activate( toNeuronCount, sums, layerOutputs );
        }
    }

    void ANNetwork::computeGradients( Core::VectorALU::const_real_array_ptr &perfect ) {
        using namespace Core;

        auto alu = Core::VectorALUFactory( );

        // output layer is a special case (compare with perfect)
        {
            const auto &dstLayer      = layers.back( );
            const auto dstNeuronIndex = dstLayer->neuronIndex;
            const auto dstNeuronCount = dstLayer->countOfNeurons( );

            alu->sub( dstNeuronCount,
                      perfect,
                      outputs + dstNeuronIndex,
                      scratchPad0 );

            auto dstGradients = gradients + dstNeuronIndex;

            dstLayer->activationFunc->differentiate( dstNeuronCount, scratchPad0, dstGradients );
//            alu->negate( dstNeuronCount, deltas, deltas );
        }

        // note: we are back propagating so last hidden layer to first hidden layer
        for( size_t i = connections.size( ) - 1; i < connections.size( ); i-- ) {
            const auto &dstLayer = connections[ i ]->from;
            const auto &srcLayer = connections[ i ]->to;


            const auto weight                  = weights + connections[ i ]->weightIndex;
            const auto weightCount             = connections[ i ]->weightCount;
            const auto numConnectionsPerNeuron = connections[ i ]->neuronConnectionCount;

            const auto srcGradients   = gradients + srcLayer->neuronIndex;
            const auto srcNeuronCount = srcLayer->countOfNeurons( );

            auto       dstGradients   = gradients + dstLayer->neuronIndex;
            const auto dstNeuronCount = dstLayer->countOfNeurons( );

            for( int j = 0; j < srcNeuronCount; ++j ) {
                alu->mul( numConnectionsPerNeuron,
                          weight + (j * numConnectionsPerNeuron),
                          *(srcGradients + j),
                          scratchPad0 );

                *(dstGradients + j) = alu->horizSum( numConnectionsPerNeuron, scratchPad0 );
            }

            dstLayer->activationFunc->differentiate( dstNeuronCount, dstGradients, dstGradients );
        }
    }

    void ANNetwork::updateWeights() {

        auto alu = Core::VectorALUFactory( );

        for( int i = 0; i < connections.size( ); ++i ) {
            const auto &srcLayer = connections[ i ]->from;
            const auto &dstLayer = connections[ i ]->to;

            const auto srcNeuronCount = srcLayer->countOfNeurons( );
            const auto srcOutputs     = outputs + srcLayer->neuronIndex;

            const auto neuronConnectionCount = connections[ i ]->neuronConnectionCount;
            auto       scratch               = scratchPad0 + connections[ i ]->weightIndex;
            const auto dstGradients          = gradients + dstLayer->neuronIndex;

            alu->replicateItems( srcNeuronCount, neuronConnectionCount, srcOutputs, scratch );

            for( int j = 0; j < neuronConnectionCount; ++j ) {
                auto scr = scratch + (j * neuronConnectionCount);
                alu->mul( neuronConnectionCount, scr, dstGradients, scr );
            }
        }

        // weight the connections (all at once)
        alu->mul( totalWeightCount, scratchPad0, etalearningRate, scratchPad0 ); // eta * grad * in (at weight rate)

        alu->fmad( totalWeightCount, deltaWeights, alphaMomentum, scratchPad0, scratchPad1 );
        alu->copy( totalWeightCount, scratchPad0, deltaWeights );
        alu->add( totalWeightCount, weights, scratchPad1, weights );
    }

/*
    void ANNetwork::updateWeights() {
        auto alu = Core::VectorALUFactory();

        // dW(t) = (learning * gradient) + (alphaMomentum *  dW(t-1))
        alu->mul( workspace->totalWeightCount, workspace->weights, alphaMomentum, workspace->scratchPad0 );
        alu->fmad( workspace->totalWeightCount, trainingWorkspace->gradients, etalearningRate, workspace->scratchPad0, workspace->weights );
    }
*/

    void ANNetwork::supervisedTrain( Core::VectorALU::const_real_array_ptr &input,
                                     Core::VectorALU::const_real_array_ptr &output ) {
        evaluate( input );
        computeGradients( output );
        updateWeights( );
    }

}
