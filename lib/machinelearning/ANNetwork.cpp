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
            scratchPad2( nullptr ),
            sums( nullptr ),
            outputs( nullptr ),
            weights( nullptr ),
            deltaWeights( nullptr ),
            nodeDeltas( nullptr ),
            gradients( nullptr ),
            etalearningRate( 0.7 ),
            alphaMomentum( 0.3 ) {
    }

    ANNetwork::~ANNetwork() {
        auto alu = Core::VectorALUFactory( );

        if( scratchPad0 != nullptr ) { alu->deleteRealVector( scratchPad0 ); }
        if( scratchPad1 != nullptr ) { alu->deleteRealVector( scratchPad1 ); }
        if( scratchPad1 != nullptr ) { alu->deleteRealVector( scratchPad2 ); }
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
            layers.push_back( layer );
        } else {
            assert( layer->getLayerType( ) != LayerType::InputLayer );
            layers.push_back( layer );
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

    void ANNetwork::setWeights( const std::vector<Core::real> &in ) {
        assert( weights != nullptr );
        assert( in.size( ) == totalWeightCount );

        size_t    i = 0;
        for( auto &&it : in ) {
            weights[ i ] = it;
            ++i;
        }
    }

    void ANNetwork::finalise( bool willTrain ) {

        assert( layers.back( )->getLayerType( ) == LayerType::OutputLayer );
        assert( connections.size( ) == (layers.size( ) - 1) );

        auto alu = Core::VectorALUFactory( );

        size_t   neuronIndex = 0;
        for( int i           = 0; i < layers.size( ); ++i ) {
            layers[ i ]->setNeuronIndex( neuronIndex );
            neuronIndex += layers[ i ]->countOfNeurons( );
        }

        size_t   weightIndex = 0;
        for( int j           = 0; j < connections.size( ); ++j ) {
            connections[ j ]->weightIndex = weightIndex;
            weightIndex += connections[ j ]->weightCount;
        }

        totalNeuronCount = neuronIndex;
        totalWeightCount = weightIndex;

        // temp buffers reused in several places through an epoch, enough for all weights
        scratchPad0 = alu->newRealVector( totalWeightCount );
        scratchPad1 = alu->newRealVector( totalWeightCount );
        scratchPad2 = alu->newRealVector( totalWeightCount );

        alu->set( totalWeightCount, Core::real( 0 ), scratchPad0 );
        alu->set( totalWeightCount, Core::real( 0 ), scratchPad1 );
        alu->set( totalWeightCount, Core::real( 0 ), scratchPad2 );

        sums    = alu->newRealVector( totalNeuronCount );
        outputs = alu->newRealVector( totalNeuronCount );


        alu->set( totalNeuronCount, Core::real( 0 ), sums );
        alu->set( totalNeuronCount, Core::real( 0 ), outputs );

        weights = alu->newRealVector( totalWeightCount );
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

    void ANNetwork::evaluate( Core::VectorALU::const_real_array_ptr input, Core::VectorALU::real_array_ptr results ) {
        using namespace Core;

        auto     alu = Core::VectorALUFactory( );

        {
            const auto &iLayer  = connections[ 0 ]->from;
            auto       iOutputs = outputs + iLayer->getNeuronIndex( );
            alu->copy( iLayer->getActualNeuronCount( ), input, iOutputs ); // neuronCount doesn't have bias in
            iOutputs[ iLayer->getActualNeuronCount( ) ] = Core::real( 1.0 ); // bias neuron
        }

        for( int i   = 0; i < connections.size( ); ++i ) {
            const auto &srcLayer = connections[ i ]->from;

            const auto srcNeuronCount = srcLayer->getActualNeuronCount( );
            const auto srcOutputs     = outputs + srcLayer->getNeuronIndex( );

            const auto neuronConnectionCount = connections[ i ]->srcNeuronConnectionCount;
            auto       scratch               = scratchPad0 + connections[ i ]->weightIndex;

            alu->replicateItems( srcNeuronCount, neuronConnectionCount, srcOutputs, scratch );
            // now add bias 1 if need
            if( srcLayer->isBiased( ) ) {
                for( int j = 0; j < neuronConnectionCount; ++j ) {
                    scratch[ (srcNeuronCount * neuronConnectionCount) + j ] = Core::real( 1.0 );
                }
            }
        }

        // weight the connections (all at once)
        alu->mul( totalWeightCount, weights, scratchPad0, scratchPad1 );

        // sum the result per layer
        for( int i = 0; i < connections.size( ); ++i ) {
            const auto &toLayer = connections[ i ]->to;

            // weights store on from layer
            const auto toNeuronCount = toLayer->getActualNeuronCount( );
            const auto toNeuronIndex = toLayer->getNeuronIndex( );

            const auto neuronConnectionCount = connections[ i ]->dstNeuronConnectionCount;
            auto       layerSums             = sums + toNeuronIndex;
            auto       layerOutputs          = outputs + toNeuronIndex;
            auto       scratch               = scratchPad1 + connections[ i ]->weightIndex;

            for( auto j = 0; j < toNeuronCount; ++j ) {
                alu->gather( neuronConnectionCount,
                             scratch + j,
                             connections[ i ]->srcNeuronConnectionCount,
                             scratchPad0 );
                *(layerSums + j) = alu->horizSum( neuronConnectionCount, scratchPad0 );
            }

            // activate each neuron in this layer
            toLayer->getActivationFunc( ).activate( toNeuronCount, layerSums, layerOutputs );
        }

        if( results != nullptr ) {
            alu->copy( layers.back( )->getActualNeuronCount( ), outputs + layers.back( )->getNeuronIndex( ), results );
        }
    }

    void ANNetwork::computeGradients( Core::VectorALU::const_real_array_ptr &perfect ) {
        using namespace Core;

        auto alu = Core::VectorALUFactory( );

        // output layer is a special case (compare with perfect)
        {
            const auto &connect  = connections.back( );
            const auto &dstLayer = connect->to;
            assert( dstLayer->getLayerType( ) == LayerType::OutputLayer );

            const auto dstNeuronIndex = dstLayer->getNeuronIndex( );
            const auto dstNeuronCount = dstLayer->countOfNeurons( );

            alu->sub( dstNeuronCount,
                      perfect,
                      outputs + dstNeuronIndex,
                      scratchPad0 );

            auto dstDeltas = nodeDeltas + dstNeuronIndex;

            dstLayer->getActivationFunc( ).differentiate( dstNeuronCount, scratchPad0, scratchPad1 );
            alu->negate( dstNeuronCount, scratchPad1, dstDeltas );
        }

        // note: we are back propagating so last hidden layer to first hidden layer
        for( size_t i = connections.size( ) - 1; i < connections.size( ); i-- ) {
            const auto &dstLayer = connections[ i ]->from;
            const auto &srcLayer = connections[ i ]->to;

            const auto weight                  = weights + connections[ i ]->weightIndex;
            const auto weightCount             = connections[ i ]->weightCount;
            const auto numConnectionsPerNeuron = connections[ i ]->dstNeuronConnectionCount;

            const auto srcDeltas      = nodeDeltas + srcLayer->getNeuronIndex( );
            const auto srcOutputs     = outputs + srcLayer->getNeuronIndex( );
            const auto srcNeuronCount = srcLayer->countOfNeurons( );

            auto       tmpDeltas      = scratchPad1 + dstLayer->getNeuronIndex( );
            auto       dstDeltas      = nodeDeltas + dstLayer->getNeuronIndex( );
            const auto dstNeuronCount = dstLayer->countOfNeurons( );

            for( int j = 0; j < srcNeuronCount; ++j ) {
                alu->mul( numConnectionsPerNeuron,
                          weight + (j * numConnectionsPerNeuron),
                          *(srcDeltas + j),
                          scratchPad0 );

                *(tmpDeltas + j) = alu->horizSum( numConnectionsPerNeuron, scratchPad0 );
            }

            dstLayer->getActivationFunc( ).differentiate( dstNeuronCount, tmpDeltas, dstDeltas );
        }
    }

    void ANNetwork::updateWeights() {

        auto alu = Core::VectorALUFactory( );

        for( int i = 0; i < connections.size( ); ++i ) {
            const auto &srcLayer = connections[ i ]->from;

            const auto srcNeuronConnectionCount = connections[ i ]->srcNeuronConnectionCount;
            const auto srcOutput                = outputs + srcLayer->getNeuronIndex( );
            auto       scratch0                 = scratchPad0 + connections[ i ]->weightIndex;
            alu->replicateItems( srcLayer->countOfNeurons( ), srcNeuronConnectionCount, srcOutput, scratch0 );

            const auto &dstLayer                = connections[ i ]->to;
            const auto dstNeuronConnectionCount = connections[ i ]->dstNeuronConnectionCount;
            const auto dstDeltas                = outputs + dstLayer->getNeuronIndex( );
            auto       scratch1                 = scratchPad1 + connections[ i ]->weightIndex;
            alu->replicateItems( dstLayer->countOfNeurons( ), dstNeuronConnectionCount, dstDeltas, scratch1 );
        }
        alu->mul( totalWeightCount, scratchPad0, scratchPad1, scratchPad2 );

        // weight the connections (all at once)
        alu->fmad( totalWeightCount,
                   scratchPad2,
                   weights,
                   -etalearningRate,
                   scratchPad0 ); // eta * grad * in (at weight rate)
        alu->fmad( totalNeuronCount, weights, alphaMomentum, scratchPad0, scratchPad1 ); // momentum

        alu->copy( totalWeightCount, scratchPad1, weights );
    }

    void ANNetwork::supervisedTrain( const std::vector<MatchingPair> &trainingSet,
                                     const std::vector<MatchingPair> &testSet ) {

        assert( trainingSet.size( ) > 0 );
        assert( testSet.size( ) > 0 );

        int epoch = 0;

        auto tmpResults = std::vector<Core::real>( connections.back( )->to->getActualNeuronCount( ), Core::real( 0 ) );

        Core::real bestErr = Core::real( 1 );

        for( int epoch = 0; epoch < 10; ++epoch ) {
            Core::real err = Core::real( 0 );

            for( auto &&ipair : trainingSet ) {
                evaluate( ipair.first, tmpResults.data( ) );
                computeGradients( ipair.second );

                err = RootMeanSquare( 1, ipair.second, tmpResults.data( ) );
                std::cout << "training " << err << "\n";
            }
            if( err < bestErr ) {
                std::cout << "Epoch " << epoch << "\n";
                updateWeights( );
                bestErr = err;
            }
        }

//        setErr /= trainingSet.size( );
    }
/*
    void ANNetwork::supervisedTrainMiniBatch( int numTraining,  Core::VectorALU::const_real_array_ptr* input,
                                                                Core::VectorALU::const_real_array_ptr* output ) {
        evaluate( input, nullptr );
        computeGradients( output );
        updateWeights( );
    }
*/


}
