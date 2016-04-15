//
// Created by Dean Calver on 15/04/2016.
//

#include "core/core.h"
#include "core/random.h"
#include <array>
#include "machinelearning/machinelearning.h"
#include "machinelearning/inputlayer.h"
#include "machinelearning/hiddenlayer.h"
#include "machinelearning/outputlayer.h"
#include "machinelearning/connections.h"
#include "machinelearning/ANNetwork.h"
#include "gtest/gtest.h"

namespace MachineLearning {

    TEST( MachineLearningTests, InputLayer ) {

        using namespace Core;
        using namespace MachineLearning;

        auto emptyLayer                    = std::make_shared<InputLayer>( 1 );
        EXPECT_EQ( emptyLayer->getLayerType( ), LayerType::InputLayer );
        EXPECT_EQ( emptyLayer->isBiased( ), true );
        EXPECT_EQ( emptyLayer->countOfNeurons( ), 2 ); // 1 input + bias neuron
        EXPECT_FLOAT_EQ( emptyLayer->getInputData( )[ 1 ], Core::real( 1.0 ) );

        std::array<Core::real, 1> input1{ real( 0.0 ) };
        std::array<Core::real, 1> input2{ real( 5.0 ) };
        std::array<Core::real, 2> input3{ real( 5.0 ), real( 10.0 ) };
        auto                      smlLayer = std::make_shared<InputLayer>( 1, input1.data( ) );

        EXPECT_FLOAT_EQ( smlLayer->getInputData( )[ 0 ], input1[ 0 ] );
        EXPECT_NE( smlLayer->getInputData( )[ 0 ], input2[ 0 ] );
        EXPECT_NO_THROW( {
                             smlLayer->setInputData( &input2[ 0 ], &input2[ 1 ] );
                         } );
        auto stdarraylLayer                = std::make_shared<InputLayer>( input3 );
        EXPECT_FLOAT_EQ( stdarraylLayer->getInputData( )[ 0 ], input3[ 0 ] );
        EXPECT_FLOAT_EQ( stdarraylLayer->getInputData( )[ 1 ], input3[ 1 ] );
        EXPECT_FLOAT_EQ( stdarraylLayer->getInputData( )[ 1 ], real( 10.0 ) );

        EXPECT_FLOAT_EQ( smlLayer->getInputData( )[ 0 ], input2[ 0 ] );
        EXPECT_NE( smlLayer->getInputData( )[ 0 ], input1[ 0 ] );
        EXPECT_NO_THROW( {
                             smlLayer->setInputData( input1 );
                         } );
        EXPECT_FLOAT_EQ( smlLayer->getInputData( )[ 0 ], input1[ 0 ] );
        EXPECT_NE( smlLayer->getInputData( )[ 0 ], input2[ 0 ] );

        EXPECT_EQ( smlLayer->getLayerType( ), LayerType::InputLayer );
        EXPECT_EQ( smlLayer->isBiased( ), true );
        EXPECT_EQ( smlLayer->countOfNeurons( ), 2 ); // 1 input + bias neuron
    }

    TEST( MachineLearningTests, OutputLayer ) {

        using namespace Core;
        using namespace MachineLearning;

        auto smlLayer = std::make_shared<OutputLayer>( 1 );


        EXPECT_EQ( smlLayer->getLayerType( ), LayerType::OutputLayer );
        EXPECT_EQ( smlLayer->isBiased( ), false );
        EXPECT_EQ( smlLayer->countOfNeurons( ), 1 ); // 1 input
    }

    TEST( MachineLearningTests, Connections ) {

        using namespace Core;
        using namespace MachineLearning;

        std::array<Core::real, 2> input1{ real( 0.0 ), real( 1.0 ) };

        auto inLayer   = std::make_shared<InputLayer>( input1 );
        auto outLayer1 = std::make_shared<OutputLayer>( 1 );
        auto outLayer2 = std::make_shared<OutputLayer>( 2 );

        auto connector = std::make_shared<Connections>( inLayer, outLayer1 );
        EXPECT_EQ( connector->getWeightCount( ), 3 * 1 );
        EXPECT_EQ( connector->getNeuronConnectionCount( ), 1 );

        auto connector2 = std::make_shared<Connections>( inLayer, outLayer2 );
        EXPECT_EQ( connector2->getWeightCount( ), 3 * 2 );
        EXPECT_EQ( connector2->getNeuronConnectionCount( ), 2 );

    }

    TEST( MachineLearningTests, ActivationFunctions ) {
        using namespace Core;
        {
            ActivationFunction        a( ActivationFunctionType::Linear );
            std::array<Core::real, 2> in{ real( 0.0 ), real( 1.0 ) };
            std::array<Core::real, 2> out;
            a.activate( 2, in.begin( ), out.begin( ) );
            EXPECT_FLOAT_EQ( out[ 0 ], real( 0.0 ) );
            EXPECT_FLOAT_EQ( out[ 1 ], real( 1.0 ) );
        }
        {
            ActivationFunction        a( ActivationFunctionType::Step );
            std::array<Core::real, 2> in{ real( 0.4 ), real( 0.6 ) };
            std::array<Core::real, 2> out;
            a.activate( 2, in.begin( ), out.begin( ) );
            EXPECT_FLOAT_EQ( out[ 0 ], real( 0.0 ) );
            EXPECT_FLOAT_EQ( out[ 1 ], real( 1.0 ) );
        }
        {
            ActivationFunction        a( ActivationFunctionType::Sigmoid );
            std::array<Core::real, 2> in{ real( 0.0 ), real( 1.0 ) };
            std::array<Core::real, 2> out;
            a.activate( 2, in.begin( ), out.begin( ) );
            EXPECT_FLOAT_EQ( out[ 0 ], real( 0.5 ) );
            EXPECT_FLOAT_EQ( out[ 1 ], real( 0.73105897 ) );
        }
        {
            ActivationFunction        a( ActivationFunctionType::HyperbolicTangent );
            std::array<Core::real, 2> in{ real( 0.0 ), real( 1.0 ) };
            std::array<Core::real, 2> out;
            a.activate( 2, in.begin( ), out.begin( ) );
            EXPECT_FLOAT_EQ( out[ 0 ], real( 0.0 ) );
            EXPECT_FLOAT_EQ( out[ 1 ], real( 0.761594176 ) );
        }
        {
            ActivationFunction        a( ActivationFunctionType::ReLU );
            std::array<Core::real, 2> in{ real( -1.0 ), real( 2.0 ) };
            std::array<Core::real, 2> out;
            a.activate( 2, in.begin( ), out.begin( ) );
            EXPECT_FLOAT_EQ( out[ 0 ], real( 0.0 ) );
            EXPECT_FLOAT_EQ( out[ 1 ], real( 2.0 ) );
        }
    }

    TEST( MachineLearningTests, ANNetworkStructureInOut ) {

        using namespace Core;
        using namespace MachineLearning;

        {
            std::array<Core::real, 2> input1{ real( 0.0 ), real( 1.0 ) };
            std::array<Core::real, 1> perfect1{ real( 4.0 ) };

            auto inLayer  = std::make_shared<InputLayer>( input1 );
            auto outLayer = std::make_shared<OutputLayer>( 1 );

            auto connector = std::make_shared<Connections>( inLayer, outLayer );

            ANNetwork ann{ };
            EXPECT_EQ( ann.scratchPad0, nullptr );
            EXPECT_EQ( ann.sums, nullptr );
            EXPECT_EQ( ann.outputs, nullptr );
            EXPECT_EQ( ann.weights, nullptr );

            ann.addLayer( inLayer );
            ann.addLayer( outLayer );
            ann.connectLayers( connector );
            ann.finalise( true );

            EXPECT_EQ( *ann.begin( ), inLayer );
            EXPECT_EQ( *(--ann.end( )), outLayer );

            EXPECT_NE( ann.scratchPad0, nullptr );
            EXPECT_NE( ann.sums, nullptr );
            EXPECT_NE( ann.outputs, nullptr );
            EXPECT_NE( ann.weights, nullptr );

            // 2 inputs nodes + 1 bias to a single output node
            EXPECT_EQ( ann.totalWeightCount, 3 );
            EXPECT_EQ( ann.totalNeuronCount, 4 );

            Core::Random::seed( 0xDEA0DEA0 ); // fixed seed for test

            ann.setRandomWeights( );

            EXPECT_FLOAT_EQ( ann.weights[ 0 ], 0.880438685 );

            ann.computeGradients( perfect1.data( ) );
        }
        {
            // XOR Neural network test
            // 2 inputs (bool) -> 1 output (bool) with 1 node hidden layer

            auto inLayer  = std::make_shared<InputLayer>( 2 );
            auto hidLayer = std::make_shared<HiddenLayer>( 1 );
            auto outLayer = std::make_shared<OutputLayer>( 1 );

            auto connIH = std::make_shared<Connections>( inLayer, hidLayer );
            auto connHO = std::make_shared<Connections>( hidLayer, outLayer );

            ANNetwork ann{ };
            ann.addLayer( inLayer );
            ann.addLayer( hidLayer );
            ann.addLayer( outLayer );
            ann.connectLayers( connIH );
            ann.connectLayers( connHO );
            ann.finalise( true );

            Core::Random::seed( 0xDEA0DEA0 ); // fixed seed for test
            ann.setRandomWeights( );
            auto ti0 = std::array<Core::real, 2>{ real( 0.0 ), real( 0.0 ) };
            auto ti1 = std::array<Core::real, 2>{ real( 1.0 ), real( 0.0 ) };
            auto ti2 = std::array<Core::real, 2>{ real( 0.0 ), real( 1.0 ) };
            auto ti3 = std::array<Core::real, 2>{ real( 1.0 ), real( 1.0 ) };

            auto to0 = std::array<Core::real, 1>{ real( 0.0 ) };
            auto to1 = std::array<Core::real, 1>{ real( 1.0 ) };
            ann.supervisedTrain( ti0.data( ), to0.data( ) );
            ann.supervisedTrain( ti1.data( ), to1.data( ) );
            ann.supervisedTrain( ti2.data( ), to1.data( ) );
            ann.supervisedTrain( ti3.data( ), to0.data( ) );

        }

    }
}