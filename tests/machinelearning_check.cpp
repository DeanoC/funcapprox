//
// Created by Dean Calver on 15/04/2016.
//

#include "core/core.h"
#include "core/random.h"
#include <array>
#include <boost/generator_iterator.hpp>
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

        auto connector2 = std::make_shared<Connections>( inLayer, outLayer2 );
        EXPECT_EQ( connector2->getWeightCount( ), 3 * 2 );

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
            EXPECT_FLOAT_EQ( out[ 1 ], real( 0.7310586 ) );
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
/*
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
        }*/
        {
            // XOR Neural network test
            // 2 inputs (bool) -> 1 output (bool) with 3 node hidden layer

            auto inLayer  = std::make_shared<InputLayer>( 2 );
            auto hidLayer = std::make_shared<HiddenLayer>( 2 );
            auto outLayer = std::make_shared<OutputLayer>( 1 );

            auto connIH = std::make_shared<Connections>( inLayer, hidLayer, 2 );
            auto connHO = std::make_shared<Connections>( hidLayer, outLayer, 1 );

            ANNetwork ann{ };
            ann.addLayer( inLayer );
            ann.addLayer( hidLayer );
            ann.addLayer( outLayer );
            ann.connectLayers( connIH );
            ann.connectLayers( connHO );
            ann.finalise( true );

            // use weights from JHeatons website to check answers
            std::vector<Core::real> wts{
                    0.13, 0.63, 0.68, 0.89, 0.94, -0.86,
                    -0.5, -0.4, 0.44
            };
            ann.setWeights( wts );


            const auto ti0 = std::array<Core::real, 2>{ real( 0.0 ), real( 0.0 ) };
            const auto ti1 = std::array<Core::real, 2>{ real( 1.0 ), real( 0.0 ) };
            const auto ti2 = std::array<Core::real, 2>{ real( 0.0 ), real( 1.0 ) };
            const auto ti3 = std::array<Core::real, 2>{ real( 1.0 ), real( 1.0 ) };
            const auto to0 = std::array<Core::real, 1>{ real( 0.0 ) };
            const auto to1 = std::array<Core::real, 1>{ real( 1.0 ) };
            using pair = std::pair<std::array<Core::real, 2>, std::array<Core::real, 1> >;

            const pair arr[] = { pair( ti0, to0 ), pair( ti1, to1 ), pair( ti2, to1 ), pair( ti3, to0 ) };

            Random::uniform_int_gen_type                            kRandGen( Random::generator,
                                                                              Random::ui_distribution_type( 0, 3 ) );
            boost::generator_iterator<Random::uniform_int_gen_type> kIter( &kRandGen );

            std::array<int, 3>                   train = { *kIter++, *kIter++, *kIter++ };
            std::vector<ANNetwork::MatchingPair> training;

//            for( auto &&it : train ) {
//                training.push_back( ANNetwork::MatchingPair( arr[ it ].first.data( ), arr[ it ].second.data( ) ) );
//            }
            training.emplace_back( ti0.data( ), to0.data( ) );
            training.emplace_back( ti1.data( ), to1.data( ) );
            training.emplace_back( ti2.data( ), to1.data( ) );
            training.emplace_back( ti3.data( ), to0.data( ) );
            training.emplace_back( ti0.data( ), to0.data( ) );
            training.emplace_back( ti1.data( ), to1.data( ) );
            training.emplace_back( ti2.data( ), to1.data( ) );
            training.emplace_back( ti2.data( ), to0.data( ) );
            ann.supervisedTrain( training, training );
        }

    }
}