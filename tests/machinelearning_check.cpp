//
// Created by Dean Calver on 15/04/2016.
//

#include "core/core.h"
#include <array>
#include "machinelearning/machinelearning.h"
#include "machinelearning/inputlayer.h"
#include "machinelearning/outputlayer.h"
#include "machinelearning/connections.h"
#include "machinelearning/ANNetwork.h"
#include "gtest/gtest.h"

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

TEST( MachineLearningTests, ANNetworkStructureInOut ) {

    using namespace Core;
    using namespace MachineLearning;

    std::array<Core::real, 2> input1{ real( 0.0 ), real( 1.0 ) };

    auto inLayer  = std::make_shared<InputLayer>( input1 );
    auto outLayer = std::make_shared<OutputLayer>( 1 );

    auto connector = std::make_shared<Connections>( inLayer, outLayer );

    ANNetwork ann;
    ann.addLayer( inLayer );
    ann.addLayer( outLayer );
    ann.connectLayers( connector );
    ann.finalise( true );

    EXPECT_EQ( *ann.begin( ), inLayer );
    EXPECT_EQ( *( --ann.end( ) ), outLayer );


}