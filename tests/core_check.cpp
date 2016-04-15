//
// Created by Dean Calver on 15/04/2016.
//

#include "core/core.h"
#include "gtest/gtest.h"

TEST( CoreTests, AlmostEqual ) {
    EXPECT_EQ( Core::almost_equal( 0.0, 0.0 ), true );
    EXPECT_EQ( Core::almost_equal( 0.0, 1.0 ), false );
    EXPECT_EQ( Core::almost_equal( 0.0, 0.0 + std::numeric_limits<double>::min( ) ), false );
    EXPECT_EQ( Core::almost_equal( 0.0, 0.0 + std::numeric_limits<double>::epsilon( ) ), false );

}