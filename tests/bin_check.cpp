//
// Created by Dean Calver on 15/04/2016.
//
#include "core/core.h"
#include "gtest/gtest.h"
#include "../bin/realfunc.h"

TEST( RealFuncTests, IsSine ) {
    RealFunc f;
    EXPECT_FLOAT_EQ( f( 0.0 ), std::sin( 0.0 ) );
    EXPECT_FLOAT_EQ( f( 1.0 ), std::sin( 1.0 ) );
    EXPECT_FLOAT_EQ( f( -1.0 ), std::sin( -1.0 ) );
}
