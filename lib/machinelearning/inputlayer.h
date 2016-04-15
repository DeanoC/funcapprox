//
// Created by Dean Calver on 14/04/2016.
//

#pragma once

#include <array>
#include "machinelearning/layer.h"

namespace MachineLearning {

    class InputLayer : public Layer {
    public:
        friend class ANNetwork;

        InputLayer( const size_t _neuronCount );

        InputLayer( const size_t _neuronCount, const Core::real *_inputData );

        template< size_t count >
        InputLayer( const std::array<Core::real, count> &array ) : InputLayer( count, array.data( ) ) { }

    public:

        const std::vector<Core::real> &getInputData() const { return inputData; }

        void setInputData( const std::vector<Core::real> &_inputData ) { inputData = _inputData; }

        void setInputData( const Core::real *start,
                           const Core::real *end ) { inputData = std::vector<Core::real>( start, end ); }

        template< size_t count >
        void setInputData( const std::array<Core::real, count> &array ) {
            inputData = std::vector<Core::real>( array.begin( ),
                                                 array.end( ) );
        }

    private:
        std::vector<Core::real> inputData;
    };
}


