//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "connections.h"


namespace MachineLearning {

    // fully connected
    Connections::Connections( const Layer::shared_ptr &_from, const Layer::shared_ptr &_to ) :
            from( _from ),
            to( _to ),
            weightCount( from->countOfNeurons( ) * to->countOfNeurons( ) ),
            neuronConnectionCount( to->countOfNeurons( ) ) {
    }

}