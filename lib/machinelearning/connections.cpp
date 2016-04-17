//
// Created by Dean Calver on 14/04/2016.
//

#include "core/core.h"
#include "connections.h"


namespace MachineLearning {

    // fully connected
    Connections::Connections( const Layer::shared_ptr _from, const Layer::shared_ptr _to, int _fromEdgesPerNeuron ) :
            from( _from ),
            to( _to ),
            weightCount( (_fromEdgesPerNeuron == -1) ?
                         (from->countOfNeurons( ) * to->countOfNeurons( )) : (from->countOfNeurons( ) *
                                                                              _fromEdgesPerNeuron) ),
            srcNeuronConnectionCount( (_fromEdgesPerNeuron == -1) ? (weightCount / from->countOfNeurons( ))
                                                                  : _fromEdgesPerNeuron ),
            dstNeuronConnectionCount( weightCount / to->getActualNeuronCount( ) ) {
    }


}