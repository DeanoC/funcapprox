//
// Created by Dean Calver on 12/04/2016.
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "core/core.h"
#include "core/vectoralu.h"
#include "machinelearning/ActivationFunction.h"

namespace MachineLearning {

    /*
     * There are a number of layers types, which the 3 basic types as Input, Hidden and Output
     */
    enum class LayerType : uint8_t {
        InputLayer = 0,
        HiddenLayer,
        OutputLayer
    };

    /*
     * Layers are internally flat arrays of real numbers, a LayerView makes them appear as N dimenstional objects
     */
    class LayerView {
    public:
        virtual size_t numberOfDimensions() const { return sizeOfDims.size( ); }

        virtual size_t sizeOfDimension( const size_t d ) const { return sizeOfDims.at( d ); }

    protected:
        std::vector<size_t> sizeOfDims;         // each dimension has N elements
    };


    /*
     * A layer has (external) arrays of reals repesenting an ANN layer, each also has a non linear activation function
     * For efficiency the actually data is stored in a large continous array and the layer just contains the offset
     * where its actualy data is.
     */
    class Layer {
    public:
        friend class ANNetwork;

        using shared_ptr = std::shared_ptr<Layer>;

        LayerType getLayerType() const { return layerType; }

        size_t countOfNeurons() const { return neuronCount + ( isBiased( ) ? 1 : 0 ); }

        bool isBiased() const {
            return biased;
        }

    protected:

        Layer( const LayerType _layerType, const size_t _neuronCount, const bool _biased = true );

        Layer() = delete;

        const LayerType layerType;

        const size_t neuronCount; // how many nuerons in this layer

        const bool biased;

        std::unique_ptr<ActivationFunction> activationFunc;

        size_t neuronIndex; // where does the neurons start for this layer in the shared array
    };
}

