//
// Created by Dean Calver on 12/04/2016.
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "core/core.h"
#include "core/vectoralu.h"

namespace MachineLearning {

    class ActivationFunction;

    enum class LayerConnections : uint8_t {
        InputLayer = 0,
        OutputLayer
    };

    class Layer {
    public:
        friend class ANNetwork;

        using vector_type = Core::VectorALU::real_array_ptr;
        using shared_ptr = std::shared_ptr<Layer>;

        virtual size_t numberOfDimensions() const { return sizeOfDims.size(); }

        virtual size_t sizeOfDimension(const size_t d) const { return sizeOfDims.at(d); }

        virtual size_t size() const { return totalSize; }

        virtual const LayerConnections getConnectionType() const = 0;

        virtual void finalise();


    protected:
        Layer() = default;

        std::shared_ptr<ActivationFunction> activationFunc;

        std::vector<size_t> sizeOfDims;         // each dimension has N elements
        size_t totalSize = 0;                   // how many in total reals in this layer
    };
}

