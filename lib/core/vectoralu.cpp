//
// Created by Dean Calver on 12/04/2016.
//

#include "core/core.h"
#include "core/vectoralu.h"
#include "core/basiccppvectoralu.h"

namespace Core {
    std::weak_ptr<VectorALU> weakSingletonVectorALU;

    std::shared_ptr<VectorALU> VectorALUFactory() {
        if (auto sptr = weakSingletonVectorALU.lock()) {
            return sptr;
        } else {
            sptr = std::make_shared<BasicCPPVectorALU>();
            weakSingletonVectorALU = static_cast<std::weak_ptr<VectorALU>>(sptr);
            return sptr;
        }
    }
}