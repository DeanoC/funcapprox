//
// Created by Dean Calver on 12/04/2016.
//

#include "core/core.h"
#include "basiccppvectoralu.h"
#include <boost/numeric/ublas/storage.hpp>

namespace Core {
    void BasicCPPVectorALU::add( const size_t numItems, const_real_array_ptr a, const_real_array_ptr b,
                                 real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av + bv; });
    }

    void BasicCPPVectorALU::sub( const size_t numItems, const_real_array_ptr a, const_real_array_ptr b,
                                 real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av - bv; });
    }

    void BasicCPPVectorALU::mul( const size_t numItems, const_real_array_ptr a, const_real_array_ptr b,
                                 real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av * bv; });
    }

    void BasicCPPVectorALU::div( const size_t numItems, const_real_array_ptr a, const_real_array_ptr b,
                                 real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av / bv; });
    }

    void BasicCPPVectorALU::add( const size_t numItems, const_real_array_ptr a, const real b, real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av + bv; });
    }

    void BasicCPPVectorALU::sub( const size_t numItems, const_real_array_ptr a, const real b, real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av - bv; });
    }

    void BasicCPPVectorALU::mul( const size_t numItems, const_real_array_ptr a, const real b, real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av * bv; });
    }

    void BasicCPPVectorALU::div( const size_t numItems, const_real_array_ptr a, const real b, real_array_ptr o ) const {
        BinOp(numItems, a, b, o, [](const real av, const real bv) -> real { return av / bv; });
    }

    void BasicCPPVectorALU::fmad( const size_t numItems, const_real_array_ptr a, const_real_array_ptr b,
                                  const_real_array_ptr c, real_array_ptr o ) const {
        TrinOp(numItems, a, b, c, o,
               [](const real av, const real bv, const real cv) -> real { return (av * bv) + cv; });
    }

    void BasicCPPVectorALU::fmad( const size_t numItems, const_real_array_ptr a, const real b, const real c,
                                  real_array_ptr o ) const {
        TrinOp(numItems, a, b, c, o,
               [](const real av, const real bv, const real cv) -> real { return (av * bv) + cv; });
    }

    void BasicCPPVectorALU::fmad( const size_t numItems, const_real_array_ptr a, const const_real_array_ptr b,
                                  const real c, real_array_ptr o ) const {
        TrinOp(numItems, a, b, c, o,
               [](const real av, const real bv, const real cv) -> real { return (av * bv) + cv; });
    }

    void BasicCPPVectorALU::fmad( const size_t numItems, const_real_array_ptr a, const real b,
                                  const const_real_array_ptr c, real_array_ptr o ) const {
        TrinOp(numItems, a, b, c, o,
               [](const real av, const real bv, const real cv) -> real { return (av * bv) + cv; });
    }

    void BasicCPPVectorALU::horizSum( const size_t numItems, const_real_array_ptr a, real &out ) const {
        out = Core::real(0);
        for (int i = 0; i < numItems; ++i) {
            out = out + a[i];
        }
    }

    real BasicCPPVectorALU::horizSum( const size_t numItems, const_real_array_ptr a ) const {
        real out;
        horizSum(numItems, a, out);
        return out;
    }

    void BasicCPPVectorALU::abs( const size_t numItems, const_real_array_ptr a, real_array_ptr o ) const {
        UnOp(numItems, a, o, [](const real av) -> real { return std::abs(av); });
    }

    void BasicCPPVectorALU::set( const size_t numItems, const real value, real_array_ptr o ) const {
        using namespace boost::numeric::ublas;
        array_adaptor<real> oa(numItems, o);
        std::fill(oa.begin(), oa.end(), value);
    }

    void BasicCPPVectorALU::replicateItems( const size_t numInItems, const size_t replAmnt, const_real_array_ptr a,
                                            real_array_ptr o ) const {
        for (auto i = 0; i < numInItems; ++i) {
            for (auto j = 0; j < replAmnt; ++j) {
                o[(i * replAmnt) + j] = a[i];
            }
        }
    }


    void BasicCPPVectorALU::negate( const size_t numItems, const_real_array_ptr a, real_array_ptr o ) const {
        UnOp(numItems, a, o, [](const real av) -> real { return -av; });
    }

    void BasicCPPVectorALU::max( const size_t numItems, const_real_array_ptr a, const real test,
                                 real_array_ptr o ) const {
        BinOp(numItems, a, test, o, [](const real av, const real bv) -> real { return std::max(av, bv); });
    }

    void BasicCPPVectorALU::min( const size_t numItems, const_real_array_ptr a, const real test,
                                 real_array_ptr o ) const {
        BinOp(numItems, a, test, o, [](const real av, const real bv) -> real { return std::min(av, bv); });
    }

    void BasicCPPVectorALU::copy( const size_t numItems, const_real_array_ptr a, real_array_ptr o ) const {
        memcpy(o, a, sizeof(real) * numItems);
    }

    void BasicCPPVectorALU::shuffle( const size_t numItems, const_real_array_ptr mixer, const_real_array_ptr a,
                                     const_real_array_ptr b, real_array_ptr o ) const {
        TrinOp(numItems, mixer, a, b, o,
               [](const real av, const real bv, const real cv) -> real { return av ? bv : cv; });
    }

    void BasicCPPVectorALU::replaceif( const size_t numItems, const_real_array_ptr chooser, const_real_array_ptr a,
                                       const real with, real_array_ptr o ) const {
        TrinOp(numItems, chooser, a, with, o, [](const real av, const real bv, const real cv) -> real {
            return Core::almost_equal(av, Core::real(1), 1) ? bv : cv;
        });
    }

    void BasicCPPVectorALU::step( const size_t numItems, const_real_array_ptr a, const real test,
                                  real_array_ptr o ) const {
        BinOp( numItems, a, test, o,
               [ ]( const real av, const real bv ) -> real { return (av > bv) ? real( 1.0 ) : real( 0.0 ); } );
    }

    void BasicCPPVectorALU::relu( const size_t numItems, const_real_array_ptr a, const real test,
                                  real_array_ptr o, const real lower ) const {
        BinOp( numItems, a, test, o,
               [ lower ]( const real av, const real bv ) -> real { return (av >= bv) ? av : lower; } );
    }

    void BasicCPPVectorALU::sigmoid( const size_t numItems, const_real_array_ptr a, real_array_ptr o ) const {
        UnOp(numItems, a, o, [](const real av) -> real { return real(1) / (real(1) + exp(-av)); });
    }

    void BasicCPPVectorALU::hyperbolicTangent( const size_t numItems, const_real_array_ptr a, real_array_ptr o ) const {
        UnOp(numItems, a, o, [](const real av) -> real { return std::tanh(av); });
    }

    real BasicCPPVectorALU::norm1( const size_t numItems, const_real_array_ptr a ) const {
        // todo remove allocation
        real_array_ptr res = newRealVector(numItems);
        abs(numItems, a, res);
        auto r = horizSum(numItems, res);
        deleteRealVector(res);
        return r;
    }

    real BasicCPPVectorALU::norm2( const size_t numItems, const_real_array_ptr a ) const {
        // todo remove allocation
        real_array_ptr res = newRealVector(numItems);
        mul(numItems, a, a, res);
        auto r = std::sqrt(horizSum(numItems, res));
        deleteRealVector(res);
        return r;
    }

    real BasicCPPVectorALU::norm3( const size_t numItems, const_real_array_ptr a ) const {
        // todo remove allocation
        real_array_ptr res = newRealVector(numItems);
        real_array_ptr res2 = newRealVector(numItems);
        mul(numItems, a, a, res);
        mul(numItems, res, res, res2);
        auto r = std::cbrt(horizSum(numItems, res2));
        deleteRealVector(res);
        deleteRealVector(res2);
        return r;
    }

    real BasicCPPVectorALU::normInfinite( const size_t numItems, const_real_array_ptr a ) const {
        // todo remove allocation
        real_array_ptr res = newRealVector(numItems);
        abs(numItems, a, res);
        auto mm = minMaxOf(numItems, res);
        deleteRealVector(res);
        return mm.second;
    }

    bool BasicCPPVectorALU::compareEquals( const size_t numItems, const_real_array_ptr a,
                                           const_real_array_ptr b ) const {
        for (int i = 0; i < numItems; ++i) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    bool BasicCPPVectorALU::compareNotEquals( const size_t numItems, const_real_array_ptr a,
                                              const_real_array_ptr b ) const {
        return !compareEquals(numItems, a, b);
    }

    bool BasicCPPVectorALU::compareAllGreater( const size_t numItems, const_real_array_ptr a,
                                               const_real_array_ptr b ) const {
        for (int i = 0; i < numItems; ++i) {
            if (a[i] <= b[i]) {
                return false;
            }
        }
        return true;
    }

    bool BasicCPPVectorALU::compareAllLess( const size_t numItems, const_real_array_ptr a,
                                            const_real_array_ptr b ) const {
        for (int i = 0; i < numItems; ++i) {
            if (a[i] >= b[i]) {
                return false;
            }
        }
        return true;
    }

    BasicCPPVectorALU::range BasicCPPVectorALU::minMaxOf( const size_t numItems, const_real_array_ptr in ) const {
        // default to min and max of the type held in the container
        real mini = std::numeric_limits<real>::max();
        real maxi = std::numeric_limits<real>::min();

        for (int i = 0; i < numItems; ++i) {
            mini = std::min(in[i], mini);
            maxi = std::max(in[i], maxi);
        }

        return range(mini, maxi);
    }

    void BasicCPPVectorALU::gather( const size_t numItems, const real *a, const size_t stride,
                                    real_array_ptr o ) const {
        for (int i = 0; i < numItems; ++i) {
            o[i] = a[i * stride];
        }
    }


    void BasicCPPVectorALU::scatter( const size_t numItems, const_real_array_ptr a, const size_t stride,
                                     real *o ) const {
        for (int i = 0; i < numItems; ++i) {
            o[i * stride] = a[i];
        }
    }

    BasicCPPVectorALU::real_array_ptr BasicCPPVectorALU::newRealVector(const size_t size) const {
        return new real[size];
    }

    void BasicCPPVectorALU::deleteRealVector(real_array_ptr &vector) const {
        delete[] vector;
        vector = nullptr;
    }
}