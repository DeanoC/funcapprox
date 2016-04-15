#pragma once

#include <cstdint>
#include <vector>
#include "core.h"

namespace Core {

    enum class VectorALUBackend : uint8_t {
        BASIC_CPP
    };

    struct VectorALU {
        // these are never used, consider this the 'interface' which each ALU backend should support at a minimum

        using real = Core::real;
        using real_array_ptr = real *;
        using const_real_array_ptr = real const *const;
        using range = std::pair<real, real>;

        virtual VectorALUBackend getBackendType() const = 0;

        virtual real_array_ptr newRealVector( const size_t size ) const         = 0;

        virtual void           deleteRealVector( real_array_ptr &vector ) const = 0;

        // vector basic ops
        virtual void add( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const = 0;

        virtual void sub( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const = 0;

        virtual void mul( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const = 0;

        virtual void div( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const = 0;

        // scalar basic ops
        virtual void add( const size_t numItems, const_real_array_ptr &a, const real b, real_array_ptr &o ) const = 0;

        virtual void sub( const size_t numItems, const_real_array_ptr &a, const real b, real_array_ptr &o ) const = 0;

        virtual void mul( const size_t numItems, const_real_array_ptr &a, const real b, real_array_ptr &o ) const = 0;

        virtual void div( const size_t numItems, const_real_array_ptr &a, const real b, real_array_ptr &o ) const = 0;

        // fused multiply accumalate
        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                           const_real_array_ptr &c, real_array_ptr &o ) const = 0;

        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const real b, const real c,
                           real_array_ptr &o ) const = 0;

        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const const_real_array_ptr &b, const real c,
                           real_array_ptr &o ) const = 0;

        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const real b, const const_real_array_ptr &c,
                           real_array_ptr &o ) const = 0;

        //
        virtual void horizSum( const size_t numItems, const_real_array_ptr &a, real &o ) const = 0;

        virtual real horizSum( const size_t numItems, const_real_array_ptr &a ) const = 0;

        virtual void min( const size_t numItems, const_real_array_ptr &a, const real test,
                          real_array_ptr &o ) const = 0;

        virtual void max( const size_t numItems, const_real_array_ptr &a, const real test,
                          real_array_ptr &o ) const = 0;

        virtual void abs( const size_t numItems, const_real_array_ptr &a, real_array_ptr &out ) const = 0;

        virtual void set( const size_t numItems, const real value, real_array_ptr &out ) const = 0;

        virtual void replicateItems( const size_t numInItems, const size_t replAmnt, const_real_array_ptr &a,
                                     real_array_ptr &o ) const = 0;

        virtual void negate( const size_t numItems, const_real_array_ptr &a, real_array_ptr &out ) const = 0;

        virtual void copy( const size_t numItems, const_real_array_ptr &a, real_array_ptr &out ) const = 0;

        virtual void shuffle( const size_t numItems, const_real_array_ptr &mixer, const_real_array_ptr &a,
                              const_real_array_ptr &b, real_array_ptr &o ) const = 0;

        virtual void replaceif( const size_t numItems, const_real_array_ptr &chooser, const_real_array_ptr &a,
                                const real with, real_array_ptr &o ) const = 0;

        virtual void step( const size_t numItems, const_real_array_ptr &a, real test, real_array_ptr &o ) const = 0;

        virtual void relu( const size_t numItems, const_real_array_ptr &a, real test, real_array_ptr &o,
                           const real lower = real( 0 ) ) const = 0;

        virtual void sigmoid( const size_t numItems, const_real_array_ptr &a, real_array_ptr &o ) const = 0;

        virtual void hyperbolicTangent( const size_t numItems, const_real_array_ptr &a, real_array_ptr &o ) const = 0;

        virtual real norm1( const size_t numItems, const_real_array_ptr &a ) const = 0;

        virtual real norm2( const size_t numItems, const_real_array_ptr &a ) const = 0;

        virtual real norm3( const size_t numItems, const_real_array_ptr &a ) const = 0;

        virtual real normInfinite( const size_t numItems, const_real_array_ptr &a ) const = 0;

        virtual range minMaxOf( const size_t numItems, const_real_array_ptr &input ) const = 0;

        virtual bool compareEquals( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b ) const = 0;

        virtual bool compareNotEquals( const size_t numItems, const_real_array_ptr &a,
                                       const_real_array_ptr &b ) const = 0;

        virtual bool compareAllGreater( const size_t numItems, const_real_array_ptr &a,
                                        const_real_array_ptr &b ) const = 0;

        virtual bool compareAllLess( const size_t numItems, const_real_array_ptr &a,
                                     const_real_array_ptr &b ) const = 0;

        virtual void gather( const size_t numItems, const real *&a, const size_t stride, real_array_ptr &o ) const = 0;

        virtual void scatter( const size_t numItems, const_real_array_ptr &a, const size_t stride, real *&o ) const = 0;
    };

    std::shared_ptr<VectorALU> VectorALUFactory();
}