#pragma once

#include <cassert>
#include <cmath>
#include "core/core.h"
#include "core/vectoralu.h"

namespace Core {

    class BasicCPPVectorALU : public VectorALU {
    public:
        VectorALUBackend getBackendType() const override final { return VectorALUBackend::BASIC_CPP; };

        virtual real_array_ptr newRealVector( const size_t size ) const override;

        virtual void deleteRealVector( real_array_ptr &vector ) const override;

        // vector basic ops
        virtual void add( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const override;

        virtual void sub( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const override;

        virtual void mul( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const override;

        virtual void div( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                          real_array_ptr &o ) const override;

        // scalar basic ops
        virtual void add( const size_t numItems, const_real_array_ptr &a, const real b,
                          real_array_ptr &o ) const override;

        virtual void sub( const size_t numItems, const_real_array_ptr &a, const real b,
                          real_array_ptr &o ) const override;

        virtual void mul( const size_t numItems, const_real_array_ptr &a, const real b,
                          real_array_ptr &o ) const override;

        virtual void div( const size_t numItems, const_real_array_ptr &a, const real b,
                          real_array_ptr &o ) const override;

        // fused multiply accumalate
        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b,
                           const_real_array_ptr &c, real_array_ptr &out ) const override;

        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const real b, const real c,
                           real_array_ptr &out ) const override;

        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const const_real_array_ptr &b, const real c,
                           real_array_ptr &out ) const override;

        virtual void fmad( const size_t numItems, const_real_array_ptr &a, const real b, const const_real_array_ptr &c,
                           real_array_ptr &out ) const override;

        virtual void horizSum( const size_t numItems, const_real_array_ptr &a, real &o ) const override;

        virtual real horizSum( const size_t numItems, const_real_array_ptr &a ) const override;

        virtual void abs( const size_t numItems, const_real_array_ptr &a, real_array_ptr &o ) const override;

        virtual void set( const size_t numItems, const real value, real_array_ptr &o ) const override;

        virtual void replicateItems( const size_t numInItems, const size_t replAmnt, const_real_array_ptr &a,
                                     real_array_ptr &o ) const override;

        virtual void negate( const size_t numItems, const_real_array_ptr &a, real_array_ptr &o ) const override;

        virtual void min( const size_t numItems, const_real_array_ptr &a, const real test,
                          real_array_ptr &o ) const override;

        virtual void max( const size_t numItems, const_real_array_ptr &a, const real test,
                          real_array_ptr &o ) const override;

        virtual void copy( const size_t numItems, const_real_array_ptr &a, real_array_ptr &o ) const override;

        virtual void shuffle( const size_t numItems, const_real_array_ptr &mixer, const_real_array_ptr &a,
                              const_real_array_ptr &b, real_array_ptr &o ) const override;

        virtual void replaceif( const size_t numItems, const_real_array_ptr &chooser, const_real_array_ptr &a,
                                const real with, real_array_ptr &o ) const override;

        virtual void step( const size_t numItems, const_real_array_ptr &a, const real test,
                           real_array_ptr &o ) const override;

        virtual void relu( const size_t numItems, const_real_array_ptr &a, const real test, real_array_ptr &o,
                           const real lower = real( 0 ) ) const override;

        virtual void sigmoid( const size_t numItems, const_real_array_ptr &a, real_array_ptr &o ) const override;

        virtual void hyperbolicTangent( const size_t numItems, const_real_array_ptr &a,
                                        real_array_ptr &o ) const override;

        virtual real norm1( const size_t numItems, const_real_array_ptr &a ) const override;

        virtual real norm2( const size_t numItems, const_real_array_ptr &a ) const override;

        virtual real norm3( const size_t numItems, const_real_array_ptr &a ) const override;

        virtual real normInfinite( const size_t numItems, const_real_array_ptr &a ) const override;

        virtual range minMaxOf( const size_t numItems, const_real_array_ptr &input ) const override;

        virtual bool compareEquals( const size_t numItems, const_real_array_ptr &a,
                                    const_real_array_ptr &b ) const override;

        virtual bool compareNotEquals( const size_t numItems, const_real_array_ptr &a,
                                       const_real_array_ptr &b ) const override;

        virtual bool compareAllGreater( const size_t numItems, const_real_array_ptr &a,
                                        const_real_array_ptr &b ) const override;

        virtual bool compareAllLess( const size_t numItems, const_real_array_ptr &a,
                                     const_real_array_ptr &b ) const override;

        virtual void gather( const size_t numItems, const real *&a, const size_t stride,
                             real_array_ptr &o ) const override;

        virtual void scatter( const size_t numItems, const_real_array_ptr &a, const size_t stride,
                              real *&o ) const override;

    protected:
        void UnOp( const size_t numItems, const_real_array_ptr &a, real_array_ptr &o,
                   const std::function<real( const real )> lambda ) const {
            for( auto i = 0; i < numItems; ++i ) {
                o[ i ] = lambda( a[ i ] );
            }
        }

        void BinOp( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b, real_array_ptr &o,
                    const std::function<real( const real, const real )> &lambda ) const {
            for( auto i = 0; i < numItems; ++i ) {
                o[ i ] = lambda( a[ i ], b[ i ] );
            }
        }

        void BinOp( const size_t numItems, const_real_array_ptr &a, const real b, real_array_ptr &o,
                    const std::function<real( const real, const real )> &lambda ) const {
            for( auto i = 0; i < numItems; ++i ) {
                o[ i ] = lambda( a[ i ], b );
            }
        }

        void TrinOp( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b, const_real_array_ptr &c,
                     real_array_ptr &o,
                     const std::function<real( const real, const real, const real )> &lambda ) const {
            for( auto i = 0; i < numItems; ++i ) {
                o[ i ] = lambda( a[ i ], b[ i ], c[ i ] );
            }
        }

        void TrinOp( const size_t numItems, const_real_array_ptr &a, const_real_array_ptr &b, const real c,
                     real_array_ptr &o,
                     const std::function<real( const real, const real, const real )> &lambda ) const {
            for( auto i = 0; i < numItems; ++i ) {
                o[ i ] = lambda( a[ i ], b[ i ], c );
            }
        }

        void TrinOp( const size_t numItems, const_real_array_ptr &a, const real b, const_real_array_ptr &c,
                     real_array_ptr &o,
                     const std::function<real( const real, const real, const real )> &lambda ) const {
            for( auto i = 0; i < numItems; ++i ) {
                o[ i ] = lambda( a[ i ], b, c[ i ] );
            }
        }

        void TrinOp( const size_t numItems, const_real_array_ptr &a, const real b, const real c, real_array_ptr &o,
                     const std::function<real( const real, const real, const real )> &lambda ) const {
            for( auto i = 0; i < numItems; ++i ) {
                o[ i ] = lambda( a[ i ], b, c );
            }
        }

    };
}