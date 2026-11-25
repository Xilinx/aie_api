// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef __AIE_API_DETAIL_AIE2_SHUFFLE__HPP__
#define __AIE_API_DETAIL_AIE2_SHUFFLE__HPP__

#include "../../vector.hpp"
#include "../../mask.hpp"

namespace aie::detail {

// Shifts a 2048-bit vector from the 1024-bit implementation: each half of the result is the shift of two consecutive
// halves of (v, fill), (v0, v1) and (v1, fill0) when n < Elems / 2, (v1, fill0) and (fill0, fill1) otherwise
template <unsigned TypeBits, typename T, unsigned Elems>
struct shuffle_down_fill_bits_halves
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, const vector_type &fill, unsigned n)
    {
        using half_type = vector<T, Elems / 2>;
        using half_impl = shuffle_down_fill_bits_impl<TypeBits, T, Elems / 2>;

        vector_type ret;

        const half_type v0 =    v.template extract<Elems / 2>(0), v1 =    v.template extract<Elems / 2>(1);
        const half_type f0 = fill.template extract<Elems / 2>(0), f1 = fill.template extract<Elems / 2>(1);

        if (chess_manifest(n >= Elems / 2)) {
            ret.template insert<Elems / 2>(0, half_impl::run(v1, f0, n - Elems / 2));
            ret.template insert<Elems / 2>(1, half_impl::run(f0, f1, n - Elems / 2));
        }
        else if (chess_manifest(n < Elems / 2)) {
            ret.template insert<Elems / 2>(0, half_impl::run(v0, v1, n));
            ret.template insert<Elems / 2>(1, half_impl::run(v1, f0, n));
        }
        else {
            const bool upper = n >= Elems / 2;
            const half_type a = upper? v1 : v0;
            const half_type b = upper? f0 : v1;
            const half_type c = upper? f1 : f0;
            const unsigned  m = upper? n - Elems / 2 : n;

            ret.template insert<Elems / 2>(0, half_impl::run(a, b, m));
            ret.template insert<Elems / 2>(1, half_impl::run(b, c, m));
        }

        return ret;
    }
};

template <typename T, unsigned Elems>
struct shuffle_down_fill_bits_impl<4, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, const vector_type &fill, unsigned n)
    {
        using next_type = utils::get_next_integer_type_t<T>;
        vector_type ret;

        if (chess_manifest(n == 0))
            return v;

        if (chess_manifest(n % 2 == 0))
            return shuffle_down_fill_bits_impl<8, next_type, Elems / 2>::run(v.template cast_to<next_type>(), fill.template cast_to<next_type>(), n / 2).template cast_to<T>();

        if constexpr (vector_type::bits() <= 1024) {
            ret = shuffle_down_fill_bits_impl<8, next_type, Elems>::run(v.unpack(), fill.unpack(), n).template pack<T>();

            return ret;
        }
        else {
            return shuffle_down_fill_bits_halves<4, T, Elems>::run(v, fill, n);
        }
    }
};

template <typename T, unsigned Elems>
struct shuffle_down_bits_impl<4, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, unsigned n)
    {
        using next_type = utils::get_next_integer_type_t<T>;
        vector_type ret;

        if (chess_manifest(n == 0))
            return v;

        if (chess_manifest(n % 2 == 0))
            return shuffle_down_bits_impl<8, next_type, Elems / 2>::run(v.template cast_to<next_type>(), n / 2).template cast_to<T>();

        if constexpr (vector_type::bits() <= 1024) {
            ret = shuffle_down_bits_impl<8, next_type, Elems>::run(v.unpack(), n).template pack<T>();

            return ret;
        }
        else {
            return shuffle_down_fill_bits_impl<4, T, Elems>::run(v, vector_type(), n);
        }
    }
};

template <unsigned TypeBits, typename T, unsigned Elems> requires(TypeBits >= 4)
struct shuffle_up_rotate_bits_impl<TypeBits, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, unsigned n)
    {
        return shuffle_down_rotate_bits_impl<TypeBits, T, Elems>::run(v, Elems - n);
    }
};

template <unsigned TypeBits, typename T, unsigned Elems> requires(TypeBits >= 4)
struct shuffle_up_fill_bits_impl<TypeBits, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, const vector_type &fill, unsigned n)
    {
        return shuffle_down_fill_bits_impl<TypeBits, T, Elems>::run(fill, v, Elems - n);
    }
};

template <unsigned TypeBits, typename T, unsigned Elems> requires(TypeBits >= 4)
struct shuffle_up_bits_impl<TypeBits, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, unsigned n)
    {
        return shuffle_up_fill_bits_impl<TypeBits, T, Elems>::run(v, vector_type(), n);
    }
};

template <unsigned TypeBits, typename T, unsigned Elems> requires(TypeBits >= 4)
struct shuffle_down_rotate_bits_impl<TypeBits, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, unsigned n)
    {
        return shuffle_down_fill_bits_impl<TypeBits, T, Elems>::run(v, v, n);
    }
};

template <unsigned TypeBits, typename T, unsigned Elems>
    requires(TypeBits >= 8)
struct shuffle_down_fill_bits_impl<TypeBits, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, const vector_type &fill, unsigned n)
    {
        vector_type ret;

#if __AIE_ARCH__ == 22
        // TODO: CRVO-11672: cast cbfloat16 to float as sel does not work properly with cbfloat16 on AIE2PS
        if constexpr (std::is_same_v<T, cbfloat16>) {
            return shuffle_down_fill_bits_impl<TypeBits, float, Elems>::run(
                v.template cast_to<float>(),
                fill.template cast_to<float>(),
                n
            ).template cast_to<cbfloat16>();
        }
#endif

        if (chess_manifest(n == 0)
#if AIE_API_NATIVE
            || (n == 0)
#endif
            )
            return v;

        if constexpr (vector_type::bits() == 128) {
            ret = shuffle_down_fill_bits_impl<TypeBits, T, Elems * 4>::run(concat_vector(            v,           fill, vector_type(), vector_type()),
                                                                           concat_vector(vector_type(),  vector_type(), vector_type(), vector_type()), n).template extract<Elems>(0);
        }
        else if constexpr (vector_type::bits() == 256) {
            ret = shuffle_down_fill_bits_impl<TypeBits, T, Elems * 2>::run(concat_vector(            v,           fill),
                                                                           concat_vector(vector_type(),  vector_type()), n).template extract<Elems>(0);
        }
        else if constexpr (vector_type::bits() == 512) {
            ret = ::shift_bytes(v, fill, n * sizeof(T));
        }
        else if constexpr (vector_type::bits() == 1024) {
            if (chess_manifest(n >= Elems / 2)) {
                ret.template insert<Elems / 2>(0, ::shift_bytes(   v.template extract<Elems / 2>(1),
                                                                fill.template extract<Elems / 2>(0), (n - Elems / 2) * sizeof(T)));
                ret.template insert<Elems / 2>(1, ::shift_bytes(fill.template extract<Elems / 2>(0),
                                                                fill.template extract<Elems / 2>(1), (n - Elems / 2) * sizeof(T)));
            }
            else if (chess_manifest(n < Elems / 2)) {
                ret.template insert<Elems / 2>(0, ::shift_bytes(   v.template extract<Elems / 2>(0),
                                                                   v.template extract<Elems / 2>(1), n * sizeof(T)));
                ret.template insert<Elems / 2>(1, ::shift_bytes(   v.template extract<Elems / 2>(1),
                                                                fill.template extract<Elems / 2>(0), n * sizeof(T)));
            }
            else {
                vector<T, Elems / 2> tmp1a, tmp1b, tmp2a, tmp2b;

                mask<Elems / 2> mask1(true), mask2(true);

                mask1 <<= Elems - n;
                mask2 <<= Elems - n;

                tmp1a = ::shift_bytes(v.template extract<Elems / 2>(0),    v.template extract<Elems / 2>(1), n * sizeof(T));
                tmp1b = ::shift_bytes(vector<T, Elems / 2>(),           fill.template extract<Elems / 2>(0), (n - Elems / 2) * sizeof(T));

                tmp2a = ::shift_bytes(v.template extract<Elems / 2>(1), fill.template extract<Elems / 2>(0), n * sizeof(T));
                tmp2b = ::shift_bytes(vector<T, Elems / 2>(),           fill.template extract<Elems / 2>(1), (n - Elems / 2) * sizeof(T));

                if constexpr (Elems > 64) {
                    ret.template insert<Elems / 2>(0, ::sel(tmp1a, tmp1b, mask1.to_uint64()));
                    ret.template insert<Elems / 2>(1, ::sel(tmp2a, tmp2b, mask2.to_uint64()));
                }
                else {
                    ret.template insert<Elems / 2>(0, ::sel(tmp1a, tmp1b, mask1.to_uint32()));
                    ret.template insert<Elems / 2>(1, ::sel(tmp2a, tmp2b, mask2.to_uint32()));
                }
            }
        }
        else {
            ret = shuffle_down_fill_bits_halves<TypeBits, T, Elems>::run(v, fill, n);
        }

        return ret;
    }
};

template <unsigned TypeBits, typename T, unsigned Elems> requires(TypeBits >= 8)
struct shuffle_down_bits_impl<TypeBits, T, Elems>
{
    using vector_type = vector<T, Elems>;

    __aie_inline
    static vector_type run(const vector_type &v, unsigned n)
    {
        vector_type ret;

        if (chess_manifest(n == 0))
            return v;

        if constexpr (vector_type::bits() == 128) {
            ret = shuffle_down_bits_impl<TypeBits, T, Elems * 4>::run(v.template grow<Elems * 4>(), n).template extract<Elems>(0);
        }
        else if constexpr (vector_type::bits() == 256) {
            ret = shuffle_down_bits_impl<TypeBits, T, Elems * 2>::run(v.template grow<Elems * 2>(), n).template extract<Elems>(0);
        }
        else if constexpr (vector_type::bits() == 512) {
            ret = ::shift_bytes(v, v, n * sizeof(T));
        }
        else if constexpr (vector_type::bits() == 1024) {
            ret.template insert<Elems / 2>(0, ::shift_bytes(v.template extract<Elems / 2>(0), v.template extract<Elems / 2>(1), n * sizeof(T)));
            ret.template insert<Elems / 2>(1, ::shift_bytes(v.template extract<Elems / 2>(1), vector<T, Elems / 2>(),           n * sizeof(T)));
        }
        else {
            ret = shuffle_down_fill_bits_impl<TypeBits, T, Elems>::run(v, vector_type(), n);
        }

        return ret;
    }
};

}

#endif
