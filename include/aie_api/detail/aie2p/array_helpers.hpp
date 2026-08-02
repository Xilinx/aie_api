// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef __AIE_API_DETAIL_AIE2P_ARRAY_HELPERS__HPP__
#define __AIE_API_DETAIL_AIE2P_ARRAY_HELPERS__HPP__

#include "../../concepts.hpp"
#include "../../vector.hpp"
#include "../../sparse_vector.hpp"
#include "../array_helpers.hpp"

namespace aie {

// Forward declarations to avoid circular dependency with aie_api/iterator.hpp
template <DecoratedElemBaseType T, unsigned Elems, aie_dm_resource Resource>
class unaligned_vector_input_buffer_stream;

template <DecoratedElemBaseType T, unsigned Elems, aie_dm_resource Resource>
class unaligned_vector_output_buffer_stream;

// Specialization for 512b/1024b input buffer streams, which can leverage the FIFO instructions

template <DecoratedElemBaseType T, unsigned Elems, aie_dm_resource Resource>
#if __AIE_API_HAS_COMPLEX_BFLOAT16_FIFO__
    requires(detail::type_bits_v<T> * Elems >= 512)
#else
// TODO: CRVO-7245/7246: cint16/cint32/bfloat16 missing for fifo intrinsics
    requires(detail::type_bits_v<T> * Elems >= 512 && !detail::is_complex_v<T> && !detail::is_floating_point_v<T>)
#endif
class unaligned_vector_input_buffer_stream<T, Elems, Resource>
{
public:
    using         elem_type = aie_dm_resource_remove_t<T>;
    using       vector_type = vector<elem_type, Elems>;

    using        value_type = vector_type;

    using       native_type = detail::native_vector_type_t<elem_type, 512 / detail::type_bits_v<elem_type>>;
    using      aliased_type = detail::add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T>>>;

    __aie_inline
    constexpr explicit unaligned_vector_input_buffer_stream(const T *ptr) :
        ptr_((aliased_type *)ptr)
    {
        RUNTIME_ASSERT(ptr != nullptr, "Input buffer stream cannot be created from nullptr");

        ::fifo_ld_reset(ptr_, fS_);
    }

    __aie_inline
    constexpr unaligned_vector_input_buffer_stream &operator>>(vector_type &v)
    {
        v = pop();
        return *this;
    }

    __aie_inline
    constexpr vector_type pop()
    {
        constexpr unsigned      num_ops = vector_type::bits() / 512;
        constexpr unsigned native_elems = Elems / num_ops;

        vector_type v;

        unroll_times<num_ops>([&](unsigned idx) __aie_inline {
            v.template insert<native_elems>(idx, ::fifo_ld_pop(ptr_, fS_));
        });

        return v;
    }

private:

    aliased_type *ptr_;
    fifo_state_t fS_;
};

// Specialization for 512b/1024b output buffer streams, which can leverage the FIFO instructions

template <DecoratedElemBaseType T, unsigned Elems, aie_dm_resource Resource>
#if __AIE_API_HAS_COMPLEX_BFLOAT16_FIFO__
    requires(detail::type_bits_v<T> * Elems >= 512)
#else
// TODO: CRVO-7245/7246: cint16/cint32/bfloat16 missing for fifo intrinsics
    requires(detail::type_bits_v<T> * Elems >= 512 && !detail::is_complex_v<T> && !detail::is_floating_point_v<T>)
#endif
class unaligned_vector_output_buffer_stream<T, Elems, Resource>
{
public:
    using         elem_type = aie_dm_resource_remove_t<T>;
    using       vector_type = vector<elem_type, Elems>;

    using        value_type = vector_type;

    using       native_type = detail::native_vector_type_t<elem_type, 512 / detail::type_bits_v<elem_type>>;
    using      aliased_type = detail::add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T>>>;

    __aie_inline
    constexpr explicit unaligned_vector_output_buffer_stream(T *ptr) :
        ptr_((aliased_type *)ptr)
    {
        RUNTIME_ASSERT(ptr != nullptr, "Output buffer stream cannot be created from nullptr");

        fS_.pos = 0;
    }

    __aie_inline
    constexpr unaligned_vector_output_buffer_stream &operator<<(const vector_type &v)
    {
        push(v);
        return *this;
    }

    __aie_inline
    constexpr void push(const vector_type &v)
    {
        constexpr unsigned      num_ops = vector_type::bits() / 512;
        constexpr unsigned native_elems = Elems / num_ops;

        unroll_times<num_ops>([&](unsigned idx) __aie_inline {
            ::fifo_st_push(ptr_, v.template extract<native_elems>(idx), fS_);
        });

        ::fifo_st_flush(ptr_, fS_);
    }

private:

    aliased_type *ptr_;
    fifo_state_t fS_;
};

namespace detail {

template <typename Vector, aie_dm_resource Resource, FifoDirection Direction, bool Restrict = false>
    requires(is_sparse_vector_v<Vector> || is_block_vector_v<Vector>)
class fifo_buffer_stream property(keep_in_registers)
{
public:
    using         vector_type = Vector;
    using           elem_type = typename vector_type::value_type;
    using          accum_type = accum<accum_tag_for_type<elem_type>, vector_type::size()>;
    using native_pointer_type = typename vector_type::native_pointer_type;
    using          value_type = vector_type;

    static constexpr unsigned native_ld_bits = 512;

    static constexpr unsigned ld_fill_ratio = []() {
        unsigned excess_bits = vector_type::memory_bits() % native_ld_bits;
        if (vector_type::memory_bits() == native_ld_bits)
            return 1u;
        else if (excess_bits == 0)
            return vector_type::memory_bits() / native_ld_bits - 1;
        else
            return native_ld_bits / excess_bits;
    }();

    __aie_inline
    void seek(int inc)
    {
        // Struct sizes for sparse vectors don't match the actual storage in memory, so we compute it manually
        ptr_ = ::byte_incr(ptr_, inc * vector_type::memory_bytes());
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek(dims_2d_t &pattern)
    {
        // Struct sizes for sparse vectors don't match the actual storage in memory, so we compute it manually
        ptr_ = ::add_2d_byte(ptr_,
                             pattern.inc2 * vector_type::memory_bytes(),
                             pattern.num1,
                             pattern.count1,
                             pattern.inc1 * vector_type::memory_bytes());
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek(dims_3d_t &pattern)
    {
        // Struct sizes for sparse vectors don't match the actual storage in memory, so we compute it manually
        ptr_ = ::add_3d_byte(ptr_,
                             pattern.inc3 * vector_type::memory_bytes(),
                             pattern.num1,
                             pattern.count1,
                             pattern.inc1 * vector_type::memory_bytes(),
                             pattern.num2,
                             pattern.count2,
                             pattern.inc2 * vector_type::memory_bytes());
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek_byte(int inc)
    {
        ptr_ = ::byte_incr(ptr_, inc);
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek_byte(dims_2d_t &pattern)
    {
        ptr_ = ::add_2d_byte(ptr_, pattern);
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek_byte(dims_3d_t &pattern)
    {
        ptr_ = ::add_3d_byte(ptr_, pattern);
        load_count_ = 0;
        state_.pos = 0;
    }

    value_type pop()
    {
        value_type ret;

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            fill(ptr_, state_);

        ret = ::fifo_ld_pop(ptr_, state_);

        load_count_ = (load_count_ + 1) % ld_fill_ratio;

        return ret;
    }

    __aie_inline
    value_type pop_seek(int inc) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            fill(ptr_, state_);

        ret = ::fifo_ld_pop_1d_byte(ptr_, state_, inc * vector_type::memory_bytes());

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek_byte(int inc) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            fill(ptr_, state_);

        ret = ::fifo_ld_pop_1d_byte(ptr_, state_, inc);

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek(dims_2d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            fill(ptr_, state_);

        ret = ::fifo_ld_pop_2d_byte(ptr_,
                                    state_,
                                    pattern.inc2 * vector_type::memory_bytes(),
                                    pattern.num1,
                                    pattern.count1,
                                    pattern.inc1 * vector_type::memory_bytes());

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek_byte(dims_2d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            fill(ptr_, state_);

        ret = ::fifo_ld_pop_2d_byte(ptr_, state_, pattern.inc2, pattern.num1, pattern.count1, pattern.inc1);

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek(dims_3d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            fill(ptr_, state_);

        ret = ::fifo_ld_pop_3d_byte(ptr_,
                                    state_,
                                    pattern.inc3 * vector_type::memory_bytes(),
                                    pattern.num1,
                                    pattern.count1,
                                    pattern.inc1 * vector_type::memory_bytes(),
                                    pattern.num2,
                                    pattern.count2,
                                    pattern.inc2 * vector_type::memory_bytes());

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek_byte(dims_3d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            fill(ptr_, state_);

        ret = ::fifo_ld_pop_3d_byte(ptr_, state_, pattern.inc3, pattern.num1, pattern.count1, pattern.inc1, pattern.num2, pattern.count2, pattern.inc2);

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    void push(const value_type &v) requires(Direction == FifoDirection::Out)
    {
        ::fifo_st_push(ptr_, v, state_);
        ::fifo_st_flush(ptr_, state_);
    }

    __aie_inline
    void push_seek(const value_type &v, int inc) requires(Direction == FifoDirection::Out)
    {
        ::fifo_st_push(ptr_, v, state_);
        ::fifo_st_flush_1d_byte(ptr_, state_, inc * vector_type::memory_bytes());
    }

    __aie_inline
    void push_seek_byte(const value_type &v, int inc) requires(Direction == FifoDirection::Out)
    {
        ::fifo_st_push(ptr_, v, state_);
        ::fifo_st_flush_1d_byte(ptr_, state_, inc);
    }

    __aie_inline
    void push_seek(const value_type &v, dims_2d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        ::fifo_st_push(ptr_, v, state_);
        ::fifo_st_flush_2d_byte(ptr_,
                                state_,
                                pattern.inc2 * vector_type::memory_bytes(),
                                pattern.num1,
                                pattern.count1,
                                pattern.inc1 * vector_type::memory_bytes());
    }

    __aie_inline
    void push_seek_byte(const value_type &v, dims_2d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        ::fifo_st_push(ptr_, v, state_);
        ::fifo_st_flush_2d_byte(ptr_, state_, pattern.inc2, pattern.num1, pattern.count1, pattern.inc1);
    }

    __aie_inline
    void push_seek(const value_type &v, dims_3d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        ::fifo_st_push(ptr_, v, state_);
        ::fifo_st_flush_3d_byte(ptr_,
                                state_,
                                pattern.inc3 * vector_type::memory_bytes(),
                                pattern.num1,
                                pattern.count1,
                                pattern.inc1 * vector_type::memory_bytes(),
                                pattern.num2,
                                pattern.count2,
                                pattern.inc2 * vector_type::memory_bytes());
    }

    __aie_inline
    void push_seek_byte(const value_type &v, dims_3d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        ::fifo_st_push(ptr_, v, state_);
        ::fifo_st_flush_3d_byte(ptr_, state_, pattern.inc3, pattern.num1, pattern.count1, pattern.inc1, pattern.num2, pattern.count2, pattern.inc2);
    }

    __aie_inline
    void push(const accum_type &acc, int shift) requires(Direction == FifoDirection::Out)
    {
        push(acc.template to_vector<elem_type>(shift));
    }

    template <typename T>
    __aie_inline
    void push_seek(const accum_type &acc, int shift, T& pattern) requires(Direction == FifoDirection::Out)
    {
        push_seek(acc.template to_vector<elem_type>(shift), pattern);
    }

    template <typename T>
    __aie_inline
    void push_seek_byte(const accum_type &acc, int shift, T& pattern) requires(Direction == FifoDirection::Out)
    {
        push_seek_byte(acc.template to_vector<elem_type>(shift), pattern);
    }

protected:
    __aie_inline
    fifo_buffer_stream(const elem_type *ptr) requires(Direction == FifoDirection::In) :
        load_count_(0),
        ptr_((ptr_type)ptr)
    {
        state_.pos = 0;
    }

    __aie_inline
    fifo_buffer_stream(elem_type *ptr) requires(Direction == FifoDirection::Out) :
        ptr_((ptr_type)ptr)
    {
        state_.pos = 0;
    }

private:
#if AIE_API_NATIVE
    using ptr_type = add_memory_bank_t<Resource, native_pointer_type> *;
#else
    using ptr_type = std::conditional_t<Restrict,
                                        add_memory_bank_t<Resource, native_pointer_type> * __restrict,
                                        add_memory_bank_t<Resource, native_pointer_type> *>;
#endif

#if __AIE_ARCH__ == 22
    __aie_inline
    void fill(ptr_type &ptr, fifo_state_t &state) requires(std::is_same_v<typename vector_type::value_type, mx4>) {}
#endif

    __aie_inline
    void fill(ptr_type &ptr, fifo_state_t &state) {
        ::fifo_ld_fill(ptr, state);
    }

    unsigned load_count_;
    ptr_type ptr_;
    fifo_state_t state_;
};

#if __AIE_ARCH__ == 22 && !__AIE_API_EMULATED_MX_LOADS__
template <typename T, unsigned M, template <typename T_, unsigned M_> class Vec, aie_dm_resource Resource, FifoDirection Direction, bool Restrict>
    requires (utils::is_one_of_v<T, mx4, mx6, mx9>)
class fifo_buffer_stream<Vec<T, M>, Resource, Direction, Restrict> property(keep_in_registers)
{
public:
    using         vector_type = Vec<T, M>;
    using           elem_type = typename vector_type::value_type;
    using native_pointer_type = typename vector_type::native_pointer_type;

    using          value_type = vector_type;

    static constexpr unsigned    native_ld_bits = 512;
    static constexpr unsigned native_load_elems = std::min(M, native_block_vector_length_v<T>);
    static constexpr unsigned         num_loads = vector_type::size() / native_load_elems;
    static constexpr bool             need_fill = vector_type::memory_bits() % native_ld_bits;
    static constexpr unsigned     ld_fill_ratio = native_ld_bits / (vector_type::memory_bits() % native_ld_bits);

    __aie_inline
    void seek(int inc)
    {
        // Struct sizes for sparse vectors don't match the actual storage in memory, so we compute it manually
        ptr_ = ::byte_incr(ptr_, inc * vector_type::memory_bytes());
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek(dims_2d_t &pattern)
    {
        // Struct sizes for sparse vectors don't match the actual storage in memory, so we compute it manually
        ptr_ = ::add_2d_byte(ptr_,
                             pattern.inc2 * vector_type::memory_bytes(),
                             pattern.num1,
                             pattern.count1,
                             pattern.inc1 * vector_type::memory_bytes());
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek(dims_3d_t &pattern)
    {
        // Struct sizes for sparse vectors don't match the actual storage in memory, so we compute it manually
        ptr_ = ::add_3d_byte(ptr_,
                             pattern.inc3 * vector_type::memory_bytes(),
                             pattern.num1,
                             pattern.count1,
                             pattern.inc1 * vector_type::memory_bytes(),
                             pattern.num2,
                             pattern.count2,
                             pattern.inc2 * vector_type::memory_bytes());
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek_byte(int inc)
    {
        ptr_ = ::byte_incr(ptr_, inc);
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek_byte(dims_2d_t &pattern)
    {
        ptr_ = ::add_2d_byte(ptr_, pattern);
        load_count_ = 0;
        state_.pos = 0;
    }

    __aie_inline
    void seek_byte(dims_3d_t &pattern)
    {
        ptr_ = ::add_3d_byte(ptr_, pattern);
        load_count_ = 0;
        state_.pos = 0;
    }

    value_type pop()
    {
        value_type ret;

        utils::unroll_times<num_loads>([&](unsigned idx) __aie_inline {
            // sparse / block vectors are larger than 512b, which is the native memory interface width
            if (!chess_manifest(load_count_ != 0))
                ::fifo_ld_fill(ptr_, state_);

            ret.template insert<native_load_elems>(idx, ::fifo_ld_pop(ptr_, state_));

            update_load_count();
        });

        return ret;
    }

    __aie_inline
    value_type pop_seek(int inc) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        utils::unroll_times<num_loads - 1>([&](unsigned idx) __aie_inline {
            // sparse / block vectors are larger than 512b, which is the native memory interface width
            if (!chess_manifest(load_count_ != 0))
                ::fifo_ld_fill(ptr_, state_);

            ret.template insert<native_load_elems>(idx, ::fifo_ld_pop(ptr_, state_));

            update_load_count();
        });

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            ::fifo_ld_fill(ptr_, state_);

        ret.template insert<native_load_elems>(num_loads - 1,
                                               ::fifo_ld_pop_1d_byte(ptr_, state_, inc * vector_type::memory_bytes()));

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek_byte(int inc) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        utils::unroll_times<num_loads - 1>([&](unsigned idx) __aie_inline {
            // sparse / block vectors are larger than 512b, which is the native memory interface width
            if (!chess_manifest(load_count_ != 0))
                ::fifo_ld_fill(ptr_, state_);

            ret.template insert<native_load_elems>(idx, ::fifo_ld_pop(ptr_, state_));

            update_load_count();
        });

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            ::fifo_ld_fill(ptr_, state_);

        ret.template insert<native_load_elems>(num_loads - 1,
                                               ::fifo_ld_pop_1d_byte(ptr_, state_, inc));

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek(dims_2d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        utils::unroll_times<num_loads - 1>([&](unsigned idx) __aie_inline {
            // sparse / block vectors are larger than 512b, which is the native memory interface width
            if (!chess_manifest(load_count_ != 0))
                ::fifo_ld_fill(ptr_, state_);

            ret.template insert<native_load_elems>(idx, ::fifo_ld_pop(ptr_, state_));

            update_load_count();
        });

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            ::fifo_ld_fill(ptr_, state_);

        ret.template insert<native_load_elems>(num_loads - 1,
                                               ::fifo_ld_pop_2d_byte(ptr_,
                                                                     state_,
                                                                     pattern.inc2 * vector_type::memory_bytes(),
                                                                     pattern.num1,
                                                                     pattern.count1,
                                                                     pattern.inc1 * vector_type::memory_bytes()));

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek_byte(dims_2d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        utils::unroll_times<num_loads - 1>([&](unsigned idx) __aie_inline {
            // sparse / block vectors are larger than 512b, which is the native memory interface width
            if (!chess_manifest(load_count_ != 0))
                ::fifo_ld_fill(ptr_, state_);

            ret.template insert<native_load_elems>(idx, ::fifo_ld_pop(ptr_, state_));

            update_load_count();
        });

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            ::fifo_ld_fill(ptr_, state_);

        ret.template insert<native_load_elems>(num_loads - 1,
                                               ::fifo_ld_pop_2d_byte(ptr_, state_, pattern.inc2, pattern.num1, pattern.count1, pattern.inc1));

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek(dims_3d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        utils::unroll_times<num_loads - 1>([&](unsigned idx) __aie_inline {
            // sparse / block vectors are larger than 512b, which is the native memory interface width
            if (!chess_manifest(load_count_ != 0))
                ::fifo_ld_fill(ptr_, state_);

            ret.template insert<native_load_elems>(idx, ::fifo_ld_pop(ptr_, state_));

            update_load_count();
        });

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            ::fifo_ld_fill(ptr_, state_);

        ret.template insert<native_load_elems>(num_loads - 1,
                                               ::fifo_ld_pop_3d_byte(ptr_,
                                                                     state_,
                                                                     pattern.inc3 * vector_type::memory_bytes(),
                                                                     pattern.num1,
                                                                     pattern.count1,
                                                                     pattern.inc1 * vector_type::memory_bytes(),
                                                                     pattern.num2,
                                                                     pattern.count2,
                                                                     pattern.inc2 * vector_type::memory_bytes()));

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    value_type pop_seek_byte(dims_3d_t &pattern) requires(Direction == FifoDirection::In)
    {
        value_type ret;

        utils::unroll_times<num_loads - 1>([&](unsigned idx) __aie_inline {
            // sparse / block vectors are larger than 512b, which is the native memory interface width
            if (!chess_manifest(load_count_ != 0))
                ::fifo_ld_fill(ptr_, state_);

            ret.template insert<native_load_elems>(idx, ::fifo_ld_pop(ptr_, state_));

            update_load_count();
        });

        // sparse / block vectors are larger than 512b, which is the native memory interface width
        if (!chess_manifest(load_count_ != 0))
            ::fifo_ld_fill(ptr_, state_);

        ret.template insert<native_load_elems>(num_loads - 1,
                                               ::fifo_ld_pop_3d_byte(ptr_, state_, pattern.inc3,
                                                                     pattern.num1, pattern.count1, pattern.inc1,
                                                                     pattern.num2, pattern.count2, pattern.inc2));

        load_count_ = 0;

        return ret;
    }

    __aie_inline
    void push(const value_type &v) requires(Direction == FifoDirection::Out)
    {
        push_common(v);
        ::fifo_st_flush(ptr_, state_);
    }

    __aie_inline
    void push_seek(const value_type &v, int inc) requires(Direction == FifoDirection::Out)
    {
        push_common(v);
        ::fifo_st_flush_1d_byte(ptr_, state_, inc * vector_type::memory_bytes());
    }

    __aie_inline
    void push_seek_byte(const value_type &v, int inc) requires(Direction == FifoDirection::Out)
    {
        push_common(v);
        ::fifo_st_flush_1d_byte(ptr_, state_, inc);
    }

    __aie_inline
    void push_seek(const value_type &v, dims_2d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        push_common(v);
        ::fifo_st_flush_2d_byte(ptr_,
                                state_,
                                pattern.inc2 * vector_type::memory_bytes(),
                                pattern.num1,
                                pattern.count1,
                                pattern.inc1 * vector_type::memory_bytes());
    }

    __aie_inline
    void push_seek_byte(const value_type &v, dims_2d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        push_common(v);
        ::fifo_st_flush_2d_byte(ptr_, state_, pattern.inc2, pattern.num1, pattern.count1, pattern.inc1);
    }

    __aie_inline
    void push_seek(const value_type &v, dims_3d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        push_common(v);
        ::fifo_st_flush_3d_byte(ptr_,
                                state_,
                                pattern.inc3 * vector_type::memory_bytes(),
                                pattern.num1,
                                pattern.count1,
                                pattern.inc1 * vector_type::memory_bytes(),
                                pattern.num2,
                                pattern.count2,
                                pattern.inc2 * vector_type::memory_bytes());
    }

    __aie_inline
    void push_seek_byte(const value_type &v, dims_3d_t &pattern) requires(Direction == FifoDirection::Out)
    {
        push_common(v);
        ::fifo_st_flush_3d_byte(ptr_, state_, pattern.inc3, pattern.num1, pattern.count1, pattern.inc1, pattern.num2, pattern.count2, pattern.inc2);
    }

protected:
    __aie_inline
    fifo_buffer_stream(const elem_type *ptr) requires(Direction == FifoDirection::In) :
        load_count_(0),
        ptr_((ptr_type)ptr)
    {
        state_.pos = 0;
    }

    __aie_inline
    fifo_buffer_stream(elem_type *ptr) requires(Direction == FifoDirection::Out) :
        ptr_((ptr_type)ptr)
    {
        state_.pos = 0;
    }

private:

    __aie_inline
    void push_common(const value_type &v) {
        utils::unroll_times<num_loads>([&](unsigned idx) __aie_inline {
            ::fifo_st_push(ptr_, v.template extract<native_load_elems>(idx), state_);
        });
    }

    __aie_inline
    void update_load_count() {
        if constexpr (need_fill) load_count_ = (load_count_ + 1) % ld_fill_ratio;
        else                     load_count_ = 1u;
    }

#if AIE_API_NATIVE
    using ptr_type = add_memory_bank_t<Resource, native_pointer_type> *;
#else
    using ptr_type = std::conditional_t<Restrict,
                                        add_memory_bank_t<Resource, native_pointer_type> * __restrict,
                                        add_memory_bank_t<Resource, native_pointer_type> *>;
#endif

    unsigned load_count_;
    ptr_type ptr_;
    fifo_state_t state_;
};
#endif

template <typename T, unsigned Elems, typename IterDescriptor, aie_dm_resource Resource>
    requires (arch::is(arch::Gen2))
class sliding_window_buffer_stream property(keep_in_registers)
{
public:
    using   elem_type = std::remove_const_t<aie_dm_resource_remove_t<T>>;
    using vector_type = vector<elem_type, Elems>;
    using  value_type = vector_type;

    using        iter_desc_t = std::decay_t<IterDescriptor>;
    using  iter_desc_storage = const iter_desc_t&;
    using iter_state_storage = iter_state<iter_desc_t>;

    __aie_inline
    constexpr sliding_window_buffer_stream(T *ptr, iter_desc_storage iter_desc) :
        ptr_((native_type*)ptr),
        iter_desc_(iter_desc),
        step_(inc_to_step(iter_desc.inc1 + 64, true)), // +64 to compensate for sliding_window_dim_* offset
        state_(chess_dont_care(fifo_state_t)) //CRVO-4372
    {
        RUNTIME_ASSERT(ptr != nullptr, "Input buffer stream cannot be created from nullptr");

        state_.pos = 0;
    }

    __aie_inline
    sliding_window_buffer_stream& operator>>(vector_type& res)
    {
        res = pop();
        return *this;
    }

    __aie_inline
    value_type pop()
    {
        if      constexpr (vector_type::bits() <= native_vector_type::bits()) {
            sparse_fifo_t chess_storage(lf1) temp = chess_copy(state_.fifo); state_.fifo = temp;  //CRVO-4372
            vector<elem_type, native_elems> ret = ::fifo_ld_popx(ptr_, state_, step_, 63);

            increment();

            return ret.template extract<Elems>(0);
        }
        else {
            //FIXME: This will need to be rolled back num_loads times...
            constexpr unsigned num_loads = vector_type::bits() / native_vector_type::bits();
            value_type ret;
            ret.insert(0, native_vector_type(::fifo_ld_popx(ptr_, state_, step_, 63)));

            utils::unroll_times<num_loads-1>([&](auto idx) __aie_inline {
                ret.insert(idx + 1, native_vector_type(::fifo_ld_pop(ptr_, state_)));
            });

            increment();

            return ret;
        }
    }

    constexpr bool operator==(const sliding_window_buffer_stream& rhs) { return ptr_ == rhs.ptr_; }
    constexpr bool operator!=(const sliding_window_buffer_stream& rhs) { return ptr_ != rhs.ptr_; }

private:
    static constexpr unsigned subbyte_elems = std::max(1u, 8 / type_bits_v<elem_type>);
    static constexpr unsigned native_elems  = native_vector_length_v<elem_type>;
    using native_vector_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<vector<elem_type, native_elems>, aie_dm_resource_get_v<T>>>;
    using native_type        = typename native_vector_type::native_type;

    static constexpr unsigned inc_to_step(unsigned inc, bool byte_input)
    {
        unsigned inc_bytes = byte_input ? inc : inc * sizeof(T) / subbyte_elems;

        REQUIRES(inc_bytes ==  8 || inc_bytes ==  16 || inc_bytes ==  32);

        return utils::ffs(inc_bytes >> 3);
    }

    __aie_inline
    constexpr void increment(void)
    {
        if      constexpr (std::is_same_v<sliding_window_dim_3d, iter_desc_t>) {
            ptr_ = ::add_3d_byte(ptr_, iter_desc_.inc3,
                                       iter_desc_.num1, iter_state_.state_.c1, iter_desc_.inc1,
                                       iter_desc_.num2, iter_state_.state_.c2, iter_desc_.inc2);
        }
        else if constexpr (std::is_same_v<sliding_window_dim_2d, iter_desc_t>) {
            ptr_ = ::add_2d_byte(ptr_, iter_desc_.inc2,
                                       iter_desc_.num1, iter_state_.state_.c1, iter_desc_.inc1);
        }
        else {
            ptr_ = ::byte_incr(ptr_, iter_desc_.inc1);
        }
    }

    native_type *ptr_;
    iter_desc_storage iter_desc_;
    iter_state_storage iter_state_;
    unsigned step_;
    fifo_state_t state_;
};

} // namespace detail

} // namespace aie

#endif
