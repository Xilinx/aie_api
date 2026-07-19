// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

/**
 * @file
 * @brief AIE API main entry point.
 */

#pragma once

#ifndef __AIE_API_AIE__HPP__
#define __AIE_API_AIE__HPP__

#if !__AIENGINE__

// Forward declaration of common AI API types that may be visible at the graph level
#include "aie_declaration.hpp"

#else // __AIENGINE__

#if !defined(__AIE_ARCH__)
#error "'__AIE_ARCH__' macro is required to compile AIE API kernels"
#endif

// AIE API currently requires C++20
#define AIE_API_CXX_VERSION 202002L

#if __cplusplus < AIE_API_CXX_VERSION
#error "C++20 or greater is required to compile AIE API kernels"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>

#include "detail/abs.hpp"
#include "detail/abs_square.hpp"
#include "detail/accumulate.hpp"
#include "detail/add.hpp"
#include "detail/bit.hpp"
#include "detail/blend.hpp"
#include "detail/cfr.hpp"
#include "detail/compare.hpp"
#include "detail/conj.hpp"
#include "detail/elementary.hpp"
#include "detail/fft.hpp"
#include "detail/filter.hpp"
#include "detail/interleave.hpp"
#include "detail/ld_st.hpp"
#include "detail/linear_approx.hpp"
#include "detail/lut.hpp"
#include "detail/max_min.hpp"
#include "detail/mmul.hpp"
#include "detail/mul.hpp"
#include "detail/neg.hpp"
#include "detail/parallel_lookup.hpp"
#include "detail/reverse.hpp"
#include "detail/shift.hpp"
#include "detail/shuffle.hpp"
#include "detail/square.hpp"
#include "detail/transpose.hpp"
#include "detail/vector_accum_cast.hpp"

#include "accum.hpp"
#include "aie_doc.hpp"
#include "aie_types.hpp"
#if AIE_API_ML_VERSION >= 210
#include "block_vector.hpp"
#endif
#include "concepts.hpp"
#include "expr.hpp"
#include "fft.hpp"
#include "iterator.hpp"
#include "mask.hpp"
#include "sliding_mul.hpp"
#include "sparse_vector.hpp"
#include "tile.hpp"
#include "types.hpp"
#include "vector.hpp"
#include "vector_elem_ref.hpp"

namespace aie {

namespace Utils = detail::utils;

template <typename T>
struct operand_base_type
{
    using type = aie_dm_resource_remove_t<T>;
};

template <typename T, unsigned Elems>
struct operand_base_type<vector_elem_ref<T, Elems>>
{
    using type = T;
};

template <typename T, unsigned Elems>
struct operand_base_type<vector_elem_const_ref<T, Elems>>
{
    using type = T;
};

template <typename Parent, Operation Op>
struct operand_base_type<unary_op<Parent, Op>>
{
    using type = typename unary_op<Parent, Op>::value_type;
};

template <typename Parent1, typename Parent2, Operation Op>
struct operand_base_type<binary_op<Parent1, Parent2, Op>>
{
    using type = typename binary_op<Parent1, Parent2, Op>::value_type;
};

template <typename T1, typename T2>
struct is_valid_elem_op
{
    static constexpr bool value = std::is_same_v<operand_base_type_t<T1>,
                                                 operand_base_type_t<T2>>;
};

template <typename T1, typename T2>
static constexpr bool is_valid_elem_op_v = is_valid_elem_op<T1, T2>::value;

#if AIE_API_ML_VERSION >= 210
template <unsigned M_Elems, unsigned K_Elems, unsigned N_Elems, ElemBaseOrBlockType TypeA, ElemBaseOrBlockType TypeB, AccumElemBaseType AccumTag, accum_ownership Ownership>
struct mmul;
#else
template <unsigned M_Elems, unsigned K_Elems, unsigned N_Elems, ElemBaseType TypeA, ElemBaseType TypeB, AccumElemBaseType AccumTag, accum_ownership Ownership>
struct mmul;
#endif

//TODO: Try to refactor this to be within detail when we re-organize the interfaces
template <typename T>
struct is_mmul
{
    static constexpr bool value = false;
};

template  <unsigned M, unsigned K, unsigned N, ElemBaseType TypeA, ElemBaseType TypeB, AccumElemBaseType AccumTag, accum_ownership Ownership>
struct is_mmul<mmul<M, K, N, TypeA, TypeB, AccumTag, Ownership>>
{
    static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_mmul_v = is_mmul<T>::value;

template <typename T>
concept Mmul = is_mmul_v<T>;

template <typename T>
static constexpr bool is_mmul_or_op_v =  is_mmul_v<aie_dm_resource_remove_t<T>> ||
                                         is_op_v<T>;

template  <unsigned M, unsigned K, unsigned N, ElemBaseType TypeA, ElemBaseType TypeB, AccumElemBaseType AccumTag, accum_ownership Ownership>
struct op_value_type_helper<mmul<M, K, N, TypeA, TypeB, AccumTag, Ownership>>
{
    using type = typename mmul<M, K, N, TypeA, TypeB, AccumTag, Ownership>::value_type;
};

template <typename T>
struct result_type_helper;

template <Accum T>
struct result_type_helper<T>
{
    using type = T;
};

template <Vector T>
struct result_type_helper<T>
{
    using type = T;
};

template <Elem T>
struct result_type_helper<T>
{
    using type = operand_base_type_t<T>;
};

template <typename T> requires is_op_v<T>
struct result_type_helper<T>
{
    using type = typename T::result_type;
};

/*
 * If the input type is Accum, Vector or Elem, returns the same type. If the input type is an operation, returns the
 * resulting type of such operation.
 */
template <typename T>
using result_type_t = typename result_type_helper<T>::type;


template <typename T>
concept MmulOrOp = is_mmul_or_op_v<T>;

template <typename T, typename T2, typename... Other>
struct is_valid_size
{
    static constexpr bool value = (is_valid_size<T, T2>::are_same() && is_valid_size<T, Other...>::value);
};

template <typename T, typename T2>
struct is_valid_size<T, T2>
{
    static constexpr bool are_same()
    {
        using detail::is_vector_or_op_v;
        using detail::is_accum_or_op_v;
        using detail::is_elem_or_op_v;
        if constexpr ((is_vector_or_op_v<T>  || is_accum_or_op_v<T>) &&
                      (is_vector_or_op_v<T2> || is_accum_or_op_v<T2>)) {
            return T::size() == T2::size();
        }
        else if constexpr ((is_vector_or_op_v<T>  && is_elem_or_op_v<T2>) ||
                           (is_vector_or_op_v<T2> && is_elem_or_op_v<T>)) {
            return true;
        }
        else if constexpr ((is_accum_or_op_v<T>   && is_elem_or_op_v<T2>) ||
                           (is_accum_or_op_v<T2>  && is_elem_or_op_v<T>)) {
            return true;
        }
        else {
            return false;
        }
    }

    static constexpr bool value = are_same();
};

template <typename T, typename T2, typename... Other>
static constexpr bool is_valid_size_v = is_valid_size<T, T2, Other...>::value;

template <Vector Vec1, Vector Vec2>
struct is_same_vector {
    static constexpr bool value = false;
};

template <typename T, unsigned N>
struct is_same_vector<vector<T, N>, vector<T, N>>
{
    static constexpr bool value = true;
};

template <typename T, unsigned N, aie_dm_resource Resource, Vector Vec>
struct is_same_vector<vector_ref<T, N, Resource>, Vec>
            : is_same_vector<typename vector_ref<T, N, Resource>::vector_type, Vec> {};

template <typename T, unsigned N, aie_dm_resource Resource, Vector Vec>
struct is_same_vector<unaligned_vector_ref<T, N, Resource>, Vec>
            : is_same_vector<typename unaligned_vector_ref<T, N, Resource>::vector_type, Vec> {};

template <typename T1, unsigned N1, typename T2, unsigned N2, aie_dm_resource Resource>
struct is_same_vector<vector<T1, N1>, vector_ref<T2, N2, Resource>>
            : is_same_vector<vector<T1, N1>, typename vector_ref<T2, N2, Resource>::vector_type> {};

template <typename T1, unsigned N1, typename T2, unsigned N2, aie_dm_resource Resource>
struct is_same_vector<vector<T1, N1>, unaligned_vector_ref<T2, N2, Resource>>
            : is_same_vector<vector<T1, N1>, typename unaligned_vector_ref<T2, N2, Resource>::vector_type> {};

template <Vector Vec1, Vector Vec2>
static constexpr bool is_same_vector_v = is_same_vector<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec2>>::value;

template <Accum Acc1, Accum Acc2>
static constexpr bool is_same_accum_v = std::is_same_v<aie_dm_resource_remove_t<Acc1>, aie_dm_resource_remove_t<Acc2>>;

/**
 * @ingroup group_basic_types_ops
 *
 * Returns an addition operation modifier for the given accumulator.
 *
 * @param acc Accumulator to which the operation is performed.
 */
template <Accum Acc>
__aie_inline
constexpr unary_op<Acc, Operation::Acc_Add> op_add(const Acc &acc)
{
    return {acc};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns a subtraction operation modifier for the given accumulator.
 *
 * @param acc Accumulator to which the operation is performed.
 */
template <Accum Acc>
__aie_inline
constexpr unary_op<Acc, Operation::Acc_Sub> op_sub(const Acc &acc)
{
    return {acc};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns an empty operation modifier for the given vector, vector element or accumulator.
 *
 * @param e Vector, vector element, or accumulator to which the operation is performed.
 *          The type must meet @ref aie::Vector, @ref aie::SparseVector, @ref aie::Elem or @ref aie::Accum.
 */
template <typename T> requires(Vector<T>
#if AIE_API_ML_VERSION >= 210
                               || BlockVector<T>
#endif
                               || SparseVector<T>
                               || Elem<T>
                               || Accum<T>)
__aie_inline
constexpr unary_op<T, Operation::None> op_none(const T &e)
{
    return {e};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns an empty operation modifier for the given tensor buffer stream
 *
 * @param e A reference to a @ref aie::TensorBufferStream
 */
template <TensorBufferStream T>
__aie_inline
unary_op<T, Operation::None> op_none(T &e)
{
    return {e};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns an empty operation modifier for the given input argument when it is not an operation already
 * and when it is it returns the input itself
 *
 * @param arg The input argument
 */
template <typename T>
__aie_inline
auto op_or_none(T &&arg) {
    if constexpr (!is_op_v<std::remove_cvref_t<T>>)
        return op_none(std::forward<T>(arg));
    else
        return std::forward<T>(arg);
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns an absolute operation modifier for the given vector or element. On some architecture versions, this operation
 * can be collapsed with a subsequent multiplication.
 *
 * @param e Vector or element to which the operation is performed. The type must meet @ref aie::RealVector
 *          or @ref aie::RealElem.
 */
template <typename T> requires(RealVector<T> || RealElem<T> || is_real_op_v<T>)
__aie_inline
constexpr unary_op<T, Operation::Abs> op_abs(const T &e)
{
    return {e};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns a conjugate operation modifier for the given complex vector or element. On some architecture versions, this
 * operation can be collapsed with a subsequent multiplication.
 *
 * @param e Vector or element to which the operation is performed. The type must meet
 *          @ref aie::ComplexVector or @ref aie::ComplexElem.
 */
template <typename T> requires(ComplexVector<T> || ComplexElem<T> || is_complex_op_v<T>)
__aie_inline
constexpr unary_op<T, Operation::Conj> op_conj(const T &e)
{
    return {e};
}

#if AIE_API_ML_VERSION >= 210

/**
 * @ingroup group_basic_types_ops
 *
 * Returns a transpose operation modifier for the given vector. On some architecture versions, this
 * operation can be collapsed with a subsequent multiplication.
 *
 * Currently this is only supported in the case of the B matrix for block floating point matrix multiplication.
 *
 * @param v Block vector to which the operation is performed. The type must meet @ref aie::BlockVector.
 */
template <BlockVector Vec>
__aie_inline
constexpr unary_op<Vec, Operation::Transpose> op_transpose(const Vec &v)
{
    return {v};
}

#endif

/**
 * @ingroup group_basic_types_ops
 *
 * Returns dynamically signed/unsigned operation modifier for the vector or element, based on the boolean parameter. On some architecture versions, this
 * operation can be collapsed with a subsequent operation.
 *
 * @param e         Vector, element or accumulator passed to the operation. The type must meet
 *                  @ref aie::Vector, @ref aie::SparseVector, @ref aie::Elem, @ref aie::Accum or be another operation,
 *                  using an integer type.
 * @param is_signed Controls if the first paramer is interpreted as signed, or not.
 */
template <typename T> requires((Vector<T> || SparseVector<T> || Elem<T> || Accum<T> || is_op_v<T>) &&
                                       (!detail::is_complex_v<typename T::value_type>)  &&
                                       (!detail::is_floating_point_v<typename T::value_type>))
__aie_inline
constexpr binary_op<T, bool, Operation::Sign> op_sign(const T &e, bool is_signed)
{
    return {e, is_signed};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns dynamically signed/unsigned operation modifier for the given tensor buffer stream
 *
 * @param e A reference to a @ref aie::TensorBufferStream
 * @param is_signed Controls if the first paramer is interpreted as signed, or not.
 */
template <TensorBufferStream T>
__aie_inline
binary_op<T, bool, Operation::Sign> op_sign(T &e, bool is_signed)
{
    return {e, is_signed};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Zeroes out the given accumulator or not depending on the zero parameter. On some architectures this operation can be collapsed with
 * a subsequent operation.
 *
 * @param acc     Accumulator to which the operation is performed.
 * @param to_zero Controls if the passed accumulator is zeroed or not.
 */
template <Accum Acc>
__aie_inline
constexpr binary_op<Acc, bool, Operation::Zero> op_zero(const Acc &acc, bool to_zero)
{
    return {acc, to_zero};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Conditionally negates the operand based on the to_negate parameter.
 * On some architectures this operation can be collapsed with a subsequent operation.
 *
 * @param e         Vector, element or accumulator passed to the operation. The type must meet
 *                  @ref aie::Vector, @ref aie::Elem, @ref aie::Accum or be another operation,
 *                  using an integer type.
 * @param to_negate Controls if the first paramer is negated, or not.
 */
template <typename T> requires(Vector<T> || Elem<T> || Accum<T> || is_op_v<T>)
__aie_inline
constexpr binary_op<T, bool, Operation::Neg> op_neg(const T &e, bool to_negate)
{
    return {e, to_negate};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns a minimum operation modifier for the given pair of vectors or elements. On some architecture versions, this
 * operation can be collapsed with a subsequent multiplication.
 *
 * @param a First vector or element passed to the operation. The type must meet @ref aie::Vector or
 *          @ref aie::Elem.
 * @param b Second vector or element passed to the operation. The type must be the same as for a.
 */
//TODO: Clarify what types we want here, and change doc. It says element/vector but the type before was Vector?
template <VectorOrOp Vec1, VectorOrOp Vec2>
__aie_inline
constexpr binary_op<Vec1, Vec2, Operation::Min> op_min(const Vec1 &a, const Vec2 &b)
{
    return {a, b};
}

/**
 * @ingroup group_basic_types_ops
 *
 * Returns a maximum operation modifier for the given pair of vectors or elements. On some architecture versions, this
 * operation can be collapsed with a subsequent multiplication.
 *
 * @param a First vector or element passed to the operation. The type must meet @ref aie::Vector or
 *          @ref aie::Elem.
 * @param b Second vector or element passed to the operation. The type must be the same as for a.
 */
//TODO: Clarify what types we want here, and change doc. It says element/vector but the type before was Vector?
template <VectorOrOp Vec1, VectorOrOp Vec2>
__aie_inline
constexpr binary_op<Vec1, Vec2, Operation::Max> op_max(const Vec1 &a, const Vec2 &b)
{
    return {a, b};
}

/**
 * @ingroup group_memory
 *
 * Specifies the minimum alignment requirement for vector loads and stores.
 *
 * Example:
 * @code
 * alignas(aie::vector_decl_align) int32 data[8];
 *
 * aie::vector<int32, 8> v;
 * v = aie::load_v<8>(data);
 * v.store(data);
 * @endcode
 *
 * @sa @ref aie::load_v
 * @sa @ref aie::vector<T, Elems>::load
 */
static constexpr unsigned vector_decl_align   = detail::vector_decl_align;

/**
 * @ingroup group_memory
 *
 * Specifies the minimum alignment requirement in bytes for vector loads and stores.
 *
 * @sa @ref aie::vector_decl_align
 * @sa @ref aie::load_v
 * @sa @ref aie::vector<T, Elems>::load
 */
template <DecoratedElemBaseType T, unsigned Elems>
static constexpr unsigned vector_ldst_align_v = detail::vector_ldst_align_v<T, Elems>;

/**
 * @ingroup group_memory
 *
 * Specifies the minimum alignment requirement in elements for vector loads and stores.
 *
 * @sa @ref aie::vector_decl_align
 * @sa @ref aie::load_v
 * @sa @ref aie::vector<T, Elems>::load
 */
template <DecoratedElemBaseType T, unsigned Elems>
static constexpr unsigned vector_ldst_align_elems_v = (detail::vector_ldst_align_v<T, Elems> * 8) / detail::type_bits_v<T>;

/**
 * @ingroup group_memory
 *
 * Load a vector of Elems size whose elements have type T. The pointer is assumed to meet the alignment requirements
 * for a vector load of this size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = ptr[i];
 * @endcode
 *
 * @tparam Elems Size of the vector to be read from memory
 *
 * @param  ptr   Address data is read from
 */
template <unsigned Elems, aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T>
__aie_inline
auto load_v(const T *ptr) -> vector<aie_dm_resource_remove_t<T>, Elems>
{
#if !AIE_API_DISABLE_ALIGNMENT_ASSERTIONS
    RUNTIME_ASSERT_NO_ASSUME(detail::check_vector_alignment<Elems>(ptr), "Insufficient alignment");
#endif

    return detail::load_vector<Elems, Resource>(ptr);
}

/**
 * @ingroup group_memory
 *
 * Load a vector whose elements have type T. The size is automatically chosen with the optimal size for the current
 * architecture. The pointer is assumed to meet the alignment requirements for a vector load of this size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = ptr[i];
 * @endcode
 *
 * @param ptr Address data is read from
 */
template <aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T>
__aie_inline
auto load_v(const T *ptr) -> vector<aie_dm_resource_remove_t<T>, native_vector_length_v<T>>
{
#if !AIE_API_DISABLE_ALIGNMENT_ASSERTIONS
    RUNTIME_ASSERT_NO_ASSUME(detail::check_vector_alignment<native_vector_length_v<T>>(ptr), "Insufficient alignment");
#endif

    return load_v<native_vector_length_v<T>, Resource>(ptr);
}

/**
 * @ingroup group_memory
 *
 * Load a vector of Elems size whose elements have type T. The pointer is assumed to be aligned to T.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = ptr[i];
 * @endcode
 *
 * @tparam Elems Size of the vector to be read from memory
 *
 * @param ptr Address data is read from
 * @param aligned_elems Number of elements the pointer is aligned to. If unspecified, the default value is 1.
 */
template <unsigned Elems, aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T>
__aie_inline
auto load_unaligned_v(const T *ptr, unsigned aligned_elems = 1) -> vector<aie_dm_resource_remove_t<T>, Elems>
{
    RUNTIME_ASSERT_NO_ASSUME(detail::check_alignment(ptr, aligned_elems), "Insufficient alignment");

    REQUIRES_MSG(detail::utils::is_powerof2(aligned_elems), "aligned_elems must be a power of two");

    if (chess_manifest(detail::check_vector_alignment<Elems>(ptr)))
        return load_v<Elems, Resource>(ptr);

    return detail::load_unaligned_vector<Elems, Resource>(ptr, aligned_elems);
}

/**
 * @ingroup group_memory
 *
 * Load a vector whose elements have type T. The size is automatically chosen with the optimal size for the current
 * architecture. The pointer is assumed to be aligned to T.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = ptr[i];
 * @endcode
 *
 * @param ptr Address data is read from
 * @param aligned_elems Number of elements the pointer is aligned to. If unspecified, the default value is 1.
 */
template <aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T>
__aie_inline
auto load_unaligned_v(const T *ptr, unsigned aligned_elems = 1) -> vector<aie_dm_resource_remove_t<T>, native_vector_length_v<T>>
{
    RUNTIME_ASSERT_NO_ASSUME(detail::check_alignment(ptr, aligned_elems), "Insufficient alignment");
    REQUIRES_MSG(detail::utils::is_powerof2(aligned_elems), "aligned_elems must be a power of two");

    return load_unaligned_v<native_vector_length_v<T>, Resource, T>(ptr, aligned_elems);
}

/**
 * @ingroup group_memory
 *
 * Load a vector of Elems size whose elements have type T. The pointer will be aligned to bytes.
 *
 * @tparam Elems Size of the vector to be read from memory
 *
 * @param ptr   Address data is read from
 * @param bytes Numbers of bytes to which the input pointer is aligned. Must be a power of two.
 */
template <unsigned Elems, aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T>
__aie_inline
auto load_floor_bytes_v(const T *ptr, size_t bytes) -> vector<aie_dm_resource_remove_t<T>, Elems>
{
    REQUIRES_MSG(detail::utils::is_powerof2(bytes), "bytes must be a power of two");

    const T *aligned_ptr;

#if AIE_API_NATIVE == 0
    // In the following cases, the HW truncates the value automatically, so no need to compute it here
    constexpr unsigned native_truncation = detail::vector_ldst_align_v<T, Elems>;

    if (chess_manifest(bytes == native_truncation))
        aligned_ptr = ptr;
    else
#endif
    aligned_ptr = (T *)(uintptr_t(ptr) & ~(bytes - 1));

#if !AIE_API_DISABLE_ALIGNMENT_ASSERTIONS
    RUNTIME_ASSERT_NO_ASSUME(detail::check_vector_alignment<Elems>(aligned_ptr), "Insufficient alignment");
#endif

    return detail::load_vector<Elems, Resource>(aligned_ptr);
}

/**
 * @ingroup group_memory
 *
 * Load a vector of Elems size whose elements have type T. The pointer will be aligned to n * sizeof(T).
 *
 * @tparam Elems Size of the vector to be read from memory
 *
 * @param ptr Address data is read from
 * @param n   Numbers of elements of type T to which the input pointer is aligned. Must be a power of two.
 */
template <unsigned Elems, aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T>
__aie_inline
auto load_floor_v(const T *ptr, unsigned n = Elems) -> vector<aie_dm_resource_remove_t<T>, Elems>
{
    REQUIRES_MSG(detail::utils::is_powerof2(n), "n must be a power of two");

    return load_floor_bytes_v<Elems, Resource>(ptr, n * sizeof(T));
}

/**
 * @ingroup group_memory
 *
 * Store a vector of Elems size whose elements have type T. The pointer is assumed to meet the alignment requirements
 * for a vector store of this size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     ptr[i] = v[i];
 * @endcode
 *
 * @param ptr Address data is written to
 * @param v Vector to be written to memory
 */
template <aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T1, ElemBaseType T2, unsigned Elems>
requires(std::is_same_v<aie_dm_resource_remove_t<T1>, aie_dm_resource_remove_t<T2>>)
__aie_inline
T1 *store_v(T1 *ptr, const vector<T2, Elems> &v)
{
#if !AIE_API_DISABLE_ALIGNMENT_ASSERTIONS
    RUNTIME_ASSERT_NO_ASSUME(detail::check_vector_alignment<Elems>(ptr), "Insufficient alignment");
#endif

    return detail::store_vector<Elems, Resource>(ptr, v);
}

/**
 * @ingroup group_memory
 *
 * Store a vector of Elems size whose elements have type T. The pointer is assumed to be aligned to T.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     ptr[i] = v[i];
 * @endcode
 *
 * @param ptr Address data is written to
 * @param v Vector to be written to memory
 * @param aligned_elems Number of elements the pointer is aligned to. If unspecified, the default value is 1.
 */
template <unsigned Elems, aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T1, ElemBaseType T2>
requires(std::is_same_v<T2, aie_dm_resource_remove_t<T1>>)
__aie_inline
T1 *store_unaligned_v(T1 *ptr, const vector<T2, Elems> &v, unsigned aligned_elems = 1)
{
    RUNTIME_ASSERT_NO_ASSUME(detail::check_alignment(ptr, aligned_elems), "Insufficient alignment");
    REQUIRES_MSG(detail::utils::is_powerof2(aligned_elems), "aligned_elems must be a power of two");

    if (chess_manifest(detail::check_vector_alignment<Elems>(ptr)))
        return store_v<Resource>(ptr, v);

    return detail::store_unaligned_vector<Elems, Resource>(ptr, v, aligned_elems);
}

/**
 * @ingroup group_memory
 *
 * Store a vector of Elems size whose elements have type T. The pointer will be aligned to bytes.
 *
 * @param ptr   Address data is written to
 * @param v     Vector to be written to memory
 * @param bytes Numbers of bytes to which the input pointer should be aligned. Must be a power of two.
 */
template <aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T1, ElemBaseType T2, unsigned Elems>
requires(std::is_same_v<aie_dm_resource_remove_t<T1>, aie_dm_resource_remove_t<T2>>)
__aie_inline
T1 *store_floor_bytes_v(T1 *ptr, const vector<T2, Elems> &v, size_t bytes)
{
    REQUIRES_MSG(detail::utils::is_powerof2(bytes), "bytes must be a power of two");

    T2 *aligned_ptr;

#if AIE_API_NATIVE == 0
    // In the following cases, the HW truncates the value automatically, so no need to compute it here
    constexpr unsigned native_truncation = detail::vector_ldst_align_v<T2, Elems>;

    if (chess_manifest(bytes == native_truncation))
        aligned_ptr = ptr;
    else
#endif
    aligned_ptr = (T2 *)(uintptr_t(ptr) & ~(bytes - 1));

#if !AIE_API_DISABLE_ALIGNMENT_ASSERTIONS
    RUNTIME_ASSERT_NO_ASSUME(detail::check_vector_alignment<Elems>(aligned_ptr), "Insufficient alignment");
#endif

    return detail::store_vector<Elems, Resource>(aligned_ptr, v);
}

/**
 * @ingroup group_memory
 *
 * Store a vector of Elems size whose elements have type T. The pointer will be aligned to n * sizeof(T).
 *
 * @param ptr Address data is written to
 * @param v   Vector to be written to memory
 * @param n   Numbers of elements of type T to which the input pointer should be aligned. Must be a power of two.
 */
template <aie_dm_resource Resource = aie_dm_resource::none, DecoratedElemBaseType T1, ElemBaseType T2, unsigned Elems>
requires(std::is_same_v<aie_dm_resource_remove_t<T1>, aie_dm_resource_remove_t<T2>>)
__aie_inline
T1 *store_floor_v(T1 *ptr, const vector<T2, Elems> &v, unsigned n = Elems)
{
    REQUIRES_MSG(detail::utils::is_powerof2(n), "n must be a power of two");

    return store_floor_bytes_v<Resource>(ptr, v, n * sizeof(T2));
}

/**
 * @ingroup group_basic_types_conversion
 *
 * Reinterpret a vector using a different element type. The returned vector has the same size in bits as the input
 * vector.
 *
 * @tparam DstT Element type for the returned vector.
 *
 * @param v Input vector.
 */
template <ElemBaseType DstT, Vector Vec>
__aie_inline
auto vector_cast(const Vec &v)
{
    return detail::vector_cast<DstT>(v);
}

/**
 * @ingroup group_basic_types_conversion
 *
 * Reinterpret a vector as an accumulator. The returned accumulator has the same size in bits as the input
 * vector.
 *
 * @tparam DstTag Accumulator tag for the returned accumulator.
 *
 * @param v Input vector.
 */
template <AccumElemBaseType DstTag, Vector Vec>
    requires (arch::is(arch::Gen2))
auto vector_cast(const Vec &v)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::vector_to_accum_cast<DstTag, T, Elems>::run(v);
}

/**
 * @ingroup group_basic_types_conversion
 *
 * Reinterpret an accumulator as a vector. The returned vector has the same size in bits as the input
 * accumulator.
 *
 * @tparam DstT Element type for the returned vector.
 *
 * @param acc Input accumulator.
 */
template <ElemBaseType DstT, Accum Acc>
    requires (arch::is(arch::Gen2))
auto vector_cast(const Acc &acc)
{
    return detail::accum_to_vector_cast<DstT, typename Acc::value_type, Acc::size()>::run(acc);
}

/**
 * @ingroup group_basic_types_conversion
 *
 * Returns a conversion of the passed vector to the next larger integer datatype, with the same number of elements.
 *
 * @param v Input vector. The type must meet @ref aie::VectorOrOp
 */
template <VectorOrOp Vec, typename T2 = Utils::get_next_integer_type_t<typename Vec::value_type>>
    requires(!detail::is_floating_point_v<Vec>)
__aie_inline
auto unpack(const Vec &v) -> vector<T2, Vec::size()>
{
    if constexpr (!is_op_v<Vec>)
        return unpack(op_none(v));
    else if constexpr (Vec::is_operation_not(Operation::Sign))
        return unpack(v());
    else {
        if constexpr (Vec::is_operation_none())
            return v.parent1().unpack();
        else
            return v.parent1().unpack_sign(v.parent2());
    }
}

/**
 * @ingroup group_basic_types_conversion
 *
 * Returns a conversion of the passed vector to the next smaller integer datatype, with the same number of elements.
 *
 * @param v Input vector. The type must meet @ref aie::VectorOrOp
 *
 * @tparam T2 Element type for the returned vector. Can only select between int8/uint8 on AIE1.
 */
template <VectorOrOp Vec, typename T2 = Utils::get_prev_integer_type_t<typename Vec::value_type>>
    requires(!detail::is_floating_point_v<Vec>)
__aie_inline
auto pack(const Vec &v) -> vector<T2, Vec::size()>
{
    if constexpr (!is_op_v<Vec>)
        return pack(op_none(v));
    else if constexpr (Vec::is_operation_not(Operation::Sign))
        return pack(v());
    else {
        if constexpr (Vec::is_operation_none())
            return v.parent1().pack();
        else
            return v.parent1().template pack_sign<T2>(v.parent2());
    }
}

/**
 * @ingroup group_basic_types_conversion
 *
 * Returns the values of the passed accumulator in a vector of the requested type. The values can be shifted down before rounding and saturation are applied
 * (does not apply to floating point accumulators).
 *
 * \note On AIE-ML/XDNA1 shift values of -4, -3, and -2 are unsafe, as they will only produce correct result if truncation
 * is selected or saturation against 0 is required.
 *
 * @param acc   Input accumulator.  The type must meet @ref aie::AccumOrOp or @ref aie::MmulOrOp.
 * @param shift Optional downshift in bits to be applied to output data. This parameter is ignored for floating-point types.
 *
 * @tparam TR Element type for the returned vector.
 */
//TODO: Add documentation to Basic Types for which conversions to/from_vector supports
template <typename TR, typename T> requires(AccumOrOp<T> || MmulOrOp<T>)
__aie_inline
vector<TR, T::size()> to_vector(const T &acc, int shift = 0)
{
    if constexpr (!is_op_v<T>)
        return to_vector<TR>(op_none(acc), shift);
    else if constexpr (T::is_operation_not(Operation::Sign))
        return to_vector<TR>(acc(), shift);
    else {
        if constexpr (T::is_operation_none())
            return acc.parent1().template to_vector<TR>(shift);
        else
            return acc.parent1().template to_vector_sign<TR>(acc.parent2(), shift);
    }
}

#if AIE_API_ML_VERSION >= 210

/**
 * @ingroup group_basic_types_conversion
 *
 * Returns the values of the passed accumulator in a block_vector of the requested type.
 *
 * @param acc Input accumulator. The type must meet @ref aie::AccumOrOp or @ref aie::MmulOrOp.
 *
 * @tparam TR Block element type for the returned vector.
 */
template <BlockType TR, typename T> requires(AccumOrOp<T> || MmulOrOp<T>)
__aie_inline
block_vector<TR, T::size()> to_vector(const T &acc)
{
    if constexpr (!is_op_v<T>)
        return acc.template to_vector<TR>();
    else
        return to_vector<TR>(acc());
}

#endif

/**
 * @ingroup group_basic_types_conversion
 *
 * Returns the values of the passed vector in an accumulator of the requested type after applying the requested upshift
 * operation.
 *
 * @param v     Input vector. The type must meet @ref aie::VectorOrOp.
 * @param shift Optional upshift in bits to be applied to output accumulator. This parameter is ignored for floating-point types.
 *
 * @tparam AccumTag Accumulator tag that specifies returned accumulator type. The class must be compatible with
 *                  the vector type (real/complex).
 */
template <AccumElemBaseType AccumTag, VectorOrOp Vec>
__aie_inline
accum<AccumTag, Vec::size()> from_vector(const Vec &v, int shift = 0)
{
    if constexpr (!is_op_v<Vec>)
        return from_vector<AccumTag>(op_none(v), shift);
    else if constexpr (Vec::is_operation_not(Operation::Sign))
        return from_vector<AccumTag>(v(), shift);
    else {
        accum<AccumTag, Vec::size()> tmp;

        if constexpr (Vec::is_operation_none())
            tmp.from_vector(v.parent1(), shift);
        else
            tmp.from_vector_sign(v.parent1(), v.parent2(), shift);

        return tmp;
    }
}

/**
 * @ingroup group_init
 *
 * Concatenate the contents of all input accumulators into a larger accumulator. All input accumulators must have the
 * same type and size.
 *
 * @param acc First input accumulator.
 * @param accums Rest of input accumulators.
 */
template <Accum Acc, Accum... Accums> requires((is_same_accum_v<Acc, Accums> && ...))
__aie_inline
auto concat(const Acc &acc, const Accums & ...accums) -> accum<typename Acc::value_type, Acc::size() * (1 + sizeof...(Accums))>
{
    return detail::concat_accum(acc, accums...);
}

/**
 * @ingroup group_init
 *
 * Concatenate the contents of all input vectors into a larger vector. All input vectors must have the same type and
 * size.
 *
 * @param v First input vector.
 * @param vectors Rest of input vectors.
 */
template <Vector Vec, Vector... Vectors> requires((is_same_vector_v<Vec, Vectors> && ...))
__aie_inline
auto concat(const Vec &v, const Vectors & ...vectors) -> vector<typename Vec::value_type, Vec::size() * (1 + sizeof...(Vectors))>
{
    return detail::concat_vector(v, vectors...);
}

/**
 * Accepts any type implementing tuple interface.
 *
 * Examples: std::array, std::pair, std::tuple.
 */
template <typename T>
concept TupleLike = requires (T t) {
    typename std::tuple_size<std::decay_t<T>>;
    typename std::tuple_element<0, std::decay_t<T>>;
    { std::get<0>(t) };
};

/**
 * \brief Concatenate the contents of all input vectors/accumulators wrapped in
 * a @ref aie::TupleLike object.
 *
 * The result is a larger vector/accumulator containing all the input elements
 * combined.
 *
 * All tuple members must have the exactly same type and size.
 *
 * This overload is useful when joining the members of tuples returned by
 * interleave operations:
 *
 * @code
 * aie::vector<int8, 64> even = ..., odd = ...;
 * aie::vector<int8, 128> mixed = aie::concat( aie::interleave_zip(even, odd, 1) );
 * @endcode
 *
 * @param t An object satisfying @ref aie::TupleLike that contains the input
 *          vectors.
 */
template <TupleLike T>
    requires(std::tuple_size_v<T> > 1)
auto concat(const T &t)
{
    return detail::utils::apply_tuple(
                [](const auto &... args) { return concat(args...); },
                t);
}

/**
 * @ingroup group_init
 *
 * Returns a vector whose elements are initialized to the given value.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a;
 * @endcode
 *
 * @tparam E     Element type of the returned vector.
 * @tparam Elems Size of the vector.
 *
 * @param a Value. The type must meet @ref aie::Elem.
 */
template <Elem E, unsigned Elems = native_vector_length_v<E>>
__aie_inline
vector<operand_base_type_t<E>, Elems> broadcast(E a)
{
    using T = operand_base_type_t<E>;

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::broadcast<T, Elems>::run(vector_elem_const_ref(a));
    else
        return detail::broadcast<T, Elems>::run(a);
}

/**
 * @ingroup group_init
 *
 * Returns a vector whose elements are initialized to a repeating set of values.
 *
 * @code
 * unsigned i = 0;
 * while (i < Elems)
 *   for (unsigned j = 0; j < N; ++j)
 *     out[i++] = a[j];
 * @endcode
 *
 * @tparam T     Element type of the result vector.
 * @tparam Elems Size of the result vector.
 * @tparam N     Number of repeating elements. The combined number of bits must be small to fit in a scalar value (64bit or smaller).
 *
 * @param a Set of values. Its elements must meet @ref aie::ElemBaseType.
 */
template <ElemBaseType T, unsigned Elems = native_vector_length_v<T>, unsigned N>
     requires(arch::is(arch::Gen2)
             && detail::utils::is_powerof2(N)
             && (detail::type_bits_v<T> * N) <= 64)
__aie_inline
vector<T, Elems> broadcast(const T (&a)[N])
{
    return detail::broadcast<T, Elems>::run(a);
}

/**
 * @ingroup group_init
 *
 * Returns a vector whose elements are initialized to a repeating set of values.
 *
 * @code
 * unsigned i = 0;
 * while (i < Elems)
 *   for (unsigned j = index * N; j < (index + 1) * N; ++j)
 *     out[i++] = v[j];
 * @endcode
 *
 * @tparam N      Number of repeating elements.
 * @tparam T      Element type of the result vector.
 * @tparam Elems  Size of the result vector.
 * @tparam Elems2 Size of the vector where to get values to be broadcast from
 *
 * @param v     Vector with the values to be broadcast
 * @param index Index of the sequence of N elements to be broadcast
 */
template <unsigned N, ElemBaseType T, unsigned Elems = native_vector_length_v<T>, unsigned Elems2>
     requires(arch::is(arch::Gen2)
             && detail::utils::is_powerof2(N)
             && N <= Elems)
__aie_inline
vector<T, Elems> broadcast_vector(const vector<T, Elems2> &v, unsigned index)
{
    return detail::broadcast_vector<T, Elems, N>::run(v, index);
}

/**
 * @ingroup group_init
 *
 * Returns a vector whose elements are initialized to zero.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = 0;
 * @endcode
 *
 * @tparam E. Element type of the returned vector.
 * @tparam Elems. Size of the vector.
 */
template <ElemBaseType T, unsigned Elems = native_vector_length_v<T>>
__aie_inline
vector<T, Elems> zeros()
{
    return detail::zeros<T, Elems>::run();
}

/**
 * @ingroup group_init
 *
 * Returns a vector whose elements are initialized to zero.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = 0;
 * @endcode
 *
 * @tparam AccumTag. Accumulator tag.
 * @tparam Elems. Size of the accumulator.
 */
template <AccumElemBaseType AccumTag, unsigned Elems = native_vector_length_v<AccumTag>>
__aie_inline
accum<AccumTag, Elems> zeros()
{
    return detail::zeros_acc<detail::accum_class_for_tag_v<AccumTag>, detail::accum_bits_for_tag_v<AccumTag>, Elems>::run();
}

/**
 * @ingroup group_compare
 *
 * Return whether all the elements of the two input vectors are equal. The vectors must have the same type and size.
 *
 * @code
 * bool out = true;
 * for (unsigned i = 0; i < Elems; ++i)
 *     out = out && (v1[i] == v2[i]);
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
constexpr bool equal(const Vec1 &v1, const Vec2 &v2)
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return true;

    return detail::equal<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Return whether all the elements of the input vector are equal to the scalar.
 *
 * @code
 * bool out = true;
 * for (unsigned i = 0; i < Elems; ++i)
 *     out = out && (a == v[i]);
 * @endcode
 *
 * @param a Input scalar. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr bool equal(E a, const Vec &v)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::equal<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::equal<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Return whether all the elements of the input vector are equal to the scalar.
 *
 * @code
 * bool out = true;
 * for (unsigned i = 0; i < Elems; ++i)
 *     out = out && (v[i] == a);
 * @endcode
 *
 * @param a Input scalar. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr bool equal(const Vec &v, E a)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::equal<T2, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::equal<T2, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Return whether some elements in the two input vectors are not equal. The vectors must have the same type and size.
 *
 * @code
 * bool out = true;
 * for (unsigned i = 0; i < Elems; ++i)
 *     out = out && (v1[i] != v2[i]);
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
constexpr bool not_equal(const Vec1 &v1, const Vec2 &v2)
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec2::size();

    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return false;

    return detail::not_equal<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Return whether some the elements of the input vector are not equal to the scalar.
 *
 * @code
 * bool out = true;
 * for (unsigned i = 0; i < Elems; ++i)
 *     out = out && (a != v[i]);
 * @endcode
 *
 * @param a Input scalar. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr bool not_equal(E a, const Vec &v)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::not_equal<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::not_equal<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Return whether some the elements of the input vector are not equal to the scalar.
 *
 * @code
 * bool out = true;
 * for (unsigned i = 0; i < Elems; ++i)
 *     out = out && (v[i] != a);
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param a Input scalar. The type must meet @ref aie::Elem.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr bool not_equal(const Vec &v, E a)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::not_equal<T2, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::not_equal<T2, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a mask that says what elements from the first vector are
 * smaller than the corresponding elements in the second vector. The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] < v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::RealVector.
 * @param v2 Second input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
mask<Vec1::size()> lt(const Vec1 &v1, const Vec2 &v2)
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return mask<Vec1::size()>();

    return detail::lt<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says if the value is smaller than the
 * elements in the vector. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a < v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealElem E, RealVector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> lt(E a, const Vec &v)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::lt<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::lt<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a mask that says if the elements in the vector
 * are smaller than the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] < a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <RealVector Vec, RealElem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> lt(const Vec &v, E a)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::lt<T2, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::lt<T2, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a mask that says what elements from the first vector are
 * greater than the corresponding elements in the second vector. The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] > v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::RealVector.
 * @param v2 Second input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
mask<Vec1::size()> gt(const Vec1 &v1, const Vec2 &v2)
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return mask<Vec1::size()>();

    return detail::lt<T, Elems>::run(v2, v1);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says if the value is greater than the
 * elements in the vector. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a > v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealElem E, RealVector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> gt(E a, const Vec &v)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::lt<T2, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::lt<T2, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a mask that says if the elements in the vector
 * are greater than the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] > a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <RealVector Vec, RealElem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> gt(const Vec &v, E a)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::lt<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::lt<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a mask that says what elements from the first vector are
 * greater or equal than the corresponding elements in the second vector. The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] >= v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::RealVector.
 * @param v2 Second input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
mask<Vec1::size()> ge(const Vec1 &v1, const Vec2 &v2)
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return mask<Vec1::size()>(true);

    return detail::ge<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says if the value is greater or equal
 * than the elements in the vector. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a >= v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealElem E, RealVector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> ge(E a, const Vec &v)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::ge<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::ge<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a mask that says if the elements in the vector
 * are greater or equal than the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] >= a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <RealVector Vec, RealElem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> ge(const Vec &v, E a)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::ge<T2, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::ge<T2, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a mask that says what elements from the first vector are
 * smaller or equal than the corresponding elements in the second vector. The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] <= v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::RealVector.
 * @param v2 Second input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
mask<Vec1::size()> le(const Vec1 &v1, const Vec2 &v2)
{
    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return mask<Vec1::size()>(true);

#if __AIE_ARCH__ == 10
    // On AIE1 changing operand order can be more expensive than flipping the bits with a scalar instruction
    return ~gt(v1, v2);
#else
    return ge(v2, v1);
#endif
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says if the value is smaller or equal
 * than the elements in the vector. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a <= v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealElem E, RealVector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> le(E a, const Vec &v)
{
#if __AIE_ARCH__ == 10
    // On AIE1 changing operand order can be more expensive than flipping the bits with a scalar instruction
    return ~gt(a, v);
#else
    return ge(v, a);
#endif
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a mask that says if the elements in the vector
 * are smaller or equal than the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] <= a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <RealVector Vec, RealElem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> le(const Vec &v, E a)
{
#if __AIE_ARCH__ == 10
    // On AIE1 changing operand order can be more expensive than flipping the bits with a scalar instruction
    return ~gt(v, a);
#else
    return ge(a, v);
#endif
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a mask that says what elements are different. The vectors
 * must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] != v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
mask<Vec1::size()> neq(const Vec1 &v1, const Vec2 &v2)
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return mask<Vec1::size()>(false);

    return detail::neq<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says what elements in the vector
 * are different than the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a != v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> neq(E a, const Vec &v)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::neq<T, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::neq<T, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says what elements in the vector
 * are different than the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] != a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param a Value. The type must meet @ref aie::Elem.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> neq(const Vec &v, E a)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::neq<T, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::neq<T, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a mask that says what elements are equal. The vectors must
 * have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] == v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
mask<Vec1::size()> eq(const Vec1 &v1, const Vec2 &v2)
{
    if (chess_manifest(Utils::is_same_object(v1, v2)))
        return mask<Vec1::size()>(true);

#if __AIE_ARCH__ == 10
    return ~neq(v1, v2);
#else
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    return detail::eq<T, Elems>::run(v1, v2);
#endif
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says what elements in the vector
 * are equal to the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a == v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> eq(E a, const Vec &v)
{
#if __AIE_ARCH__ == 10
    return ~neq(a, v);
#else
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::eq<T, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::eq<T, Elems>::run(a, v);
#endif
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a mask that says what elements in the vector
 * are equal to the value. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] == a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param a Value. The type must meet @ref aie::Elem.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
mask<Vec::size()> eq(const Vec &v, E a)
{
#if __AIE_ARCH__ == 10
    return ~neq(v, a);
#else
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::eq<T, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::eq<T, Elems>::run(v, a);
#endif
}

template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
bool contains(E a, const Vec &v)
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::contains<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::contains<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_reshape
 *
 * Combines the values of the two input vectors into a vector of the same size by using a mask that specifies which is
 * the source input for each element of the output vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = m[i] == 0? v1[i] : v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 * @param m  Mask that specifies the source input for each output element. The type must meet @ref aie::Mask.
 */
template <Vector Vec1, Vector Vec2, Mask M> requires(is_same_vector_v<Vec1, Vec2> && Vec1::size() == M::size())
__aie_inline
auto select(const Vec1 &v1, const Vec2 &v2, const M &m) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return v1;

    if (chess_manifest(m == M(false)))
        return v1;
    if (chess_manifest(m == M(true)))
        return v2;

    return detail::select<T, Elems>::run(v1, v2, m);
}

/**
 * @ingroup group_reshape
 *
 * Combines the values of the input value and vector into a vector of the same size by using a mask that specifies which
 * is the source input for each element of the output vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = m[i] == 0? a : v[i];
 * @endcode
 *
 * @param a Input value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param m Mask that specifies the source input for each output element. The type must meet @ref aie::Mask.
 */
template <Elem E, Vector Vec, Mask M> requires(is_valid_elem_op_v<E, typename Vec::value_type> &&
                                                       Vec::size() == M::size())
__aie_inline
auto select(E a, const Vec &v, const M &m) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if (chess_manifest(m == M(true)))
        return v;

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::select<T2, Elems>::run(vector_elem_const_ref(a), v, m);
    else
        return detail::select<T2, Elems>::run(a, v, m);
}

/**
 * @ingroup group_reshape
 *
 * Combines the values of the input vector and value into a vector of the same size by using a mask that specifies which
 * is the source input for each element of the output vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = m[i] == 0? v[i] : a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param a Input value. The type must meet @ref aie::Elem.
 * @param m Mask that specifies the source input for each output element. The type must meet @ref aie::Mask.
 */
template <Vector Vec, Elem E, Mask M> requires(is_valid_elem_op_v<E, typename Vec::value_type> &&
                                                       Vec::size() == M::size())
__aie_inline
auto select(const Vec &v, E a, const M &m) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if (chess_manifest(m == M(false)))
        return v;

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::select<T2, Elems>::run(v, vector_elem_const_ref(a), m);
    else
        return detail::select<T2, Elems>::run(v, a, m);
}

/**
 * @ingroup group_reshape
 *
 * Combines the values of the input vector and value into a vector of the same size by using a mask that specifies which
 * is the source input for each element of the output vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = m[i] == 0? a : b;
 * @endcode
 *
 * @param a Input value. The type must meet @ref aie::Elem.
 * @param b Input value. The type must meet @ref aie::Elem.
 * @param m Mask that specifies the source input for each output element. The type must meet @ref aie::Mask.
 */
template <Elem E1, Elem E2, Mask M> requires(is_valid_elem_op_v<E1, E2>)
__aie_inline
vector<operand_base_type_t<E1>, M::size()> select(const E1 &a, const E2 &b, const M &m)
{
    constexpr unsigned Elems = M::size();

    using T = operand_base_type_t<E1>;

    if      constexpr (detail::is_vector_elem_ref_v<E1> && detail::is_vector_elem_ref_v<E2>)
        return detail::select<T, Elems>::run(vector_elem_const_ref(a), vector_elem_const_ref(b), m);
    else if constexpr (detail::is_vector_elem_ref_v<E1>)
        return detail::select<T, Elems>::run(vector_elem_const_ref(a), b, m);
    else if constexpr (detail::is_vector_elem_ref_v<E2>)
        return detail::select<T, Elems>::run(a, vector_elem_const_ref(b), m);
    else
        return detail::select<T, Elems>::run(a, b, m);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a vector with the maximum values. The vectors must have
 * the same type and size.
 *
 * @param v1 First input vector. The type must meet @ref aie::RealVector.
 * @param v2 Second input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto max(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return v1;

    return detail::max<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a vector with the maximum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealElem E, RealVector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto max(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::max<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::max<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a vector with the maximum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <RealVector Vec, RealElem E> requires(is_valid_elem_op_v<typename Vec::value_type, E>)
__aie_inline
auto max(const Vec &v, E a) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::max<T2, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::max<T2, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a vector with the maximum values. The vectors must have
 * the same type and size.
 *
 * @param v1   First input vector. The type must meet @ref aie::RealVector.
 * @param v2   Second input vector. The type must meet @ref aie::RealVector.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealVector Vec1, RealVector Vec2> requires(arch::is(arch::Gen2) &&
                                                     is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto max(const Vec1 &v1, const Vec2 &v2, bool sign) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return v1;

    return detail::max<T, Elems>::run(v1, v2, sign);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a vector with the maximum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param a    Value. The type must meet @ref aie::RealElem.
 * @param v    Input vector. The type must meet @ref aie::RealVector.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealElem E, RealVector Vec> requires(arch::is(arch::Gen2) &&
                                               is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto max(E a, const Vec &v, bool sign) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::max<T2, Elems>::run(vector_elem_const_ref(a), v, sign);
    else
        return detail::max<T2, Elems>::run(a, v, sign);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a vector with the maximum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param v    Input vector. The type must meet @ref aie::RealVector.
 * @param a    Value. The type must meet @ref aie::RealElem.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealVector Vec, RealElem E> requires(arch::is(arch::Gen2) &&
                                               is_valid_elem_op_v<typename Vec::value_type, E>)
__aie_inline
auto max(const Vec &v, E a, bool sign) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::max<T2, Elems>::run(v, vector_elem_const_ref(a), sign);
    else
        return detail::max<T2, Elems>::run(v, a, sign);
}

/**
 * @ingroup group_reduce
 *
 * Returns the element from the input vector with the largest value.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec>
__aie_inline
typename Vec::value_type reduce_max(const Vec &v)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::max_reduce<T, Elems>::run(v);
}

template <ElemBaseType T, unsigned Elems>
    requires(RealElem<T>)
__aie_inline
T reduce_max(const T &max, const aie::vector<T, Elems> &v)
{
    return detail::max_reduce<T, Elems>::run(max, v);
}

template <RealVector Vec>
[[deprecated("Use reduce_max instead")]]
__aie_inline
typename Vec::value_type max_reduce(const Vec &v)
{
    return reduce_max(v);
}

template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
auto max_cmp(const Vec1 &v1, const Vec2 &v2) -> std::tuple<aie_dm_resource_remove_t<Vec1>, mask<Vec1::size()>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    return detail::max_cmp<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a vector with the minimum values. The vectors must have
 * the same type and size.
 *
 * @param v1 First input vector. The type must meet @ref aie::RealVector.
 * @param v2 Second input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto min(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return v1;

    return detail::min<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a vector with the minimum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealElem E, RealVector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto min(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::min<T2, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::min<T2, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a vector with the minimum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <RealVector Vec, RealElem E> requires(is_valid_elem_op_v<typename Vec::value_type, E>)
__aie_inline
auto min(const Vec &v, E a) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::min<T2, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::min<T2, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a vector with the minimum values. The vectors must have
 * the same type and size.
 *
 * @param v1   First input vector. The type must meet @ref aie::RealVector.
 * @param v2   Second input vector. The type must meet @ref aie::RealVector.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealVector Vec1, RealVector Vec2> requires(arch::is(arch::Gen2) &&
                                                     is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto min(const Vec1 &v1, const Vec2 &v2, bool sign) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return v1;

    return detail::min<T, Elems>::run(v1, v2, sign);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a vector with the minimum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param a    Value. The type must meet @ref aie::RealElem.
 * @param v    Input vector. The type must meet @ref aie::RealVector.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealElem E, RealVector Vec> requires(arch::is(arch::Gen2) &&
                                               is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto min(E a, const Vec &v, bool sign) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::min<T2, Elems>::run(vector_elem_const_ref(a), v, sign);
    else
        return detail::min<T2, Elems>::run(a, v, sign);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a vector with the minimum values. The type of
 * the value and the type of the vector elements must be the same.
 *
 * @param v    Input vector. The type must meet @ref aie::RealVector.
 * @param a    Value. The type must meet @ref aie::RealElem.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealVector Vec, RealElem E> requires(arch::is(arch::Gen2) &&
                                               is_valid_elem_op_v<typename Vec::value_type, E>)
__aie_inline
auto min(const Vec &v, E a, bool sign) -> aie_dm_resource_remove_t<Vec>
{
    using T2 = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::min<T2, Elems>::run(v, vector_elem_const_ref(a), sign);
    else
        return detail::min<T2, Elems>::run(v, a, sign);
}

/**
 * @ingroup group_reduce
 *
 * Returns the element from the input vector with the smallest value.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec>
__aie_inline
typename Vec::value_type reduce_min(const Vec &v)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::min_reduce<T, Elems>::run(v);
}

template <ElemBaseType T, unsigned Elems>
    requires(RealElem<T>)
__aie_inline
T reduce_min(const T &min, const aie::vector<T, Elems> &v)
{
    return detail::min_reduce<T, Elems>::run(min, v);
}

template <RealVector Vec>
[[deprecated("Use reduce_min instead")]]
__aie_inline
typename Vec::value_type min_reduce(const Vec &v)
{
    return reduce_min(v);
}

template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto min_cmp(const Vec1 &v1, const Vec2 &v2) -> std::tuple<aie_dm_resource_remove_t<Vec1>, mask<Vec1::size()>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    return detail::min_cmp<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a vector whose values are their difference if it is
 * positive, or zero otherwise. The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] > v2[i]? v1[i] - v2[i] : 0;
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::RealVector.
 * @param v2 Second input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec1, RealVector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto maxdiff(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return zeros<T, Elems>();

    return detail::maxdiff<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a vector whose values are their difference if it
 * is positive, or zero otherwise. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a > v[i]? a - v[i] : 0;
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <RealElem E, RealVector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto maxdiff(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::maxdiff<T, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::maxdiff<T, Elems>::run(a, v);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a vector whose values are their difference if it
 * is positive, or zero otherwise. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] > a? v[i] - a : 0;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param a Value. The type must meet @ref aie::Elem.
 */
template <RealVector Vec, RealElem E> requires(is_valid_elem_op_v<typename Vec::value_type, E>)
__aie_inline
auto maxdiff(const Vec &v, E a) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::maxdiff<T, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::maxdiff<T, Elems>::run(v, a);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the two input vectors and returns a vector whose values are their difference if it is
 * positive, or zero otherwise. The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] > v2[i]? v1[i] - v2[i] : 0;
 * @endcode
 *
 * @param v1   First input vector. The type must meet @ref aie::RealVector.
 * @param v2   Second input vector. The type must meet @ref aie::RealVector.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealVector Vec1, RealVector Vec2> requires(arch::is(arch::Gen2) &&
                                                     is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto maxdiff(const Vec1 &v1, const Vec2 &v2, bool sign) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return zeros<T, Elems>();

    return detail::maxdiff<T, Elems>::run(v1, v2, sign);
}

/**
 * @ingroup group_compare
 *
 * Compares a value with the elements of the input vector and returns a vector whose values are their difference if it
 * is positive, or zero otherwise. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a > v[i]? a - v[i] : 0;
 * @endcode
 *
 * @param a    Value. The type must meet @ref aie::Elem.
 * @param v    Input vector. The type must meet @ref aie::Vector.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealElem E, RealVector Vec> requires(arch::is(arch::Gen2) &&
                                               is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto maxdiff(E a, const Vec &v, bool sign) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::maxdiff<T, Elems>::run(vector_elem_const_ref(a), v, sign);
    else
        return detail::maxdiff<T, Elems>::run(a, v, sign);
}

/**
 * @ingroup group_compare
 *
 * Compares the elements of the input vector with a value and returns a vector whose values are their difference if it
 * is positive, or zero otherwise. The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] > a? v[i] - a : 0;
 * @endcode
 *
 * @param v    Input vector. The type must meet @ref aie::Vector.
 * @param a    Value. The type must meet @ref aie::Elem.
 * @param sign Whether the inputs need to be interpreted as signed values.
 */
template <RealVector Vec, RealElem E> requires(arch::is(arch::Gen2) &&
                                               is_valid_elem_op_v<typename Vec::value_type, E>)
__aie_inline
auto maxdiff(const Vec &v, E a, bool sign) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::maxdiff<T, Elems>::run(v, vector_elem_const_ref(a), sign);
    else
        return detail::maxdiff<T, Elems>::run(v, a, sign);
}

/**
 * @ingroup group_compare
 *
 * Returns a vector whose elements are within specified lower and upper limits.
 *
 * The result of the operation is unspecified if the value of `lo` argument is greater than `hi`.
 * The vector elements and the boundary values must have the same type.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i) {
 *     auto e = max(m[i], lo);
 *     out[i] = min(e,    hi);
 * }
 * @endcode
 *
 * @param v  Input vector. The type must meet @ref aie::RealVector.
 * @param lo Smallest value. The type must meet @ref aie::RealElem.
 * @param hi Greatest value. The type must meet @ref aie::RealElem.
 */
template <RealVector Vec, RealElem E1, RealElem E2> requires(is_valid_elem_op_v<E1, typename Vec::value_type> &&
                                                             is_valid_elem_op_v<E2, typename Vec::value_type>)
__aie_inline
auto clamp(const Vec &v, E1 lo, E2 hi) -> aie_dm_resource_remove_t<Vec>
{
    REQUIRES_MSG(hi >= lo, "`lo` argument must not be greater than `hi`");
    return min(max(v, lo), hi);
}

/**
 * @ingroup group_compare
 *
 * Returns a vector whose elements are within specified lower and upper limits.
 *
 * The result of the operation is unspecified if the value of `lo` argument is greater than `hi`.
 * The input and the boundary vectors must have the same type.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i) {
 *     auto e = max(m[i], lo[i]);
 *     out[i] = min(e,    hi[i]);
 * }
 * @endcode
 *
 * @param v  Values vector. The type must meet @ref aie::RealVector.
 * @param lo Lower bounds vector. The type must meet @ref aie::RealVector.
 * @param hi Upper bounds vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec, RealVector VecL, RealVector VecH>
    requires(is_same_vector_v<Vec, VecL> && is_same_vector_v<Vec, VecH>)
__aie_inline
auto clamp(const Vec &v, const VecL &lo, const VecH &hi) -> aie_dm_resource_remove_t<Vec>
{
    REQUIRES_MSG(lt(hi, lo).empty(), "`lo` argument must not be greater than `hi`");
    return min(max(v, lo), hi);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but in reverse order.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = in[Elems - i - 1];
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec>
__aie_inline
auto reverse(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::reverse<T, Elems>::run(v);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted up by n. Elements do not wrap around
 * and new elements are undefined.
 *
 * @code
 * for (unsigned i = 0; i < Elems - n; ++i)
 *     out[i + n] = in[i]
 *
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param n Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_up(const Vec &v, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0))
        return v;
    else if (chess_manifest(n == Elems))
        return Vec{};

    return detail::shuffle_up<T, Elems>::run(v, n);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted up by n (elements wrap around):
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = in[(i - n) % Elems]
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param n Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_up_rotate(const Vec &v, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0 || n == Elems))
        return v;

    return detail::shuffle_up_rotate<T, Elems>::run(v, n);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted up by n. Elements do not wrap around
 * and new elements are copies of the first element of the input vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems - n; ++i)
 *     out[i + n] = in[i]
 *
 * for (unsigned i = 0; i < n; ++i)
 *     out[i] = in[0];
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param n Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_up_replicate(const Vec &v, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0))
        return v;

    return detail::shuffle_up_replicate<T, Elems>::run(v, n);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted up by n. Elements do not wrap around
 * and new elements are filled from a second vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems - n; ++i)
 *     out[i + n] = in[i]
 *
 * for (unsigned i = 0; i < n; ++i)
 *     out[i] = fill[i + Elems - n];
 * @endcode
 *
 * @param v    Input vector. The type must meet @ref aie::Vector.
 * @param fill Second input vector used to fill the elements in the lower part of the output vector.
 * @param n    Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_up_fill(const Vec &v, const Vec &fill, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0))
        return v;
    else if (chess_manifest(n == Elems))
        return fill;

    return detail::shuffle_up_fill<T, Elems>::run(v, fill, n);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted down by n. Elements do not wrap around
 * and new elements are undefined.
 *
 * @code
 * for (unsigned i = 0; i < Elems - n; ++i)
 *     out[i] = in[i + n]
 * @endcode
 *
 * @param v    Input vector. The type must meet @ref aie::Vector.
 * @param n    Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_down(const Vec &v, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0))
        return v;
    else if (chess_manifest(n == Elems))
        return Vec{};

    return detail::shuffle_down<T, Elems>::run(v, n);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted down by n (elements wrap around):
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = in[(i + n) % Elems]
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param n Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_down_rotate(const Vec &v, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0 || n == Elems))
        return v;

    return detail::shuffle_down_rotate<T, Elems>::run(v, n);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted down by n. Elements do not wrap around
 * and new elements are filled from a second vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems - n; ++i)
 *     out[i] = in[i + n]
 *
 * for (unsigned i = 0; i < n; ++i)
 *     out[i + Elems - n] = fill[i]
 * @endcode
 *
 * @param v    Input vector. The type must meet @ref aie::Vector.
 * @param fill Second input vector used to fill the elements in the upper part of the output vector.
 * @param n    Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_down_fill(const Vec &v, const Vec & fill, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0))
        return v;
    else if (chess_manifest(n == Elems))
        return fill;

    return detail::shuffle_down_fill<T, Elems>::run(v, fill, n);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector whose contents are the same as the input vector, but shifted down by n. Elements do not wrap around
 * and new elements are copies of the last element of the input vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems - n; ++i)
 *     out[i] = in[i + n]
 *
 * for (unsigned i = Elems - n; i < Elems; ++i)
 *     out[i] = in[Elems - 1];
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param n Distance for the elements to be shifted.
 */
template <Vector Vec>
__aie_inline
auto shuffle_down_replicate(const Vec &v, unsigned n) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(n <= Elems, "n must be smaller or equal than the vector length");

    if (chess_manifest(n == 0))
        return v;

    return detail::shuffle_down_replicate<T, Elems>::run(v, n);
}

/**
 * @ingroup group_reshape
 *
 * The input vector is interpreted as a row-major matrix of the dimensions specified by Row and Col, the
 * function returns a vector ordered as the transpose of the specified matrix. For Row equal to 1, the
 * input vector is returned.
 *
 * @param v   Input vector. The type must meet @ref aie::Vector, with number of elements equal to Row * Col.
 * @param Row Number of rows of the input matrix shape.
 * @param Col Number of cols of the input matrix shape.
 */
template <Vector Vec>
__aie_inline
auto transpose(const Vec &v, unsigned Row, unsigned Col) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(Row * Col == Elems, "Requires Row * Col equal to number of vector elements");
    REQUIRES_MSG(Row > 0, "The number of rows must be greater than 0");
    REQUIRES_MSG(Col > 0, "The number of columns must be greater than 0");

    return detail::transpose<T, Elems>::run(v, Row, Col);
}

/**
 * @ingroup group_reshape
 *
 * Enables precomputation of the interleave element selection.
 *
 * Allows to efficiently choose from multiple element selection alternatives at
 * execution time, when performing an interleave_unzip operation.
 *
 * Example:
 * @code
 * bool condition = evaluated_at_runtime();
 * auto m = condition? aie::interleave_zip_mode<int8, 64>(4)
 *                   : aie::interleave_zip_mode<int8, 64>(2);
 * aie::vector<int8, 64> v = ..., w = ...;
 * auto [low, high] = aie::interleave_zip(v, w, m);
 * @endcode
 *
 * @tparam T     Element type of the input vectors to be used in the interleave.
 * @tparam Elems Number of elements of the input vectors to be used in the
 *               interleave.
 *
 * @sa interleave_unzip_mode
 * @sa interleave_zip(const Vec1&, const Vec2&, const interleave_zip_mode&)
 */
template <ElemBaseType T, unsigned Elems>
    requires(arch::is(arch::Gen2))
class interleave_zip_mode : public detail::shuffle_mode<detail::type_bits_v<T>, Elems> {
    static constexpr unsigned type_bits = detail::type_bits_v<T>;
public:
    /**
     * Constructor for class interleave_zip_mode
     *
     * @param chunk_size Number of contiguous elements taken in an interleave_zip operation.
     *                   It must be a power of two.
     */
    __aie_inline
    explicit constexpr interleave_zip_mode(unsigned chunk_size) noexcept :
        detail::shuffle_mode<type_bits, Elems>(chunk_size, detail::shuffle_zip_tag{})
    {
        REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
        REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
        REQUIRES_MSG(type_bits * chunk_size >= 8, "Minimum chunk_size granularity is 8bit");
    }

    /** Copy constructor for class interleave_zip_mode */
    constexpr interleave_zip_mode(const interleave_zip_mode &) = default;
};

/**
 * @ingroup group_reshape
 *
 * Enables precomputation of the interleave element selection.
 *
 * Allows to efficiently choose from multiple element selection alternatives at
 * execution time, when performing an interleave_unzip operation.
 *
 * Example:
 * @code
 * bool condition = evaluated_at_runtime();
 * auto m = condition? aie::interleave_unzip_mode<int8, 64>(4)
 *                   : aie::interleave_unzip_mode<int8, 64>(2);
 * aie::vector<int8, 64> v = ..., w = ...;
 * auto [even, odd] = aie::interleave_unzip(v, w, m);
 * @endcode
 *
 * @tparam T     Element type of the input vectors to be used in the interleave.
 * @tparam Elems Number of elements of the input vectors to be used in the
 *               interleave.
 *
 * @sa interleave_zip_mode
 * @sa filter_mode
 * @sa interleave_unzip(const Vec1&, const Vec2&, const interleave_unzip_mode&)
 */
template <ElemBaseType T, unsigned Elems>
    requires(arch::is(arch::Gen2))
class interleave_unzip_mode : public detail::shuffle_mode<detail::type_bits_v<T>, Elems> {
    static constexpr unsigned type_bits = detail::type_bits_v<T>;
public:
    /**
     * Constructor for class interleave_unzip_mode
     *
     * @param chunk_size Number of contiguous elements taken in an interleave_unzip operation.
     *                   It must be a power of two.
     */
    __aie_inline
    explicit constexpr interleave_unzip_mode(unsigned chunk_size) noexcept :
        detail::shuffle_mode<type_bits, Elems>(chunk_size, detail::shuffle_unzip_tag{})
    {
        REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
        REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
        REQUIRES_MSG(type_bits * chunk_size >= 8, "Minimum chunk_size granularity is 8bit");
    }

    /** Copy constructor for class interleave_unzip_mode */
    constexpr interleave_unzip_mode(const interleave_unzip_mode &) = default;
};


/**
 * @ingroup group_reshape
 *
 * Picks elements alternatively from the input vectors and writes them sequentially into the output vectors.
 *
 * @code
 * out = { v[0:chunk_size], w[0:chunk_size], v[chunk_size:2*chunk_size], w[chunk_size:2*chunk_size], ... }
 * @endcode
 *
 * @param v          First input vector. Must meet @ref aie::Vector.
 * @param w          Second input vector. Must meet @ref aie::Vector.
 * @param chunk_size Number of contiguous elements taken from each input vector. It must be a power of two.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto interleave_zip(const Vec1 &v, const Vec2 &w, unsigned chunk_size)
    -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
    REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
    REQUIRES_MSG(detail::type_bits_v<T> * chunk_size >= 8, "Minimum chunk_size granularity is 8bit");

    return detail::interleave_zip<T, Elems>::run(v, w, chunk_size);
}

/**
 * @ingroup group_reshape
 *
 * Picks elements alternatively from the input vectors and writes them
 * sequentially into the output vectors.
 *
 * In cases where the interleave chunk size is picked at execution time, this
 * overload allows the user to precompute the interleave mode in advance (it is
 * an expensive operation).
 *
 * @param v    First input vector. Must meet @ref aie::Vector.
 * @param w    Second input vector. Must meet @ref aie::Vector.
 * @param mode Precomputed interleave mode, specifying the number of contiguous
 *             elements.
 *
 * @sa interleave_zip_mode
 * @sa interleave_zip(const Vec1&, const Vec2&, unsigned)
 */
template <Vector Vec1, Vector Vec2, ElemBaseType T, unsigned Elems>
    requires(arch::is(arch::Gen2)
             && is_same_vector_v<Vec1, Vec2>
             && std::is_same_v<typename Vec1::value_type, T>
             && Vec1::size() == Elems)
__aie_inline
auto interleave_zip(const Vec1 &v, const Vec2 &w, const interleave_zip_mode<T, Elems> &mode)
    -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    return detail::interleave_zip<T, Elems>::run(v, w, mode);
}

/**
 * @ingroup group_reshape
 *
 * Picks elements sequentially from the input vectors and writes them alternatively into the output vectors.
 *
 * @code
 * out = { v[0:chunk_size], v[2*chunk_size:3*chunk_size], ..., w[0:chunk_size], w[2*chunk_size:3*chunk_size], ..., v[chunk_size:2*chunk_size], v[3*chunk_size:4*chunk_size], ..., w[chunk_size:2*chunk_size], w[3*chunk_size:4*chunk_size], ... }
 * @endcode
 *
 * @param v          First input vector. Must meet @ref aie::Vector.
 * @param w          Second input vector. Must meet @ref aie::Vector.
 * @param chunk_size Number of contiguous elements taken from each input vector. It must be a power of two.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto interleave_unzip(const Vec1 &v, const Vec2 &w, unsigned chunk_size)
    -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
    REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
    REQUIRES_MSG(detail::type_bits_v<T> * chunk_size >= 8, "Minimum chunk_size granularity is 8bit");

    return detail::interleave_unzip<T, Elems>::run(v, w, chunk_size);
}

/**
 * @ingroup group_reshape
 *
 * Picks elements sequentially from the input vectors and writes them
 * alternatively into the output vectors.
 *
 * In cases where the interleave chunk size is picked at execution time, this
 * overload allows the user to precompute the interleave mode in advance (it is
 * an expensive operation).
 *
 * @param v1   First input vector. Must meet @ref aie::Vector.
 * @param v2   Second input vector. Must meet @ref aie::Vector.
 * @param mode Precomputed interleave mode, specifying the number of contiguous
 *             elements.
 *
 * @sa interleave_unzip_mode
 * @sa interleave_unzip(const Vec1&, const Vec2&, unsigned)
 */
template <Vector Vec1, Vector Vec2, ElemBaseType T, unsigned Elems>
    requires(arch::is(arch::Gen2)
             && is_same_vector_v<Vec1, Vec2>
             && std::is_same_v<typename Vec1::value_type, T>
             && Vec1::size() == Elems)
__aie_inline
auto interleave_unzip(const Vec1 &v1, const Vec2 &v2, const interleave_unzip_mode<T, Elems> &mode)
    -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    return detail::interleave_unzip<T, Elems>::run(v1, v2, mode);
}

template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto interleave_butterfly(const Vec1 &v1, const Vec2 &v2, unsigned chunk_size)
    -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
    REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");

    return detail::interleave_butterfly<T, Elems>::run(v1, v2, chunk_size);
}

template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto interleave_butterfly_half(const Vec1 &v1, const Vec2 &v2, unsigned chunk_size)
    -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
    REQUIRES_MSG(chunk_size && !(chunk_size & (chunk_size - 1)), "chunk_size must be a power of two");

    return detail::interleave_butterfly_half<T, Elems>::run(v1, v2, chunk_size);
}

template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto interleave_crossover(const Vec1 &v1, const Vec2 &v2, unsigned chunk_size)
    -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
    REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");

    return detail::interleave_crossover<T, Elems>::run(v1, v2, chunk_size);
}

template <unsigned S0,  unsigned S1,  unsigned S2,  unsigned S3,  unsigned S4,  unsigned S5,  unsigned S6,  unsigned S7,
          unsigned S8,  unsigned S9,  unsigned S10, unsigned S11, unsigned S12, unsigned S13, unsigned S14, unsigned S15,
          unsigned S16, unsigned S17, unsigned S18, unsigned S19, unsigned S20, unsigned S21, unsigned S22, unsigned S23,
          unsigned S24, unsigned S25, unsigned S26, unsigned S27, unsigned S28, unsigned S29, unsigned S30, unsigned S31,
          Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto interleave_custom_static(const Vec1 &v1, const Vec2 &v2) -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec2::size();

    return detail::interleave_custom_static<T, Elems, S0,  S1,  S2,  S3,  S4,  S5,  S6,  S7,  S8,  S9,  S10, S11, S12, S13, S14, S15,
                                                      S16, S17, S18, S19, S20, S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31>::run(v1, v2);
}

template <Vector Vec1, Vector Vec2, typename... Select> requires(is_same_vector_v<Vec1, Vec2> &&
                                                                         sizeof...(Select) == Vec1::size())
__aie_inline
auto interleave_custom(const Vec1 &v1, const Vec2 &v2, Select... select) -> std::pair<aie_dm_resource_remove_t<Vec1>, aie_dm_resource_remove_t<Vec1>>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    return detail::interleave_custom<T, Elems>::run(v1, v2, unsigned(select)...);
}

/**
 * @ingroup group_reshape
 *
 * Enables precomputation of the filter element selection.
 *
 * Allows to efficiently choose from multiple element selection alternatives at
 * execution time, including odd/even, when performing a filter operation.
 *
 * Example:
 * @code
 * bool condition = evaluated_at_runtime();
 * auto f = condition? aie::filter_mode<int8, 64>::odd(4)
 *                   : aie::filter_mode<int8, 64>::even(2);
 * aie::vector<int8, 128> v = ...;
 * aie::vector<int8, 64> w = aie::filter(v, f);
 * @endcode
 *
 * @tparam T     Element type of the input vectors to be used in the interleave.
 * @tparam Elems Number of elements of the input vectors to be used in the
 *               interleave.
 *
 * @sa interleave_unzip_mode
 * @sa interleave_zip_mode
 * @sa filter(const Vec &v, filter_mode<T, Elems> mode)
 */
template <ElemBaseType T, unsigned Elems>
    requires(arch::is(arch::Gen2))
class filter_mode : public detail::filter_mode<detail::type_bits_v<T>, Elems> {
private:
    static constexpr unsigned type_bits = detail::type_bits_v<T>;

    __aie_inline
    constexpr filter_mode(const detail::filter_mode<type_bits, Elems> &m) noexcept :
        detail::filter_mode<type_bits, Elems>(m)
    {
    }

public:
    /** Copy constructor for class filter_mode */
    constexpr filter_mode(const filter_mode &) = default;

    /** Construts a filter_mode object that specifies the same pattern used in
     * @ref filter_odd
     *
     * @param chunk_size Number of contiguous elements taken from the input
     *                   vector. It must be a power of two.
     *
     * @sa filter_odd
     */
    static constexpr filter_mode odd(unsigned chunk_size) noexcept
    {
        detail::filter_mode<type_bits, Elems> mode{chunk_size, detail::filter_odd_tag{}};
        REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
        REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
        REQUIRES_MSG(type_bits * chunk_size >= 8, "Minimum chunk_size granularity is 8bit");
        return filter_mode(mode);
    }

    /** Construts a filter_mode object that specifies the same pattern used in
     * @ref filter_even
     *
     * @param chunk_size Number of contiguous elements taken from the input
     *                   vector. It must be a power of two.
     *
     * @sa filter_even
     */
    static constexpr filter_mode even(unsigned chunk_size) noexcept
    {
        detail::filter_mode<type_bits, Elems> mode{chunk_size, detail::filter_even_tag{}};
        REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
        REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
        REQUIRES_MSG(type_bits * chunk_size >= 8, "Minimum chunk_size granularity is 8bit");
        return filter_mode(mode);
    }
};

/**
 * @ingroup group_reshape
 *
 * Returns a vector of half the size. The output vector's contents may follow
 * different patterns depending on how @ref filter_mode argument was constructed.
 *
 * @param v    Input vector. It must meet @ref aie::Vector.
 * @param mode Specifies what pattern will be used to pick the contents for the output vector.
 *
 * @sa filter_mode
 * @sa filter_even
 * @sa filter_odd
 */
template <Vector Vec, ElemBaseType T, unsigned Elems>
    requires(std::is_same_v<typename Vec::value_type, T>
             && Vec::size() == Elems)
__aie_inline
auto filter(const Vec &v, filter_mode<T, Elems> mode) -> vector<T, Elems / 2>
{
    return detail::filter<T, Elems, detail::FilterOp::Dynamic>::run(v, mode);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector of half the size whose contents follow the following pattern.
 *
 * @code
 * out = { v[chunk_size:2*chunk_size), v[3*chunk_size:4*chunk_size), ... }
 * @endcode
 *
 * For example
 *
 * @code
 * aie::filter_odd(v, 1) = { v[1], v[3], ... }
 * aie::filter_odd(v, 2) = { v[2], v[3], v[6], v[7], ... }
 * @endcode
 *
 * @param v          Input vector. Must meet @ref aie::Vector.
 * @param chunk_size Number of contiguous elements taken from the input vector. Must be a power of two.
 *
 * @sa filter_mode
 * @sa filter(const Vec &v, filter_mode<T, Elems> mode)
 */
template <Vector Vec>
__aie_inline
auto filter_odd(const Vec &v, unsigned chunk_size = 1) -> vector<typename Vec::value_type, Vec::size() / 2>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(chunk_size > 0, "chunk_size must be greater than 0");
    REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
    REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
    REQUIRES_MSG(detail::type_bits_v<T> * chunk_size >= 8, "Minimum chunk granularity is 8bit");

    return detail::filter<T, Elems, detail::FilterOp::Odd>::run(v, chunk_size);
}

/**
 * @ingroup group_reshape
 *
 * Returns a vector of half the size whose contents follow the following pattern.
 *
 * @code
 * out = { v[0:chunk_size), v[2*chunk_size:3*chunk_size), ... }
 * @endcode
 *
 * For example
 *
 * @code
 * aie::filter_even(v, 1) = { v[0], v[2], ... }
 * aie::filter_even(v, 2) = { v[0], v[1], v[4], v[5], ... }
 * @endcode
 *
 * @param v          Input vector. Must meet @ref aie::Vector.
 * @param chunk_size Number of contiguous elements taken from the input vector. Must be a power of two.
 *
 * @sa filter_mode
 * @sa filter(const Vec &v, filter_mode<T, Elems> mode)
 */
template <Vector Vec>
__aie_inline
auto filter_even(const Vec &v, unsigned chunk_size = 1) -> vector<typename Vec::value_type, Vec::size() / 2>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(chunk_size > 0, "chunk_size must be greater than 0");
    REQUIRES_MSG(chunk_size <= Elems, "chunk_size must be smaller or equal than the vector length");
    REQUIRES_MSG(detail::utils::is_powerof2(chunk_size), "chunk_size must be a power of two");
    REQUIRES_MSG(detail::type_bits_v<T> * chunk_size >= 8, "Minimum chunk granularity is 8bit");

    return detail::filter<T, Elems, detail::FilterOp::Even>::run(v, chunk_size);
}

/**
 * @ingroup group_reshape
 *
 * Return the real component of a complex value
 *
 * @param e Complex input
 */
template <ComplexElem Elem>
__aie_inline
auto real(Elem e) -> detail::remove_complex_t<operand_base_type_t<Elem>>
{
    using T_base = operand_base_type_t<Elem>;
    T_base value = e;
    return value.real;
}

/**
 * @ingroup group_reshape
 *
 * Return the imaginary component of a complex value
 *
 * @param e Complex input
 */
template <ComplexElem Elem>
__aie_inline
auto imag(Elem e) -> detail::remove_complex_t<operand_base_type_t<Elem>>
{
    using T_base = operand_base_type_t<Elem>;
    T_base value = e;
    return value.imag;
}

/**
 * @ingroup group_reshape
 *
 * Gather and return the real components of a complex vector
 *
 * @param v Complex input vector
 */
template <ComplexVector Vec>
__aie_inline
auto real(const Vec &v) -> detail::remove_complex_t<Vec>
{
    using real_elem_t = detail::remove_complex_t<typename Vec::value_type>;
    return filter_even(v.template cast_to<real_elem_t>());
}

/**
 * @ingroup group_reshape
 *
 * Gather and return the imaginary components of a complex vector
 *
 * @param v Complex input vector
 */
template <ComplexVector Vec>
__aie_inline
auto imag(const Vec &v) -> detail::remove_complex_t<Vec>
{
    using real_elem_t = detail::remove_complex_t<typename Vec::value_type>;
    return filter_odd(v.template cast_to<real_elem_t>());
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with the bit-wise AND of the elements of the two input vectors. The vectors must have the same type
 * and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] & v2[i];
 * @endcode
 *
 * \note This is a scalar operation on AIE
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2> &&
                                             (!detail::is_complex_v<typename Vec1::value_type>)  &&
                                             (!detail::is_floating_point_v<typename Vec1::value_type>))
__aie_inline
auto bit_and(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec2::size();

    if (chess_manifest(&v1 == &v2))
        return v1;

    return detail::bit<T, Elems, detail::BitOp::And>::run(v1, v2);
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with the bit-wise AND of a value and all the elements of the input vector. The type of the value
 * and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a & v[i];
 * @endcode
 *
 * \note This is a scalar operation on AIE
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type> &&
                                       (!detail::is_complex_v<typename Vec::value_type>)  &&
                                       (!detail::is_floating_point_v<typename Vec::value_type>))
__aie_inline
auto bit_and(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>) {
        return detail::bit<T, Elems, detail::BitOp::And>::run(vector_elem_const_ref(a), v);
    }
    else {
        if (chess_manifest(a == (T)-1))
            return v;

        return detail::bit<T, Elems, detail::BitOp::And>::run(a, v);
    }
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with the bit-wise OR of the elements of the two input vectors. The vectors must have the same type
 * and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] | v2[i];
 * @endcode
 *
 * \note This is a scalar operation on AIE
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2> &&
                                             (!detail::is_complex_v<typename Vec1::value_type>)  &&
                                             (!detail::is_floating_point_v<typename Vec1::value_type>))
__aie_inline
auto bit_or(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    if (chess_manifest(&v1 == &v2))
        return v1;

    return detail::bit<T, Elems, detail::BitOp::Or>::run(v1, v2);
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with the bit-wise OR of a value and all the elements of the input vector. The type of the value
 * and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a | v[i];
 * @endcode
 *
 * \note This is a scalar operation on AIE
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type> &&
                                       (!detail::is_complex_v<typename Vec::value_type>)  &&
                                       (!detail::is_floating_point_v<typename Vec::value_type>))
__aie_inline
auto bit_or(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>) {
        return detail::bit<T, Elems, detail::BitOp::Or>::run(vector_elem_const_ref(a), v);
    }
    else {
        if (chess_manifest(a == 0))
            return v;

        return detail::bit<T, Elems, detail::BitOp::Or>::run(a, v);
    }
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with the bit-wise XOR of the elements of the two input vectors. The vectors must have the same type
 * and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] ^ v2[i];
 * @endcode
 *
 * \note This is a scalar operation on AIE
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2> &&
                                             (!detail::is_complex_v<typename Vec1::value_type>)  &&
                                             (!detail::is_floating_point_v<typename Vec1::value_type>))
__aie_inline
auto bit_xor(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    return detail::bit<T, Elems, detail::BitOp::Xor>::run(v1, v2);
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with the bit-wise XOR of a value and all the elements of the input vector. The type of the value
 * and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a ^ v[i];
 * @endcode
 *
 * \note This is a scalar operation on AIE
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type> &&
                                       (!detail::is_complex_v<typename Vec::value_type>)  &&
                                       (!detail::is_floating_point_v<typename Vec::value_type>))
__aie_inline
auto bit_xor(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::bit<T, Elems, detail::BitOp::Xor>::run(vector_elem_const_ref(a), v);
    else
        return detail::bit<T, Elems, detail::BitOp::Xor>::run(a, v);
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with the bit-wise NEG of the elements of the input vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = ~v[i];
 * @endcode
 *
 * \note This is a scalar operation on AIE
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec> requires((!detail::is_complex_v<typename Vec::value_type>)  &&
                               (!detail::is_floating_point_v<typename Vec::value_type>))
__aie_inline
auto bit_not(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::bit<T, Elems, detail::BitOp::Not>::run(v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the element-wise addition of the two input vectors. The vectors must have the same type and
 * size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] + v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto add(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    return detail::add<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the element-wise addition of the two input vectors.
 * The behaviour of the addition is dictated by the saturation mode
 * The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] + v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2> && !detail::is_floating_point_v<typename Vec1::value_type>)
__aie_inline
auto saturating_add(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    constexpr bool is_complex = Vec1::is_complex();

    using T = typename Vec1::value_type;
    using U = detail::remove_complex_t<T>;

    constexpr unsigned Elems = is_complex ? 2 * Vec1::size() : Vec1::size();

    accum<detail::accum_tag_for_type<U>, Elems> acc(v1.template cast_to<U>());
    acc = mac(acc, v2.template cast_to<U>(), U(1));

    return acc.template to_vector<U>().template cast_to<T>();
}

/*
 * Returns whether the it can be determined at compile time that the given value is zero
 */
template <typename T, Elem E>
__aie_inline
constexpr bool is_ct_elem_zero(E a)
{
    if constexpr (detail::is_complex_v<T>) {
        if (chess_manifest(a == T(0, 0)))
            return true;
    }
    else {
        if (chess_manifest(a == T(0)))
            return true;
    }

    return false;
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the addition of a value to all the elements of the input vector. The type of the value and
 * the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a + v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto add(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;

    if (is_ct_elem_zero<T>(a))
        return v;

    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::add<T, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::add<T, Elems>::run(a, v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the addition of a value to all the elements of the input vector. The type of the value and
 * the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] + a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param a Value. The type must meet @ref aie::Elem.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<typename Vec::value_type, E>)
__aie_inline
auto add(const Vec &v, E a) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;

    if (is_ct_elem_zero<T>(a))
        return v;

    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::add<T, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::add<T, Elems>::run(v, a);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the addition of a value to all the elements of the input vector.
 * The behaviour of the addition is dictated by the saturation mode
 * The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a + v[i];
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::Elem.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto saturating_add(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if (is_ct_elem_zero<T>(a))
        return v;

    return saturating_add(broadcast<T, Elems>(a), v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the addition of a value to all the elements of the input vector.
 * The behaviour of the addition is dictated by the saturation mode
 * The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] + a;
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 * @param a Value. The type must meet @ref aie::Elem.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto saturating_add(const Vec &v, E a) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if (is_ct_elem_zero<T>(a))
        return v;

    return saturating_add(v, broadcast<T, Elems>(a));
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise addition of the two input accumulators
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc1[i] + acc2[i];
 * @endcode
 *
 * @param acc1 First input accumulator. The type must meet @ref aie::AccumOrOp.
 * @param acc2 Second input accumulator. The type must meet @ref aie::AccumOrOp.
 */
template <AccumOrOp Acc1, AccumOrOp Acc2> requires(arch::is(arch::Gen2) &&
                                                   is_same_accum_v<result_type_t<Acc1>, result_type_t<Acc2>>)
__aie_inline
auto add(const Acc1 &acc1, const Acc2 &acc2) -> result_type_t<Acc1>
{
    if      constexpr (!is_op_v<Acc1>) {
        return add(op_none(acc1), acc2);
    }
    else if constexpr (!is_op_v<Acc2>) {
        return add(acc1, op_none(acc2));
    }
    else {
        if      constexpr (Acc1::is_operation_not(Operation::Zero))
            return add(acc1(), acc2);
        else if constexpr (!Acc2::is_operation_none())
            return add(acc1, acc2());
        else {
            using acc_type = typename Acc1::result_type;

            bool zero_acc1 = false;

            if constexpr (Acc1::is_operation(Operation::Zero))
                zero_acc1 = acc1.parent2();

            return detail::add_accum<typename acc_type::value_type, acc_type::size()>::run(acc1.parent1(), zero_acc1, acc2.parent1());
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise addition of the input accumulator and vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] + v[i];
 * @endcode
 *
 * @param acc Input accumulator
 * @param v Input vector
 */
// TODO: add requirements to check for Acc/Vec compatibility
template <AccumOrOp Acc, Vector Vec> requires(Acc::size() == Vec::size())
__aie_inline
auto add(const Acc &acc, const Vec &v) -> result_type_t<Acc>
{
    using T = typename Vec::value_type;

    if     constexpr (!is_op_v<Acc>) {
        return add(op_none(acc), v);
    }
    else {
#if __AIE_ARCH__ == 10
        // AIE1 does not have support for dynamic zeroization
        if constexpr (!Acc::is_operation_none())
            return add(acc(), v);
#else
        if constexpr (Acc::is_operation_not(Operation::Zero))
            return add(acc(), v);
#endif
        else {
            using acc_type = typename Acc::result_type;

            bool zero_acc = false;

            if constexpr (Acc::is_operation(Operation::Zero))
                zero_acc = acc.parent2();

            return detail::add_accum_vector<acc_type::accum_bits(), T, acc_type::size()>::run(acc.parent1(), zero_acc, v);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the addition of a value to all the elements of the input vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] + a;
 * @endcode
 *
 * @param acc Input accumulator
 * @param a   Value. The type must meet @ref aie::Elem.
 */
// TODO: add requirements to check for Acc/E compatibility
template <AccumOrOp Acc, Elem E>
__aie_inline
auto add(const Acc &acc, E a) -> result_type_t<Acc>
{
    using T = operand_base_type_t<E>;

    if (is_ct_elem_zero<T>(a)) {
        if constexpr (is_op_v<Acc>) return acc();
        else                        return acc;
    }

    if     constexpr (!is_op_v<Acc>) {
        return add(op_none(acc), a);
    }
    else {
#if __AIE_ARCH__ == 10
        // AIE1 does not have support for dynamic zeroization
        if constexpr (!Acc::is_operation_none())
            return add(acc(), a);
#else
        if constexpr (Acc::is_operation_not(Operation::Zero))
            return add(acc(), a);
#endif
        else {
            using acc_type = typename Acc::result_type;

            bool zero_acc = false;

            if constexpr (Acc::is_operation(Operation::Zero))
                zero_acc = acc.parent2();

            if constexpr (detail::is_vector_elem_ref_v<E>)
                return detail::add_accum_vector<acc_type::accum_bits(), T, acc_type::size()>::run(acc.parent1(), zero_acc, vector_elem_const_ref(a));
            else
                return detail::add_accum_vector<acc_type::accum_bits(), T, acc_type::size()>::run(acc.parent1(), zero_acc, a);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Compute the absolute square of each element in the given complex vector.
 *
 * For a vector of N complex elements, returns a vector of N integers
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i].real² + v[i].imag²;
 * @endcode
 *
 * @param v     Input complex vector
 * @param shift Optional parameter to control the shift value used in the srs internally
 *
 * @tparam TR Type of returned integer vector, default to int32 but can be set to int16
 */
template <typename TR = int32, ComplexIntegralVector Vec> requires(Utils::is_one_of_v<TR, int32, int16>)
__aie_inline
constexpr auto abs_square(const Vec &v, int shift = 0)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::abs_square<T, TR, Elems>::run(v, shift);
}

/**
 * @ingroup group_arithmetic
 *
 * Compute the absolute square of each element in the given complex vector.
 *
 * For a vector of N complex elements, returns a vector of N integers
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i].real² + v[i].imag²;
 * @endcode
 *
 * @param v     Input complex vector
 */
template <typename TR = float, ComplexFloatVector Vec>
    requires (arch::is(arch::AIE) || (arch::is(arch::AIE_ML, arch::AIE_MLv2) && __AIE_API_COMPLEX_FP32_EMULATION__ == 1))
__aie_inline
constexpr auto abs_square(const Vec &v)
{
    return detail::abs_square<typename Vec::value_type, TR, Vec::size()>::run(v, 0);
}

/**
 * @ingroup group_elementary
 *
 * Compute the absolute square of a scalar complex value
 *
 * \f$ f(a + b i) = a^2 + b^2 \f$
 *
 * @param v     Input complex scalar
 */
template <ComplexElem E>
__aie_inline
constexpr auto abs_square(const E &v) -> detail::remove_complex_t<operand_base_type_t<E>>
{
    using T_base = operand_base_type_t<E>;
    T_base value = v;
    return value.real * value.real + value.imag * value.imag;
}

/**
 * @ingroup group_arithmetic
 *
 * Compute the absolute value for each element in the given vector.
 *
 * @param v Input vector
 */
template <Vector Vec>
__aie_inline
constexpr auto abs(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_signed_v<T>)
        return detail::abs<T, Elems>::run(v);

    return v;
}

/**
 * @ingroup group_arithmetic
 *
 * Compute the absolute value of a value.
 *
 * @param a Input value
 */
template <Elem E>
__aie_inline
constexpr auto abs(const E &a) -> operand_base_type_t<E>
{
    using base_t = operand_base_type_t<E>;
    const base_t _a = a;

    if constexpr (detail::is_signed_v<base_t>)
        return (base_t)std::abs(_a);

    return _a;
}

static constexpr int max_shift = detail::max_shift;
static constexpr int min_shift = detail::min_shift;

/**
 * @ingroup group_bit
 *
 * Returns a vector with all values upshifted by specified number of bits
 *
 * @param v Input vector
 * @param shift Number of bits to upshift by
 */
template <Vector Vec>
__aie_inline
auto upshift(const Vec &v, unsigned shift) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(shift <= detail::max_shift, "Shift value cannot exceed aie::max_shift");

    return detail::shift<T, Elems>::run(v, shift, 0);
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with all values arithmetically downshifted by specified number of bits
 *
 * @param v Input vector
 * @param shift Number of bits to downshift by
 */
template <Vector Vec>
__aie_inline
auto downshift(const Vec &v, unsigned shift) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    REQUIRES_MSG(shift <= detail::max_shift, "Shift value cannot exceed aie::max_shift");

    scoped_mode rnd_mode{rounding_mode::floor};

    return detail::shift<T, Elems>::run(v, 0, shift);
}

/**
 * @ingroup group_bit
 *
 * Returns a vector with all values logically downshifted by specified number of bits
 *
 * @param v Input vector
 * @param shift Number of bits to downshift by
 */
template <Vector Vec>
__aie_inline
auto logical_downshift(const Vec &v, unsigned shift) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;

    REQUIRES_MSG(shift <= detail::max_shift, "Shift value cannot exceed aie::max_shift");

    constexpr unsigned RealElems = Vec::size() * (Vec::is_complex() ? 2 : 1);
    using RealT = detail::remove_complex_t<T>;

#if __AIE_ARCH__ == 10
    using RealVec = vector<RealT, RealElems>;

    constexpr RealT top_bit  = RealT(1 << (detail::type_bits_v<RealT> - 1));
              RealT top_bits = top_bit >> shift;

    // AIE1 Does not support unsigned types so we emulate the logical downshift
    // by running an arithmetic shift and zeroing the upper shift bits
    RealVec tmp  = detail::shift<RealT, RealElems>::run(v.template cast_to<RealT>(), 0, shift);
    RealVec zero = zeros<RealT, RealElems>();
    return detail::sub<RealT, RealElems>::run(tmp.template cast_to<RealT>(), select(zero, top_bits, lt(tmp, zero))).template cast_to<T>();
#else
    scoped_mode rnd_mode{rounding_mode::floor};

    using U  = typename detail::utils::make_unsigned_t<RealT>;

    return detail::shift<U, RealElems>::run(v.template cast_to<U>(), 0, shift).template cast_to<T>();
#endif
}

/**
 * @ingroup group_arithmetic
 *
 * For vectors with signed types, return a vector whose elements are the same as in the given vector but with the sign
 * flipped. If the input type is unsigned, the input vector is returned.
 *
 * @param v Input vector
 */
template <Vector Vec>
__aie_inline
constexpr auto neg(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_signed_v<T>)
        return detail::neg<T, Elems>::run(v);

    return v;
}

/**
 * @ingroup group_arithmetic
 *
 * For values with signed types, return the input value with the sign flipped. If the input type is unsigned, the input
 * value is returned.
 *
 * @param a Input value
 */
template <Elem E>
__aie_inline
constexpr auto neg(const E &a) -> operand_base_type_t<E>
{
    using base_t = operand_base_type_t<E>;
    const base_t _a = a;

    if constexpr (detail::is_signed_v<base_t>)
        return (base_t)-_a;

    return _a;
}

/**
 * @ingroup group_arithmetic
 *
 * For accumulators, return an accumulator whose elements are the same as in the given accumulator but with the sign
 * flipped.
 *
 * @param acc Input accumulator
 */
template <Accum Acc>
    requires (arch::is(arch::Gen2))
__aie_inline
constexpr auto neg(const Acc &acc) -> aie_dm_resource_remove_t<Acc>
{
    using T = typename Acc::value_type;
    constexpr unsigned Elems = Acc::size();

    return detail::neg_acc<Acc::accum_bits(), T, Elems>::run(acc);

}

/**
 * @ingroup group_arithmetic
 *
 * Compute the conjugate for each element in the given vector of complex elements.
 *
 * @param v Input vector
 */
template <ComplexVector Vec>
__aie_inline
auto conj(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::conj<T, Elems>::run(v);
}

/**
 * @ingroup group_arithmetic
 *
 * Compute the conjugate in the given complex value.
 *
 * @param a Input value
 */
template <ComplexElem E>
__aie_inline
constexpr auto conj(const E &a) -> operand_base_type_t<E>
{
    using base_t = operand_base_type_t<E>;
    const base_t _a = a;

    return base_t{ _a.real, (decltype(base_t::imag)) -_a.imag };
}

/**
 * @ingroup group_reduce
 *
 * Returns the sum of all input vector elements. Higher precision intermediate values may be used to perform this
 * reduction.
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec>
__aie_inline
typename Vec::value_type reduce_add(const Vec &v)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::add_reduce<T, Elems>::run(v);
}

template <ElemBaseType T, unsigned Elems>
__aie_inline
T reduce_add(const T &total, const aie::vector<T, Elems> &v)
{
    return detail::add_reduce<T, Elems>::run(total, v);
}

/**
 * @ingroup group_reduce
 *
 * Returns the sum of all input vector elements. This overload is useful when the result must be converted to a lower
 * precision format and the input values are already available in an accumulator (for example, as a result of a previous
 * operation such as a multiplication).
 *
 * @param acc  Input accumulator. The type must meet @ref aie::Accum.
 * @tparam Acc Input accumulator type.
 * @tparam T   Result type after reduction. The type must be compatible with the input accumulator.
 *
 * Example:
 * @code
 * aie::accum<accfloat, 16> acc = ...;
 * bfloat16 result = aie::reduce_add<bfloat16>(acc);
 * @endcode
 */
template <ElemBaseType T, Accum Acc>
__aie_inline
T reduce_add(const Acc &acc)
    requires(Acc::value_class() == detail::accum_class_for_type_v<T>)
{
    constexpr unsigned Elems = Acc::size();

    return detail::add_reduce<T, Elems>::run(acc);
}

template <Vector Vec>
[[deprecated("Use reduce_add instead")]]
__aie_inline
typename Vec::value_type add_reduce(const Vec &v)
{
    return reduce_add(v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns the sums of the elements in the input vectors. The sum of each input vector is stored in an element of the
 * output vector.
 *
 * @param v      First input vector. The type must meet @ref aie::Vector.
 * @param others Remaining input vectors. The type must meet @ref aie::Vector.
 */
template <Vector Vec, Vector... Others>
__aie_inline
auto reduce_add_v(const Vec &v, const Others &... others) -> aie_dm_resource_remove_t<Vec>
{
    static_assert((is_same_vector_v<Vec, Others> && ...));
    static_assert(sizeof...(Others) < 4, "Only 4 vectors can be reduced in parallel");

    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::add_reduce_v<T, Elems>::run(v, others...);
}

template <Vector Vec, Vector... Others>
__aie_inline
auto add_reduce_v(const Vec &v, const Others &... others) -> aie_dm_resource_remove_t<Vec>
{
    return reduce_add_v(v, others...);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the element-wise subtraction of the two input vectors. The vectors must have the same type and
 * size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] - v2[i];
 * @endcode
 *
 * @param v1 First input vector (minuend). The type must meet @ref aie::Vector.
 * @param v2 Second input vector (subtrahend). The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2>)
__aie_inline
auto sub(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    using T = typename Vec1::value_type;
    constexpr unsigned Elems = Vec1::size();

    return detail::sub<T, Elems>::run(v1, v2);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the element-wise subtraction of the two input vectors.
 * The behaviour of the subtraction is dictated by the saturation mode
 * The vectors must have the same type and size.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v1[i] - v2[i];
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::Vector.
 * @param v2 Second input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec1, Vector Vec2> requires(is_same_vector_v<Vec1, Vec2> && !detail::is_floating_point_v<typename Vec1::value_type>)
__aie_inline
auto saturating_sub(const Vec1 &v1, const Vec2 &v2) -> aie_dm_resource_remove_t<Vec1>
{
    constexpr bool is_complex = Vec1::is_complex();

    using T = typename Vec1::value_type;
    using U = detail::remove_complex_t<T>;

    constexpr unsigned Elems = is_complex ? 2 * Vec1::size() : Vec1::size();

    accum<detail::accum_tag_for_type<U>, Elems> acc(v1.template cast_to<U>());
    acc = msc(acc, v2.template cast_to<U>(), U(1));

    return acc.template to_vector<U>().template cast_to<T>();
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the subtraction of a value and all the elements of the input vector. The type of the value and
 * the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a - v[i];
 * @endcode
 *
 * @param a Value (minuend). The type must meet @ref aie::Elem.
 * @param v Input vector (subtrahend). The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto sub(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::sub<T, Elems>::run(vector_elem_const_ref(a), v);
    else
        return detail::sub<T, Elems>::run(a, v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the subtraction of the elements of the input vector and a value. The type of the value and
 * the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] - a;
 * @endcode
 *
 * @param v Input vector (subtrahend). The type must meet @ref aie::Vector.
 * @param a Value (minuend). The type must meet @ref aie::Elem.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto sub(const Vec &v, E a) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;

    if (is_ct_elem_zero<T>(a))
        return v;

    constexpr unsigned Elems = Vec::size();

    if constexpr (detail::is_vector_elem_ref_v<E>)
        return detail::sub<T, Elems>::run(v, vector_elem_const_ref(a));
    else
        return detail::sub<T, Elems>::run(v, a);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the subtraction of a value to all the elements of the input vector.
 * The behaviour of the subtraction is dictated by the saturation mode
 * The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a - v[i];
 * @endcode
 *
 * @param a Value (minuend). The type must meet @ref aie::Elem.
 * @param v Input vector (subtrahend). The type must meet @ref aie::Vector.
 */
template <Elem E, Vector Vec> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto saturating_sub(E a, const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return saturating_sub(broadcast<T, Elems>(a), v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns a vector with the subtraction of the elements of the input vector and a value.
 * The behaviour of the subtraction is dictated by the saturation mode
 * The type of the value and the type of the vector elements must be the same.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] - a;
 * @endcode
 *
 * @param v Input vector (subtrahend). The type must meet @ref aie::Vector.
 * @param a Value (minuend). The type must meet @ref aie::Elem.
 */
template <Vector Vec, Elem E> requires(is_valid_elem_op_v<E, typename Vec::value_type>)
__aie_inline
auto saturating_sub(const Vec &v, E a) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if (is_ct_elem_zero<T>(a))
        return v;

    return saturating_sub(v, broadcast<T, Elems>(a));
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise subtraction of the two input accumulators
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc1[i] - acc2[i];
 * @endcode
 *
 * @param acc1 First input accumulator. The type must meet @ref aie::AccumOrOp.
 * @param acc2 Second input accumulator. The type must meet @ref aie::AccumOrOp.
 */
template <AccumOrOp Acc1, AccumOrOp Acc2> requires(arch::is(arch::Gen2) &&
                                                   is_same_accum_v<result_type_t<Acc1>, result_type_t<Acc2>>)
__aie_inline
auto sub(const Acc1 &acc1, const Acc2 &acc2) -> result_type_t<Acc1>
{
    if      constexpr (!is_op_v<Acc1>) {
        return sub(op_none(acc1), acc2);
    }
    else if constexpr (!is_op_v<Acc2>) {
        return sub(acc1, op_none(acc2));
    }
    else {
        if      constexpr (Acc1::is_operation_not(Operation::Zero))
            return sub(acc1(), acc2);
        else if constexpr (!Acc2::is_operation_none())
            return sub(acc1, acc2());
        else {
            using acc_type = typename Acc1::result_type;

            bool zero_acc1 = false;

            if constexpr (Acc1::is_operation(Operation::Zero))
                zero_acc1 = acc1.parent2();

            return detail::sub_accum<typename acc_type::value_type, acc_type::size()>::run(acc1.parent1(), zero_acc1, acc2.parent1());
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise subtraction of the input accumulator and vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] - v[i];
 * @endcode
 *
 * @param acc Input accumulator (minuend)
 * @param v Input vector (subtrahend)
 */
// TODO: add requirements to check for Acc/Vec compatibility
template <AccumOrOp Acc, Vector Vec> requires(Acc::size() == Vec::size())
__aie_inline
constexpr auto sub(const Acc &acc, const Vec &v) -> result_type_t<Acc>
{
    using T = typename Vec::value_type;

    if     constexpr (!is_op_v<Acc>) {
        return sub(op_none(acc), v);
    }
    else {
#if __AIE_ARCH__ == 10
        // AIE1 does not have support for dynamic zeroization
        if constexpr (!Acc::is_operation_none())
            return sub(acc(), v);
#else
        if constexpr (Acc::is_operation_not(Operation::Zero))
            return sub(acc(), v);
#endif
        else {
            using acc_type = typename Acc::result_type;

            bool zero_acc = false;

            if constexpr (Acc::is_operation(Operation::Zero))
                zero_acc = acc.parent2();

            return detail::sub_accum_vector<acc_type::accum_bits(), T, acc_type::size()>::run(acc.parent1(), zero_acc, v);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the subtraction of all the elements of the input accumulator and a value.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] - a;
 * @endcode
 *
 * @param acc Input accumulator (minuend)
 * @param a   Value (subtrahend). The type must meet @ref aie::Elem.
 */
// TODO: add requirements to check for Acc/E compatibility
template <AccumOrOp Acc, Elem E>
__aie_inline
auto sub(const Acc &acc, E a) -> result_type_t<Acc>
{
    using T = operand_base_type_t<E>;

    if (is_ct_elem_zero<T>(a)) {
        if constexpr (is_op_v<Acc>) return acc();
        else                        return acc;
    }

    if     constexpr (!is_op_v<Acc>) {
        return sub(op_none(acc), a);
    }
    else {
#if __AIE_ARCH__ == 10
        // AIE1 does not have support for dynamic zeroization
        if constexpr (!Acc::is_operation_none())
            return sub(acc(), a);
#else
        if constexpr (Acc::is_operation_not(Operation::Zero))
            return sub(acc(), a);
#endif
        else {
            using acc_type = typename Acc::result_type;

            bool zero_acc = false;

            if constexpr (Acc::is_operation(Operation::Zero))
                zero_acc = acc.parent2();

            if constexpr (detail::is_vector_elem_ref_v<E>)
                return detail::sub_accum_vector<acc_type::accum_bits(), T, acc_type::size()>::run(acc.parent1(), zero_acc, vector_elem_const_ref(a));
            else
                return detail::sub_accum_vector<acc_type::accum_bits(), T, acc_type::size()>::run(acc.parent1(), zero_acc, a);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator of the requested type with the element-wise multiplication of the two input vectors. The
 * vectors must have the same size and compatible types. Vectors can also be lazy operations (see @ref
 * group_basic_types_ops). On some AIE architectures certain operations can be collapsed with the multiplication into
 * a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = op1(v1[i]) * op2(v2[i]);
 * @endcode
 *
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with
 *                  the vector types (real/complex). Refer to \ref DefaultAccumTag "default accumulator tag" table for
 *                  the minimum accumulator element tag required for each combination of input element types.
 *
 * @param v1 First input vector. The type must meet @ref aie::VectorOrOp.
 * @param v2 Second input vector. The type must meet @ref aie::VectorOrOp.
 */

template <AccumElemBaseType AccumTag, VectorOrOp Vec1, VectorOrOp Vec2>
    requires(is_valid_size_v<Vec1, Vec2> &&
                     is_valid_mul_op_v<typename Vec1::value_type, typename Vec2::value_type>)
__aie_inline
constexpr auto mul(const Vec1 &v1, const Vec2 &v2) -> accum<AccumTag, Vec1::size()>
{
    if      constexpr (!is_op_v<Vec1>) {
        return mul<AccumTag>(op_none(v1), v2);
    }
    else if constexpr (!is_op_v<Vec2>) {
        return mul<AccumTag>(v1, op_none(v2));
    }
    else {
        using T1 = typename Vec1::value_type;
        using T2 = typename Vec2::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec1::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return mul<AccumTag>(v1(), v2);
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Abs, Operation::Max, Operation::Min))
            return mul<AccumTag>(v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj))
            return mul<AccumTag>(v1, v2());
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && Vec1::is_operation_not(Operation::Neg))
            return mul<AccumTag>(v1(), v2);
        else if constexpr (detail::is_floating_point_v<T2> && Vec2::is_operation_not(Operation::Neg))
            return mul<AccumTag>(v1, v2());
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mul<AccumTag>(v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mul<AccumTag>(v1, v2());
#endif
        else {
            constexpr Operation Op1 = detail::evaluate_mul_operation<Vec1>();
            constexpr Operation Op2 = detail::evaluate_mul_operation<Vec2>();

            constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();
            constexpr auto op = detail::to_mul_macro_op<Op1, Op2>();

            bool sub_mul = detail::eval_sub_mul(v1, v2);

            if constexpr (Vec1::is_operation(Operation::Min, Operation::Max))
                return detail::mul_maxmin<op, accum_bits, T1, T2>::run(v1.parent1(), v1.parent2(), v2.parent1());
            else
                return detail::mul<op, accum_bits, T1, T2>::run(v1.parent1(), detail::get_mul_sign(v1),
                                                                v2.parent1(), detail::get_mul_sign(v2),
                                                                sub_mul);
        }
    }
}


/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator of the requested type with the element-wise multiplication of a value and all the elements
 * of the input vector. The type of the value and the type of the vector elements must be compatible. Operands
 * can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain operations can be
 * collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = op1(a) * op2(v[i]);
 * @endcode
 *
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with
 *                  the vector types (real/complex). Refer to \ref DefaultAccumTag "default accumulator tag" table for
 *                  the minimum accumulator element tag required for each combination of input element types.
 *
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 */
template <AccumElemBaseType AccumTag, ElemOrOp E, VectorOrOp Vec>
    requires(is_valid_size_v<Vec, E> && is_valid_mul_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr auto mul(E a, const Vec &v) -> accum<AccumTag, Vec::size()>
{
    if      constexpr (!is_op_v<E>) {
        return mul<AccumTag>(op_none(a), v);
    }
    else if constexpr (!is_op_v<Vec>) {
        return mul<AccumTag>(a, op_none(v));
    }
    else {
        using T1 = operand_base_type_t<typename E::value_type>;
        using T2 = typename Vec::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (E::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return mul<AccumTag>(a(), v);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Abs))
            return mul<AccumTag>(a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj))
            return mul<AccumTag>(a, v());
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && E::is_operation_not(Operation::Neg))
            return mul<AccumTag>(a(), v);
        else if constexpr (detail::is_floating_point_v<T2> && Vec::is_operation_not(Operation::Neg))
            return mul<AccumTag>(a, v());
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mul<AccumTag>(a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mul<AccumTag>(a, v());
#endif
        else {
            constexpr Operation Op1 = detail::evaluate_mul_operation<E>();
            constexpr Operation Op2 = detail::evaluate_mul_operation<Vec>();

            constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();
            constexpr auto op = detail::to_mul_macro_op<Op1, Op2>();

            bool sub_mul = detail::eval_sub_mul(a, v);

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>)
                return detail::mul<op, accum_bits, T1, T2>::run(vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                                v.parent1(),                        detail::get_mul_sign(v),
                                                                sub_mul);
            else
                return detail::mul<op, accum_bits, T1, T2>::run(a.parent1(), detail::get_mul_sign(a),
                                                                v.parent1(), detail::get_mul_sign(v),
                                                                sub_mul);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator of the requested type with the element-wise multiplication of a value and all the elements
 * of the input vector. The type of the value and the type of the vector elements must be compatible. Operands
 * can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain operations can be
 * collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = op1(v[i]) * op2(a);
 * @endcode
 *
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with
 *                  the vector types (real/complex). Refer to \ref DefaultAccumTag "default accumulator tag" table for
 *                  the minimum accumulator element tag required for each combination of input element types.
 *
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 */
template <AccumElemBaseType AccumTag, VectorOrOp Vec, ElemOrOp E>
    requires(is_valid_size_v<Vec, E> && is_valid_mul_op_v<typename Vec::value_type, E>)
__aie_inline
constexpr auto mul(const Vec &v, E a) -> accum<AccumTag, Vec::size()>
{
    if constexpr (!is_op_v<Vec>) {
        return mul<AccumTag>(op_none(v), a);
    }
    else if constexpr (!is_op_v<E>) {
        return mul<AccumTag>(v, op_none(a));
    }
    else {
        using T1 = typename Vec::value_type;
        using T2 = operand_base_type_t<typename E::value_type>;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return mul<AccumTag>(v(), a);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Abs))
            return mul<AccumTag>(v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj))
            return mul<AccumTag>(v, a());
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && Vec::is_operation_not(Operation::Neg))
            return mul<AccumTag>(v(), a);
        else if constexpr (detail::is_floating_point_v<T2> && E::is_operation_not(Operation::Neg))
            return mul<AccumTag>(v, a());
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mul<AccumTag>(v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mul<AccumTag>(v, a());
#endif
        else {
            constexpr Operation Op1 = detail::evaluate_mul_operation<Vec>();
            constexpr Operation Op2 = detail::evaluate_mul_operation<E>();

            constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();
            constexpr auto op = detail::to_mul_macro_op<Op1, Op2>();

            bool sub_mul = detail::eval_sub_mul(v, a);

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>)
                return detail::mul<op, accum_bits, T1, T2>::run(v.parent1(),                        detail::get_mul_sign(v),
                                                                vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                                sub_mul);
            else
                return detail::mul<op, accum_bits, T1, T2>::run(v.parent1(), detail::get_mul_sign(v),
                                                                a.parent1(), detail::get_mul_sign(a),
                                                                sub_mul);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise multiplication of the two input vectors. The default accumulator accuracy
 * is used. The vectors must have the same size and compatible types. Operands can also be lazy operations (see @ref
 * group_basic_types_ops). On some AIE architectures certain operations can be collapsed with the multiplication into
 * a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = op1(v1[i]) * op2(v2[i]);
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::VectorOrOp.
 * @param v2 Second input vector. The type must meet @ref aie::VectorOrOp.
 */
// TODO: document default accumulation
// TODO: document compatible types.
template <VectorOrOp Vec1, VectorOrOp Vec2>
    requires(is_valid_size_v<Vec1, Vec2> && is_valid_mul_op_v<typename Vec1::value_type, typename Vec2::value_type>)
__aie_inline
constexpr auto mul(const Vec1 &v1, const Vec2 &v2)
{
    using T1 = typename Vec1::value_type;
    using T2 = typename Vec2::value_type;
    using T3 = detail::default_accum_tag_t<T1, T2>;

    return mul<T3>(v1, v2);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise multiplication of a value and all the elements of a vector. The default
 * accumulator accuracy is used of the input vector. The type of the value and the type of the vector elements must be
 * compatible. Operands can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain
 * operations can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = op1(a) * op2(v[i]);
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 */
template <ElemOrOp E, VectorOrOp Vec>
    requires(is_valid_size_v<E, Vec> && is_valid_mul_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr auto mul(E a, const Vec &v)
{
    using T1 = operand_base_type_t<E>;
    using T2 = typename Vec::value_type;
    using T3 = detail::default_accum_tag_t<T1, T2>;

    return mul<T3>(a, v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise multiplication of a value and all the elements of a vector. The default
 * accumulator accuracy is used of the input vector. The type of the value and the type of the vector elements must be
 * compatible. Operands can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain
 * operations can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = op1(v[i]) * op2(a);
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 */
template <VectorOrOp Vec, ElemOrOp E>
    requires(is_valid_size_v<E, Vec> && is_valid_mul_op_v<typename Vec::value_type, E>)
__aie_inline
constexpr auto mul(const Vec &v, E a)
{
    using T1 = typename Vec::value_type;
    using T2 = operand_base_type_t<E>;
    using T3 = detail::default_accum_tag_t<T1, T2>;

    return mul<T3>(v, a);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise multiply-add of the two input vectors and accumulator. The vectors and
 * the accumulator must have the same size and their types must be compatible. Operands can also be lazy operations
 * (see @ref group_basic_types_ops). On some AIE architectures certain operations can be collapsed with the
 * multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] + op1(v1[i]) * op2(v2[i]);
 * @endcode
 *
 * @param acc Accumulator to which the result of the multiplication is added (or subtracted). The type must meet
 *            AccumOrOp.
 * @param v1  First input vector. The type must meet @ref aie::VectorOrOp.
 * @param v2  Second input vector. The type must meet @ref aie::VectorOrOp.
 */
// TODO: document compatible types.
template <AccumOrOp Acc, VectorOrOp Vec1, VectorOrOp Vec2>
    requires(is_valid_size_v<Acc, Vec1, Vec2> && is_valid_mul_op_v<typename Vec1::value_type, typename Vec2::value_type>)
__aie_inline
constexpr auto mac(const Acc &acc, const Vec1 &v1, const Vec2 &v2) -> operand_base_type_t<Acc>
{
    if constexpr (!is_op_v<Acc>) {
        return mac(op_add(acc), v1, v2);
    }
    else if constexpr (!is_op_v<Vec1>) {
        return mac(acc, op_none(v1), v2);
    }
    else if constexpr (!is_op_v<Vec2>) {
        return mac(acc, v1, op_none(v2));
    }
    else {
        using T1 = typename Vec1::value_type;
        using T2 = typename Vec2::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec1::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return mac(acc, v1(), v2);
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Abs, Operation::Max, Operation::Min))
            return mac(acc, v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj))
            return mac(acc, v1, v2());
        else if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg))
            return mac(acc(), v1, v2);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && Vec1::is_operation_not(Operation::Neg))
            return mac(acc, v1(), v2);
        else if constexpr (detail::is_floating_point_v<T2> && Vec2::is_operation_not(Operation::Neg))
            return mac(acc, v1, v2());
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mac(acc, v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mac(acc, v1, v2());
        //FIXME: CRVO-10673 cfloat zeroization
        else if constexpr (Acc::is_operation(Operation::Zero) && (std::is_same_v<T1, cfloat> || std::is_same_v<T2, cfloat>))
            return mac(acc(), v1, v2);
#endif
        else {
            constexpr Operation Op1     = detail::evaluate_mul_operation<Vec1>();
            constexpr Operation Op2     = detail::evaluate_mul_operation<Vec2>();
            constexpr unsigned acc_bits = Acc::result_type::accum_bits();

            if constexpr (Vec1::is_operation(Operation::Min, Operation::Max)) {
                constexpr auto op = detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
                return detail::mul_maxmin<op, acc_bits, T1, T2>::run(v1.parent1(), v1.parent2(),
                                                                     v2.parent1(), acc.parent1());
            }
            else {
                bool zero_acc = false;
                bool sub_mul  = detail::eval_sub_mul(v1, v2);
                bool sub_acc  = false;

                if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg)) {
                    constexpr auto op = detail::to_mul_macro_op<Operation::Acc_Add, Op1, Op2>();
                    if constexpr (Acc::is_operation(Operation::Zero)) zero_acc = acc.parent2();
                    if constexpr (Acc::is_operation(Operation::Neg))  sub_acc  = acc.parent2();

                    return detail::mul<op, acc_bits, T1, T2>::run(v1.parent1(), detail::get_mul_sign(v1),
                                                                  v2.parent1(), detail::get_mul_sign(v2),
                                                                  sub_mul, sub_acc,
                                                                  zero_acc, acc.parent1());
                }
                else {
                    constexpr auto op = detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
                    return detail::mul<op, acc_bits, T1, T2>::run(v1.parent1(), detail::get_mul_sign(v1),
                                                                  v2.parent1(), detail::get_mul_sign(v2),
                                                                  sub_mul, sub_acc,
                                                                  zero_acc, acc.parent1());
                }
            }
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise multiply-add of value, input vector and accumulator. The vector and
 * the accumulator must have the same size and the types of all operands must be compatible. Operands can also be lazy
 * operations (see @ref group_basic_types_ops). On some AIE architectures certain operations can be collapsed with the
 * multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] + op1(a) * op2(v[i]);
 * @endcode
 *
 * @param acc Accumulator to which the result of the multiplication is added (or subtracted). The type must meet
 *            AccumOrOp.
 * @param a   Input value. The type must meet @ref aie::ElemOrOp.
 * @param v   Input vector. The type must meet @ref aie::VectorOrOp.
 */
// TODO: document compatible types.
template <AccumOrOp Acc, ElemOrOp E, VectorOrOp Vec>
    requires(is_valid_size_v<Acc, E, Vec> && is_valid_mul_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr auto mac(const Acc &acc, E a, const Vec &v) -> operand_base_type_t<Acc>
{
    if      constexpr (!is_op_v<Acc>) {
        return mac(op_add(acc), a, v);
    }
    else if constexpr (!is_op_v<E>) {
        return mac(acc, op_none(a), v);
    }
    else if constexpr (!is_op_v<Vec>) {
        return mac(acc, a, op_none(v));
    }
    else {
        using T1 = operand_base_type_t<typename E::value_type>;
        using T2 = typename Vec::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (E::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return mac(acc, a(), v);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Abs))
            return mac(acc, a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj))
            return mac(acc, a, v());
        else if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg))
            return mac(acc(), a, v);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && E::is_operation_not(Operation::Neg))
            return mac(acc, a(), v);
        else if constexpr (detail::is_floating_point_v<T2> && Vec::is_operation_not(Operation::Neg))
            return mac(acc, a, v());
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mac(acc, a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mac(acc, a, v());
        //FIXME: CRVO-10673 cfloat zeroization
        else if constexpr (Acc::is_operation(Operation::Zero) && (std::is_same_v<T1, cfloat> || std::is_same_v<T2, cfloat>))
            return mac(acc(), a, v);
#endif
        else {
            constexpr Operation Op1     = detail::evaluate_mul_operation<E>();
            constexpr Operation Op2     = detail::evaluate_mul_operation<Vec>();
            constexpr unsigned acc_bits = Acc::result_type::accum_bits();

            bool zero_acc = false;
            bool sub_mul  = detail::eval_sub_mul(a, v);
            bool sub_acc  = false;

            if constexpr (Acc::is_operation(Operation::Zero)) zero_acc = acc.parent2();
            if constexpr (Acc::is_operation(Operation::Neg))  sub_acc  = acc.parent2();

            constexpr auto op = [&](){
                if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg)) return detail::to_mul_macro_op<Operation::Acc_Add, Op1, Op2>();
                else                                                              return detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
            }();

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>) {
                return detail::mul<op, acc_bits, T1, T2>::run(vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                                                    v.parent1(),  detail::get_mul_sign(v),
                                                                                    sub_mul, sub_acc,
                                                                                    zero_acc, acc.parent1());
            }
            else {
                return detail::mul<op, acc_bits, T1, T2>::run(a.parent1(), detail::get_mul_sign(a),
                                                              v.parent1(), detail::get_mul_sign(v),
                                                              sub_mul, sub_acc,
                                                              zero_acc, acc.parent1());
            }
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise multiply-add of input vector, value and accumulator. The vector and
 * the accumulator must have the same size and the types of all operands must be compatible. Operands can also be lazy
 * operations (see @ref group_basic_types_ops). On some AIE architectures certain operations can be collapsed with the
 * multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] + op1(v[i]) * op2(a);
 * @endcode
 *
 * @param acc Accumulator to which the result of the multiplication is added (or subtracted). The type must meet
 *            AccumOrOp.
 * @param v   Input vector. The type must meet @ref aie::VectorOrOp.
 * @param a   Input value. The type must meet @ref aie::ElemOrOp.
 */
// TODO: document compatible types.
template <AccumOrOp Acc, VectorOrOp Vec, ElemOrOp E>
    requires(is_valid_size_v<Acc, Vec, E> && is_valid_mul_op_v<typename Vec::value_type, E>)
__aie_inline
constexpr auto mac(const Acc &acc, const Vec &v, E a) -> operand_base_type_t<Acc>
{
    if      constexpr (!is_op_v<Acc>) {
        return mac(op_add(acc), v, a);
    }
    else if constexpr (!is_op_v<Vec>) {
        return mac(acc, op_none(v), a);
    }
    else if constexpr (!is_op_v<E>) {
        return mac(acc, v, op_none(a));
    }
    else {
        using T1 = typename Vec::value_type;
        using T2 = operand_base_type_t<typename E::value_type>;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return mac(acc, v(), a);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Abs))
            return mac(acc, v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj))
            return mac(acc, v, a());
        else if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg))
            return mac(acc(), v, a);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && Vec::is_operation_not(Operation::Neg))
            return mac(acc, v(), a);
        else if constexpr (detail::is_floating_point_v<T2> && E::is_operation_not(Operation::Neg))
            return mac(acc, v, a());
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mac(acc, v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return mac(acc, v, a());
        //FIXME: CRVO-10673 cfloat zeroization
        else if constexpr (Acc::is_operation(Operation::Zero) && (std::is_same_v<T1, cfloat> || std::is_same_v<T2, cfloat>))
            return mac(acc(), v, a);
#endif
        else {
            constexpr Operation Op1     = detail::evaluate_mul_operation<Vec>();
            constexpr Operation Op2     = detail::evaluate_mul_operation<E>();
            constexpr unsigned acc_bits = Acc::result_type::accum_bits();

            bool zero_acc = false;
            bool sub_mul  = detail::eval_sub_mul(v, a);
            bool sub_acc  = false;

            if constexpr (Acc::is_operation(Operation::Zero)) zero_acc = acc.parent2();
            if constexpr (Acc::is_operation(Operation::Neg))  sub_acc  = acc.parent2();

            constexpr auto op = [&](){
                if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg)) return detail::to_mul_macro_op<Operation::Acc_Add, Op1, Op2>();
                else                                                              return detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
            }();

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>)
                return detail::mul<op, acc_bits, T1, T2>::run(v.parent1(),                        detail::get_mul_sign(v),
                                                              vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                              sub_mul, sub_acc,
                                                              zero_acc, acc.parent1());
            else
                return detail::mul<op, acc_bits, T1, T2>::run(v.parent1(), detail::get_mul_sign(v),
                                                              a.parent1(), detail::get_mul_sign(a),
                                                              sub_mul, sub_acc,
                                                              zero_acc, acc.parent1());
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Performs an element-wise multiplication of two input vectors, then returns the subtraction of an accumulator with the
 * resulting product. The vectors and the accumulator must have the same size and the types of all operands must be
 * compatible.
 *
 * Operands can also be lazy operations (see @ref group_basic_types_ops).
 *
 * On some AIE architectures certain operations can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] - op1(v1[i]) * op2(v2[i]);
 * @endcode
 *
 * @param acc Accumulator from which the result of the multiplication is subtracted. The type must meet @ref aie::Accum.
 * @param v1  First input vector. The type must meet @ref aie::VectorOrOp.
 * @param v2  Second input vector. The type must meet @ref aie::VectorOrOp.
 */
// TODO: document compatible types.
template <AccumOrOp Acc, VectorOrOp Vec1, VectorOrOp Vec2>
    requires(is_valid_size_v<Acc, Vec1, Vec2> && is_valid_mul_op_v<typename Vec1::value_type, typename Vec2::value_type>)
__aie_inline
constexpr auto msc(const Acc &acc, const Vec1 &v1, const Vec2 &v2) -> operand_base_type_t<Acc>
{
    if constexpr (!is_op_v<Acc>) {
        return msc(op_sub(acc), v1, v2);
    }
    else if constexpr (!is_op_v<Vec1>) {
        return msc(acc, op_none(v1), v2);
    }
    else if constexpr (!is_op_v<Vec2>) {
        return msc(acc, v1, op_none(v2));
    }
    else {
        using T1 = typename Vec1::value_type;
        using T2 = typename Vec2::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec1::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return msc(acc, v1(), v2);
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Abs, Operation::Max, Operation::Min))
            return msc(acc, v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj))
            return msc(acc, v1, v2());
        else if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg))
            return msc(acc(), v1, v2);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && Vec1::is_operation_not(Operation::Neg))
            return msc(acc, v1(), v2);
        else if constexpr (detail::is_floating_point_v<T2> && Vec2::is_operation_not(Operation::Neg))
            return msc(acc, v1, v2());
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return msc(acc, v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return msc(acc, v1, v2());
        //FIXME: CRVO-10673 cfloat zeroization
        else if constexpr (Acc::is_operation(Operation::Zero) && (std::is_same_v<T1, cfloat> || std::is_same_v<T2, cfloat>))
            return msc(acc(), v1, v2);
#endif
        else {
            constexpr Operation Op1     = detail::evaluate_mul_operation<Vec1>();
            constexpr Operation Op2     = detail::evaluate_mul_operation<Vec2>();
            constexpr unsigned acc_bits = Acc::result_type::accum_bits();

            if constexpr (Vec1::is_operation(Operation::Min, Operation::Max)) {
                constexpr auto op = detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
                return detail::mul_maxmin<op, acc_bits, T1, T2>::run(v1.parent1(), v1.parent2(),
                                                                     v2.parent1(), acc.parent1());
            }
            else {
                bool zero_acc = false;
                bool sub_mul  = detail::eval_sub_mul(v1, v2);
                bool sub_acc  = false;

                if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg)) {
                    constexpr auto op = detail::to_mul_macro_op<Operation::Acc_Sub, Op1, Op2>();
                    if constexpr (Acc::is_operation(Operation::Zero)) zero_acc = acc.parent2();
                    if constexpr (Acc::is_operation(Operation::Neg))  sub_acc  = acc.parent2();

                    return detail::mul<op, acc_bits, T1, T2>::run(v1.parent1(), detail::get_mul_sign(v1),
                                                                  v2.parent1(), detail::get_mul_sign(v2),
                                                                  sub_mul, sub_acc,
                                                                  zero_acc, acc.parent1());
                }
                else {
                    constexpr auto op = detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
                    return detail::mul<op, acc_bits, T1, T2>::run(v1.parent1(), detail::get_mul_sign(v1),
                                                                  v2.parent1(), detail::get_mul_sign(v2),
                                                                  sub_mul, sub_acc,
                                                                  zero_acc, acc.parent1());
                }
            }
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Performs an element-wise multiplication of a scalar and a vector, then returns the subtraction of an accumulator with
 * the resulting product. The vectors and the accumulator must have the same size and the types of all operands must be
 * compatible.
 *
 * Operands can also be lazy operations (see @ref group_basic_types_ops).
 *
 * On some AIE architectures certain operations can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] - op1(a) * op2(v[i]);
 * @endcode
 *
 * @param acc Accumulator from which the result of the multiplication is subtracted. The type must meet @ref aie::Accum.
 * @param a   Input value. The type must meet @ref aie::ElemOrOp.
 * @param v   Input vector. The type must meet @ref aie::VectorOrOp.
 */
// TODO: document compatible types.
template <AccumOrOp Acc, ElemOrOp E, VectorOrOp Vec>
    requires(is_valid_size_v<Acc, E, Vec> && is_valid_mul_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr auto msc(const Acc &acc, E a, const Vec &v) -> operand_base_type_t<Acc>
{
    if      constexpr (!is_op_v<Acc>) {
        return msc(op_sub(acc), a, v);
    }
    else if constexpr (!is_op_v<E>) {
        return msc(acc, op_none(a), v);
    }
    else if constexpr (!is_op_v<Vec>) {
        return msc(acc, a, op_none(v));
    }
    else {
        using T1 = operand_base_type_t<typename E::value_type>;
        using T2 = typename Vec::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (E::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return msc(acc, a(), v);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Abs))
            return msc(acc, a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj))
            return msc(acc, a, v());
        else if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg))
            return msc(acc(), a, v);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && E::is_operation_not(Operation::Neg))
            return msc(acc, a(), v);
        else if constexpr (detail::is_floating_point_v<T2> && Vec::is_operation_not(Operation::Neg))
            return msc(acc, a, v());
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return msc(acc, a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return msc(acc, a, v());
        //FIXME: CRVO-10673 cfloat zeroization
        else if constexpr (Acc::is_operation(Operation::Zero) && (std::is_same_v<T1, cfloat> || std::is_same_v<T2, cfloat>))
            return msc(acc(), a, v);
#endif
        else {
            constexpr Operation Op1     = detail::evaluate_mul_operation<E>();
            constexpr Operation Op2     = detail::evaluate_mul_operation<Vec>();
            constexpr unsigned acc_bits = Acc::result_type::accum_bits();

            bool zero_acc = false;
            bool sub_mul  = detail::eval_sub_mul(a, v);
            bool sub_acc  = false;

            if constexpr (Acc::is_operation(Operation::Zero)) zero_acc = acc.parent2();
            if constexpr (Acc::is_operation(Operation::Neg))  sub_acc  = acc.parent2();

            constexpr auto op = [&](){
                if constexpr (Acc::is_operation(Operation::Zero)) return detail::to_mul_macro_op<Operation::Acc_Sub, Op1, Op2>();
                else                                              return detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
            }();

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>) {
                return detail::mul<op, acc_bits, T1, T2>::run(vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                                                    v.parent1(),  detail::get_mul_sign(v),
                                                                                    sub_mul, sub_acc,
                                                                                    zero_acc, acc.parent1());
            }
            else {
                return detail::mul<op, acc_bits, T1, T2>::run(a.parent1(), detail::get_mul_sign(a),
                                                              v.parent1(), detail::get_mul_sign(v),
                                                              sub_mul, sub_acc,
                                                              zero_acc, acc.parent1());
            }
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Performs an element-wise multiplication of a scalar and a vector, then returns the subtraction of an accumulator with
 * the resulting product. The vectors and the accumulator must have the same size and the types of all operands must be
 * compatible.
 *
 * Operands can also be lazy operations (see @ref group_basic_types_ops).
 *
 * On some AIE architectures certain operations can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] - op1(v[i]) * op2(a);
 * @endcode
 *
 * @param acc Accumulator from which the result of the multiplication is subtracted. The type must meet @ref aie::Accum.
 * @param v   Input vector. The type must meet @ref aie::VectorOrOp.
 * @param a   Input value. The type must meet @ref aie::ElemOrOp.
 */
// TODO: document compatible types.
template <AccumOrOp Acc, VectorOrOp Vec, ElemOrOp E>
    requires(is_valid_size_v<Acc, Vec, E> && is_valid_mul_op_v<typename Vec::value_type, E>)
__aie_inline
constexpr auto msc(const Acc &acc, const Vec &v, E a) -> operand_base_type_t<Acc>
{
    if      constexpr (!is_op_v<Acc>) {
        return msc(op_sub(acc), v, a);
    }
    else if constexpr (!is_op_v<Vec>) {
        return msc(acc, op_none(v), a);
    }
    else if constexpr (!is_op_v<E>) {
        return msc(acc, v, op_none(a));
    }
    else {
        using T1 = typename Vec::value_type;
        using T2 = operand_base_type_t<typename E::value_type>;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return msc(acc, v(), a);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Abs))
            return msc(acc, v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj))
            return msc(acc, v, a());
        else if constexpr (Acc::is_operation(Operation::Zero, Operation::Neg))
            return msc(acc(), v, a);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && Vec::is_operation_not(Operation::Neg))
            return msc(acc, v(), a);
        else if constexpr (detail::is_floating_point_v<T2> && E::is_operation_not(Operation::Neg))
            return msc(acc, v, a());
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return msc(acc, v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return msc(acc, v, a());
        //FIXME: CRVO-10673 cfloat zeroization
        else if constexpr (Acc::is_operation(Operation::Zero) && (std::is_same_v<T1, cfloat> || std::is_same_v<T2, cfloat>))
            return msc(acc(), v, a);
#endif
        else {
            constexpr Operation Op1     = detail::evaluate_mul_operation<Vec>();
            constexpr Operation Op2     = detail::evaluate_mul_operation<E>();
            constexpr unsigned acc_bits = Acc::result_type::accum_bits();

            bool zero_acc = false;
            bool sub_mul  = detail::eval_sub_mul(v, a);
            bool sub_acc  = false;

            if constexpr (Acc::is_operation(Operation::Zero)) zero_acc = acc.parent2();
            if constexpr (Acc::is_operation(Operation::Neg))  sub_acc  = acc.parent2();

            constexpr auto op = [&](){
                if constexpr (Acc::is_operation(Operation::Zero)) return detail::to_mul_macro_op<Operation::Acc_Sub, Op1, Op2>();
                else                                              return detail::to_mul_macro_op<Acc::operation, Op1, Op2>();
            }();

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>)
                return detail::mul<op, acc_bits, T1, T2>::run(v.parent1(),                        detail::get_mul_sign(v),
                                                              vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                              sub_mul, sub_acc,
                                                              zero_acc, acc.parent1());
            else
                return detail::mul<op, acc_bits, T1, T2>::run(v.parent1(), detail::get_mul_sign(v),
                                                              a.parent1(), detail::get_mul_sign(a),
                                                              sub_mul, sub_acc,
                                                              zero_acc, acc.parent1());
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator of the requested type with the negate of the element-wise multiplication of the two input
 * vectors. The vectors must have the same size and compatible types. Vectors can also be lazy operations (see @ref
 * group_basic_types_ops). On some AIE architectures certain operations can be collapsed with the multiplication into
 * a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = -(op1(v1[i]) * op1(v2[i]));
 * @endcode
 *
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with
 *                  the vector types (real/complex). Refer to \ref DefaultAccumTag "default accumulator tag" table for
 *                  the minimum accumulator element tag required for each combination of input element types.
 *
 * @param v1 First input vector. The type must meet @ref aie::VectorOrOp.
 * @param v2 Second input vector. The type must meet @ref aie::VectorOrOp.
 */
template <AccumElemBaseType AccumTag, VectorOrOp Vec1, VectorOrOp Vec2>
    requires(Vec1::size() == Vec2::size() && is_valid_mul_op_v<typename Vec1::value_type, typename Vec2::value_type>)
__aie_inline
constexpr auto negmul(const Vec1 &v1, const Vec2 &v2) -> accum<AccumTag, Vec1::size()>
{
    if      constexpr (!is_op_v<Vec1>) {
        return negmul<AccumTag>(op_none(v1), v2);
    }
    else if constexpr (!is_op_v<Vec2>) {
        return negmul<AccumTag>(v1, op_none(v2));
    }
    else {
        using T1 = typename Vec1::value_type;
        using T2 = typename Vec2::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec1::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return negmul<AccumTag>(v1(), v2);
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Abs, Operation::Max, Operation::Min))
            return negmul<AccumTag>(v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj))
            return negmul<AccumTag>(v1, v2());
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
        if      constexpr (detail::is_floating_point_v<T1> && Vec1::is_operation_not(Operation::Neg))
            return negmul<AccumTag>(v1(), v2);
        else if constexpr (detail::is_floating_point_v<T2> && Vec2::is_operation_not(Operation::Neg))
            return negmul<AccumTag>(v1, v2());
        else if constexpr (Vec1::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return negmul<AccumTag>(v1(), v2);
        else if constexpr (Vec2::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return negmul<AccumTag>(v1, v2());
#endif
        else {
            constexpr Operation Op1 = detail::evaluate_mul_operation<Vec1>();
            constexpr Operation Op2 = detail::evaluate_mul_operation<Vec2>();

            constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();
            constexpr auto op = detail::to_negmul_macro_op<Op1, Op2>();

            bool sub_mul = detail::eval_sub_mul(v1, v2);

            if constexpr (is_binary_op_v<Vec1>)
                return detail::mul_maxmin<detail::to_mul_macro_op<Op1, Op2>(), accum_bits, T1, T2>::run(v1.parent1(), v1.parent2(), v2.parent1());
            else
                return detail::mul<op, accum_bits, T1, T2>::run(v1.parent1(), detail::get_mul_sign(v1),
                                                                v2.parent1(), detail::get_mul_sign(v2),
                                                                sub_mul);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator of the requested type with the negate of the element-wise multiplication of a value and all
 * the elements of the input vector. The type of the value and the type of the vector elements must be compatible. The
 * input arguments can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain
 * operations can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = -(op1(a) * op2(v[i]));
 * @endcode
 *
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with
 *                  the vector types (real/complex). Refer to \ref DefaultAccumTag "default accumulator tag" table for
 *                  the minimum accumulator element tag required for each combination of input element types.
 *
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 */
template <AccumElemBaseType AccumTag, ElemOrOp E, VectorOrOp Vec>
    requires(is_valid_mul_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr auto negmul(E a, const Vec &v) -> accum<AccumTag, Vec::size()>
{
    if      constexpr (!is_op_v<E>) {
        return negmul<AccumTag>(op_none(a), v);
    }
    else if constexpr (!is_op_v<Vec>) {
        return negmul<AccumTag>(a, op_none(v));
    }
    else {
        using T1 = operand_base_type_t<typename E::value_type>;
        using T2 = typename Vec::value_type;

#if __AIE_ARCH__ == 10
        if      constexpr (E::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return negmul<AccumTag>(a(), v);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Abs))
            return negmul<AccumTag>(a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj))
            return negmul<AccumTag>(a, v());
#else
        if      constexpr (detail::is_floating_point_v<T1> && E::is_operation_not(Operation::Neg))
            return negmul<AccumTag>(a(), v);
        else if constexpr (detail::is_floating_point_v<T2> && Vec::is_operation_not(Operation::Neg))
            return negmul<AccumTag>(a, v());
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return negmul<AccumTag>(a(), v);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return negmul<AccumTag>(a, v());
#endif
        else {
            constexpr Operation Op1 = detail::evaluate_mul_operation<E>();
            constexpr Operation Op2 = detail::evaluate_mul_operation<Vec>();

            constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();
            constexpr auto op = detail::to_negmul_macro_op<Op1, Op2>();

            bool sub_mul = detail::eval_sub_mul(a, v);

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>)
                return detail::mul<op, accum_bits, T1, T2>::run(vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                                v.parent1(),                        detail::get_mul_sign(v),
                                                                sub_mul);
            else
                return detail::mul<op, accum_bits, T1, T2>::run(a.parent1(), detail::get_mul_sign(a),
                                                                v.parent1(), detail::get_mul_sign(v),
                                                                sub_mul);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator of the requested type with the negate of the element-wise multiplication of all the elements
 * of the input vector and a value. The type of the value and the type of the vector elements must be compatible. The
 * input arguments can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain
 * operations can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = -(op1(v[i]) * op2(a));
 * @endcode
 *
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with
 *                  the vector types (real/complex). Refer to \ref DefaultAccumTag "default accumulator tag" table for
 *                  the minimum accumulator element tag required for each combination of input element types.
 *
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 */
template <AccumElemBaseType AccumTag, VectorOrOp Vec, ElemOrOp E>
    requires(is_valid_mul_op_v<typename Vec::value_type, E>)
__aie_inline
constexpr auto negmul(const Vec &v, E a) -> accum<AccumTag, Vec::size()>
{
    if      constexpr (!is_op_v<Vec>) {
        return negmul<AccumTag>(op_none(v), a);
    }
    else if constexpr (!is_op_v<E>) {
        return negmul<AccumTag>(v, op_none(a));
    }
    else {
        using T1 = typename Vec::value_type;
        using T2 = operand_base_type_t<typename E::value_type>;

#if __AIE_ARCH__ == 10
        if      constexpr (Vec::is_operation_not(Operation::Conj) && detail::is_floating_point_v<T1>)
            return negmul<AccumTag>(v(), a);
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Abs))
            return negmul<AccumTag>(v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj))
            return negmul<AccumTag>(v, a());
#else
        if      constexpr (detail::is_floating_point_v<T1> && Vec::is_operation_not(Operation::Neg))
            return negmul<AccumTag>(v(), a);
        else if constexpr (detail::is_floating_point_v<T2> && E::is_operation_not(Operation::Neg))
            return negmul<AccumTag>(v, a());
        else if constexpr (Vec::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return negmul<AccumTag>(v(), a);
        else if constexpr (E::is_operation_not(Operation::Conj, Operation::Sign, Operation::Neg))
            return negmul<AccumTag>(v, a());
#endif
        else {
            constexpr Operation Op1 = detail::evaluate_mul_operation<Vec>();
            constexpr Operation Op2 = detail::evaluate_mul_operation<E>();

            constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();
            constexpr auto op = detail::to_negmul_macro_op<Op1, Op2>();

            bool sub_mul = detail::eval_sub_mul(v, a);

            if constexpr (detail::is_vector_elem_ref_v<typename E::parent1_type>)
                return detail::mul<op, accum_bits, T1, T2>::run(v.parent1(),                        detail::get_mul_sign(v),
                                                                vector_elem_const_ref(a.parent1()), detail::get_mul_sign(a),
                                                                sub_mul);
            else
                return detail::mul<op, accum_bits, T1, T2>::run(v.parent1(), detail::get_mul_sign(v),
                                                                a.parent1(), detail::get_mul_sign(a),
                                                                sub_mul);
        }
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the negate of the element-wise multiplication of the two input vectors. The default
 * accumulator accuracy is used. The vectors must have the same size and compatible types. Operands can also be lazy
 * operations (see @ref group_basic_types_ops). On some AIE architectures certain operations can be collapsed with the
 * multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = -(op1(v1[i]) * op2(v2[i]));
 * @endcode
 *
 * @param v1 First input vector. The type must meet @ref aie::VectorOrOp.
 * @param v2 Second input vector. The type must meet @ref aie::VectorOrOp.
 */
// TODO: document default accumulation
// TODO: document compatible types.
template <VectorOrOp Vec1, VectorOrOp Vec2>
    requires((Vec1::size() == Vec2::size()) && is_valid_mul_op_v<typename Vec1::value_type, typename Vec2::value_type>)
__aie_inline
constexpr auto negmul(const Vec1 &v1, const Vec2 &v2)
{
    using T1 = typename Vec1::value_type;
    using T2 = typename Vec2::value_type;
    using T3 = detail::default_accum_tag_t<T1, T2>;

    return negmul<T3>(v1, v2);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the negate of the element-wise multiplication of a value and all the elements of the
 * input vector. The default accumulator accuracy is used. The vectors must have the same size and compatible types.
 * Operands can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain operations
 * can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = -(op1(a) * op2(v[i]));
 * @endcode
 *
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 */
template <ElemOrOp E, VectorOrOp Vec> requires(is_valid_mul_op_v<E, typename Vec::value_type>)
__aie_inline
constexpr auto negmul(E a, const Vec &v)
{
    using T1 = operand_base_type_t<E>;
    using T2 = typename Vec::value_type;
    using T3 = detail::default_accum_tag_t<T1, T2>;

    return negmul<T3>(a, v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the negate of the element-wise multiplication of all the elements of the input vector
 * and a value. The default accumulator accuracy is used. The vectors must have the same size and compatible types.
 * Operands can also be lazy operations (see @ref group_basic_types_ops). On some AIE architectures certain operations
 * can be collapsed with the multiplication into a single instruction.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = -(op1(v[i]) * op2(a)));
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::VectorOrOp.
 * @param a Value. The type must meet @ref aie::ElemOrOp.
 */
template <VectorOrOp Vec, ElemOrOp E> requires(is_valid_mul_op_v<typename Vec::value_type, E>)
__aie_inline
constexpr auto negmul(const Vec &v, E a)
{
    using T1 = typename Vec::value_type;
    using T2 = operand_base_type_t<E>;
    using T3 = detail::default_accum_tag_t<T1, T2>;

    return negmul<T3>(v, a);
}

template <AccumElemBaseType AccumTag, Vector Vec>
__aie_inline
typename Vec::value_type reduce_mul(const Vec &v)
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();

    return detail::mul_reduce<accum_bits, T, Elems>::run(v);
}

/**
 * @ingroup group_reduce
 *
 * Returns a product of the elements in the input vector.
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec>
__aie_inline
constexpr typename Vec::value_type reduce_mul(const Vec &v)
{
    using T = typename Vec::value_type;

    using accum_tag = detail::default_accum_tag_t<T, T>;

    return reduce_mul<accum_tag>(v);
}

template <AccumElemBaseType AccumTag, Vector Vec>
[[deprecated("Use reduce_add instead")]]
__aie_inline
typename Vec::value_type mul_reduce(const Vec &v)
{
    return reduce_mul<AccumTag>(v);
}

template <Vector Vec>
[[deprecated("Use reduce_add instead")]]
__aie_inline
constexpr typename Vec::value_type mul_reduce(const Vec &v)
{
    return reduce_mul(v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator of the requested type with the element-wise square of the input vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] * v[i];
 * @endcode
 *
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with
 *                  the vector types (real/complex). Refer to \ref DefaultAccumTag "default accumulator tag" table for
 *                  the minimum accumulator element tag required for each combination of input element types.
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <AccumElemBaseType AccumTag, Vector Vec>
__aie_inline
constexpr auto mul_square(const Vec &v) -> accum<AccumTag, Vec::size()>
{
    using T = typename Vec::value_type;

    constexpr unsigned accum_bits = detail::to_native_accum_bits<AccumTag>();

    return detail::square<detail::MulMacroOp::Mul, accum_bits, T>::run(v, false);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the element-wise square of the input vector. The default accumulator accuracy is used.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = v[i] * v[i];
 * @endcode
 *
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <Vector Vec>
__aie_inline
constexpr auto mul_square(const Vec &v)
{
    using T = typename Vec::value_type;

    using accum_tag = detail::default_accum_tag_t<T, T>;

    return mul_square<accum_tag>(v);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the addition or subtraction of the given accumulator and the element-wise square of the
 * input vector. The vector and the accumulator must have the same size and their types must be compatible.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] + v[i] * v[i];
 * @endcode
 *
 * @param acc Input accumulator. The type must meet @ref aie::AccumOrOp.
 * @param v Input vector. The type must meet @ref aie::Vector.
 */
template <AccumOrOp Acc, Vector Vec> requires(Vec::size() == Acc::size())
__aie_inline
constexpr auto mac_square(const Acc &acc, const Vec &v)
{
    if constexpr (!is_op_v<Acc>) {
        return mac_square(op_add(acc), v);
    }
#if __AIE_ARCH__ == 10
    else if constexpr (Acc::is_operation(Operation::Zero)) {
        return mac_square(acc(), v);
    }
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    //FIXME: CRVO-10673 cfloat zeroization
    else if constexpr (Acc::is_operation(Operation::Zero) && std::is_same_v<typename Vec::value_type, cfloat>) {
        return mac_square(acc(), v);
    }
#endif
    else {
        using T = typename Vec::value_type;

        bool zero_acc = false;

        constexpr auto get_op = [&](){
            if constexpr (Acc::is_operation(Operation::Zero)) return Operation::Acc_Add;
            else                                              return Acc::operation;
        };

        if constexpr (Acc::is_operation(Operation::Zero))
            zero_acc = acc.parent2();

        constexpr auto op = detail::to_mul_macro_op<get_op(), Operation::None, Operation::None>();

        return detail::square<op, Acc::result_type::accum_bits(), T>::run(v, zero_acc, acc.parent1());
    }
}

/**
 * @ingroup group_arithmetic
 *
 * Returns an accumulator with the subtraction of the given accumulator and the element-wise square of the input
 * vector. The vector and the accumulator must have the same size and their types must be compatible.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = acc[i] - v[i] * v[i];
 * @endcode
 *
 * @param acc Input accumulator. The type must meet @ref aie::Accum.
 * @param v   Input vector. The type must meet @ref aie::Vector.
 */
template <AccumOrOp Acc, Vector Vec> requires(Vec::size() == Acc::size())
__aie_inline
constexpr auto msc_square(const Acc &acc, const Vec &v)
{
    if constexpr (!is_op_v<Acc>) {
        return msc_square(op_sub(acc), v);
    }
#if __AIE_ARCH__ == 10
    else if constexpr (Acc::is_operation(Operation::Zero)) {
        return msc_square(acc(), v);
    }
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    //FIXME: CRVO-10673 cfloat zeroization
    else if constexpr (Acc::is_operation(Operation::Zero) && std::is_same_v<typename Vec::value_type, cfloat>) {
        return msc_square(acc(), v);
    }
#endif
    else {
        using T = typename Vec::value_type;

        bool zero_acc = false;

        constexpr auto get_op = [&](){
            if constexpr (Acc::is_operation(Operation::Zero)) return Operation::Acc_Sub;
            else                                              return Acc::operation;
        };

        if constexpr (Acc::is_operation(Operation::Zero))
            zero_acc = acc.parent2();

        constexpr auto op = detail::to_mul_macro_op<get_op(), Operation::None, Operation::None>();

        return detail::square<op, Acc::result_type::accum_bits(), T>::run(v, zero_acc, acc.parent1());
    }
}

/**
 * @ingroup group_mul_special
 *
 * \brief Performs a weighted addition over multiple vectors and accumulates the result into an existing accumulator.
 *
 * Given a finite number of vectors \f$ d_1, d_2, ..., d_C \f$ and a weight vector
 * \f$ coef = ( \alpha_1, \alpha_2, ..., \alpha_n ) \f$ for \f$ n \geq (C + start) \f$ and an accumulator \f$ acc \f$,
 * the result is an accumulator of the form \f$ out = acc + \sum_{i=1}^{C} \alpha_{i + start} * d_i \f$.
 *
 * Each of the data vectors is scaled by its corresponding element of coeff vector.
 *
 * The operation consists on a parametrized multiplication that implements the following compute pattern:
 *
 * @code
 *
 * L   = Lanes
 * C   = sizeof...(next_data) + 1
 * c_s = coeff_start
 *
 * out[0]   = acc[0]   + coeff[c_s] * data[0]   + coeff[c_s + 1] * next_data_0[0]   + ... + coeff[c_s + (C-1)] * next_data_C-2[0]
 * out[1]   = acc[1]   + coeff[c_s] * data[1]   + coeff[c_s + 1] * next_data_0[1]   + ... + coeff[c_s + (C-1)] * next_data_C-2[1]
 * ...
 * out[L-1] = acc[L-1] + coeff[c_s] * data[L-1] + coeff[c_s + 1] * next_data_0[L-1] + ... + coeff[c_s + (C-1)] * next_data_C-2[L-1]
 *
 * @endcode
 *
 * In the following example, accumulate multiplies data0 by 1, data1 by 2 and data2 by 3 respectively:
 * @code{.cpp}
 * aie::accum<acc80, 8> acc;
 * aie::vector<int, 4> coeffs{1, 2, 3};
 * aie::vector<int, 8> data0{...}, data1{...}, data2{...};
 *
 * acc = aie::accumulate(acc, coeffs, 0, data0, data1, data2);
 * @endcode
 *
 * @tparam Lanes       Number of output elements.
 *
 * @param  acc         Accumulator to which the result of the accumulation is added (or subtracted).
 *                     The type must meet @ref aie::AccumOrOp.
 * @param  coeff       Vector of coefficients. Size is limited to 256b and 512b on AIE and AIE-ML/XDNA1 respectively.
 * @param  coeff_start First element from the coeff vector to be used.
 * @param  data        First vector of data.
 * @param  next_data   Remaining data vectors.
 */
template <unsigned Lanes,
          AccumOrOp Acc,
          Vector VecCoeff,
          Vector VecData,
          Vector... NextVecData> requires((is_same_vector_v<VecData, NextVecData> && ...) &&
                                          Acc::size() == Lanes &&
                                          is_valid_mul_op_v<typename VecCoeff::value_type, typename VecData::value_type>)
__aie_inline
auto accumulate(const Acc &acc,
                const VecCoeff &coeff,
                unsigned coeff_start,
                const VecData  &data,
                const NextVecData & ...next_data) -> operand_base_type_t<Acc>
{
    using CoeffType = typename VecCoeff::value_type;
    using DataType  = typename VecData::value_type;

    if constexpr (detail::is_accum_v<Acc>) {
        return accumulate<Lanes>(op_add(acc), coeff, coeff_start, data, next_data...);
    }
    else {
        return detail::accumulator<Acc::result_type::accum_bits(), Lanes, CoeffType, VecCoeff::size(), DataType>::template run<Acc::operation, 1>(acc.parent1(),
                                                                                                                                                  coeff,
                                                                                                                                                  coeff_start,
                                                                                                                                                  data,
                                                                                                                                                  next_data...);
    }
}

/**
 * @ingroup group_mul_special
 *
 * \brief Performs a weighted addition over multiple vectors and accumulates the result into an existing accumulator.
 *
 * Given a finite number of vectors \f$ d_1, d_2, ..., d_C \f$ and a weight vector
 * \f$ coef = ( \alpha_1, \alpha_2, ..., \alpha_n ) \f$ for \f$ n \geq (C + start) \f$ and an accumulator \f$ acc \f$,
 * the result is an accumulator of the form \f$ out = acc + \sum_{i=1}^{C} \alpha_{i + start} * d_i \f$.
 *
 * Each of the data vectors is scaled by its corresponding element of coeff vector.
 *
 * The operation consists on a parametrized multiplication that implements the following compute pattern:
 *
 * @code
 *
 * L   = Lanes
 * C   = sizeof...(next_data) + 1
 * c_s = coeff_start
 *
 * out[0]   = acc[0]   + coeff[c_s] * data[0]   + coeff[c_s + 1] * next_data_0[0]   + ... + coeff[c_s + (C-1)] * next_data_C-2[0]
 * out[1]   = acc[1]   + coeff[c_s] * data[1]   + coeff[c_s + 1] * next_data_0[1]   + ... + coeff[c_s + (C-1)] * next_data_C-2[1]
 * ...
 * out[L-1] = acc[L-1] + coeff[c_s] * data[L-1] + coeff[c_s + 1] * next_data_0[L-1] + ... + coeff[c_s + (C-1)] * next_data_C-2[L-1]
 *
 * @endcode
 *
 * In the following example, accumulate multiplies data0 by 1, data1 by 2 and data2 by 3 respectively:
 * @code{.cpp}
 * aie::accum<acc80, 8> acc;
 * aie::vector<int, 4> coeffs{1, 2, 3};
 * aie::vector<int, 8> data0{...}, data1{...}, data2{...};
 * std::array<aie::vector<int, 8>, 3> data{data0, data1, data2};
 *
 * acc = aie::accumulate(acc, coeffs, 0, data);
 * @endcode
 *
 * @tparam Lanes       Number of output elements.
 *
 * @param  acc         Accumulator to which the result of the accumulation is added (or subtracted).
 *                     The type must meet @ref aie::AccumOrOp.
 * @param  coeff       Vector of coefficients. Size is limited to 256b and 512b on AIE and AIE-ML/XDNA1 respectively.
 * @param  coeff_start First element from the coeff vector to be used.
 * @param  data        Array of data vectors
 */
template <unsigned Lanes,
          AccumOrOp Acc,
          Vector VecCoeff,
          Vector VecData,
          size_t N> requires((N >= 1)             &&
                             Acc::size() == Lanes &&
                             is_valid_mul_op_v<typename VecCoeff::value_type, typename VecData::value_type>)
__aie_inline
auto accumulate(const Acc &acc,
                const VecCoeff &coeff,
                unsigned coeff_start,
                const std::array<VecData, N> &data) -> operand_base_type_t<Acc>
{
    constexpr auto run_accumulate = []<size_t... Idx>(const Acc &acc,
                                                      const VecCoeff &coeff,
                                                      unsigned coeff_start,
                                                      const std::array<VecData, N> &data,
                                                      std::index_sequence<Idx...>){
        return accumulate<Lanes>(acc, coeff, coeff_start, data[Idx]...);
    };

    return run_accumulate(acc, coeff, coeff_start, data, std::make_index_sequence<N>{});
}

/**
 * @ingroup group_mul_special
 *
 * \brief Performs a weighted addition over multiple vectors.
 *
 * Given a finite number of vectors \f$ d_1, d_2, ..., d_C \f$ and a weight vector
 * \f$ coef = ( \alpha_1, \alpha_2, ..., \alpha_n ) \f$, for \f$ n \geq (C + start) \f$, the result is an accumulator
 * of the form \f$ out = \sum_{i=1}^{C} \alpha_{i + start} * d_i \f$.
 *
 * Each of the data vectors is scaled by its corresponding element of coeff vector.
 *
 * This function provides a parametrized multiplication that implements the following compute pattern:
 *
 * @code
 *
 * L   = Lanes
 * C   = sizeof...(next_data) + 1
 * c_s = coeff_start
 *
 * out[0]   = coeff[c_s] * data_0[0]   + coeff[c_s + 1] * next_data_0[0]   + ... + coeff[c_s + (C-1)] * next_data_C-2[0]
 * out[1]   = coeff[c_s] * data_0[1]   + coeff[c_s + 1] * next_data_0[1]   + ... + coeff[c_s + (C-1)] * next_data_C-2[1]
 * ...
 * out[L-1] = coeff[c_s] * data_0[L-1] + coeff[c_s + 1] * next_data_0[L-1] + ... + coeff[c_s + (C-1)] * next_data_C-2[L-1]
 *
 * @endcode
 *
 * @tparam Lanes    Number of output elements.
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with the
 *                  result of the multiplication of the coefficient and data types (real/complex). Refer to
 *                  \ref DefaultAccumTag "default accumulator tag" table for the minimum accumulator element tag
 *                  required for each combination of input element types.
 * @param coeff Vector of coefficients. Vectors limited to 256b and 512b on AIE and AIE-ML/XDNA1 respectively.
 * @param coeff_start First element from the coeff vector to be used.
 * @param data First vector of data.
 * @param next_data Rest of the data vectors.
 */
template <unsigned Lanes,
          AccumElemBaseType AccumTag = accauto,
          Vector VecCoeff,
          Vector VecData,
          Vector... NextVecData> requires((is_same_vector_v<VecData, NextVecData> && ...))
__aie_inline
auto accumulate(const VecCoeff &coeff,
                unsigned coeff_start,
                const VecData  &data,
                const NextVecData & ...next_data) -> accum<detail::accum_tag_or_default_t<AccumTag,
                                                                                          typename VecCoeff::value_type,
                                                                                          typename VecData::value_type>, Lanes>
{
    using CoeffType = typename VecCoeff::value_type;
    using DataType  = typename VecData::value_type;

    constexpr unsigned accum_bits = detail::to_native_accum_bits_for_mul_types_tag<CoeffType, DataType, AccumTag>();

    return detail::accumulator<accum_bits, Lanes, CoeffType, VecCoeff::size(), DataType>::template run<1>(coeff,
                                                                                                          coeff_start,
                                                                                                          data,
                                                                                                          next_data...);
}

/**
 * @ingroup group_mul_special
 *
 * \brief Performs a weighted addition over multiple vectors.
 *
 * Given a finite number of vectors \f$ d_1, d_2, ..., d_C \f$ and a weight vector
 * \f$ coef = ( \alpha_1, \alpha_2, ..., \alpha_n ) \f$, for \f$ n \geq (C + start) \f$, the result is an accumulator
 * of the form \f$ out = \sum_{i=1}^{C} \alpha_{i + start} * d_i \f$.
 *
 * Each of the data vectors is scaled by its corresponding element of coeff vector.
 *
 * This function provides a parametrized multiplication that implements the following compute pattern:
 *
 * @code
 *
 * L   = Lanes
 * C   = sizeof...(next_data) + 1
 * c_s = coeff_start
 *
 * out[0]   = coeff[c_s] * data_0[0]   + coeff[c_s + 1] * next_data_0[0]   + ... + coeff[c_s + (C-1)] * next_data_C-2[0]
 * out[1]   = coeff[c_s] * data_0[1]   + coeff[c_s + 1] * next_data_0[1]   + ... + coeff[c_s + (C-1)] * next_data_C-2[1]
 * ...
 * out[L-1] = coeff[c_s] * data_0[L-1] + coeff[c_s + 1] * next_data_0[L-1] + ... + coeff[c_s + (C-1)] * next_data_C-2[L-1]
 *
 * @endcode
 *
 * @tparam Lanes    Number of output elements.
 * @tparam AccumTag Accumulator tag that specifies the required accumulation bits. The class must be compatible with the
 *                  result of the multiplication of the coefficient and data types (real/complex). Refer to
 *                  \ref DefaultAccumTag "default accumulator tag" table for the minimum accumulator element tag
 *                  required for each combination of input element types.
 * @param coeff Vector of coefficients. Vectors limited to 256b and 512b on AIE and AIE-ML/XDNA1 respectively.
 * @param coeff_start First element from the coeff vector to be used.
 * @param data Array of data vectors
 */
template <unsigned Lanes,
          AccumElemBaseType AccumTag = accauto,
          Vector VecCoeff,
          Vector VecData,
          size_t N> requires (N >= 1)
__aie_inline
auto accumulate(const VecCoeff &coeff,
                unsigned coeff_start,
                const std::array<VecData, N> &data) -> accum<detail::accum_tag_or_default_t<AccumTag,
                                                                                            typename VecCoeff::value_type,
                                                                                            typename VecData::value_type>, Lanes>
{
    constexpr auto run_accumulate = []<size_t... Idx>(const VecCoeff &coeff,
                                                      unsigned coeff_start,
                                                      const std::array<VecData, N> &data,
                                                      std::index_sequence<Idx...>){
        return accumulate<Lanes>(coeff, coeff_start, data[Idx]...);
    };

    return run_accumulate(coeff, coeff_start, data, std::make_index_sequence<N>{});
}

/**
 * @ingroup group_mul_special
 *
 * @deprecated This function is deprecated. Use accumulate(const VecCoeff & coeff, unsigned coeff_start, const VecData & data, const NextVecData &... next_data ) instead.
 *
 * @see accumulate(const VecCoeff & coeff, unsigned coeff_start, const VecData & data, const NextVecData &... next_data )
 */
template <unsigned Lanes,
          unsigned CoeffStart = 0,
          AccumElemBaseType AccumTag = accauto,
          Vector VecCoeff,
          Vector VecData,
          Vector... NextVecData>
[[deprecated("Use the variant with coeff_start as an argument")]]
__aie_inline
auto accumulate(const VecCoeff &coeff,
                const VecData  &data,
                const NextVecData & ...next_data)
{
    return accumulate<Lanes>(coeff, CoeffStart, data, next_data...);
}

/**
 * @ingroup group_mmul
 *
 * Type that encapsulates a blocked matrix multiplication C = A x B
 *
 * Objects of this type encapsulate the current result of the multiplication. The first result is computed with the mul
 * method. New multiplications can be accumulated using the mac method.
 *
 * @tparam M_Elems   Rows in matrix A.
 * @tparam K_Elems   Columns in matrix A / Rows in matrix B.
 * @tparam N_Elems   Columns in matrix B.
 * @tparam TypeA     Type of the elements in matrix A. It must meet @ref aie::ElemBaseType.
 * @tparam TypeB     Optional. Type of the elements in matrix B. By default is the same as TypeA. It must meet
 *                   @ref aie::ElemBaseType.
 * @tparam AccumTag  Optional. Type of the elements of the accumulator that contains the results to be written in matrix
 *                   C. It must meet @ref aie::AccumElemBaseType. If not specified, it uses the
 *                   \ref DefaultAccumTag "default accumulation type" for multiplications of TypeA x TypeB.
 * @tparam Ownership Optional. If set to owned, the mmul object references a user-provided accumulator instead of owning one.
 *                   When set as ref, the object must be constructed with an accumulator reference. Defaults to owned.
 */
template <unsigned M_Elems, unsigned K_Elems, unsigned N_Elems,
#if AIE_API_ML_VERSION >= 210
          ElemBaseOrBlockType TypeA, ElemBaseOrBlockType TypeB = TypeA,
#else
          ElemBaseType TypeA, ElemBaseType TypeB = TypeA,
#endif
          AccumElemBaseType AccumTag = accauto,
          accum_ownership Ownership = accum_ownership::owned>
struct mmul : public detail::mmul<M_Elems, K_Elems, N_Elems, TypeA, TypeB,
                                  detail::to_native_accum_bits_for_mul_types_tag<TypeA, TypeB, AccumTag>(), Ownership>
{
    using mmul_impl = detail::mmul<M_Elems, K_Elems, N_Elems, TypeA, TypeB,
                                   detail::to_native_accum_bits_for_mul_types_tag<TypeA, TypeB, AccumTag>(), Ownership>;

    /**
     * \brief Number of rows in matrix A.
     */
    static constexpr unsigned M = M_Elems;

    /**
     * \brief Number of columns in matrix A, and number of rows in matrix B.
     */
    static constexpr unsigned K = K_Elems;

    /**
     * \brief Number of columns in matrix B.
     */
    static constexpr unsigned N = N_Elems;

    /**
     * \brief Number of elements in matrix A
     */
    static constexpr unsigned size_A = M * K;

    /**
     * \brief Number of elements in matrix B
     */
    static constexpr unsigned size_B = K * N;

    /**
     * \brief Number of elements in matrix C
     */
    static constexpr unsigned size_C = M * N;

    /**
     * \brief Returns number of elements in matrix C
     */
    static constexpr unsigned size() { return size_C; }

    /**
     * \brief Number of bits in the accumulator.
     */
    static constexpr unsigned accum_bits = detail::to_native_accum_bits_for_mul_types_tag<TypeA, TypeB, AccumTag>();

    using accum_type = typename mmul_impl::accum_type;

    /**
     * \brief Default constructor. Data will qualify as zero initialised for the first operation.
     *
     * @note The result may not be explicitly zero initialized before the first @ref mul or @ref mac operation takes place.
     */
    __aie_inline
    mmul() requires (Ownership == accum_ownership::owned)
    {}

    /**
     * \brief Constructor. Data will qualify as zero/non zero initialised for the first operation.
     *
     * @param to_zero Indicates whether to zero initialize the accumulator data.
     */
    __aie_inline
    mmul(bool to_zero) requires (Ownership == accum_ownership::owned) : mmul_impl(to_zero)
    {}

    /**
     * \brief Constructor. Data is initialized from the given accumulator in row-major order.
     *
     * @param acc Accumulator data is initialized from.
     */
    __aie_inline
    mmul(const accum_type &acc) requires (Ownership == accum_ownership::owned) : mmul_impl(acc)
    {}

    /**
     * \brief Constructor. Data is initialized from the given operation modifier.
     *
     * @param op @ref aie::op_add operation.
     *
     * @sa aie::op_add
     */
    __aie_inline
    mmul(const unary_op<accum_type, Operation::Acc_Add> &op) requires (Ownership == accum_ownership::owned) : mmul_impl(op.parent1())
    {}

    /**
     * \brief Constructor. Data is initialized from the given operation modifier.
     *
     * This modifier conditionally qualifies the data to be represented as if it were zero for the first operation.
     *
     * @param op @ref aie::op_zero operation.
     *
     * @sa aie::op_zero
     */
    __aie_inline
    mmul(const binary_op<accum_type, bool, Operation::Zero> &op) requires (Ownership == accum_ownership::owned) : mmul_impl(op.parent1(), op.parent2())
    {}

    /**
     * \brief Reinitialize the mmul object using the given accumulator.
     *
     * @param acc Accumulator data is initialized from.
     */
    __aie_inline
    mmul &operator=(const accum_type &acc) requires (Ownership == accum_ownership::owned)
    {
        *this = mmul(acc);

        return *this;
    }

    /**
     * \brief Constructor. Data is initialized from the given vector.
     *
     * Data is expected to be row-major layout.
     *
     * @param v Vector data is initialized from.
     * @param shift Upshift in bits to be applied to input data. This parameter is ignored for floating-point types.
     */
    template <typename T>
    __aie_inline
    mmul(const vector<T, size_C> &v, int shift = 0) requires (Ownership == accum_ownership::owned) : mmul_impl(v, shift)
    {}

    /**
     * \brief Return the result of the multiplication as an accumulator.
     */
    __aie_inline
    accum_type to_accum() const
    {
        return mmul_impl::to_accum();
    }

    /**
     * \brief Conversion operator to accumulator.
     */
    __aie_inline
    operator accum_type() const
    {
        return mmul_impl::to_accum();
    }

    /**
     * \brief Return the result of the multiplication as a vector of the requested type.
     *
     * @param shift Downshift in bits to be applied to output data. This parameter is ignored for floating-point types.
     */
    template <typename T>
    __aie_inline
    vector<T, size_C> to_vector(int shift = 0) const
    {
        return mmul_impl::template to_vector<T>(shift);
    }

    //TODO: implement dynamic sign support
    /**
     * \brief Initialize the result value with the multiplication of the two given matrices.
     *
     * Data is overwritten regardless how it was initialised.
     *
     * @param a Represents the A input matrix with row-major data layout.
     *          The number of elements must be mmul::size_A (M * K). It must meet @ref aie::VectorOrOp.
     * @param b Represents the B input matrix with row-major data layout.
     *          The number of elements must be mmul::size_B (K * N). It must meet @ref aie::VectorOrOp.
     */
    template <VectorOrOp VecA, VectorOrOp VecB>
    requires(VecA::size() == size_A && VecB::size() == size_B
                     && std::is_same_v<typename VecA::value_type, TypeA>
                     && std::is_same_v<typename VecB::value_type, TypeB>)
    __aie_inline
    mmul& mul(const VecA &a, const VecB &b)
    {
        if      constexpr (!is_op_v<VecA>)
            return mul(op_none(a), b);
        else if constexpr (!is_op_v<VecB>)
            return mul(a, op_none(b));
        else
            mmul_impl::mul(a.parent1(), detail::get_mul_sign(a), b.parent1(), detail::get_mul_sign(b));
        return *this;
    }

    /**
     * \brief Multiply the two given matrices and add it to the result.
     *
     * @param a Represents the A input matrix with row-major data layout.
     *          The number of elements must be mmul::size_A (M * K). It must meet @ref aie::VectorOrOp.
     * @param b Represents the B input matrix with row-major data layout.
     *          The number of elements must be mmul::size_B (K * N). It must meet @ref aie::VectorOrOp.
     */
    template <VectorOrOp VecA, VectorOrOp VecB>
    requires(VecA::size() == size_A && VecB::size() == size_B
                     && std::is_same_v<typename VecA::value_type, TypeA>
                     && std::is_same_v<typename VecB::value_type, TypeB>)
    __aie_inline
    mmul& mac(const VecA &a, const VecB &b)
    {
        if      constexpr (!is_op_v<VecA>)
            return mac(op_none(a), b);
        else if constexpr (!is_op_v<VecB>)
            return mac(a, op_none(b));
        else
            mmul_impl::mac(a.parent1(), detail::get_mul_sign(a), b.parent1(), detail::get_mul_sign(b));
        return *this;
    }

    /**
     * \brief Initialize the result value with the multiplication of the two given matrices. Matrix B is sparse.
     *
     * Data is overwritten regardless how it was initialised.
     *
     * @param a Vector that represents the A input matrix.
     *          The number of elements must be mmul::size_A (M * K). It must meet @ref aie::VectorOrOp.
     * @param b Sparse vector that represents the B input matrix.
     *          The number of elements must be mmul::size_B (K * N).
     */
    // Note: template is necessary to avoid a type instantiation error when this function is not usable
    // (for example, there is no valid sparse_vector with size_B elements).
    template <VectorOrOp VecA, SparseVectorOrOp VecB>
        requires(arch::is(arch::Gen2)      &&
                        VecA::size() == size_A && VecB::size() == size_B &&
                        std::is_same_v<typename VecA::value_type, TypeA> &&
                        std::is_same_v<typename VecB::value_type, TypeB>)
    __aie_inline
    mmul& mul(const VecA &a, const VecB &b)
    {
        if      constexpr (!is_op_v<VecA>)
            return mul(op_none(a), b);
        else if constexpr (!is_op_v<VecB>)
            return mul(a, op_none(b));
        else
            mmul_impl::mul(a.parent1(), detail::get_mul_sign(a), b.parent1(), detail::get_mul_sign(b));
        return *this;
    }

    /**
     * \brief Multiply the two given matrices and add it to the result. Matrix B is sparse.
     *
     * @param a Vector that represents the A input matrix.
     *          The number of elements must be mmul::size_A (M * K). It must meet @ref aie::VectorOrOp.
     * @param b Sparse vector that represents the B input matrix.
     *          The number of elements must be mmul::size_B (K * N).
     */
    // Note: template is necessary to avoid a type instantiation error when this function is not usable
    // (for example, there is no valid sparse_vector with size_B elements).
    template <VectorOrOp VecA, SparseVectorOrOp VecB>
        requires(arch::is(arch::Gen2)      &&
                        VecA::size() == size_A && VecB::size() == size_B &&
                        std::is_same_v<typename VecA::value_type, TypeA> &&
                        std::is_same_v<typename VecB::value_type, TypeB>)
    __aie_inline
    mmul& mac(const VecA &a, const VecB &b)
    {
        if      constexpr (!is_op_v<VecA>)
            return mac(op_none(a), b);
        else if constexpr (!is_op_v<VecB>)
            return mac(a, op_none(b));
        else
            mmul_impl::mac(a.parent1(), detail::get_mul_sign(a), b.parent1(), detail::get_mul_sign(b));
        return *this;
    }

#if AIE_API_ML_VERSION >= 210
    /**
     * \brief Initialize the result value with the result of A * transpose(B) for the two given block matrices.
     *
     * Data is overwritten regardless how it was initialised.
     *
     * @param a Block vector that represents the A input matrix.
     *          The number of elements must be mmul::size_A (M * K).
     * @param b Block vector that represents the B input matrix, will be transposed and must be passed in with op_transpose.
     *          The number of elements must be mmul::size_B (K * N).
     */
    template <typename VecA, BlockVectorOp VecB>
    requires((Vector<VecA> || BlockVector<VecA>)
                     && VecA::size() == size_A && VecB::size() == size_B
                     && std::is_same_v<typename VecA::value_type, TypeA>
                     && std::is_same_v<typename VecB::value_type, TypeB>
                     && VecB::is_operation(Operation::Transpose))
    __aie_inline
    mmul& mul(const VecA &a, const VecB &b)
    {
        mmul_impl::mul(a, true, b.parent1(), true);
        return *this;
    }

    /**
     * \brief Compute the result value of A * transpose(B) for the two given block matrices, and add it to the result.
     *
     * @param a Block vector that represents the A input matrix.
     *          The number of elements must be mmul::size_A (M * K).
     * @param b Block vector that represents the B input matrix, will be transposed and must be passed in with op_transpose.
     *          The number of elements must be mmul::size_B (K * N).
     */
    template <typename VecA, BlockVectorOp VecB>
    requires((Vector<VecA> || BlockVector<VecA>)
                     && VecA::size() == size_A && VecB::size() == size_B
                     && std::is_same_v<typename VecA::value_type, TypeA>
                     && std::is_same_v<typename VecB::value_type, TypeB>
                     && VecB::is_operation(Operation::Transpose))
    __aie_inline
    mmul& mac(const VecA &a, const VecB &b)
    {
        mmul_impl::mac(a, true, b.parent1(), true);
        return *this;
    }

    /**
     * \brief Return the result of the multiplication as a block vector of the requested type.
     */
    template <BlockType T>
    __aie_inline
    block_vector<T, size_C> to_vector() const
    {
        return mmul_impl::template to_vector<T>();
    }
#endif
};

template <unsigned M, unsigned K, unsigned N,
          AccumElemBaseType AccumTag = accauto,
          VectorOrOp VecA, VectorOrOp VecB>
    requires((VecA::size() == M * K) &&
                     (VecB::size() == K * N) &&
                     is_valid_mul_op_v<typename VecA::value_type, typename VecB::value_type>)
__aie_inline
auto mmul_fn(const VecA &a, const VecB &b)
{
    if      constexpr (!is_op_v<VecA>)
        return mmul_fn<M, K, N>(op_none(a), b);
    else if constexpr (!is_op_v<VecB>)
        return mmul_fn<M, K, N>(a, op_none(b));
    else {
        using TypeA = typename VecA::value_type;
        using TypeB = typename VecB::value_type;

        using accum_type = detail::accum_tag_or_default_t<AccumTag, TypeA, TypeB>;

        mmul<M, K, N, TypeA, TypeB, accum_type> mm;

        mm.mul(a, b);

        return mm;
    }
}

template <unsigned M, unsigned K, unsigned N,
          AccumOrOp Acc, VectorOrOp VecA, VectorOrOp VecB>
    requires((Acc::size() == M * N) &&
                     (VecA::size() == M * K) &&
                     (VecB::size() == K * N) &&
                     is_valid_mul_op_v<typename VecA::value_type, typename VecB::value_type>)
__aie_inline
auto mmac_fn(const Acc &c, const VecA &a, const VecB &b)
{
    if      constexpr (!is_op_v<Acc>)
        return mmac_fn<M, K, N>(op_add(c), a, b);
    else if constexpr (!is_op_v<VecA>)
        return mmac_fn<M, K, N>(c, op_none(a), b);
    else if constexpr (!is_op_v<VecB>)
        return mmac_fn<M, K, N>(c, a, op_none(b));
    else {
        using    TypeA = typename VecA::value_type;
        using    TypeB = typename VecB::value_type;
        using AccumTag = typename Acc::result_type::value_type;

        mmul<M, K, N, TypeA, TypeB, AccumTag> mm(c);

        mm.mac(a, b);

        return mm;
    }
}

template <unsigned M, unsigned K, unsigned N,
          typename Acc, VectorOrOp VecA, VectorOrOp VecB>
    requires((Acc::accum_type::size() == M * N) &&
                     (VecA::size() == M * K) &&
                     (VecB::size() == K * N) &&
                     is_valid_mul_op_v<typename VecA::value_type, typename VecB::value_type> &&
                     is_mmul_v<Acc>)
__aie_inline
auto mmac_fn(Acc &c, const VecA &a, const VecB &b)
{
    c.mac(a, b);

    return c;
}

//TODO: Document
template <typename T>
using cfr = detail::cfr<T>;

using detail::lut_oor_policy;

template <unsigned ParallelAccesses, typename OffsetType, typename SlopeType>
    requires (arch::is(arch::Gen2))
struct lut;

template <typename T>
struct is_lut
{
    static constexpr bool value() { return false; }
};

template <unsigned ParallelAccesses, typename OffsetType, typename SlopeType>
struct is_lut<lut<ParallelAccesses, OffsetType, SlopeType>>
{
    static constexpr bool value() { return true; }
};

template <typename T>
static constexpr bool is_lut_v = is_lut<T>::value();

/**
 * @ingroup group_basic_types_concepts
 *
 * @concept aie::ParallelLUT
 * Concept for parallel access LUT types. Accepts any @ref aie::ParallelLUT type.
 */
template <typename T>
concept ParallelLUT = is_lut_v<T>;

/**
 * @ingroup group_lut
 *
 * Abstraction to represent a LUT that is stored in memory, instantiated with pointer(s) to the already appropriately populated memory and the number of elements.
 *
 * The requirement on memory layout is that for degree N parallel accesses, N copies of the LUT data are required; i.e.
 *
 * - For a single load without parallelism, the values required to be stored linearly in memory.
 * - For 2 loads in parallel, the LUT needs to have 2 copies of the LUT values with repetition every bank width.
 *   For example with 32b values and a 128b bank width, in memory we would have the first 4 values (128b), then the same 4 again, then the next 4, which then repeat, etc.
 * - For 4 loads in parallel, we require the same layout as for 2 loads, but two distinct copies in this layout, placed in different memory banks.
 *
 * Currently the only supported implementation on this architecture is for 4 parallel accesses.
 *
 * @tparam ParallelAccesses Defines how many parallel accesses will be done in a single LUT access, possibilities depend on the hardware available for the given architecture
 * @tparam OffsetType Type of values stored within the lookup table.
 * @tparam SlopeType Optional template parameter, only needed in certain cases of linear approximation where the offset/slope value pair uses two different types.
 *
 */
template <unsigned ParallelAccesses, typename OffsetType, typename SlopeType=OffsetType>
    requires (arch::is(arch::Gen2))
struct lut : public detail::lut<ParallelAccesses, OffsetType, SlopeType>
{
    using offset_type = OffsetType;
    using slope_type  = SlopeType;
    using lut_impl = detail::lut<ParallelAccesses, OffsetType, SlopeType>;

    /**
     * Constructor for 4 parallel accesses. Each pointer points to an equivalent LUT populated within which the values are repeated twice, interleaved at a bank width granularity.
     * In total the same values need to be present 4 times in memory to allow for the 4 parallel accesses.
     *
     * For example, with a 128b bank width:
     *
     * @code
     * constexpr unsigned size = 8;
     * alignas(aie::vector_decl_align) const int32 lut_ab[size*2] = {value0, value1, value2, value3,
     *                                                               value0, value1, value2, value3, //note 128b duplication
     *                                                               value4, value5, value6, value7,
     *                                                               value4, value5, value6, value7};
     * alignas(aie::vector_decl_align) const int32 lut_cd[size*2] = {value0, value1, value2, value3,
     *                                                               value0, value1, value2, value3,
     *                                                               value4, value5, value6, value7,
     *                                                               value4, value5, value6, value7};
     * aie::lut<4, int32> lookup_table(size, lut_ab, lut_cd);
     * @endcode
     *
     * @warning Input table pointers must satisfy the alignment requirements of @ref aie::vector_decl_align
     *
     * @param LUT_elems Number elements in the LUT (not accounting for repetition).
     * @param LUT_ab    First two copies of the data, with the values repeated and interleaved at bank width granularity.
     * @param LUT_cd    Next two copies of the data, with the values repeated and interleaved at bank width granularity.
     */
    lut(unsigned LUT_elems, const void* LUT_ab, const void* LUT_cd) requires(ParallelAccesses == 4)
    : lut_impl(LUT_elems, LUT_ab, LUT_cd)
    {}

    /**
     * Constructor for two parallel accesses.
     *
     * For example, with a 128b bank width:
     *
     * @code
     * constexpr unsigned size = 8;
     * alignas(aie::vector_decl_align) const int32 lut_ab[size*2] = {value0, value1, value2, value3,
     *                                                               value0, value1, value2, value3, //note 128b duplication
     *                                                               value4, value5, value6, value7,
     *                                                               value4, value5, value6, value7};
     * aie::lut<2, int32> lookup_table(size, lut_ab);
     * @endcode
     *
     * @warning Input table pointers must satisfy the alignment requirements of @ref aie::vector_decl_align
     *
     * @param LUT_elems Number of elements in the LUT (not accounting for repetition).
     * @param LUT_ab    Two copies of the data, with the values interleaved at bank width granularity.
     */
    lut(unsigned LUT_elems, const void* LUT_ab) requires(ParallelAccesses == 2)
    : lut_impl(LUT_elems, LUT_ab)
    {}

    /**
     * Constructor for singular access.
     *
     * For example,
     *
     * @code
     * constexpr unsigned size = 8;
     * alignas(aie::vector_decl_align) const int32 lut_a[size] = {value0, value1, value2, value3,
     *                                                            value4, value5, value6, value7};
     * aie::lut<1, int32> lookup_table(size, lut_a);
     * @endcode
     *
     * @warning Input table pointers must satisfy the alignment requirements of @ref aie::vector_decl_align
     *
     * @param LUT_elems Number of elements in the LUT.
     * @param LUT_a     Pointer to the LUT values.
     */
    lut(unsigned LUT_elems, const void* LUT_a) requires(ParallelAccesses == 1)
    : lut_impl(LUT_elems, LUT_a)
    {}
};

/**
 * @ingroup group_lut
 *
 * \note
 * Linear approximation functionality is only available from AIE-ML/XDNA1
 *
 * Type to support a linear approximation via interpolation with slope/offset values stored in a lookup table.
 *
 * The offset values are simply the samples of the function to be approximated.
 * The slope values, which are the slopes of the function at the corresponding sample, are used in conjunction with the input to
 * more accurately estimate the function value between sample points.
 *
 * The logical steps of the computation for an integer based linear approximation are:
 *  - index = (input >> step_bits) + bias
 *  - slope/offset pair read from LUT based on index
 *  - output = slope * (input & ((1 << step_bits) - 1)) + (offset << shift_offset)
 *
 * while the steps for a floating point based approximation are:
 *  - index = (int(floor(input)) >> step_bits) + bias
 *  - slope/offset pair read from LUT based on index
 *  - output = slope * input + offset
 *
 * Note that for integer based linear approximations, the slope is multiplied by an integer value in the range
 * [0, 1 << step_bits) and therefore tweaking of the LUT values or linear_approx parameters may be required to
 * ensure that offset[i] + slope[i] * ((1 << step_bits) - 1) approximately equals offset[i+1].
 *
 * The slope and offset values are expected to be placed adjacent in memory. The number of achieved lookups per cycle
 * is determined by the \ref aie::lut object that encapsulates the contents of the lookup table.
 * The following example shows the memory layout of a 128b bank width lookup table with 16b values and slopes, which achieves 4 lookups per cycle:
 *
 * @code
 * constexpr unsigned size = 8;
 * alignas(aie::vector_decl_align) const int16 lut_ab[size*2*2] = {slope0, offset0, slope1, offset1, slope2, offset2, slope3, offset3,
 *                                                                 slope0, offset0, slope1, offset1, slope2, offset2, slope3, offset3, //note 128b duplication
 *                                                                 slope4, offset4, slope5, offset5, slope6, offset6, slope7, offset7,
 *                                                                 slope4, offset4, slope5, offset5, slope6, offset6, slope7, offset7};
 * alignas(aie::vector_decl_align) const int16 lut_cd[size*2*2] = {slope0, offset0, slope1, offset1, slope2, offset2, slope3, offset3,
 *                                                                 slope0, offset0, slope1, offset1, slope2, offset2, slope3, offset3,
 *                                                                 slope4, offset4, slope5, offset5, slope6, offset6, slope7, offset7,
 *                                                                 slope4, offset4, slope5, offset5, slope6, offset6, slope7, offset7};
 * aie::lut<4, int16, int16> lookup_table(size, lut_ab, lut_cd);
 * @endcode
 *
 * <table>
 * <caption>Supported linear approximation types</caption>
 * <tr><th>Input<th>Offset<th>Slope<th>Accumulator type<th>Lanes<th>Minumum step_bits required
 * <tr><td>%int8<td>%int8 <td>%int8 <td>%acc32 <td>32 <td> 2
 * <tr><td>%int16<td>%int16 <td>%int16 <td>%acc64 <td>16 <td> 3
 * <tr><td>%int16<td>%int32 <td>%int32 <td>%acc64 <td>16 <td> 4
 * <tr><td>%bfloat16<td>%float<td>%bfloat16 <td>%accfloat <td>16 <td> 0
 * </table>
 *
 * Note that while the floating point linear approx requires the offset data to be 32b floats, the slope data is required to be bfloat16.
 * However, it is required that all values in the LUT be 32b to ensure the LUT is correctly aligned.
 * While it is safe to use floats as the storage type for the lookup table, it is required that the low 16 mantissa bits of the floating
 * point slope value be zero.
 *
 * @tparam T Type of the input vector, containing values used to index the lookup table.
 * @tparam MyLUT Definition of the LUT type, using the @ref lut type.
 */
template <typename T, ParallelLUT MyLUT>
    requires (arch::is(arch::Gen2))
struct linear_approx
{
    /**
     * Constructor, configures aspects of how the approximation is performed.
     *
     * @param l            LUT containing the stored slope/offset value pairs used for the linear approximation. Each value in the LUT has the slope in the LSB, the offset in the MSB.
     * @param step_bits    Lower bits that won't be used from the input to index the LUT. For integer input, these will be the remainder multiplied by the slope value at each point. For float values, the input values are used directly in the multiplication
     * @param bias         Optional offset added to the input values used to index, for example to center on 0 by adding half the number of LUT elements.
     * @param shift_offset Optional scaling factor applied to the offset before adding it (to avoid loss of precision).
     */
    __aie_inline
    linear_approx(const MyLUT &l, unsigned step_bits, int bias = 0, int shift_offset = 0) : linear_approx_(l, step_bits, bias, shift_offset)
    {}

    /**
     * Performs a linear approximation for the input values with the configured lookup table.
     * An accumulator of the same number of elements as the input is returned.
     * <br/>Input values are interpreted from MSB to LSB: headroom | LUT elements | step_bits
     *
     * @param input Vector of input values that are used to index the look-up table.
     */
    template <Vector Vec>
    __aie_inline
    auto compute(const Vec &input) {
        return linear_approx_.compute(input);
    }

private:
    using linear_approx_impl = detail::linear_approx<T, typename MyLUT::lut_impl>;
    linear_approx_impl linear_approx_;
};

/**
 * @ingroup group_lut
 *
 * \note
 * Parallel lookup functionality is only available from AIE-ML/XDNA1
 *
 * Type with functionality to directly index a LUT based on input vector of values.
 * The number of achieved lookups per cycle is determined by the \ref aie::lut object that encapsulates the contents of the lookup table.
 * Refer to \ref aie::lut for more details.
 *
 * Real signed and unsigned integer types (>=8b) are supported as indices.
 * All types (>=8b) are supported as value types, including bfloat16, real, and complex types.
 * \note 8b value type lookups require the data to be stored in the lookup tables as 16b values due to the granularity of the memory accesses.
 *
 * @tparam T          Type of the input vector, containing values used to index the lookup table.
 * @tparam MyLUT      Definition of the LUT type, using the @ref lut type
 * @tparam oor_policy Defines the "out of range policy" for when index values on the input go beyond the size of the LUT. It can either saturate, taking on the min/max valid index, or truncate, retaining the lower bits for unsigned indicies or wrapping in the interval [-bias,lut_size-bias) for signed indices. Saturating is the default behaviour, but for certain non-linear functions which repeat after an interval truncation may be required.
 */
template <typename T, ParallelLUT MyLUT, lut_oor_policy oor_policy = lut_oor_policy::saturate>
    requires (arch::is(arch::Gen2))
struct parallel_lookup
{
    /**
     * Constructor for signed input types, configures aspects of how the lookup is performed.
     *
     * Note that usage of step_bits requires either:
     * - The rounding mode is set to the default `aie::rounding_mode::floor`
     * - The lowest step_bits of the index are zero
     *
     * @param l         LUT containing the stored values used for the linear approximation.
     * @param step_bits Optional lower bits that will be ignored for indexing the LUT.
     * @param bias      Optional offset added to the input values used to index, for example to center on 0 by adding half the number of LUT elements. This value, if supplied, must be a power of 2.
     */
    template <typename U = T> requires(std::is_signed_v<T>)
    __aie_inline
    parallel_lookup(const MyLUT &l, unsigned step_bits = 0, unsigned bias = 0) : parallel_lookup_(l, step_bits, bias)
    {}

    /**
     * Constructor for unsigned input types, configures aspects of how the lookup is performed.
     *
     * Note that usage of step_bits requires either:
     * - The rounding mode is set to the default `aie::rounding_mode::floor`
     * - The lowest step_bits of the index are zero
     *
     * @param l         LUT containing the stored values used for the linear approximation.
     * @param step_bits Optional lower bits that will be ignored for indexing the LUT.
     */
    template <typename U = T> requires(std::is_unsigned_v<T>)
    __aie_inline
    parallel_lookup(const MyLUT &l, unsigned step_bits = 0) : parallel_lookup_(l, step_bits, 0)
    {}

    /**
     * Accesses the lookup table based on the provided input values, will return a vector of the same number of elements as the input vector.
     * <br/>Input values are interpreted from MSB to LSB: headroom | LUT elements | step_bits
     *
     * Note the step_bits are required to be zeroed if the rounding mode is set to anything other than `aie::rounding_mode::floor`.
     *
     * @param input Vector of input values that are used to index the look-up table.
     */
    template <Vector Vec, unsigned N = Vec::size()>
    __aie_inline
    vector<typename MyLUT::offset_type, N> fetch(const Vec &input)
    {
        return parallel_lookup_.fetch(input);
    }

    /**
     * Accesses the lookup table based on the provided input values.
     * This overload allows the size of the returned vector to be specified as a template parameter.
     * This may be required when mapping small index types to large value types as a direct mapping may not be valid.
     * For example, mapping `int8` to `cint32` on a given architecture may require `input` to be 16 elements.
     * `fetch(input)` would therefore deduce a return type of `aie::vector<cint32, 16>`, which may be unsupported.
     * However, returning `aie::vector<cint32, 8>` by calling `fetch<8>(input)` may be valid.
     * <br/>Input values are interpreted from MSB to LSB: headroom | LUT elements | step_bits
     *
     * Note the step_bits are required to be zeroed if the rounding mode is set to anything other than `aie::rounding_mode::floor`.
     *
     * @tparam N The number of elements to lookup, which may be less than the input vector size
     * @param input Vector of input values that are used to index the look-up table.
     */
    template <unsigned N, Vector Vec>
    __aie_inline
    vector<typename MyLUT::offset_type, N> fetch(const Vec &input)
    {
        return parallel_lookup_.template fetch<Vec, N>(input);
    }

private:
    using parallel_lookup_impl = detail::parallel_lookup<T, typename MyLUT::lut_impl, oor_policy>;
    parallel_lookup_impl parallel_lookup_;
};

/**
 * @ingroup group_elementary
 *
 * Returns the square root of a value.
 *
 * \f$ f(a) = \sqrt{a} \f$
 *
 * @param a Input value. It must meet @ref aie::Elem.
 */
template <RealElem E>
__aie_inline
auto sqrt(E a)
{
    using T_base = operand_base_type_t<E>;
    return detail::elementary<detail::ElementaryOp::Sqrt, T_base, T_base>::run(a);
}

/**
 * @ingroup group_elementary
 *
 * Returns the inverse value.
 *
 * \f$ f(a) = a^{-1} \f$
 *
 * @param a Input value. It must meet @ref aie::RealElem.
 */
template <RealElem E>
__aie_inline
auto inv(E a)
{
    using T_base = operand_base_type_t<E>;
    return detail::elementary<detail::ElementaryOp::Inv, T_base, T_base>::run(a);
}

/**
 * @ingroup group_elementary
 *
 * Returns the inverse value of a complex number.
 *
 * \f$ z^{-1} = \frac{1}{a + bi} = \frac{a - bi}{a^2 + b^2} \f$
 *
 * @param a Input value. It must meet @ref aie::ComplexElem.
 */
template <ComplexElem C>
__aie_inline
auto inv(C a)
{
    using T_base = operand_base_type_t<C>;
    T_base num = conj(a);
    auto den = abs_square(a);
    return T_base{num.real / den, num.imag / den};
}

/**
 * @ingroup group_elementary
 *
 * Performs an approximation of the inversed square root of a single value.
 *
 * \f$ f(a) \approx \frac{1}{\sqrt{a}} \f$
 *
 * \note Due to the variations in support across architectures, the results of this function may differ slightly across devices.
 *
 * @param a Input value. The type must meet @ref aie::Elem.
 */
template <RealElem E>
__aie_inline
auto invsqrt(E a)
{
    using T_base = operand_base_type_t<E>;
    return detail::elementary<detail::ElementaryOp::InvSqrt, T_base, T_base>::run(a);
}

/**
 * @ingroup group_elementary
 *
 * Performs a sine operation on a single value. The input can either be a float value in radians or an integer. The
 * floating-point range is [-Pi, Pi]. Integer values are handled as a fixed-point input value in Q1.31 format scaled
 * with 1/Pi (input value 2^31 corresponds to Pi). In this case only the upper 20-bit of the input value are used.
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <typename E> requires(arch::is(arch::AIE) &&
                               (RealElem<E> || std::is_integral_v<E>))
__aie_inline
auto sin(E a)
{
    using T_base = operand_base_type_t<E>;

    if constexpr (std::is_same_v<T_base, float>)
        return detail::elementary<detail::ElementaryOp::Sin, float, float>::run(a);
    else
        return detail::elementary<detail::ElementaryOp::Sin, T_base, uint32>::run((uint32) T_base(a));
}

/**
 * @ingroup group_elementary
 *
 * Performs a cosine operation on a single value. The input can either be a float value in radians or an integer. The
 * floating-point range is [-Pi, Pi]. Integer values are handled as a fixed-point input value in Q1.31 format scaled
 * with 1/Pi (input value 2^31 corresponds to Pi). In this case only the upper 20-bit of the input value are used.
 *
 * According to input type, the returned value is either a float or a signed Q.15 fixed-point format.
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <typename E> requires(arch::is(arch::AIE) &&
                               (RealElem<E> || std::is_integral_v<E>))
__aie_inline
auto cos(E a)
{
    using T_base = operand_base_type_t<E>;

    if constexpr (std::is_same_v<T_base, float>)
        return detail::elementary<detail::ElementaryOp::Cos, float, float>::run(a);
    else
        return detail::elementary<detail::ElementaryOp::Cos, T_base, uint32>::run((uint32) T_base(a));
}

// TODO: should this change to match the output order in sincos_complex?
/**
 * @ingroup group_elementary
 *
 * Performs both sin and cos, and returns both values as an std::pair (sin first, then cos). The input can either be a
 * float value in radians or an integer. The floating-point range is [-Pi, Pi]. Integer values are handled as a
 * fixed-point input value in Q1.31 format scaled with 1/Pi (input value 2^31 corresponds to Pi). In this case only the
 * upper 20-bit of the input value are used.
 *
 * According to input type, the returned values are either float or signed Q.15 fixed-point format.
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <typename E> requires(arch::is(arch::AIE) &&
                               (RealElem<E> || std::is_integral_v<E>))
__aie_inline
auto sincos(E a)
{
    using T_base = operand_base_type_t<E>;

    if constexpr (std::is_same_v<T_base, float>)
        return detail::elementary<detail::ElementaryOp::SinCos, float, float>::run(a);
    else
        return detail::elementary<detail::ElementaryOp::SinCos, T_base, uint32>::run((uint32) T_base(a));
}

/**
 * @ingroup group_elementary
 *
 * Same as sincos, but returns both values as the real and imaginary parts of a complex number (cos in the real part,
 * sin in the imaginary part).
 *
 * Each part of the returned complex number is either a float or a signed Q.15 fixed-point format.
 *
 * @param a Value. The type must meet @ref aie::RealElem.
 */
template <typename E> requires(arch::is(arch::AIE) &&
                               (RealElem<E> || std::is_integral_v<E>))
__aie_inline
auto sincos_complex(E a) -> std::conditional_t<detail::is_floating_point_v<operand_base_type_t<E>>, cfloat, cint16>
{
    using T_base = operand_base_type_t<E>;

    if constexpr (std::is_same_v<T_base, float>)
        return detail::elementary<detail::ElementaryOp::SinCosComplex, float, float>::run(a);
    else
        return detail::elementary<detail::ElementaryOp::SinCosComplex, cint16, uint32>::run((uint32) T_base(a));
}

/**
 * @ingroup group_elementary
 *
 * Returns the square root of all values in a vector.
 *
 * \f$ f(v_i) = \sqrt{v_i} \f$
 *
 * \note This is a scalar operation. Even though the function operates with vectors, the actual operation is applied to each element individually.
 *
 * @param v Input value. It must meet @ref aie::Vector.
 */
template <RealVector Vec>
__aie_inline
auto sqrt(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::elementary_vector<detail::ElementaryOp::Sqrt, T, T, Elems>::run(v);
}

/**
 * @ingroup group_elementary
 *
 * Returns the inverse value of every element in a vector.
 *
 * \f$ f(v_{i}) = v_{i}^{-1} \f$
 *
 * \note This is a scalar operation. Even though the function operates with vectors, the actual operation is applied to each element individually.
 *
 * @param v Input values. It must meet @ref aie::Vector
 */
template <Vector Vec>
__aie_inline
auto inv(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::elementary_vector<detail::ElementaryOp::Inv, T, T, Elems>::run(v);
}

/**
 * @ingroup group_elementary
 *
 * Performs an approximation of the inversed square root of every element in a vector.
 *
 * \f$ f(v_{i}) \approx \frac{1}{\sqrt{v_{i}}} \f$
 *
 * \note Due to the variations in support across architectures, the results of this function may differ slightly across devices.
 *
 * @param v Input values. The type must meet @ref aie::Vector.
 */
template <RealVector Vec>
__aie_inline
auto invsqrt(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    return detail::elementary_vector<detail::ElementaryOp::InvSqrt, T, T, Elems>::run(v);
}

/**
 * @ingroup group_elementary
 *
 * Performs a cosine operation on all elements in the input vector. The input vector can either be of float values in
 * radians or integers. The floating-point range is [-Pi, Pi]. Integer values are handled as a fixed-point input value
 * in Q1.31 format scaled with 1/Pi (input value 2^31 corresponds to Pi). In this case only the upper 20-bit of the
 * input value are used.
 *
 * According to input type, returns a vector of float or of signed Q.15 fixed-point format.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 *
 * \note This is a scalar operation. Even though the function operates with vectors, the actual operation is applied to each element individually.
 */
template <RealVector Vec> requires (arch::is(arch::AIE))
__aie_inline
auto cos(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (Vec::is_floating_point())
        return detail::elementary_vector<detail::ElementaryOp::Cos, float, float, Elems>::run(v);
    else
        return detail::elementary_vector<detail::ElementaryOp::Cos, T, T, Elems>::run(v);
}

/**
 * @ingroup group_elementary
 *
 * Performs a sine operation on all elements in the input vector. The input vector can either be of float values in
 * radians or integers. The floating-point range is [-Pi, Pi]. Integer values are handled as a fixed-point input value
 * in Q1.31 format scaled with 1/Pi (input value 2^31 corresponds to Pi). In this case only the upper 20-bit of the
 * input value are used.
 *
 * According to input type, returns a vector of float or of signed Q.15 fixed-point format.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 *
 * \note This is a scalar operation. Even though the function operates with vectors, the actual operation is applied to each element individually.
 */
template <RealVector Vec> requires (arch::is(arch::AIE))
__aie_inline
auto sin(const Vec &v) -> aie_dm_resource_remove_t<Vec>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (Vec::is_floating_point())
        return detail::elementary_vector<detail::ElementaryOp::Sin, float, float, Elems>::run(v);
    else
        return detail::elementary_vector<detail::ElementaryOp::Sin, T, T, Elems>::run(v);
}

/**
 * @ingroup group_elementary
 *
 * Same as sin and cos, but performs both operations and returns a std::pair of vectors of result values.
 * The first vector contains the sine values, the second contains the cosine values
 *
 * According to input type, returns a pair of vectors of float or of signed Q.15 fixed-point format.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 *
 * \note This is a scalar operation. Even though the function operates with vectors, the actual operation is applied to each element individually.
 */
template <RealVector Vec> requires (arch::is(arch::AIE))
__aie_inline
auto sincos(const Vec &v) -> std::pair<aie_dm_resource_remove_t<Vec>, aie_dm_resource_remove_t<Vec>>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (Vec::is_floating_point())
        return detail::elementary_vector<detail::ElementaryOp::SinCos, float, float, Elems>::run(v);
    else
        return detail::elementary_vector<detail::ElementaryOp::SinCos, T, T, Elems>::run(v);
}

/**
 * @ingroup group_elementary
 *
 * Same as sincos, but returns both values as the real and imaginary parts in a vector of complex values
 * (cos in the real part, sin in the imaginary).
 *
 * According to input type, returns a vector of float or of signed Q.15 fixed-point format.
 *
 * @param v Input vector. The type must meet @ref aie::RealVector.
 */
template <RealVector Vec> requires (arch::is(arch::AIE))
__aie_inline
auto sincos_complex(const Vec &v) -> vector<std::conditional_t<Vec::is_floating_point(), cfloat, cint16>, Vec::size()>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (Vec::is_floating_point())
        return detail::elementary_vector<detail::ElementaryOp::SinCosComplex, cfloat, float, Elems>::run(v);
    else
        return detail::elementary_vector<detail::ElementaryOp::SinCosComplex, cint16, T, Elems>::run(v);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert a floating-point value into a fixed-point value, rounded to the nearest integer.
 *
 * @param a     Input value.
 * @param shift Position of the point in the output value.
 *
 * @tparam TR   Type of the returned fixed-point value.
 *
 * \note Global settings, such as rounding and saturation modes, do not affect this API.
 *       The conversion relies on a hardware @c float2fix() implementation that applies a
 *       fixed round-to-nearest behavior (e.g. @ref aie::set_rounding "aie::set_rounding(aie::rounding_mode::floor)" has no effect here).
 */
template <typename TR = int32> requires(Utils::is_one_of_v<TR, int8, int16, int32>)
__aie_inline
auto to_fixed(float a, int shift = 0)
{
    return detail::elementary<detail::ElementaryOp::Float2Fix, TR, float>::run(a, shift);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert a complex floating-point value into a complex fixed-point value, with each component rounded to the nearest integer.
 *
 * @param a     Input value.
 * @param shift Position of the point in the output value.
 *
 * @tparam TR   Type of the returned fixed-point value.
 *
 * \note Global settings, such as rounding and saturation modes, do not affect this API.
 *       The conversion relies on a hardware @c float2fix() implementation that applies a
 *       fixed round-to-nearest behavior and does not observe software-configured rounding modes
 *       (e.g. @ref aie::set_rounding "aie::set_rounding(aie::rounding_mode::floor)" has no effect here).
 */
template <typename TR = cint32> requires(arch::is(arch::AIE, arch::AIE_ML) &&
                                         Utils::is_one_of_v<TR, cint16, cint32>)
__aie_inline
auto to_fixed(cfloat a, int shift = 0)
{
    return TR{ to_fixed<decltype(TR::real)>(a.real, shift),
               to_fixed<decltype(TR::imag)>(a.imag, shift) };
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a floating point vector into fixed-point values.
 *
 * @param v     Input vector.
 * @param shift Position of the point in the output values.
 *
 * @tparam TR   Type of the returned fixed-point values.
 */
template <typename TR = int32, unsigned Elems>
    requires((arch::is(arch::Gen2) && Utils::is_one_of_v<TR, int4, uint4, int8, uint8, int16, uint16, int32, uint32>) ||
             (arch::is(arch::Gen1) && Utils::is_one_of_v<TR, int16, int32>))
__aie_inline
auto to_fixed(const vector<float, Elems> &v, int shift = 0) -> vector<TR, Elems>
{
    return detail::elementary_vector<detail::ElementaryOp::Float2Fix, TR, float, Elems>::run(v, shift);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a floating point vector into fixed-point values.
 *
 * @param v     Input vector.
 * @param shift Position of the point in the output values.
 *
 * @tparam TR   Type of the returned fixed-point values.
 */
template <typename TR = cint32, unsigned Elems> requires((arch::is(arch::AIE) || (arch::is(arch::AIE_ML) && __AIE_API_COMPLEX_FP32_EMULATION__ == 1)) &&
                                                         Utils::is_one_of_v<TR, cint16, cint32>)
__aie_inline
auto to_fixed(const vector<cfloat, Elems> &v, int shift = 0) -> vector<TR, Elems>
{
    return to_fixed<decltype(TR::real)>(v.template cast_to<float>(), shift).template cast_to<TR>();
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a bfloat16 vector into fixed-point values.
 *
 * @param v     Input vector.
 * @param shift Position of the point in the output values.
 *
 * @tparam TR   Type of the returned fixed-point values.
 */
template <typename TR = int32, typename T, unsigned Elems>
    requires (arch::is(arch::Gen2) && std::is_same_v<T, bfloat16>)
__aie_inline
auto to_fixed(const vector<T, Elems> &v, int shift = 0) -> vector<TR, Elems>
{
    return detail::elementary_vector<detail::ElementaryOp::Float2Fix, TR, T, Elems>::run(v, shift);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a floating point accumulator into a vector of fixed-point values.
 *
 * @param v     Input accumulator.
 * @param shift Position of the point in the output values.
 *
 * @tparam TR   Type of the returned fixed-point values.
 */
template <typename TR = int32, unsigned Elems>
    requires(arch::is(arch::Gen2) && Utils::is_one_of_v<TR, int4, uint4, int8, uint8, int16, uint16, int32, uint32>)
__aie_inline
auto to_fixed(const accum<accfloat, Elems> &v, int shift = 0) -> vector<TR, Elems>
{
    return to_fixed<TR, Elems>(v.template to_vector<float>(), shift);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a floating point vector into fixed-point values.
 *
 * @param v     Input vector.
 * @param shift Position of the point in the output values.
 *
 * @tparam TR   Return element type of the result vector. Must be a signed integer type.
 */
template <typename TR = int32, typename T, unsigned Elems>
    requires (!detail::is_floating_point_v<TR>
              && detail::is_signed_v<TR>
              && detail::is_floating_point_v<T>
              && detail::type_bits_v<TR> <= detail::type_bits_v<T>
              && arch::is(arch::Gen2))
__aie_inline
auto to_fixed_sign(const vector<T, Elems> &v, bool v_sign, int shift = 0) -> vector<TR, Elems>
{
    return detail::elementary_vector<detail::ElementaryOp::Float2Fix, TR, T, Elems>::run(v, shift, v_sign);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a floating point accumulator into a vector of fixed-point values.
 *
 * @param v      Input accumulator.
 * @param v_sign Dynamic sign of the point in the output values.
 * @param shift  Position of the point in the output values.
 *
 * @tparam TR   Type of the returned fixed-point values.
 */
template <typename TR = int32, unsigned Elems>
    requires(arch::is(arch::Gen2) && Utils::is_one_of_v<TR, int4, int8, int16, int32>)
__aie_inline
auto to_fixed_sign(const accum<accfloat, Elems> &v, bool v_sign, int shift = 0) -> vector<TR, Elems>
{
    return detail::elementary_vector<detail::ElementaryOp::Float2Fix, TR, float, Elems>::run(v.template to_vector<float>(), shift, v_sign);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a bfloat16 vector into floored fixed-point values.
 *
 * @param v     Input vector.
 * @param shift Position of the point in the output values.
 *
 * @tparam TR   Type of the returned fixed-point values.
 */
template <typename TR = int32, typename T, unsigned Elems>
    requires (arch::is(arch::Gen2)
              && Utils::is_one_of_v<TR, int16, int32>
              && (arch::is(arch::Gen2) && std::is_same_v<T, bfloat16>))
__aie_inline
auto to_fixed_floor(const vector<T, Elems> &v, int shift = 0) -> vector<TR, Elems>
{
    return detail::elementary_vector<detail::ElementaryOp::Float2FixFloor, TR, T, Elems>::run(v, shift);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert a fixed-point value into a floating point value.
 *
 * @param a     Input value.
 * @param shift Position of the point in the input value.
 */
template <typename TR = float>
__aie_inline
auto to_float(int a, int shift = 0)
{
    return detail::elementary<detail::ElementaryOp::Fix2Float, TR, int>::run(a, shift);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert a complex fixed-point value into a floating point value.
 *
 * @param a     Input value.
 * @param shift Position of the point in the input value.
 */
template <typename TR = cfloat, typename T> requires(Utils::is_one_of_v<T, cint16, cint32>)
__aie_inline
auto to_float(T a, int shift = 0)
{
    return cfloat{ to_float<decltype(TR::real)>(a.real, shift),
                   to_float<decltype(TR::imag)>(a.imag, shift) };
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a real fixed-point vector into floating point values.
 *
 * @param v     Input vector.
 * @param shift Position of the point in the input values.
 */
template <typename TR = float, VectorOrOp Vec>
    requires((arch::is(arch::Gen2) && Utils::is_one_of_v<typename Vec::value_type, int4, uint4, int8, uint8, int16, uint16, int32, uint32>) ||
             (arch::is(arch::Gen1) && Utils::is_one_of_v<typename Vec::value_type, int16, int32>))
__aie_inline
auto to_float(const Vec &v, int shift = 0) -> vector<TR, Vec::size()>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    if constexpr (!is_op_v<Vec>)
        return to_float<TR>(op_none(v), shift);
    else if constexpr (Vec::is_operation_not(Operation::Sign))
        return to_float<TR>(v(), shift);
    else {
        using impl = detail::elementary_vector<detail::ElementaryOp::Fix2Float, TR, T, Elems>;
#if __AIE_ARCH__ == 10
        return impl::run(v.parent1(), shift, detail::is_signed_v<T>);
#else
        if constexpr (Vec::is_operation_none())
            return impl::run(v.parent1(), shift, detail::is_signed_v<T>);
        else
            return impl::run(v.parent1(), shift, v.parent2());
#endif
    }
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a real fixed-point accumulator into floating point values.
 *
 * @param acc   Input accumulator.
 * @param shift Position of the point in the input values.
 */
template <typename TR = float, Accum Acc>
    requires(std::is_same_v<TR, float> && arch::is(arch::Gen2))
__aie_inline
auto to_float(const Acc &acc, int shift = 0) -> vector<TR, Acc::size()>
{
    using T = typename Acc::value_type;
    constexpr unsigned Elems = Acc::size();

    return detail::elementary_acc<detail::ElementaryOp::Fix2Float, TR, T, Elems>::run(acc, shift);
}

/**
 * @ingroup group_fp_conversion
 *
 * Convert the elements in a complex fixed-point vector into floating point values.
 *
 * @param v     Input vector.
 * @param shift Position of the point in the input values.
 */
template <typename TR = cfloat, typename T, unsigned Elems> requires(Utils::is_one_of_v<T, cint16, cint32>)
__aie_inline
auto to_float(const vector<T, Elems> &v, int shift = 0) -> vector<TR, Elems>
{
    return to_float<decltype(TR::real)>(v.template cast_to<decltype(T::real)>(), shift).template cast_to<TR>();
}

// Namespace scalar fp overloads
/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns whether the first value is smaller than the second value.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
bool lt(T a, T b) requires(detail::is_floating_point_v<T> && !detail::is_complex_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return detail::lt<T, 256 / (sizeof(T) * 8)>::run(v_a, v_b).test(0);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator addition
    return a < b;
#endif
}

/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns whether the first value is larger than the second value.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
bool gt(T a, T b) requires(detail::is_floating_point_v<T> && !detail::is_complex_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return detail::lt<T, 256 / (sizeof(T) * 8)>::run(v_b, v_a).test(0);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator addition
    return a > b;
#endif
}

/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns whether the first value is smaller or equal than the second value.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
bool le(T a, T b) requires(detail::is_floating_point_v<T> && !detail::is_complex_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return !gt(a, b);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator addition
    return a <= b;
#endif
}

/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns whether the first value is larger or equal than the second value.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
bool ge(T a, T b) requires(detail::is_floating_point_v<T> && !detail::is_complex_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return detail::ge<T, 256 / (sizeof(T) * 8)>::run(v_a, v_b).test(0);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator addition
    return a >= b;
#endif
}

/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns whether they are not equal.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
bool neq(T a, T b) requires(detail::is_floating_point_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return detail::neq<T, 256 / (sizeof(T) * 8)>::run(v_a, v_b).test(0);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator addition
    return a != b;
#endif
}

/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns whether they are equal.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
bool eq(T a, T b) requires(detail::is_floating_point_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return !detail::neq<T, 256 / (sizeof(T) * 8)>::run(v_a, v_b).test(0);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator addition
    return a == b;
#endif
}

/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns the maximum value.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
T max(T a, T b) requires(detail::is_floating_point_v<T>)
{
    constexpr unsigned n = detail::native_vector_length_v<T>;
    return detail::max<T, n>::run(vector<T, n>(a), vector<T, n>(b))[0];
}

/**
 * @ingroup group_fp_scalar
 * Compares the two input values and returns the minimum value.
 *
 * @param a First value.
 * @param b Second value.
 */
template <typename T>
__aie_inline
T min(T a, T b) requires(detail::is_floating_point_v<T>)
{
    constexpr unsigned n = detail::native_vector_length_v<T>;
    return detail::min<T, n>::run(vector<T, n>(a), vector<T, n>(b))[0];
}

template <typename T>
__aie_inline
T add(T a, T b) requires(detail::is_floating_point_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return detail::add<T, 256 / (sizeof(T) * 8)>::run(v_a, v_b)[0];
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator addition
    return a + b;
#endif
}

template <typename T>
__aie_inline
T sub(T a, T b) requires(detail::is_floating_point_v<T>)
{
#if __AIE_ARCH__ == 10
    const vector<T, 256 / (sizeof(T) * 8)> v_a(a);
    const vector<T, 256 / (sizeof(T) * 8)> v_b(b);

    return detail::sub<T, 256 / (sizeof(T) * 8)>::run(v_a, v_b)[0];
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    // Implement with accumulator subtraction
    return a - b;
#endif
}

template <typename T1, typename T2>
__aie_inline
auto mul(T1 a, T2 b) requires(detail::is_floating_point_v<T1> &&
                                      detail::is_floating_point_v<T2>)
{
#if __AIE_ARCH__ == 10
    constexpr unsigned elems = std::max(256 / (sizeof(T1) * 8), 256 / (sizeof(T2) * 8));

    const vector<T1, elems> v_a(a);
    const vector<T2, elems> v_b(b);

    return detail::mul<detail::MulMacroOp::Mul, 32, T1, T2>::run(v_a, detail::is_signed_v<T1>, v_b, detail::is_signed_v<T2>, false).to_vector().get(0);
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    return a * b;
#endif
}

template <typename TR, typename T1, typename T2>
__aie_inline
TR mac(TR c, T1 a, T2 b) requires(detail::is_floating_point_v<T1> &&
                                          detail::is_floating_point_v<T2> &&
                                          std::is_same_v<decltype(mul(a, b)), TR>)
{
#if __AIE_ARCH__ == 10
    constexpr unsigned elems = std::max(256 / (sizeof(T1) * 8), 256 / (sizeof(T2) * 8));

    const vector<T1, elems> v_a(a);
    const vector<T2, elems> v_b(b);
    const vector<TR, elems> v_c(c);

    using acc_type = std::conditional_t<std::is_same_v<TR, float>, accfloat, caccfloat>;
    const accum<acc_type, elems> acc(v_c);

    return detail::mul<detail::MulMacroOp::Add_Mul, 32, T1, T2>::run(v_a, detail::is_signed_v<T1>, v_b, detail::is_signed_v<T2>, false, acc).to_vector()[0];
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    return c + a * b;
#endif
}

template <typename TR, typename T1, typename T2>
__aie_inline
TR msc(TR c, T1 a, T2 b) requires(detail::is_floating_point_v<T1> &&
                                          detail::is_floating_point_v<T2> &&
                                          std::is_same_v<decltype(mul(a, b)), TR>)
{
#if __AIE_ARCH__ == 10
    constexpr unsigned elems = std::max(256 / (sizeof(T1) * 8), 256 / (sizeof(T2) * 8));

    const vector<T1, elems> v_a(a);
    const vector<T2, elems> v_b(b);
    const vector<TR, elems> v_c(c);

    using acc_type = std::conditional_t<std::is_same_v<TR, float>, accfloat, caccfloat>;
    const accum<acc_type, elems> acc(v_c);

    return detail::mul<detail::MulMacroOp::Sub_Mul, 32, T1, T2>::run(v_a, detail::is_signed_v<T1>, v_b, detail::is_signed_v<T2>, false, acc).to_vector()[0];
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    return c - a * b;
#endif
}

__aie_inline
inline float div(float a, float b)
{
#if __AIE_ARCH__ == 10
    const vector<float, 8>  v_a(a);
    const vector<float, 8>  v_b(inv(b));

    return mul(v_a, v_b).to_vector()[0];
#elif __AIE_ARCH__ == 20 || __AIE_ARCH__ == 21 || __AIE_ARCH__ == 22
    return a / b;
#endif
}

/**
 * @ingroup group_arithmetic
 *
 * Returns the quotients of the element-wise division of each value of the first input vector by the corresponding
 * element in the second input vector.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a[i] / b[i];
 * @endcode
 *
 * @param a Vector of dividends. The type must meet @ref aie::Vector.
 * @param b Vector of divisors. The type must meet @ref aie::RealVector.
 */
template <Vector Vec1, RealVector Vec2>
    requires(Vec1::size() == Vec2::size() && detail::is_floating_point_v<typename Vec2::value_type>)
__aie_inline
auto div(const Vec1 &a, const Vec2 &b)
    -> accum<detail::accum_tag_for_mul_types<typename Vec1::value_type, typename Vec2::value_type>, Vec1::size()>
{
    return mul(a, inv(b));
}

/**
 * @ingroup group_arithmetic
 *
 * Returns the quotients of the element-wise division of each value of the first input vector by a scalar.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a[i] / b;
 * @endcode
 *
 * @param a Vector of dividends. The type must meet @ref aie::Vector.
 * @param b Divisor. The type must meet @ref aie::RealElem.
 */
template <Vector Vec, RealElem E>
    requires(
        detail::is_floating_point_v<typename Vec::value_type>
        && detail::is_valid_element_type_v<E>
        && detail::is_floating_point_v<E>)
__aie_inline
auto div(const Vec &a, E b)
    -> accum<detail::accum_tag_for_mul_types<typename Vec::value_type, E>, Vec::size()>
{
    auto divisor = broadcast<E, Vec::size()>(inv(b));
    return mul(a, divisor);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns the quotients of the element-wise division of two complex vectors 'a' and 'b'.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a[i] / b[i];
 * @endcode
 *
 * Complex division is achieved by multiplying both dividend and divisor by the conjugate of the divisor:
 *
 * @code
 * a = A + Bi; b = C + Di
 * a/b = (A + Bi) / (C + Di) = ((A + Bi) * (C - Di)) / ((C + Di) * (C - Di))
 *     = ((A + Bi) * (C - Di)) /  (C^2 + D^2)
 * @endcode
 *
 * @param a Vector of dividends. The type must meet @ref aie::ComplexVector.
 * @param b Vector of divisors. The type must meet @ref aie::ComplexVector.
 */
template <ComplexVector Vec>
    requires(arch::is(arch::AIE) || (arch::is(arch::AIE_ML, arch::AIE_MLv2) && __AIE_API_COMPLEX_FP32_EMULATION__ == 1))
__aie_inline
auto div(const Vec &a, const Vec &b) -> accum<detail::accum_tag_for_mul_types<
                                                    typename Vec::value_type,
                                                    detail::remove_complex_t<typename Vec::value_type>>,
                                              Vec::size()>
{
    using T = typename Vec::value_type;

    auto den = inv(abs_square(b));
    auto num = mul(aie::op_conj(b), a).template to_vector<T>();
    return mul(num, den);
}

/**
 * @ingroup group_arithmetic
 *
 * Returns the quotients of the element-wise division of a complex vector 'a' by a complex scalar 'b'.
 *
 * @code
 * for (unsigned i = 0; i < Elems; ++i)
 *     out[i] = a[i] / b;
 * @endcode
 *
 * @param a Vector of dividends. The type must meet @ref aie::ComplexVector.
 * @param b Divisor. The type must meet @ref aie::ComplexElem.
 */
template <ComplexVector Vec, ComplexElem E>
    requires(arch::is(arch::AIE) || (arch::is(arch::AIE_ML, arch::AIE_MLv2) && __AIE_API_COMPLEX_FP32_EMULATION__ == 1))
__aie_inline
auto div(const Vec &a, const E &b)
    -> accum<detail::accum_tag_for_mul_types<typename Vec::value_type, detail::remove_complex_t<E>>,
        Vec::size()>
{
    using T = typename Vec::value_type;
    constexpr unsigned Elems = Vec::size();

    auto vb = broadcast<T, Elems>(b);
    auto den = inv(abs_square(vb)[0]);
    auto num = mul(op_conj(vb), a).template to_vector<T>();

    return mul(num, broadcast<decltype(den), Elems>(den));
}

/**
 * @ingroup group_elementary
 *
 * Returns the square root of all values in an accumulator.
 *
 * \f$ f(v_i) = \sqrt{v_i} \f$
 *
 * \note This is a scalar operation. Even though the function operates with accumulators, the actual operation is applied to each element individually.
 *
 * @param v Input values.
 */
// TODO: simplify using concepts
template <unsigned Elems> requires (arch::is(arch::AIE))
__aie_inline
accum<accfloat, Elems> sqrt(const accum<accfloat, Elems> &v)
{
    const accum<accfloat, Elems> ret(sqrt(v.to_vector()));

    return ret;
}

/**
 * @ingroup group_elementary
 *
 * Returns the inverse value of every element in an accumulator.
 *
 * \f$ f(v_{i}) = v_{i}^{-1} \f$
 *
 * \note This is a scalar operation. Even though the function operates with accumulators, the actual operation is applied to each element individually.
 *
 * @param v Input values.
 */
template <unsigned Elems> requires (arch::is(arch::AIE))
__aie_inline
accum<accfloat, Elems> inv(const accum<accfloat, Elems> &v)
{
    const accum<accfloat, Elems> ret(inv(v.to_vector()));

    return ret;
}

/**
 * @ingroup group_elementary
 *
 * Performs an approximation of the inversed square root of every element in a floating point accumulator.
 *
 * \f$ f(v_{i}) \approx \frac{1}{\sqrt{v_{i}}} \f$
 *
 * @param v Input values.
 */
template <unsigned Elems> requires (arch::is(arch::AIE))
__aie_inline
accum<accfloat, Elems> invsqrt(const accum<accfloat, Elems> &v)
{
    const accum<accfloat, Elems> ret(invsqrt(v.to_vector()));

    return ret;
}

/**
 * @ingroup group_elementary
 *
 * Compute the hyperbolic tangent function on the elements of the input vector.
 *
 * @param v     Input vector.
 */
template <typename TR = bfloat16, unsigned Elems>
    requires (arch::is(arch::XDNA2, arch::AIE_MLv2))
__aie_inline
auto tanh(const vector<float, Elems> &v) -> vector<TR, Elems>
{
    return detail::elementary_vector<detail::ElementaryOp::Tanh, TR, float, Elems>::run(v);
}

/**
 * @ingroup group_elementary
 *
 * Compute the 2 raised to the power function for all elements in a vector.
 *
 * \f$ f(v_{i}) = 2^{v_{i}} \f$
 *
 * @param v Input values.
 */
template <typename TR = bfloat16, unsigned Elems>
    requires((arch::is(arch::XDNA2) && Utils::is_one_of_v<TR, bfloat16>) ||
             (arch::is(arch::AIE_MLv2) && Utils::is_one_of_v<TR, bfloat16, float>))
__aie_inline
auto exp2(const vector<float, Elems> &v) -> vector<TR, Elems>
{
    return detail::elementary_vector<detail::ElementaryOp::Exp2, TR, float, Elems>::run(v);
}

/**
 * Returns the zero value for the requested data type.
 *
 * @tparam Requested data type
 */
template <ElemBaseType T>
__aie_inline
constexpr T zero()
{
    if constexpr (detail::is_complex_v<T>)
        return T{0, 0};
    else
        return T{0};
}

/**
 * @ingroup group_config
 *
 * Set the saturation mode.
 *
 * @param m Saturation mode to be set
 */
__aie_inline
static inline void set_saturation(saturation_mode m)
{
    aie::tile::current().set_saturation(m);
}

/**
 * @ingroup group_config
 *
 * Get the current saturation mode.
 */
__aie_inline
static inline saturation_mode get_saturation()
{
    return aie::tile::current().get_saturation();
}

/**
 * @ingroup group_config
 *
 * Set the saturation mode and return the previously set mode.
 *
 * @param m Saturation mode to be set
 */
__aie_inline
static inline saturation_mode swap_saturation(saturation_mode m)
{
    saturation_mode ret = get_saturation();

    set_saturation(m);

    return ret;
}

/**
 * @ingroup group_config
 *
 * Set the rounding mode used in accumulator to vector conversions.
 *
 * @param m Rounding mode to be set
 */
__aie_inline
static inline void set_rounding(rounding_mode m)
{
    aie::tile::current().set_rounding(m);
}

/**
 * @ingroup group_config
 *
 * Get the current rounding mode used in accumulator to vector conversions.
 */
__aie_inline
static inline rounding_mode get_rounding()
{
    return aie::tile::current().get_rounding();
}

/**
 * @ingroup group_config
 *
 * Set the rounding mode used in accumulator to vector conversions and return the previously set mode.
 *
 * @param m Rounding mode to be set
 */
__aie_inline
static inline rounding_mode swap_rounding(rounding_mode m)
{
    rounding_mode ret = get_rounding();

    set_rounding(m);

    return ret;
}

UNARY_OP_IMPL(Abs)
{
    return abs(this->parent1());
}

UNARY_OP_IMPL(Conj)
{
    return conj(this->parent1());
}

BINARY_OP_IMPL(Zero)
{
    if(this->parent2() == true) {
        using acc_type = decltype(this->parent1());
        return zeros<typename acc_type::value_type, acc_type::size()>();
    }
    else{
        return this->parent1();
    }
}

BINARY_OP_IMPL(Max)
{
    return max(this->parent1(), this->parent2());
}

BINARY_OP_IMPL(Min)
{
    return min(this->parent1(), this->parent2());
}

BINARY_OP_IMPL(Neg)
{
    return this->parent2() ? neg(this->parent1()) : this->parent1();
}

}

// aie_adf.hpp brings in the ADF stream API, which includes <adf.h>. That header
// is part of the Vitis ADF framework and is not present in bare-metal, IRON, or
// mlir-aie builds. Only pull it in when it is actually available, so aie_api can
// be used outside the ADF flow; when __has_include is unavailable, fall back to
// the previous behavior of including it under __AIENGINE__.
#if defined(__AIENGINE__) && (!defined(__has_include) || __has_include(<adf.h>))
#include "aie_adf.hpp"
#endif
#include "operators.hpp"

#endif

#endif
