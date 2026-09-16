// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef __AIE_API_CASCADE__HPP__
#define __AIE_API_CASCADE__HPP__

#include "accum.hpp"
#include "vector.hpp"

// Adf-free accessor for the AIE core-to-core cascade.
//
// aie_api otherwise exposes the cascade only through the ADF stream API in
// aie_api/adf/, which includes <adf.h>. That header is part of the Vitis ADF
// framework and is absent in bare-metal, IRON, and mlir-aie builds, so a kernel
// compiled outside the ADF flow cannot issue a cascade put or get through
// aie_api, even though the hardware datapath is reachable through the compiler
// intrinsics directly.
//
// The accessors bridge aie_api's vector and accum types onto the compiler's
// cascade stream intrinsics put_mcd/get_scd (aie2p intrinsic layer), which
// enable the cascade with en = 1 by default. aie_api otherwise reaches those
// intrinsics only through the ADF stream types.
//
// Availability follows aie_api's usual capability model. The cascade datapath
// exists across the AIE generations, but the intrinsics are compiler and
// architecture specific (chess vs Peano; aie2/aie2p/aie2ps). __AIE_API_HAS_CASCADE__
// (detail/<arch>/config.hpp) is 1 exactly where an implementation is present:
// today aie2p under Peano. put_mcd/get_scd are the standard stream intrinsic
// names, so extending to another compiler or architecture is expected to be a
// matter of confirming the intrinsics there and setting __AIE_API_HAS_CASCADE__;
// callers that guard on the macro then pick it up with no source change.

namespace aie {

#if __AIE_API_HAS_CASCADE__

/**
 * \brief Write a 16-lane int32 vector to the master cascade datapath.
 *
 * Adf-free: reaches the cascade without the ADF stream API.
 */
inline void cascade_out(const vector<int32_t, 16> &v)
{
    put_mcd(v.to_native());
}

/**
 * \brief Read a 16-lane int32 vector from the slave cascade datapath.
 */
inline vector<int32_t, 16> cascade_in_i32()
{
    return get_scd_v16int32();
}

/**
 * \brief Write a 16-lane acc32 accumulator to the master cascade datapath.
 *
 * The accumulator path is the one a fused cross-core K-reduction uses, since
 * partial sums travel as accumulators. The accum to native conversion stays
 * inside the public accum API through accum::to_native(), so no reinterpret_cast
 * is needed and the storage representation is not assumed.
 */
inline void cascade_out(const accum<acc32, 16> &a)
{
    put_mcd(a.to_native());
}

/**
 * \brief Read a 16-lane acc32 accumulator from the slave cascade datapath.
 *
 * The native accumulator the intrinsic returns is rebuilt into an accum through
 * the implicit accum(storage_t) constructor.
 */
inline accum<acc32, 16> cascade_in_acc32()
{
    return accum<acc32, 16>(get_scd_v16acc32());
}

#endif // __AIE_API_HAS_CASCADE__

} // namespace aie

#endif // __AIE_API_CASCADE__HPP__
