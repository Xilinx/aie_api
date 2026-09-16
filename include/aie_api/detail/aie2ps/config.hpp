// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef __AIE_API_DETAIL_AIE2PS_CONFIG__HPP__
#define __AIE_API_DETAIL_AIE2PS_CONFIG__HPP__

#define AIE_API_PLATFORM_VERSION 200
#define AIE_API_ML_VERSION       210
#define AIE_API_MATH_VERSION     0

#define __AIE_API_REGISTER_ATTR_DEFINED__     0

#define __AIE_API_COMPR_CONST_PTR__           0

#define __AIE_API_EMULATED_MUL_INTRINSICS__   0

#define __AIE_API_MUL_CONJUGATE_32BIT_INTRINSICS__ 0

#define __AIE_API_FP32_EMULATION__            (__AIE_MODEL_VERSION__ >= 3000)

// The adf-free cascade accessor (aie_api/cascade.hpp) is currently implemented
// only for aie2p under Peano; it is not available on this architecture.
#define __AIE_API_HAS_CASCADE__               0

#define __AIE_API_FP32_SUPPORT__              0

#define __AIE_API_HAS_EXTRACT_V64BFP16__      0

#define __AIE_API_HAS_32B_MUL__               0

#define __AIE_API_8_LANE_MUL_ELEM_8__         (__AIE_MODEL_VERSION__ >= 1800)

#define __AIE_API_HAS_128_FIFO_POP_ADDR___    0

#define __AIE_API_HAS_COMPLEX_BFLOAT16_FIFO__ (__AIE_MODEL_VERSION__ >= 2800)

#define __AIE_API_CONSTEXPR_BFLOAT16__        (__AIE_MODEL_VERSION__ >= 3200)

#define __AIE_API_COMPLEX_FP32_EMULATION__    (__AIE_MODEL_VERSION__ >= 3200) && (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_FP16_SUPPORT__              (__AIE_MODEL_VERSION__ >= 3200)

#define __AIE_API_FP8_SUPPORT__               (__AIE_MODEL_VERSION__ >= 3200)

#define __AIE_API_BF8_SUPPORT__               (__AIE_MODEL_VERSION__ >= 3200)

#define __AIE_API_MIXED_BF8_FP8_SUPPORT__     (__AIE_MODEL_VERSION__ >= 3200)

#define __AIE_API_MIXED_BF16_FP16_SUPPORT__   (__AIE_MODEL_VERSION__ >= 3200)

#define __AIE_API_CBF16_SUPPORT__             (__AIE_MODEL_VERSION__ >= 10200) && (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_SCALAR_MX_TYPES__           (__AIE_MODEL_VERSION__ >= 10100)

#define __AIE_API_EMULATED_MX_LOADS__         0

#define __AIE_API_CINT_SUPPORT__              (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_CFP32_SUPPORT__             __AIE_API_COMPLEX_FP32_EMULATION__

// Workaround disabled by default. Define to 1 to enable workaround.
// If enabled, replaces ::concat with ::insert for aie::concat(accum...) and accum::upd_all(accum...)
#ifndef __AIE_API_WORKAROUND_CR_1223259__
#define __AIE_API_WORKAROUND_CR_1223259__     0
#endif

#define __AIE_API_CRVO_9906__                 (__AIE_MODEL_VERSION__ >= 10300)

#define __AIE_API_USE_UPS_TO__                0
#define __AIE_API_USE_SRS_TO__                0

#define __AIE_API_USE_MX_UPDATE__             (__AIE_MODEL_VERSION__ <= 10400)

#endif
