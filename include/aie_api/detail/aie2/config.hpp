// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef __AIE_API_DETAIL_AIE2_CONFIG__HPP__
#define __AIE_API_DETAIL_AIE2_CONFIG__HPP__

#define AIE_API_PLATFORM_VERSION 200
#define AIE_API_ML_VERSION       200
#define AIE_API_MATH_VERSION     0

#define __AIE_API_REGISTER_ATTR_DEFINED__           0

#define __AIE_API_COMPR_CONST_PTR__                 0

#define __AIE_API_MUL_CONJUGATE_32BIT_INTRINSICS__  (__AIE_MODEL_VERSION__ >= 10200)

#define __AIE_API_FP32_EMULATION__                  (__AIE_MODEL_VERSION__ >= 3000)

// The adf-free cascade accessor (aie_api/cascade.hpp) is currently implemented
// only for aie2p under Peano; it is not available on this architecture.
#define __AIE_API_HAS_CASCADE__                     0

#define __AIE_API_FP32_SUPPORT__                    0

#define __AIE_API_32ELEM_FLOAT_SRS_UPS__            (__AIE_MODEL_VERSION__ >= 10300)

#define __AIE_API_CONSTEXPR_BFLOAT16__              0

#define __AIE_API_COMPLEX_FP32_EMULATION__          (__AIE_MODEL_VERSION__ >= 10300) && (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_COMPLEX_FP32_CONF__               (__AIE_MODEL_VERSION__ >= 10700)

#define __AIE_API_CFP_TO_FP_CONVERSIONS__           (__AIE_MODEL_VERSION__ >= 10400) && (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_CBF16_SUPPORT__                   (__AIE_MODEL_VERSION__ >= 10500) && (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_CBF16_NO_SHIFT16__                (__AIE_MODEL_VERSION__ >= 10700)

#define __AIE_API_CINT_SUPPORT__                    (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_CFP32_SUPPORT__                   __AIE_API_COMPLEX_FP32_EMULATION__ && (__AIE_API_COMPLEX_VECTOR_SUPPORT__)


#define __AIE_API_CR_1269001_FFT_CBF16_WORKAROUND__ 0

#endif
