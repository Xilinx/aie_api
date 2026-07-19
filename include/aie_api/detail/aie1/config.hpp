// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

#pragma once

#ifndef __AIE_API_DETAIL_AIE1_CONFIG__HPP__
#define __AIE_API_DETAIL_AIE1_CONFIG__HPP__

#define AIE_API_PLATFORM_VERSION   100
#define AIE_API_ML_VERSION         100
#define AIE_API_MATH_VERSION       100

// The adf-free cascade accessor (aie_api/cascade.hpp) is currently implemented
// only for aie2p under Peano; it is not available on this architecture.
#define __AIE_API_HAS_CASCADE__               0

// >= 900000 is used to detect the master development branch

#define __AIE_API_REGISTER_ATTR_DEFINED__ ((__AIE_MODEL_VERSION__ >= 900000)                                                                   || \
                                           (__AIE_MODEL_VERSION__ >= 20000 && __AIE_MODEL_VERSION__ < 30000 && __AIE_MODEL_VERSION__ >= 21800) || \
                                           (__AIE_MODEL_VERSION__ >= 10000 && __AIE_MODEL_VERSION__ < 20000 && __AIE_MODEL_VERSION__ >= 11800))

#define __AIE_API_FLOAT_CONVERSION_VECTORIZED__   ((__AIE_MODEL_VERSION__ >= 900000)                                                                   || \
                                                   (__AIE_MODEL_VERSION__ >= 20000 && __AIE_MODEL_VERSION__ < 30000 && __AIE_MODEL_VERSION__ >= 22400) || \
                                                   (__AIE_MODEL_VERSION__ >= 10000 && __AIE_MODEL_VERSION__ < 20000 && __AIE_MODEL_VERSION__ >= 12400))

#define __AIE_API_CINT_SUPPORT__              (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_CFP32_SUPPORT__             (__AIE_API_COMPLEX_VECTOR_SUPPORT__)

#define __AIE_API_FP32_SUPPORT__              1

#define __AIE_API_COMPLEX_FP32_EMULATION__    0

#endif
