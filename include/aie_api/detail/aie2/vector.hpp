// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.

#pragma once

#ifndef __AIE_API_DETAIL_AIE2_VECTOR__HPP__
#define __AIE_API_DETAIL_AIE2_VECTOR__HPP__

#include <algorithm>

#include "../ld_st.hpp"

namespace aie::detail {

template <typename T>
struct native_vector_length
{
    static constexpr unsigned value = 512 / type_bits_v<T>;
};

template <unsigned Elems, typename T> static auto vector_extract(const T &v, unsigned idx);

template <> inline auto vector_extract<64,  v128int8>(const  v128int8 &v, unsigned idx) { return ::extract_v64int8(v, idx);   };
template <> inline auto vector_extract<64, v128uint8>(const v128uint8 &v, unsigned idx) { return ::extract_v64uint8(v, idx);  };
template <> inline auto vector_extract<32,   v64int8>(const  v64int8 &v, unsigned idx)  { return ::extract_v32int8(v, idx);   };
template <> inline auto vector_extract<32,  v64uint8>(const v64uint8 &v, unsigned idx)  { return ::extract_v32uint8(v, idx);  };
template <> inline auto vector_extract<16,   v64int8>(const  v64int8 &v, unsigned idx)  { return ::extract_v16int8(v, idx);   };
template <> inline auto vector_extract<16,  v64uint8>(const v64uint8 &v, unsigned idx)  { return ::extract_v16uint8(v, idx);  };
template <> inline auto vector_extract<16,   v32int8>(const  v32int8 &v, unsigned idx)  { return ::extract_v16int8(v, idx);   };
template <> inline auto vector_extract<16,  v32uint8>(const v32uint8 &v, unsigned idx)  { return ::extract_v16uint8(v, idx);  };

template <> inline auto vector_extract<32,  v64int16>(const  v64int16 &v, unsigned idx) { return ::extract_v32int16(v, idx);  };
template <> inline auto vector_extract<32, v64uint16>(const v64uint16 &v, unsigned idx) { return ::extract_v32uint16(v, idx); };
template <> inline auto vector_extract<16,  v32int16>(const  v32int16 &v, unsigned idx) { return ::extract_v16int16(v, idx);  };
template <> inline auto vector_extract<16, v32uint16>(const v32uint16 &v, unsigned idx) { return ::extract_v16uint16(v, idx); };
template <> inline auto vector_extract<8,   v32int16>(const  v32int16 &v, unsigned idx) { return ::extract_v8int16(v, idx);   };
template <> inline auto vector_extract<8,  v32uint16>(const v32uint16 &v, unsigned idx) { return ::extract_v8uint16(v, idx);  };
template <> inline auto vector_extract<8,   v16int16>(const  v16int16 &v, unsigned idx) { return ::extract_v8int16(v, idx);   };
template <> inline auto vector_extract<8,  v16uint16>(const v16uint16 &v, unsigned idx) { return ::extract_v8uint16(v, idx);  };

template <> inline auto vector_extract<16,  v32int32>(const  v32int32 &v, unsigned idx) { return ::extract_v16int32(v, idx);   };
template <> inline auto vector_extract<16, v32uint32>(const v32uint32 &v, unsigned idx) { return ::extract_v16uint32(v, idx);  };
template <> inline auto vector_extract<8,   v16int32>(const  v16int32 &v, unsigned idx) { return ::extract_v8int32(v, idx);    };
template <> inline auto vector_extract<8,  v16uint32>(const v16uint32 &v, unsigned idx) { return ::extract_v8uint32(v, idx);   };
template <> inline auto vector_extract<4,   v16int32>(const  v16int32 &v, unsigned idx) { return ::extract_v4int32(v, idx);    };
template <> inline auto vector_extract<4,  v16uint32>(const v16uint32 &v, unsigned idx) { return ::extract_v4uint32(v, idx);   };
template <> inline auto vector_extract<4,    v8int32>(const   v8int32 &v, unsigned idx) { return ::extract_v4int32(v, idx);    };
template <> inline auto vector_extract<4,   v8uint32>(const  v8uint32 &v, unsigned idx) { return ::extract_v4uint32(v, idx);   };

template <> inline auto vector_extract<16, v32cint16>(const v32cint16 &v, unsigned idx) { return ::extract_v16cint16(v, idx);  };
template <> inline auto vector_extract<8,  v16cint16>(const v16cint16 &v, unsigned idx) { return ::extract_v8cint16(v, idx);   };
template <> inline auto vector_extract<4,  v16cint16>(const v16cint16 &v, unsigned idx) { return ::extract_v4cint16(v, idx );  };
template <> inline auto vector_extract<4,   v8cint16>(const  v8cint16 &v, unsigned idx) { return ::extract_v4cint16(v, idx);   };

template <> inline auto vector_extract<8,  v16cint32>(const  v16cint32 &v, unsigned idx) { return ::extract_v8cint32(v, idx);  };
template <> inline auto vector_extract<4,   v8cint32>(const   v8cint32 &v, unsigned idx) { return ::extract_v4cint32(v, idx);  };
template <> inline auto vector_extract<2,   v8cint32>(const   v8cint32 &v, unsigned idx) { return ::extract_v2cint32(v, idx ); };
template <> inline auto vector_extract<2,   v4cint32>(const   v4cint32 &v, unsigned idx) { return ::extract_v2cint32(v, idx);  };

template <> inline auto vector_extract<128,  v256int4>(const  v256int4 &v, unsigned idx) { return ::extract_v128int4(v, idx);  };
template <> inline auto vector_extract<128, v256uint4>(const v256uint4 &v, unsigned idx) { return ::extract_v128uint4(v, idx); };
template <> inline auto vector_extract<64,   v128int4>(const  v128int4 &v, unsigned idx) { return ::extract_v64int4(v, idx);   };
template <> inline auto vector_extract<64,  v128uint4>(const v128uint4 &v, unsigned idx) { return ::extract_v64uint4(v, idx);  };
template <> inline auto vector_extract<32,   v128int4>(const  v128int4 &v, unsigned idx) { return ::extract_v32int4(v, idx );  };
template <> inline auto vector_extract<32,  v128uint4>(const v128uint4 &v, unsigned idx) { return ::extract_v32uint4(v, idx ); };
template <> inline auto vector_extract<32,    v64int4>(const   v64int4 &v, unsigned idx) { return ::extract_v32int4(v, idx);   };
template <> inline auto vector_extract<32,   v64uint4>(const  v64uint4 &v, unsigned idx) { return ::extract_v32uint4(v, idx);  };

template <> inline auto vector_extract<32, v64bfloat16>(const v64bfloat16 &v, unsigned idx) { return ::extract_v32bfloat16(v, idx); };
template <> inline auto vector_extract<16, v32bfloat16>(const v32bfloat16 &v, unsigned idx) { return ::extract_v16bfloat16(v, idx); };
template <> inline auto vector_extract<8,  v32bfloat16>(const v32bfloat16 &v, unsigned idx) { return ::extract_v8bfloat16(v, idx);  };
template <> inline auto vector_extract<8,  v16bfloat16>(const v16bfloat16 &v, unsigned idx) { return ::extract_v8bfloat16(v, idx);  };

#if __AIE_API_FP32_EMULATION__
template <> inline auto vector_extract<16, v32float>(const v32float &v, unsigned idx) { return ::extract_v16float(v, idx); };
template <> inline auto vector_extract<8,  v32float>(const v32float &v, unsigned idx) { return ::extract_v8float(v, idx);  };
template <> inline auto vector_extract<8,  v16float>(const v16float &v, unsigned idx) { return ::extract_v8float(v, idx);  };
template <> inline auto vector_extract<4,  v16float>(const v16float &v, unsigned idx) { return ::extract_v4float(v, idx);  };
template <> inline auto vector_extract<4,   v8float>(const v8float &v,  unsigned idx) { return ::extract_v4float(v, idx);  };
#endif

#if __AIE_API_COMPLEX_FP32_EMULATION__
#if __AIE_API_CBF16_SUPPORT__
template <> inline auto vector_extract<16, v32cbfloat16>(const v32cbfloat16 &v, unsigned idx) { return ::extract_v16cbfloat16(v, idx); };
template <> inline auto vector_extract<8,  v16cbfloat16>(const v16cbfloat16 &v, unsigned idx) { return ::extract_v8cbfloat16(v, idx);  };
template <> inline auto vector_extract<4,  v16cbfloat16>(const v16cbfloat16 &v, unsigned idx) { return ::extract_v4cbfloat16(v, idx);  };
template <> inline auto vector_extract<4,   v8cbfloat16>(const  v8cbfloat16 &v, unsigned idx) { return ::extract_v4cbfloat16(v, idx);  };
#endif

template <> inline auto vector_extract<8, v16cfloat>(const v16cfloat &v, unsigned idx) { return ::extract_v8cfloat(v, idx); };
template <> inline auto vector_extract<4, v16cfloat>(const v16cfloat &v, unsigned idx) { return ::extract_v4cfloat(v, idx); };
template <> inline auto vector_extract<4,  v8cfloat>(const v8cfloat &v,  unsigned idx) { return ::extract_v4cfloat(v, idx); };
template <> inline auto vector_extract<2,  v8cfloat>(const v8cfloat &v,  unsigned idx) { return ::extract_v2cfloat(v, idx); };
template <> inline auto vector_extract<2,  v4cfloat>(const v4cfloat &v,  unsigned idx) { return ::extract_v2cfloat(v, idx); };
#endif

template <typename T, unsigned Elems> struct vector_set;
template <> struct vector_set<int8,      64> { static v64int8     run(const v16int8     &v, unsigned idx) { return ::set_v64int8(idx, v);     }
                                               static v64int8     run(const v32int8     &v, unsigned idx) { return ::set_v64int8(idx, v);     } };
template <> struct vector_set<uint8,     64> { static v64uint8    run(const v16uint8    &v, unsigned idx) { return ::set_v64uint8(idx, v);    }
                                               static v64uint8    run(const v32uint8    &v, unsigned idx) { return ::set_v64uint8(idx, v);    } };
template <> struct vector_set<int8,      32> { static v32int8     run(const v16int8     &v, unsigned idx) { return ::set_v32int8(idx, v);     } };
template <> struct vector_set<uint8,     32> { static v32uint8    run(const v16uint8    &v, unsigned idx) { return ::set_v32uint8(idx, v);    } };

template <> struct vector_set<int16,     32> { static v32int16    run(const v8int16     &v, unsigned idx) { return ::set_v32int16(idx, v);    }
                                               static v32int16    run(const v16int16    &v, unsigned idx) { return ::set_v32int16(idx, v);    } };
template <> struct vector_set<uint16,    32> { static v32uint16   run(const v8uint16    &v, unsigned idx) { return ::set_v32uint16(idx, v);   }
                                               static v32uint16   run(const v16uint16   &v, unsigned idx) { return ::set_v32uint16(idx, v);   } };
template <> struct vector_set<int16,     16> { static v16int16    run(const v8int16     &v, unsigned idx) { return ::set_v16int16(idx, v);    } };
template <> struct vector_set<uint16,    16> { static v16uint16   run(const v8uint16    &v, unsigned idx) { return ::set_v16uint16(idx, v);   } };

template <> struct vector_set<int32,     16> { static v16int32    run(const v4int32     &v, unsigned idx) { return ::set_v16int32(idx, v);    }
                                               static v16int32    run(const v8int32     &v, unsigned idx) { return ::set_v16int32(idx, v);    } };
template <> struct vector_set<uint32,    16> { static v16uint32   run(const v4uint32    &v, unsigned idx) { return ::set_v16uint32(idx, v);   }
                                               static v16uint32   run(const v8uint32    &v, unsigned idx) { return ::set_v16uint32(idx, v);   } };
template <> struct vector_set<int32,      8> { static v8int32     run(const v4int32     &v, unsigned idx) { return ::set_v8int32(idx, v);     } };
template <> struct vector_set<uint32,     8> { static v8uint32    run(const v4uint32    &v, unsigned idx) { return ::set_v8uint32(idx, v);    } };

template <> struct vector_set<cint16,    16> { static v16cint16   run(const v4cint16    &v, unsigned idx) { return ::set_v16cint16(idx, v);   }
                                               static v16cint16   run(const v8cint16    &v, unsigned idx) { return ::set_v16cint16(idx, v);   } };
template <> struct vector_set<cint16,     8> { static v8cint16    run(const v4cint16    &v, unsigned idx) { return ::set_v8cint16(idx, v);    } };

template <> struct vector_set<cint32,     8> { static v8cint32    run(const v2cint32    &v, unsigned idx) { return ::set_v8cint32(idx, v);    }
                                               static v8cint32    run(const v4cint32    &v, unsigned idx) { return ::set_v8cint32(idx, v);    } };
template <> struct vector_set<cint32,     4> { static v4cint32    run(const v2cint32    &v, unsigned idx) { return ::set_v4cint32(idx, v);    } };

template <> struct vector_set<bfloat16,  32> { static v32bfloat16 run(const v8bfloat16  &v, unsigned idx) { return ::set_v32bfloat16(idx, v); }
                                               static v32bfloat16 run(const v16bfloat16 &v, unsigned idx) { return ::set_v32bfloat16(idx, v); } };
template <> struct vector_set<bfloat16,  16> { static v16bfloat16 run(const v8bfloat16  &v, unsigned idx) { return ::set_v16bfloat16(idx, v); } };

#if __AIE_API_FP32_EMULATION__
//TODO: Add v4float into v32float when intrinsic available (CRVO-4522)
template <> struct vector_set<float,     32> { static v32float    run(const v8float     &v, unsigned idx) { return ::set_v32float(idx, v);    }
                                               static v32float    run(const v16float    &v, unsigned idx) { return ::set_v32float(idx, v);    } };
template <> struct vector_set<float,     16> { static v16float    run(const v8float     &v, unsigned idx) { return ::set_v16float(idx, v);    }
                                               static v16float    run(const v4float     &v, unsigned idx) { return ::set_v16float(idx, v);    } };
template <> struct vector_set<float,     8>  { static v8float     run(const v4float     &v, unsigned idx) { return ::set_v8float(idx, v);     } };
#endif

#if __AIE_API_COMPLEX_FP32_EMULATION__
#if __AIE_API_CBF16_SUPPORT__
template <> struct vector_set<cbfloat16,  16> { static v16cbfloat16 run(const v4cbfloat16 &v, unsigned idx) { return ::set_v16cbfloat16(idx, v); }
                                                static v16cbfloat16 run(const v8cbfloat16 &v, unsigned idx) { return ::set_v16cbfloat16(idx, v); } };
template <> struct vector_set<cbfloat16,   8> { static v8cbfloat16  run(const v4cbfloat16 &v, unsigned idx) { return ::set_v8cbfloat16(idx, v); } };
#endif

//TODO: Add v2cfloat into v32float when intrinsic available (CRVO-4522)
template <> struct vector_set<cfloat,    16> { static v16cfloat   run(const v4cfloat    &v, unsigned idx) { return ::set_v16cfloat(idx, v);   }
                                               static v16cfloat   run(const v8cfloat    &v, unsigned idx) { return ::set_v16cfloat(idx, v);   } };
template <> struct vector_set<cfloat,     8> { static v8cfloat    run(const v4cfloat    &v, unsigned idx) { return ::set_v8cfloat(idx, v);    }
                                               static v8cfloat    run(const v2cfloat    &v, unsigned idx) { return ::set_v8cfloat(idx, v);    } };
template <> struct vector_set<cfloat,     4> { static v4cfloat    run(const v2cfloat    &v, unsigned idx) { return ::set_v4cfloat(idx, v);    } };
#endif

template <> struct vector_set<int4,     128> { static v128int4    run(const v32int4     &v, unsigned idx) { return ::set_v128int4(idx, v);    }
                                               static v128int4    run(const v64int4     &v, unsigned idx) { return ::set_v128int4(idx, v);    } };
template <> struct vector_set<uint4,    128> { static v128uint4   run(const v32uint4    &v, unsigned idx) { return ::set_v128uint4(idx, v);   }
                                               static v128uint4   run(const v64uint4    &v, unsigned idx) { return ::set_v128uint4(idx, v);   } };
template <> struct vector_set<int4,      64> { static v64int4     run(const v32int4     &v, unsigned idx) { return ::set_v64int4(idx, v);     } };
template <> struct vector_set<uint4,     64> { static v64uint4    run(const v32uint4    &v, unsigned idx) { return ::set_v64uint4(idx, v);    } };

template <typename DstT, unsigned DstElems, typename T>
__aie_inline
static vector_storage_t<DstT, DstElems> vector_cast_helper(T &&from)
{
    static constexpr unsigned native_elems = native_vector_length<DstT>::value;
    static constexpr unsigned Blocks       = DstElems / native_elems;

    if constexpr (std::is_same_v<DstT, int8> && DstElems == 16) return v16int8(from);
    if constexpr (std::is_same_v<DstT, int8> && DstElems == 32) return v32int8(from);
    if constexpr (std::is_same_v<DstT, int8> && DstElems == 64) return v64int8(from);
    if constexpr (std::is_same_v<DstT, int8> && DstElems >= 128) return utils::make_array<Blocks>([](auto f) __aie_inline { return v64int8(f); }, from);

    if constexpr (std::is_same_v<DstT, uint8> && DstElems ==  16) return v16uint8(from);
    if constexpr (std::is_same_v<DstT, uint8> && DstElems ==  32) return v32uint8(from);
    if constexpr (std::is_same_v<DstT, uint8> && DstElems ==  64) return v64uint8(from);
    if constexpr (std::is_same_v<DstT, uint8> && DstElems >= 128) return utils::make_array<Blocks>([](auto f) __aie_inline { return v64uint8(f); }, from);

    if constexpr (std::is_same_v<DstT, int16> && DstElems ==  8) return  v8int16(from);
    if constexpr (std::is_same_v<DstT, int16> && DstElems == 16) return v16int16(from);
    if constexpr (std::is_same_v<DstT, int16> && DstElems == 32) return v32int16(from);
    if constexpr (std::is_same_v<DstT, int16> && DstElems >= 64) return utils::make_array<Blocks>([](auto f) __aie_inline { return v32int16(f); }, from);

    if constexpr (std::is_same_v<DstT, uint16> && DstElems ==  8) return  v8uint16(from);
    if constexpr (std::is_same_v<DstT, uint16> && DstElems == 16) return v16uint16(from);
    if constexpr (std::is_same_v<DstT, uint16> && DstElems == 32) return v32uint16(from);
    if constexpr (std::is_same_v<DstT, uint16> && DstElems >= 64) return utils::make_array<Blocks>([](auto f) __aie_inline { return v32uint16(f); }, from);

    if constexpr (std::is_same_v<DstT, int32> && DstElems ==  4) return v4int32(from);
    if constexpr (std::is_same_v<DstT, int32> && DstElems ==  8) return v8int32(from);
    if constexpr (std::is_same_v<DstT, int32> && DstElems == 16) return v16int32(from);
    if constexpr (std::is_same_v<DstT, int32> && DstElems >= 32) return utils::make_array<Blocks>([](auto f) __aie_inline { return v16int32(f); }, from);

    if constexpr (std::is_same_v<DstT, uint32> && DstElems ==  4) return v4uint32(from);
    if constexpr (std::is_same_v<DstT, uint32> && DstElems ==  8) return v8uint32(from);
    if constexpr (std::is_same_v<DstT, uint32> && DstElems == 16) return v16uint32(from);
    if constexpr (std::is_same_v<DstT, uint32> && DstElems >= 32) return utils::make_array<Blocks>([](auto f) __aie_inline { return v16uint32(f); }, from);

    if constexpr (std::is_same_v<DstT, cint16> && DstElems ==  4) return v4cint16(from);
    if constexpr (std::is_same_v<DstT, cint16> && DstElems ==  8) return v8cint16(from);
    if constexpr (std::is_same_v<DstT, cint16> && DstElems == 16) return v16cint16(from);
    if constexpr (std::is_same_v<DstT, cint16> && DstElems >= 32) return utils::make_array<Blocks>([](auto f) __aie_inline { return v16cint16(f); }, from);

    if constexpr (std::is_same_v<DstT, cint32> && DstElems ==  2) return v2cint32(from);
    if constexpr (std::is_same_v<DstT, cint32> && DstElems ==  4) return v4cint32(from);
    if constexpr (std::is_same_v<DstT, cint32> && DstElems ==  8) return v8cint32(from);
    if constexpr (std::is_same_v<DstT, cint32> && DstElems >= 16) return utils::make_array<Blocks>([](auto f) __aie_inline { return v8cint32(f); }, from);

    if constexpr (std::is_same_v<DstT, int4> && DstElems ==  32) return v32int4(from);
    if constexpr (std::is_same_v<DstT, int4> && DstElems ==  64) return v64int4(from);
    if constexpr (std::is_same_v<DstT, int4> && DstElems == 128) return v128int4(from);
    if constexpr (std::is_same_v<DstT, int4> && DstElems >= 256) return utils::make_array<Blocks>([](auto f) __aie_inline { return v128int4(f); }, from);

    if constexpr (std::is_same_v<DstT, uint4> && DstElems ==  32) return v32uint4(from);
    if constexpr (std::is_same_v<DstT, uint4> && DstElems ==  64) return v64uint4(from);
    if constexpr (std::is_same_v<DstT, uint4> && DstElems == 128) return v128uint4(from);
    if constexpr (std::is_same_v<DstT, uint4> && DstElems >= 256) return utils::make_array<Blocks>([](auto f) __aie_inline { return v128uint4(f); }, from);

    if constexpr (std::is_same_v<DstT, bfloat16> && DstElems ==  8) return  v8bfloat16(from);
    if constexpr (std::is_same_v<DstT, bfloat16> && DstElems == 16) return v16bfloat16(from);
    if constexpr (std::is_same_v<DstT, bfloat16> && DstElems == 32) return v32bfloat16(from);
    if constexpr (std::is_same_v<DstT, bfloat16> && DstElems >= 64) return utils::make_array<Blocks>([](auto f) __aie_inline { return v32bfloat16(f); }, from);

#if __AIE_API_FP32_EMULATION__
#if !__AIE_API_CFP_TO_FP_CONVERSIONS__
    if constexpr (std::is_same_v<DstT, float> && DstElems ==  4) return v4float(v4int32(from));
    if constexpr (std::is_same_v<DstT, float> && DstElems ==  8) return v8float(v8int32(from));
#else
    if constexpr (std::is_same_v<DstT, float> && DstElems ==  4) return v4float(from);
    if constexpr (std::is_same_v<DstT, float> && DstElems ==  8) return v8float(from);
#endif
    if constexpr (std::is_same_v<DstT, float> && DstElems == 16) return v16float(from);
    if constexpr (std::is_same_v<DstT, float> && DstElems >= 32) return utils::make_array<Blocks>([](auto f) __aie_inline { return v16float(f); }, from);
#endif

#if __AIE_API_COMPLEX_FP32_EMULATION__
#if __AIE_API_CBF16_SUPPORT__
    if constexpr (std::is_same_v<DstT, cbfloat16> && DstElems ==  4) return  v4cbfloat16(from);
    if constexpr (std::is_same_v<DstT, cbfloat16> && DstElems ==  8) return  v8cbfloat16(from);
    if constexpr (std::is_same_v<DstT, cbfloat16> && DstElems == 16) return v16cbfloat16(from);
    if constexpr (std::is_same_v<DstT, cbfloat16> && DstElems == 32) return { v16cbfloat16(from[0]), v16cbfloat16(from[1]) };
#endif

#if !__AIE_API_CFP_TO_FP_CONVERSIONS__
    if constexpr (std::is_same_v<DstT, cfloat> && DstElems ==  2) return v2cfloat(v4int32(from));
    if constexpr (std::is_same_v<DstT, cfloat> && DstElems ==  4) return v4cfloat(v8int32(from));
#else
    if constexpr (std::is_same_v<DstT, cfloat> && DstElems ==  2) return v2cfloat(from);
    if constexpr (std::is_same_v<DstT, cfloat> && DstElems ==  4) return v4cfloat(from);
#endif
    if constexpr (std::is_same_v<DstT, cfloat> && DstElems ==  8) return v8cfloat(from);
    if constexpr (std::is_same_v<DstT, cfloat> && DstElems >= 16) return utils::make_array<Blocks>([](auto f) __aie_inline { return v8cfloat(f); }, from);
#endif
}

/**
 * Provides compile time access to the vector element type resulting from vector<T>::unpack operation.
 */
template <typename T>
struct unpacked_type {
    using type = utils::get_next_integer_type_t<T>;
};

/** Helper type alias for unpacked_type<T> */
template <typename T>
using unpacked_type_t = typename unpacked_type<T>::type;

/**
 * Provides compile time access to the vector element type resulting from vector<T>::pack operation.
 */
template <typename T>
struct packed_type {
    using type = utils::get_prev_integer_type_t<T>;
};

/** Helper type alias for packed_type<T> */
template <typename T>
using packed_type_t = typename packed_type<T>::type;

/**
 * Architecture-specific implementation of the vector data type
 *
 * @tparam T Type of the elements contained in the vector.
 * @tparam Elems Number of elements in the vector.
 */
template <typename T, unsigned Elems>
class vector_base
{
private:
    template <typename T2, unsigned E2> friend class vector_base;

    using vector_storage_type = vector_storage<T, Elems>;

public:
    using        derived_type = vector<T, Elems>;
    using         native_type = native_vector_type_t<T, Elems>;
    using          value_type = T;
    using           storage_t = typename vector_storage_type::type;

    static constexpr unsigned type_bits()
    {
        return type_bits_v<T>;
    }

    static constexpr unsigned size()
    {
        return Elems;
    }

    static constexpr unsigned bits()
    {
        return type_bits() * Elems;
    }

    static constexpr bool is_signed()
    {
        return is_signed_v<T>;
    }

    static constexpr bool is_complex()
    {
        return is_complex_v<T>;
    }

    static constexpr bool is_real()
    {
        return !is_complex();
    }

    static constexpr bool is_integral()
    {
        return is_integral_v<T>;
    }

    static constexpr bool is_floating_point()
    {
        return is_floating_point_v<T>;
    }

    __aie_inline
    vector_base() :
        data(vector_storage_type::undef())
    {
    }

    __aie_inline
    vector_base(storage_t data) :
        data(data)
    {
        // Input is taken by value to avoid losing chess_storage qualifiers
    }

    /**
     * Construct from internal types. This is a special case for 1024b internal vector types, which need to be broken
     * into two 512b vectors
     *
     * @param v Data used to construct the vector from
     */
    __aie_inline
    vector_base(const native_type &v) requires(bits() == 1024)
    {
        data[0] = vector_extract<Elems / 2>(v, 0);
        data[1] = vector_extract<Elems / 2>(v, 1);
    }

    template <typename... Values>
    __aie_inline
    explicit vector_base(value_type v, const Values &... values) :
        data(vector_storage_type::undef())
    {
        constexpr unsigned num_params = sizeof...(values) + 1;

        if constexpr (type_bits() == 4) {
            if constexpr (bits() >= 1024) {
                const auto t = std::make_tuple(v, values...);

                utils::unroll_times<std::max(1u, num_params / native_elems)>([&]<unsigned Idx>(std::integral_constant<unsigned, Idx>) __aie_inline {
                    using next_type = utils::get_next_integer_type_t<T>;
                    constexpr unsigned next_native_elems = native_vector_length_v<next_type>;

                    constexpr unsigned Start1    = Idx * native_elems;
                    constexpr unsigned TotalSize = std::min(native_elems, num_params - (Idx * native_elems));
                    constexpr unsigned Size1     = std::min(next_native_elems, TotalSize);

                    vector_base<next_type, next_native_elems> tmp1, tmp2;
                    tmp1 = init_from_values<Start1, Size1, 0>(tmp1, t);
                    if constexpr (TotalSize > Size1) {
                        constexpr unsigned Start2 = Start1 + Size1;
                        constexpr unsigned Size2  = TotalSize - Size1;
                        tmp2 = init_from_values<Start2, Size2, 0>(tmp2, t);
                    }
                    data[Idx] = ::concat(tmp1.pack(), tmp2.pack());
                });
            }
            else {
                vector_base<utils::get_next_integer_type_t<T>, Elems> tmp(v, values...);
                data = tmp.pack();
            }
        }
        else if constexpr (bits() >= 1024) {
            const auto t = std::make_tuple(v, values...);

            utils::unroll_times<std::max(1u, num_params / native_elems)>([&]<unsigned Idx>(std::integral_constant<unsigned, Idx>) __aie_inline {
                constexpr unsigned Start = Idx * native_elems;
                constexpr unsigned Size  = std::min(native_elems, num_params - (Idx * native_elems));
                data[Idx] = init_from_values<Start, Size, 0>(data[Idx], t);
            });
        }
        else {
            constexpr unsigned v512_elems = 512 / type_bits();
            vector_base<value_type, v512_elems> tmp;

            tmp.data = init_from_values(tmp.data, v, values...);

            data = tmp.template extract<Elems>(0).data;
        }
    }

    template <typename DstT>
    __aie_inline
    auto cast_to() const
    {
        constexpr unsigned DstSize  = type_bits_v<DstT>;
        constexpr unsigned DstElems = (DstSize <= type_bits())? Elems * (type_bits() / DstSize) :
                                                                Elems / (    DstSize / type_bits());

        const vector_base<DstT, DstElems> ret = vector_cast_helper<DstT, DstElems>(data);

        return ret;
    }

    __aie_inline
    vector_base &push(value_type v)
    {
        if constexpr (type_bits() == 4) {
            if constexpr (bits() > 1024) {
                T m2 = get(native_elems - 1);

                insert(0, extract<native_elems>(0).unpack().push(v).pack());

                utils::unroll_for<unsigned, 1u, Elems / native_elems>([&](unsigned idx) __aie_inline {
                    const T m3 = get(native_elems * (idx + 1) - 1);
                    insert(idx, ::shiftr_elem(extract<native_elems>(idx).unpack().push(m2).pack(), m2));
                    m2 = m3;
                });
            }
            else if constexpr (bits() == 1024) {
                const auto v2 = get(Elems / 2 - 1);

                insert(0, extract<native_elems>(0).unpack().push(v).pack());
                insert(1, extract<native_elems>(1).unpack().push(v2).pack());
            }
            else {
                data = unpack().push(v).pack();
            }
        }
        else {
            if constexpr (bits() > 1024) {
                T m2 = get(native_elems - 1);

                insert(0, ::shiftr_elem(extract<native_elems>(0), v));

                utils::unroll_for<unsigned, 1u, Elems / native_elems>([&](unsigned idx) __aie_inline {
                    const T m3 = get(native_elems * (idx + 1) - 1);
                    insert(idx, ::shiftr_elem(extract<native_elems>(idx), m2));
                    m2 = m3;
                });
            }
            else if constexpr (bits() == 1024) {
                const T m = get(native_elems - 1);
                insert(0, ::shiftr_elem(extract<native_elems>(0), v));
                insert(1, ::shiftr_elem(extract<native_elems>(1), m));
            }
            else if constexpr (bits() == 512) {
                data = ::shiftr_elem(data, v);
            }
            else if constexpr (bits() < 512) {
                data = vector_extract<size()>(::shiftr_elem(grow<native_elems>(), v), 0);
            }
        }

        return *this;
    }

    template <unsigned ElemsOut>
    __aie_inline
    vector_base<T, ElemsOut> grow(unsigned idx = 0) const
    {
        constexpr unsigned output_bits  = type_bits() * ElemsOut;
        constexpr unsigned growth_ratio = ElemsOut / Elems;

        static_assert(output_bits >= bits());
        static_assert(utils::is_powerof2(output_bits));

        vector_base<T, ElemsOut> ret;

        constexpr unsigned in_storage_elems  = num_storage_elems;
        constexpr unsigned out_storage_elems = ret.num_storage_elems;

        if constexpr (growth_ratio == 1) {
            ret = data;
        }
        else if constexpr (in_storage_elems == 1 && out_storage_elems == 1) {
            ret = vector_set<value_type, ElemsOut>::run(data, idx);
        }
        else if constexpr (in_storage_elems == 1 && out_storage_elems > 1) {
            constexpr unsigned to_native_ratio = native_elems / Elems;

            // Other elements are default initialized to undef() already
            if constexpr (to_native_ratio == 1)
                ret.data[idx / to_native_ratio] = data;
            else
                ret.data[idx / to_native_ratio] = vector_set<value_type, native_elems>::run(data, idx % to_native_ratio);
        }
        else {
            // Other elements are default initialized to undef() already
            utils::unroll_times<in_storage_elems>([&](unsigned elem) __aie_inline {
                ret.data[idx * in_storage_elems + elem] = data[elem];
            });
        }

        return ret;
    }

    /**
     * Returns a copy of the current vector in a larger vector. The contents of the vector are replicated as many
     * times as required to fill the output vector
     *
     * @tparam ElemsOut Size of the output vector.
     */
    template <unsigned ElemsOut>
    __aie_inline
    constexpr vector_base<T, ElemsOut> grow_replicate() const
    {
        constexpr unsigned output_bits  = type_bits() * ElemsOut;
        constexpr unsigned growth_ratio = ElemsOut / Elems;

        static_assert(output_bits >= bits());
        static_assert(output_bits >= 128 && utils::is_powerof2(output_bits));

        if constexpr (growth_ratio == 1) {
            return *this;
        }
        else if constexpr (bits() == 128) {
            vector_base<T, Elems * 4> tmp;

            tmp = ::shuffle(this->grow<Elems * 4>(), this->grow<Elems * 4>(), T128_2x4_lo);

            if constexpr (growth_ratio == 2) {
                // Return is 256b
                return tmp.template extract<Elems * 2>(0);
            }
            else {
                // Return is 512b at least
                tmp = ::shuffle(tmp, tmp, T128_2x4_lo);

                return tmp.template grow_replicate<ElemsOut>();
            }
        }
        else if constexpr (bits() == 256) {
            vector_base<T, Elems * 2> tmp;

            tmp.data = ::shuffle(this->grow<Elems * 2>(), this->grow<Elems * 2>(), T256_2x2_lo);

            return tmp.template grow_replicate<ElemsOut>();
        }
        else if constexpr (bits() == 512) {
            vector_base<T, ElemsOut> ret;

            utils::unroll_times<growth_ratio>([&](unsigned idx) __aie_inline {
                ret.data[idx] = *this;
            });

            return ret;
        }
    }

    __aie_inline
    void set(value_type v, unsigned idx)
    {
        REQUIRES_MSG(idx < Elems, "idx needs to be a valid element index");

        if constexpr (utils::is_one_of_v<T, int4, uint4>) {
            if constexpr (size() <= 64) {
                vector_base<utils::get_next_integer_type_t<T>, size()> tmp_vec = unpack();
                tmp_vec.set(v, idx);

                data = tmp_vec.pack().data;
            }
            else if constexpr (size() == 128) {
                const bool odd = idx % 2;
                const auto tmp_vec = cast_to<utils::get_next_integer_type_t<T>>();
                auto tmp = tmp_vec.get(idx / 2);

                tmp = odd? (tmp & 0x0f) | (v << 4) :
                           (tmp & 0xf0) | (int8)v;

                data = (native_type)::insert(tmp_vec, idx / 2, tmp);
            }
            else if constexpr (size() >= 256) {
                const bool odd = idx % 2;
                const auto tmp_vec = extract<native_elems>(idx / native_elems).template cast_to<utils::get_next_integer_type_t<T>>();
                auto tmp = tmp_vec.get((idx % native_elems) / 2);

                tmp = odd? (tmp & 0x0f) | (v << 4) :
                           (tmp & 0xf0) | (int8)v;

                data[idx / native_elems] = (typename vector_base<T, native_elems>::native_type)::insert(tmp_vec, (idx % native_elems) / 2, tmp);
            }
        }
        else {
            if constexpr (bits() == 128) {
                data = vector_extract<size()>(::insert(vector_set<value_type, size() * 4>::run(data, 0), idx, v), 0);
            }
            else if constexpr (bits() == 256) {
                data = vector_extract<size()>(::insert(vector_set<value_type, size() * 2>::run(data, 0), idx, v), 0);
            }
            else if constexpr (bits() == 512) {
                data = ::insert(data, idx, v);
            }
            else if constexpr (bits() >= 1024) {
                unsigned     i = idx / native_elems;
                unsigned sub_i = idx % native_elems;
                data[i] = ::insert(data[i], sub_i, v);
            }
        }
    }

    __aie_inline
    value_type get(unsigned idx) const
    {
        REQUIRES_MSG(idx < Elems, "idx needs to be a valid element index");

        if constexpr (utils::is_one_of_v<T, int4, uint4>) {
            if constexpr (size() <= 64) {
                return unpack().get(idx);
            }
            else if constexpr (size() == 128) {
                vector_base<utils::get_next_integer_type_t<T>, 64> tmp_vec;

                // Unpack 256b subvector if index known at compile time. Otherwise use shift to get the right subvector
                if (chess_const(idx)) {
                    tmp_vec = this->extract<64>(idx / 64).unpack();
                }
                else {
                    const vector<T, Elems> tmp = SHIFT_BYTES(data, vector_base(), 32 * (idx / 64));
                    tmp_vec = tmp.template extract<64>(0).unpack();
                }

                return (T)tmp_vec.get(idx % 64);
            }
            else if constexpr (size() == 256) {
                vector_base<utils::get_next_integer_type_t<T>, 64> tmp_vec;

                // Unpack 256b subvector if index known at compile time. Otherwise use shift to get the right subvector
                if (chess_const(idx)) {
                    tmp_vec = this->extract<64>(idx / 64).unpack();
                }
                else {
                    const vector<T, Elems / 2> tmp = SHIFT_BYTES(data[0], data[1], 64 * (idx / 128));
                    return (T)tmp.unpack().get(idx % 128);
                }

                return (T)tmp_vec.get(idx % 64);
            }
            else {
                return (T)vector(data[idx / 128]).unpack().get(idx % 128);
            }
        }
        else {
            if constexpr (bits() == 128) {
                return ::extract_elem(vector_set<value_type, size() * 4>::run(data, 0), idx);
            }
            else if constexpr (bits() == 256) {
                return ::extract_elem(vector_set<value_type, size() * 2>::run(data, 0), idx);
            }
            else if constexpr (bits() == 512) {
                return ::extract_elem(data, idx);
            }
            else if constexpr (bits() == 1024) {
                // Get 512b subvector if index known at compile time. Otherwise use shift to get the right subvector
                if (chess_manifest(idx < size() / 2))
                    return ::extract_elem(data[0], idx);
                else if (chess_manifest(idx >= size() / 2))
                    return ::extract_elem(data[1], idx - size() / 2);
                else {
                    const vector<T, Elems / 2> tmp = SHIFT_BYTES(data[0], data[1], idx * sizeof(T));
                    return tmp.get(0);
                }
            }
            else {
                constexpr unsigned n = native_vector_length<T>::value;
                return ::extract_elem(data[idx / n], idx % n);
            }
        }
    }

    template <typename... SubVectors>
    __aie_inline
    vector_base &upd_all(SubVectors && ...subvectors)
    {
        constexpr unsigned num_subvectors  = sizeof...(subvectors);
        constexpr unsigned subvector_bits  = bits() / num_subvectors;
        constexpr unsigned subvector_elems = size() / num_subvectors;

        static_assert(sizeof...(subvectors) > 1);
        static_assert((std::is_base_of_v<vector_base<T, subvector_elems>, utils::remove_all_t<SubVectors>> && ...));

        static_assert(bits() == subvector_bits * num_subvectors);

        static_assert(subvector_bits >= 128 && utils::is_powerof2(subvector_bits));

        const std::array arr = {std::forward<SubVectors>(subvectors)...};

        if constexpr (bits() >= 1024) {
            if constexpr (subvector_bits == 128) {
                utils::unroll_times<num_subvectors / 4>([&](unsigned idx) __aie_inline {
#if __AIE_API_128_BIT_INSERT_CONCAT__
                    data[idx] = ::concat(arr[4 * idx].data, arr[4 * idx + 1].data, arr[4 * idx + 2].data, arr[4 * idx + 3].data);
#else
                    this->insert(4 * idx + 0, arr[4 * idx + 0]);
                    this->insert(4 * idx + 1, arr[4 * idx + 1]);
                    this->insert(4 * idx + 2, arr[4 * idx + 2]);
                    this->insert(4 * idx + 3, arr[4 * idx + 3]);
#endif
                });
            }
            else if constexpr (subvector_bits == 256) {
                utils::unroll_times<num_subvectors / 2>([&](unsigned idx) __aie_inline {
                    data[idx] = ::concat(arr[2 * idx].data, arr[2 * idx + 1].data);
                });
            }
            else if constexpr (subvector_bits == 512) {
                utils::unroll_times<num_subvectors>([&](unsigned idx) __aie_inline {
                    data[idx] = arr[idx].data;
                });
            }
            else if constexpr (subvector_bits > 512) {
                constexpr unsigned chunks = subvector_bits / 512;
                utils::unroll_times_2d<num_subvectors, chunks>([&](unsigned idx, unsigned j) __aie_inline {
                    data[idx * chunks + j] = arr[idx].data[j];
                });
            }
        }
    #if __AIE_API_128_BIT_INSERT_CONCAT__ == 0
        else if constexpr (bits() == 512) {
            if constexpr (num_subvectors == 2) {
                data = ::concat(std::forward<SubVectors>(subvectors)...);
            }
            else if constexpr (num_subvectors == 4) {
                this->insert(0, arr[0]);
                this->insert(1, arr[1]);
                this->insert(2, arr[2]);
                this->insert(3, arr[3]);
            }
        }
        else if constexpr (bits() == 256) {
            if constexpr (num_subvectors == 2) {
                this->insert(0, arr[0]);
                this->insert(1, arr[1]);
            }
        }
    #else
        else {
            data = ::concat(std::forward<SubVectors>(subvectors)...);
        }
    #endif

        return *this;
    }

    template <unsigned ElemsIn>
    __aie_inline
    vector_base &insert(unsigned idx, const vector_base<T, ElemsIn> &v)
    {
        REQUIRES_MSG(idx < Elems / ElemsIn, "idx needs to be a valid subvector index");

        insert_helper<ElemsIn>(idx, v);

        return *this;
    }

    template <typename U> requires (std::is_same_v<T, typename native_vector_traits<U>::value_type>)
    __aie_inline
    vector_base &insert(unsigned idx, const U &v)
    {
        constexpr unsigned elems_in = native_vector_traits<U>::size;

        return insert(idx, vector_base<T, elems_in>(v));
    }

    template <unsigned ElemsOut>
    __aie_inline
    vector_base<T, ElemsOut> extract(unsigned idx) const
    {
        REQUIRES_MSG(idx < Elems / ElemsOut, "idx needs to be a valid subvector index");

        return extract_helper<ElemsOut>(idx);
    }

    template <unsigned ElemsOut>
    __aie_inline
    vector_base<value_type, ElemsOut> grow_extract(unsigned idx) const
    {
        if constexpr (ElemsOut > Elems)
            return grow<ElemsOut>();
        else
            return extract<ElemsOut>(idx);
    }

    template <unsigned ElemsOut>
    __aie_inline
    auto split() const
    {
        constexpr unsigned output_bits = type_bits() * ElemsOut;

        static_assert(output_bits <= bits() && utils::is_powerof2(output_bits));

        if constexpr (output_bits == bits())
            return *this;
        else
            return split_helper<ElemsOut>(std::make_integer_sequence<unsigned, Elems / ElemsOut>{});
    }

    template <typename T2 = typename unpacked_type<T>::type>
        requires(utils::is_one_of_v<T2, int8, uint8, int16, uint16>
                && type_bits_v<T2> / type_bits_v<T> == 2)
    __aie_inline
    auto unpack_sign(bool v_sign) const -> vector_base<T2, size()>
    {
        using next_vector_type = vector_base<T2, size()>;
        next_vector_type ret;

        if constexpr (bits() == 128) {
            ret = this->template grow<size() * 2>().template unpack_sign<T2>(v_sign).template extract<size()>(0);
        }
        else if constexpr (bits() == 256) {
            ret = ::unpack(data, v_sign);
        }
        else if constexpr (bits() >= 512) {
            utils::unroll_times<Elems / (native_elems / 2)>([&](unsigned idx) __aie_inline {
                ret.insert(idx, this->template extract<native_elems / 2>(idx).template unpack_sign<T2>(v_sign));
            });
        }

        return ret;
    }

    template <typename T2 = typename unpacked_type<T>::type>
        requires(utils::is_one_of_v<T2, int8, uint8, int16, uint16>
                && type_bits_v<T2> / type_bits_v<T> == 2)
    __aie_inline
    auto unpack() const -> vector_base<T2, size()>
    {
        return unpack_sign<T2>(is_signed());
    }

    template <typename T2 = typename packed_type<T>::type>
        requires(utils::is_one_of_v<T2, int4, uint4, int8, uint8>
                && type_bits_v<T> / type_bits_v<T2> == 2)
    __aie_inline
    auto pack_sign(bool v_sign) const -> vector_base<T2, size()>
    {
        using prev_vector_type = vector_base<T2, size()>;
        prev_vector_type ret;

        if constexpr (bits() == 256) {
            ret = this->template grow<size() * 2>().template pack_sign<T2>(v_sign).template extract<size()>(0);
        }
        else if constexpr (bits() == 512) {
            ret = ::pack(data, v_sign);
        }
        else if constexpr (bits() >= 1024) {
            utils::unroll_times<Elems / native_elems>([&](unsigned idx) __aie_inline {
                ret.insert(idx, this->template extract<native_elems>(idx).template pack_sign<T2>(v_sign));
            });
        }

        return ret;
    }

    template <typename T2 = typename packed_type<T>::type>
        requires(utils::is_one_of_v<T2, int4, uint4, int8, uint8>
                && type_bits_v<T> / type_bits_v<T2> == 2)
    __aie_inline
    auto pack() const -> vector_base<T2, size()>
    {
        return pack_sign<T2>(is_signed());
    }

    template <aie_dm_resource Resource = aie_dm_resource::none, typename T2> requires(std::is_same_v<aie_dm_resource_remove_t<T2>, value_type>)
    __aie_inline
    void load(const T2 *ptr)
    {
        if constexpr (bits() >= 1024) {
            using native_type = native_vector_type_t<value_type, native_elems>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            utils::unroll_times<Elems / native_elems>([&](unsigned idx) __aie_inline {
                data[idx] = ((const aliased_type *) ptr)[idx];
            });
        }
        else {
            using native_type = native_vector_type_t<value_type, Elems>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            data = *(const aliased_type *)ptr;
        }
    }

    template <aie_dm_resource Resource = aie_dm_resource::none, typename T2> requires(std::is_same_v<aie_dm_resource_remove_t<T2>, value_type>)
    __aie_inline
    void store(T2 *ptr) const
    {
        if constexpr (bits() >= 1024) {
            using native_type = native_vector_type_t<value_type, native_elems>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            utils::unroll_times<Elems / native_elems>([&](unsigned idx) __aie_inline {
                ((aliased_type *)ptr)[idx] = data[idx];
            });
        }
        else {
            using native_type = native_vector_type_t<value_type, Elems>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            *(aliased_type *)ptr = data;
        }
    }

    template <aie_dm_resource Resource = aie_dm_resource::none, typename T2> requires(std::is_same_v<aie_dm_resource_remove_t<T2>, value_type>)
    __aie_inline
    void load_unaligned(const T2 *ptr, unsigned aligned_elems)
    {
        constexpr unsigned subbyte_elems = type_bits() == 4 ? 2 : 1;
        const unsigned aligned_bits = aligned_elems * type_bits();

        constexpr unsigned required_alignment = vector_ldst_align<T, Elems>::value * 8;

        if (chess_manifest(aligned_bits >= required_alignment)) {
            load<Resource>(ptr);
        }
        else if constexpr (bits() == 128) {
            using native_type = native_vector_type_t<value_type, Elems * 2>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            const unsigned frac = uintptr_t(ptr) & 31;

            ptr = utils::floor_ptr<Elems * 2>(ptr);

            vector_base<value_type, Elems * 4> tmp;

            tmp = vector_set<value_type, Elems * 4>::run(*(aliased_type *)ptr, 0); ptr += Elems * 2 / subbyte_elems;
            if (!chess_manifest(frac <= 16))
                tmp.template insert<Elems * 2>(1, *(aliased_type *)ptr);

            tmp = SHIFT_BYTES(tmp, vector_base<value_type, Elems * 4>(), frac);

            data = tmp.template extract<Elems>(0);
        }
        else if constexpr (bits() == 256) {
            using native_type = native_vector_type_t<value_type, Elems>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            const unsigned frac = uintptr_t(ptr) & 31;

            ptr = utils::floor_ptr<Elems>(ptr);

            vector_base<value_type, Elems * 2> tmp;

            tmp = vector_set<value_type, Elems * 2>::run(*(aliased_type *)ptr, 0); ptr += Elems / subbyte_elems;
            tmp.template insert<Elems>(1, *(aliased_type *)ptr);

            tmp = SHIFT_BYTES(tmp, vector_base<value_type, Elems * 2>(), frac);

            data = tmp.template extract<Elems>(0);
        }
        else if constexpr (bits() == 512) {
            using native_type = native_vector_type_t<value_type, Elems / 2>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            const unsigned frac = uintptr_t(ptr) & 31;

            ptr = utils::floor_ptr<Elems / 2>(ptr);

            vector_base<value_type, Elems> tmp1, tmp2;

            tmp1 = vector_set<value_type, Elems>::run(*(aliased_type *)ptr, 0); ptr += Elems / 2 / subbyte_elems;
            tmp1.template insert<Elems / 2>(1, *(aliased_type *)ptr);           ptr += Elems / 2 / subbyte_elems;

            tmp2 = vector_set<value_type, Elems>::run(*(aliased_type *)ptr, 0);

            data = SHIFT_BYTES(tmp1, tmp2, frac);
        }
        else if constexpr (bits() == 1024) {
            using native_type = native_vector_type_t<value_type, Elems / 4>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            const unsigned frac = uintptr_t(ptr) & 31;

            ptr = utils::floor_ptr<Elems / 4>(ptr);

            vector_base<value_type, Elems / 2> tmp1, tmp2;

            tmp1 = vector_set<value_type, Elems / 2>::run(*(aliased_type *)ptr, 0); ptr += Elems / 4 / subbyte_elems;
            tmp1.template insert<Elems / 4>(1, *(aliased_type *)ptr);               ptr += Elems / 4 / subbyte_elems;

            tmp2 = vector_set<value_type, Elems / 2>::run(*(aliased_type *)ptr, 0); ptr += Elems / 4 / subbyte_elems;

            data[0] = SHIFT_BYTES(tmp1, tmp2, frac);

            tmp2.template insert<Elems / 4>(1, *(aliased_type *)ptr);               ptr += Elems / 4 / subbyte_elems;

            tmp1 = vector_set<value_type, Elems / 2>::run(*(aliased_type *)ptr, 0);

            data[1] = SHIFT_BYTES(tmp2, tmp1, frac);
        }
    }

    template <aie_dm_resource Resource = aie_dm_resource::none, typename T2> requires(std::is_same_v<aie_dm_resource_remove_t<T2>, value_type>)
    __aie_inline
    void store_unaligned(T2 *ptr, unsigned aligned_elems) const
    {
        constexpr unsigned subbyte_elems = type_bits() == 4 ? 2 : 1;
        const unsigned aligned_bits = aligned_elems * type_bits();

        constexpr unsigned required_alignment = vector_ldst_align<T, Elems>::value * 8;

        if (chess_manifest(aligned_bits >= required_alignment)) {
            store<Resource>(ptr);
        }
        else if constexpr (bits() == 128 || bits() == 256) {
            constexpr unsigned vector_native_bytes = bits() / 8;
            constexpr unsigned    mem_native_bytes = 256 / 8;
            using  native_type = native_vector_type_t<int8, mem_native_bytes>;
            using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<native_type, aie_dm_resource_get_v<T2>>>;

            vector_base<int8, vector_native_bytes> vec = this->cast_to<int8>();

            const unsigned frac = uintptr_t(ptr) & (mem_native_bytes - 1);
            ptr = utils::floor_ptr<mem_native_bytes / sizeof(T2)>(ptr);

            aliased_type *native_ptr = (aliased_type *)ptr;

            uint64_t            m = ((1ULL << vector_native_bytes) - 1) << frac;
            vector_base<int8, 64>    x = SHIFT_BYTES(::undef_v64int8(), ::set_v64int8(0, vec), 64 - frac);
            vector_base<int8, 64>    y = ::set_v64int8(0, native_ptr[0]);
            y = ::insert(y, 1, native_ptr[1]);
            y = ::sel(y, x, m);

            native_ptr[0] = y.extract<mem_native_bytes>(0);
            native_ptr[1] = y.extract<mem_native_bytes>(1);
        }
        else if constexpr (bits() >= 512) {
            // Reuse 256b implementation for simplicity at the cost of additional loads
            //
            // TODO: provide custom implementation for 1024b
            constexpr unsigned elems_per_store = 256u / type_bits_v<T2>;
            constexpr unsigned num_stores      = Elems / elems_per_store;

            utils::unroll_times<num_stores>([&](unsigned idx) __aie_inline {
                this->template extract<elems_per_store>(idx)
                     .template store_unaligned<Resource, T2>(ptr + idx * elems_per_store / subbyte_elems, aligned_elems);
            });
        }
    }

    __aie_inline
    native_type to_native() const requires (bits() <= 1024)
    {
        // TODO: verify performance of the ::concat approach
        if constexpr (utils::num_elems_v<vector_storage_t<T, Elems>> == 2)
            return ::concat(data[0], data[1]);
        else
            return data;
    }

    __aie_inline
    operator native_type() const requires (bits() <= 1024)
    {
        return to_native();
    }

private:
    template <typename Data, typename... Values>
    __aie_inline
    Data init_from_values(Data d, const value_type &v, const Values &... values)
    {
        if constexpr (sizeof...(values) > 0)
            d = init_from_values(d, values...);

        return ::shiftr_elem(d, v);
    }

    template <unsigned Offset, unsigned Size, unsigned Index, typename Data, typename ValuesTuple>
    __aie_inline
    Data init_from_values(Data d, const ValuesTuple &values)
    {
        if constexpr ((Index + 1) < Size) {
            d = init_from_values<Offset, Size, Index + 1>(d, values);
        }

        return ::shiftr_elem(d, std::get<Offset + Index>(values));
    }

    template <unsigned ElemsOut, unsigned... Indices>
    __aie_inline
    std::array<vector_base<T, ElemsOut>, Elems/ElemsOut> split_helper(std::integer_sequence<unsigned, Indices...> &&) const
    {
        return {extract<ElemsOut>(Indices)...};
    }

    template <unsigned Elems2>
    __aie_inline
    void insert_helper(unsigned idx, const vector_base<T, Elems2> &v)
    {
        constexpr unsigned input_bits = type_bits() * Elems2;

        static_assert(input_bits <= bits());
        static_assert(input_bits >= 128 && utils::is_powerof2(input_bits));

        if constexpr (input_bits == bits()) {
            data = v.data;
            return;
        }
        else if constexpr (bits() >= 1024) {
            if constexpr (input_bits > 512) {
                constexpr unsigned Blocks = Elems2 / native_elems;

                utils::unroll_times<Blocks>([&](unsigned j) __aie_inline {
                    data[(Blocks * idx) + j] = v.data[j];
                });
            }
            else if constexpr (input_bits == 512) {
                data[idx] = v;
            }
            else if constexpr (input_bits == 256) {
                data[idx / 2] = ::insert(data[idx / 2], idx % 2, v);
            }
            else if constexpr (input_bits == 128) {
                vector_base<int32, 16> tmp = (v16int32)data[idx / 4];

                const unsigned local_idx = idx % 4;
                constexpr unsigned mask_elems = 4;
                constexpr unsigned mask_base = (1u << mask_elems) - 1u;
                const     unsigned mask      = mask_base << (mask_elems * local_idx);

                if (chess_manifest(local_idx == 0)) {
                    tmp = ::sel(tmp, v.template cast_to<int32>().template grow<16>(), mask);

                    data[idx / 4] = tmp.template cast_to<value_type>().data;
                }
                else {
                    vector_base<value_type, Elems2 * 4> v_tmp = v.template grow<Elems2 * 4>();

                    v_tmp = SHIFT_BYTES(v_tmp, v_tmp, 64 - 16 * local_idx);
                    tmp = ::sel(tmp, v_tmp.template cast_to<int32>(), mask);

                    data[idx / 4] = tmp.template cast_to<value_type>().data;
                }
            }
        }
        else {
            if constexpr (input_bits == 256) {
                data = ::insert(data, idx, v);
            }
            else if constexpr (input_bits == 128) {
                constexpr unsigned v512_elems = 512 / type_bits();

                vector_base<int32, 16> tmp = (v16int32)grow<v512_elems>().data;

                constexpr unsigned mask_elems = 4;
                constexpr unsigned mask_base = (1u << mask_elems) - 1u;
                const     unsigned mask      = mask_base << (mask_elems * idx);

                if (chess_manifest(idx == 0)) {
                    tmp = ::sel(tmp, v.template cast_to<int32>().template grow<16>(), mask);

                    data = tmp.template cast_to<value_type>().template extract<Elems>(0).data;
                }
                else {
                    vector_base<value_type, Elems2 * 4> v_tmp = v.template grow<Elems2 * 4>();

                    v_tmp = SHIFT_BYTES(v_tmp, v_tmp, 64 - 16 * idx);
                    tmp = ::sel(tmp, v_tmp.template cast_to<int32>(), mask);

                    data = tmp.template cast_to<value_type>().template extract<Elems>(0).data;
                }
            }
        }
    }

    template <unsigned N>
    __aie_inline
    vector_base<value_type, N> extract_helper(unsigned idx) const
    {
        constexpr unsigned output_bits = type_bits() * N;

        static_assert(output_bits <= bits());
        static_assert(output_bits >= 128 && utils::is_powerof2(output_bits));

        if constexpr (output_bits == bits()) {
            return *this;
        }
        else {
            if constexpr (bits() >= 1024) {
                if constexpr (output_bits > 512) {
                    constexpr unsigned Blocks = N / native_elems;

                    vector_base<value_type, N> ret;
                    
                    utils::unroll_times<Blocks>([&](unsigned j) __aie_inline {
                        ret.data[j] = data[(Blocks * idx) + j];
                    });

                    return ret;
                }
                else if constexpr (output_bits == 512) {
                    return data[idx];
                }
                else if constexpr (output_bits == 256) {
                    return vector_extract<N>(data[idx / 2], idx % 2);
                }
                else if constexpr (output_bits == 128) {
                    return vector_extract<N>(data[idx / 4], idx % 4);
                }
            }
            else if constexpr (bits() == 512) {
                if constexpr (output_bits == 256) {
                    return vector_extract<N>(data, idx);
                }
                else if constexpr (output_bits == 128) {
                    return vector_extract<N>(data, idx);
                }
            }
            else if constexpr (bits() == 256) {
                if constexpr (output_bits == 128) {
                    return vector_extract<N>(data, idx);

                }
            }
        }
    }

    template <typename ScalarType, aie_dm_resource Resource, typename T2> requires(std::is_same_v<aie_dm_resource_remove_t<T2>, value_type>)
    void store_unaligned_scalar(T2 *ptr) const
    {
        constexpr unsigned num_elems = bits() / type_bits_v<ScalarType>;
        const vector_base<ScalarType, num_elems> tmp = vector_cast_helper<ScalarType, num_elems>(data);

        using aliased_type = add_memory_bank_t<Resource, aie_dm_resource_set_t<ScalarType, aie_dm_resource_get_v<T2>>>;

        aliased_type *ptr2 = (aliased_type *)ptr;

        if constexpr (type_bits_v<ScalarType> > 8) {
            for (unsigned i = 0; i < num_elems; ++i) chess_unroll_loop()
                ptr2[i] = tmp.get(i);
        }
        else {
            for (unsigned i = 0; i < num_elems; ++i) chess_prepare_for_pipelining
                ptr2[i] = tmp.get(i);
        }
    }

    static constexpr unsigned native_elems      = native_vector_length_v<T>;
    static constexpr unsigned num_storage_elems = utils::num_elems_v<vector_storage_t<T, Elems>>;
    static constexpr bool is_compound_storage   = num_storage_elems > 1;

#ifdef AIE_API_EMULATION
    std::array<T, Elems> data;
#else
    vector_storage_t<T, Elems> data;
#endif
};

} // namespace aie::detail

#endif // __AIE_API_DETAIL_AIE2_VECTOR__HPP__