// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

#include <aie_api/aie.hpp>

// The adf-free cascade accessors are available where __AIE_API_HAS_CASCADE__ is
// set (today: aie2p under Peano; see aie_api/cascade.hpp). Guarding the example
// on the capability macro keeps a default build green on architectures and
// compilers that do not implement it yet, and exercises the accessors where it
// is available, e.g. clang++ --target=aie2p-none-unknown-elf -I../include -c cascade.cpp.
#if __AIE_API_HAS_CASCADE__

//![Adf-free cascade round-trip]
// Put a 16-lane int32 vector to the master cascade, then read one back from the
// slave cascade. On real hardware the two ends are on adjacent cores; a fused
// cross-core reduction issues the put on one core and the get on the next.
extern "C" aie::vector<int32_t, 16> cascade_passthrough_i32(aie::vector<int32_t, 16> x)
{
    aie::cascade_out(x);
    return aie::cascade_in_i32();
}

// The accumulator path, which a fused cross-core K-reduction uses (partial sums
// travel as accumulators).
extern "C" aie::accum<acc32, 16> cascade_passthrough_acc32(aie::accum<acc32, 16> a)
{
    aie::cascade_out(a);
    return aie::cascade_in_acc32();
}
//![Adf-free cascade round-trip]

#endif // __AIE_API_HAS_CASCADE__
