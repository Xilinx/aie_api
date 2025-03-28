<!--
SPDX-License-Identifier: MIT
Copyright (C) 2022 Xilinx, Inc.
Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
-->

# AIE API

AIE API is a portable programming interface for AIE accelerators. It is implemented as a C++ header-only library that provides types and operations that get translated into efficient low-level intrinsics. 
The API also provides higher-level abstractions such as iterators and multi-dimensional arrays.

## Using AIE API in your application

Since AIE API is a header-only library, you only need to add the AIE API to your include path and include the corresponding header in your application.

`#include <aie_api/aie.hpp>`

## Documentation

View our [documentation site](https://xilinx.github.io/aie_api/).

You can also browse the Doxygen documentation under the doc folder.
