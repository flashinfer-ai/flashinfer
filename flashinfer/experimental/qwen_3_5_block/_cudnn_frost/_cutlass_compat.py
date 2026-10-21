# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTeDSL imports shared by kernels supporting both 4.6.2 and newer wheels.

Prefer the owning modules introduced in 4.8. The old utility exports warn on
every access in newer wheels, but remain necessary on the supported 4.6.2 floor.
This module is imported by kernel modules only, never by the package entry point.
"""

try:
    from cutlass.tensor_utils import LayoutEnum
except ModuleNotFoundError as error:
    if error.name != "cutlass.tensor_utils":
        raise
    from cutlass.utils import LayoutEnum

try:
    from cutlass.memory import SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols, get_smem_capacity_in_bytes
except ModuleNotFoundError as error:
    if error.name != "cutlass.memory":
        raise
    from cutlass.utils import SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols, get_smem_capacity_in_bytes

try:
    from cutlass.base_dsl.enums import Arch
except ModuleNotFoundError as error:
    if error.name != "cutlass.base_dsl.enums":
        raise
    from cutlass.base_dsl.arch import Arch

from cutlass.cute import FastDivmodDivisorV2 as FastDivmodDivisor
from cutlass.cute import fast_divmod_create_divisor_v2 as fast_divmod_create_divisor
from cutlass.cute.arch import setmaxregister_decrease, setmaxregister_increase
from cutlass.cute.nvgpu import OperandMajorMode
