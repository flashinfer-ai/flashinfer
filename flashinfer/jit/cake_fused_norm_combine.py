# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generated programs of the eight-peer fused residual-add, two-track RMSNorm and BF16
combine (hidden size 2560, sm_100a and sm_103a), one per dispatch band of the token
count: the one-shot exchange with one CTA per token (with a one-token build), the same
exchange over eight CTAs per token, and the two-round owner reduce as a persistent
streaming grid and as a persistent software-pipelined grid at four and at three CTAs per
SM, all three launched cooperatively. The host module
``flashinfer.comm.cake_fused_norm_combine`` owns the peer-mapped workspace, validates
the tensors, selects the program and the grid and launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

PARALLEL_SINGLE = cake_program.CakeProgram(
    name="cake_fused_norm_combine_5d69eac5f72e62976f4e",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_5d69eac5f72e62976f4e",
    sources=(
        "cake_fused_norm_combine_5d69eac5f72e62976f4e_kernel.cu",
        "cake_fused_norm_combine_5d69eac5f72e62976f4e_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
        "rank",
        "tokens",
        "epsilon",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
    ),
    block=(320, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=10496,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PARALLEL = cake_program.CakeProgram(
    name="cake_fused_norm_combine_f745596d3f90db224372",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_f745596d3f90db224372",
    sources=(
        "cake_fused_norm_combine_f745596d3f90db224372_kernel.cu",
        "cake_fused_norm_combine_f745596d3f90db224372_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
        "rank",
        "tokens",
        "epsilon",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
    ),
    block=(320, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=10496,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PARALLEL_SPLIT = cake_program.CakeProgram(
    name="cake_fused_norm_combine_3def0164ca6263e62e56",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_3def0164ca6263e62e56",
    sources=(
        "cake_fused_norm_combine_3def0164ca6263e62e56_kernel.cu",
        "cake_fused_norm_combine_3def0164ca6263e62e56_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
        "rank",
        "tokens",
        "epsilon",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
    ),
    block=(320, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=10496,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

OWNER_STREAM = cake_program.CakeProgram(
    name="cake_fused_norm_combine_b18e0a01ecd652ae20b9",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_b18e0a01ecd652ae20b9",
    sources=(
        "cake_fused_norm_combine_b18e0a01ecd652ae20b9_kernel.cu",
        "cake_fused_norm_combine_b18e0a01ecd652ae20b9_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
        "rank",
        "tokens",
        "epsilon",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
    ),
    block=(320, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=256,
    use_pdl=True,
    cooperative=True,
    needs_device_linking=False,
)

OWNER_PIPELINED = cake_program.CakeProgram(
    name="cake_fused_norm_combine_3e5044114a64d435bf4e",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_3e5044114a64d435bf4e",
    sources=(
        "cake_fused_norm_combine_3e5044114a64d435bf4e_kernel.cu",
        "cake_fused_norm_combine_3e5044114a64d435bf4e_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
        "rank",
        "tokens",
        "epsilon",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
    ),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=128,
    use_pdl=True,
    cooperative=True,
    needs_device_linking=False,
)

OWNER_PIPELINED_EARLY = cake_program.CakeProgram(
    name="cake_fused_norm_combine_8d72efaafce77c1ae5f9",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_8d72efaafce77c1ae5f9",
    sources=(
        "cake_fused_norm_combine_8d72efaafce77c1ae5f9_kernel.cu",
        "cake_fused_norm_combine_8d72efaafce77c1ae5f9_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
        "rank",
        "tokens",
        "epsilon",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
    ),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=128,
    use_pdl=True,
    cooperative=True,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    PARALLEL_SINGLE,
    PARALLEL,
    PARALLEL_SPLIT,
    OWNER_STREAM,
    OWNER_PIPELINED,
    OWNER_PIPELINED_EARLY,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "PARALLEL_SINGLE",
    "PARALLEL",
    "PARALLEL_SPLIT",
    "OWNER_STREAM",
    "OWNER_PIPELINED",
    "OWNER_PIPELINED_EARLY",
    "PROGRAMS",
    "aot_specs",
]
