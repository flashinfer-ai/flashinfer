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
    name="cake_fused_norm_combine_102ff2a7592bbb88f200",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_102ff2a7592bbb88f200",
    sources=(
        "cake_fused_norm_combine_102ff2a7592bbb88f200_kernel.cu",
        "cake_fused_norm_combine_102ff2a7592bbb88f200_binding.cu",
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
    name="cake_fused_norm_combine_997eebeeff40e290e056",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_997eebeeff40e290e056",
    sources=(
        "cake_fused_norm_combine_997eebeeff40e290e056_kernel.cu",
        "cake_fused_norm_combine_997eebeeff40e290e056_binding.cu",
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
    name="cake_fused_norm_combine_45d4f681e01eef5ea0d0",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_45d4f681e01eef5ea0d0",
    sources=(
        "cake_fused_norm_combine_45d4f681e01eef5ea0d0_kernel.cu",
        "cake_fused_norm_combine_45d4f681e01eef5ea0d0_binding.cu",
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
    name="cake_fused_norm_combine_426f8d7edfb8a57040ce",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_426f8d7edfb8a57040ce",
    sources=(
        "cake_fused_norm_combine_426f8d7edfb8a57040ce_kernel.cu",
        "cake_fused_norm_combine_426f8d7edfb8a57040ce_binding.cu",
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
    dynamic_smem_bytes=128,
    use_pdl=True,
    cooperative=True,
    needs_device_linking=False,
)

OWNER_PIPELINED = cake_program.CakeProgram(
    name="cake_fused_norm_combine_986ac9f50a9e95d43781",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_986ac9f50a9e95d43781",
    sources=(
        "cake_fused_norm_combine_986ac9f50a9e95d43781_kernel.cu",
        "cake_fused_norm_combine_986ac9f50a9e95d43781_binding.cu",
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
    name="cake_fused_norm_combine_5550b992bddbdf98a615",
    operator_dir="cake_fused_norm_combine",
    kernel="kernel_cake_fused_norm_combine_5550b992bddbdf98a615",
    sources=(
        "cake_fused_norm_combine_5550b992bddbdf98a615_kernel.cu",
        "cake_fused_norm_combine_5550b992bddbdf98a615_binding.cu",
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
