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
"""Generated programs of the SM120 block-64 block-sparse attention of the Cake VSA module:
the BF16 and FP16 builds of two persistent kernels, shipped for sm_120a and sm_121a:
VSA_SM120_BLK64_BF16 / _F16 (two 128-thread CTAs per SM) and VSA_SM120_BLK64_PAIR_BF16 /
_F16 (one CTA per SM of two 128-thread groups). Both walk the host plan's work items
over two virtual CTAs per SM (one per (head, query block) row, a long row split into
near-equal parts whose last arrival merges the partial states) over the row's KV-block
list, head dim 128, with the natural-log LSE written when asked, to bitwise-equal
outputs. The host module ``flashinfer.cake_vsa_sm120`` computes the plan, picks the
faster kernel for it and its grid, and launches it behind
BlockSparseAttentionWrapper(backend="vsa_sm120_blk64").

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

VSA_SM120_BLK64_BF16 = cake_program.CakeProgram(
    name="cake_vsa_sm120_c59bc964e94a39a04842",
    operator_dir="cake_vsa_sm120",
    kernel="kernel_cake_vsa_sm120_c59bc964e94a39a04842",
    sources=(
        "cake_vsa_sm120_c59bc964e94a39a04842_kernel.cu",
        "cake_vsa_sm120_c59bc964e94a39a04842_binding.cu",
    ),
    targets=("sm_120a", "sm_121a"),
    compile_options=(("sm_120a", ("--fmad=true",)), ("sm_121a", ("--fmad=true",))),
    args=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
        "num_heads",
        "group_size",
        "total_work",
        "write_lse",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
    ),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=50176,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

VSA_SM120_BLK64_F16 = cake_program.CakeProgram(
    name="cake_vsa_sm120_5645b831eaa1960fa011",
    operator_dir="cake_vsa_sm120",
    kernel="kernel_cake_vsa_sm120_5645b831eaa1960fa011",
    sources=(
        "cake_vsa_sm120_5645b831eaa1960fa011_kernel.cu",
        "cake_vsa_sm120_5645b831eaa1960fa011_binding.cu",
    ),
    targets=("sm_120a", "sm_121a"),
    compile_options=(("sm_120a", ("--fmad=true",)), ("sm_121a", ("--fmad=true",))),
    args=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
        "num_heads",
        "group_size",
        "total_work",
        "write_lse",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
    ),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=50176,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

VSA_SM120_BLK64_PAIR_BF16 = cake_program.CakeProgram(
    name="cake_vsa_sm120_34e8f811b2d4fd8b01fd",
    operator_dir="cake_vsa_sm120",
    kernel="kernel_cake_vsa_sm120_34e8f811b2d4fd8b01fd",
    sources=(
        "cake_vsa_sm120_34e8f811b2d4fd8b01fd_kernel.cu",
        "cake_vsa_sm120_34e8f811b2d4fd8b01fd_binding.cu",
    ),
    targets=("sm_120a", "sm_121a"),
    compile_options=(("sm_120a", ("--fmad=true",)), ("sm_121a", ("--fmad=true",))),
    args=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
        "num_heads",
        "group_size",
        "total_work",
        "write_lse",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=99328,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

VSA_SM120_BLK64_PAIR_F16 = cake_program.CakeProgram(
    name="cake_vsa_sm120_aeffa6c83df0348168ae",
    operator_dir="cake_vsa_sm120",
    kernel="kernel_cake_vsa_sm120_aeffa6c83df0348168ae",
    sources=(
        "cake_vsa_sm120_aeffa6c83df0348168ae_kernel.cu",
        "cake_vsa_sm120_aeffa6c83df0348168ae_binding.cu",
    ),
    targets=("sm_120a", "sm_121a"),
    compile_options=(("sm_120a", ("--fmad=true",)), ("sm_121a", ("--fmad=true",))),
    args=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
        "num_heads",
        "group_size",
        "total_work",
        "write_lse",
        "softmax_scale_log2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "q_map",
        "k_map",
        "v_map",
        "out",
        "lse",
        "partial_ws",
        "split_counters",
        "q2k_index",
        "work",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=99328,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    VSA_SM120_BLK64_BF16,
    VSA_SM120_BLK64_F16,
    VSA_SM120_BLK64_PAIR_BF16,
    VSA_SM120_BLK64_PAIR_F16,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "VSA_SM120_BLK64_BF16",
    "VSA_SM120_BLK64_F16",
    "VSA_SM120_BLK64_PAIR_BF16",
    "VSA_SM120_BLK64_PAIR_F16",
    "PROGRAMS",
    "aot_specs",
]
