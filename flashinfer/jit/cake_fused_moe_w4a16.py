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
"""Generated programs of the Blackwell BF16 x NVFP4 routed-expert MoE route (BF16
activations against E2M1 codes with E4M3 block scales, decoded in the kernel): FC1
(gated SwiGLU to BF16) and FC2 (to BF16) on the moe_sort metadata, one program per route
cell (8, 16, 32 or 64 rows per tile as one CTA, 128 rows as a 2-CTA pair), each shipped
for sm_100a and sm_103a. The host module ``flashinfer.fused_moe.cake_w4a16`` selects the
cell from the token count and the routing geometry, computes the grid and launches both
programs with programmatic dependent launch.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

FC1_N8 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_7a2eedb15ab3ca64aa28",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_7a2eedb15ab3ca64aa28",
    sources=(
        "cake_fused_moe_w4a16_7a2eedb15ab3ca64aa28_kernel.cu",
        "cake_fused_moe_w4a16_7a2eedb15ab3ca64aa28_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "swiglu_limit",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "permuted_idx_to_expanded_idx",
        "top_k_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "permuted_idx_to_expanded_idx",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=228480,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC1_N16 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_16715f6889ca11019f15",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_16715f6889ca11019f15",
    sources=(
        "cake_fused_moe_w4a16_16715f6889ca11019f15_kernel.cu",
        "cake_fused_moe_w4a16_16715f6889ca11019f15_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "swiglu_limit",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=218240,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC1_N32 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_813fc13553f0bd7f9a94",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_813fc13553f0bd7f9a94",
    sources=(
        "cake_fused_moe_w4a16_813fc13553f0bd7f9a94_kernel.cu",
        "cake_fused_moe_w4a16_813fc13553f0bd7f9a94_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "swiglu_limit",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=218240,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC1_N64 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_c3d23c9abab97a0cf985",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_c3d23c9abab97a0cf985",
    sources=(
        "cake_fused_moe_w4a16_c3d23c9abab97a0cf985_kernel.cu",
        "cake_fused_moe_w4a16_c3d23c9abab97a0cf985_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "swiglu_limit",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=214144,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC1_N128_CG2 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_d889b3a0afa52e8d642b",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_d889b3a0afa52e8d642b",
    sources=(
        "cake_fused_moe_w4a16_d889b3a0afa52e8d642b_kernel.cu",
        "cake_fused_moe_w4a16_d889b3a0afa52e8d642b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "swiglu_limit",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=214144,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC2_N8 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_0b00013b06119e105c73",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_0b00013b06119e105c73",
    sources=(
        "cake_fused_moe_w4a16_0b00013b06119e105c73_kernel.cu",
        "cake_fused_moe_w4a16_0b00013b06119e105c73_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=230528,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC2_N16 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_a958db7e4d400d13f13e",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_a958db7e4d400d13f13e",
    sources=(
        "cake_fused_moe_w4a16_a958db7e4d400d13f13e_kernel.cu",
        "cake_fused_moe_w4a16_a958db7e4d400d13f13e_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=222336,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC2_N32 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_b86a08401b8b925ec8d0",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_b86a08401b8b925ec8d0",
    sources=(
        "cake_fused_moe_w4a16_b86a08401b8b925ec8d0_kernel.cu",
        "cake_fused_moe_w4a16_b86a08401b8b925ec8d0_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=226432,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC2_N64 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_2f8952e35d5b9d82dcc8",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_2f8952e35d5b9d82dcc8",
    sources=(
        "cake_fused_moe_w4a16_2f8952e35d5b9d82dcc8_kernel.cu",
        "cake_fused_moe_w4a16_2f8952e35d5b9d82dcc8_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=222336,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

FC2_N128_CG2 = cake_program.CakeProgram(
    name="cake_fused_moe_w4a16_b90ebd08fea00013ca1d",
    operator_dir="cake_fused_moe_w4a16",
    kernel="kernel_cake_fused_moe_w4a16_b90ebd08fea00013ca1d",
    sources=(
        "cake_fused_moe_w4a16_b90ebd08fea00013ca1d_kernel.cu",
        "cake_fused_moe_w4a16_b90ebd08fea00013ca1d_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
        "K",
        "num_m_tiles",
        "m_tile_div",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "W",
        "S",
        "X",
        "C",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
        "alpha",
    ),
    block=(512, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=222336,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    FC1_N8,
    FC1_N16,
    FC1_N32,
    FC1_N64,
    FC1_N128_CG2,
    FC2_N8,
    FC2_N16,
    FC2_N32,
    FC2_N64,
    FC2_N128_CG2,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "FC1_N8",
    "FC1_N16",
    "FC1_N32",
    "FC1_N64",
    "FC1_N128_CG2",
    "FC2_N8",
    "FC2_N16",
    "FC2_N32",
    "FC2_N64",
    "FC2_N128_CG2",
    "PROGRAMS",
    "aot_specs",
]
