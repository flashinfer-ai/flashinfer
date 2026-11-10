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
"""Generated programs of the MiniMax-H3 QKV quantize-and-pack kernel: one program per
output format (NVFP4, MXFP8) and destination partition count P (1, 2, 4, 8), each
shipped for sm_100a and sm_103a. The host module
``flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack`` computes the grid and launches
them.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

NVFP4_P1 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_1263f0007700395c8a00",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_1263f0007700395c8a00",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_1263f0007700395c8a00_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_1263f0007700395c8a00_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_global_scale",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_global_scale", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

NVFP4_P2 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_e889fd83c9b72f5d677f",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_e889fd83c9b72f5d677f",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_e889fd83c9b72f5d677f_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_e889fd83c9b72f5d677f_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_global_scale",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_global_scale", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

NVFP4_P4 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_a38424f3cf2fcfdf3c77",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_a38424f3cf2fcfdf3c77",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_a38424f3cf2fcfdf3c77_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_a38424f3cf2fcfdf3c77_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_global_scale",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_global_scale", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

NVFP4_P8 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_24c77d41c7fa9da4416f",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_24c77d41c7fa9da4416f",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_24c77d41c7fa9da4416f_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_24c77d41c7fa9da4416f_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_global_scale",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_global_scale", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

MXFP8_P1 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_1701b69f5a55c002e899",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_1701b69f5a55c002e899",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_1701b69f5a55c002e899_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_1701b69f5a55c002e899_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

MXFP8_P2 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_49e4c48b4f723c622503",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_49e4c48b4f723c622503",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_49e4c48b4f723c622503_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_49e4c48b4f723c622503_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

MXFP8_P4 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_984b36ddc10c63bf8bf0",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_984b36ddc10c63bf8bf0",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_984b36ddc10c63bf8bf0_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_984b36ddc10c63bf8bf0_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

MXFP8_P8 = cake_program.CakeProgram(
    name="cake_minimax_h3_qkv_quantize_pack_14c89d4596a3b1552b0b",
    operator_dir="cake_minimax_h3_qkv_quantize_pack",
    kernel="kernel_cake_minimax_h3_qkv_quantize_pack_14c89d4596a3b1552b0b",
    sources=(
        "cake_minimax_h3_qkv_quantize_pack_14c89d4596a3b1552b0b_kernel.cu",
        "cake_minimax_h3_qkv_quantize_pack_14c89d4596a3b1552b0b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out_q",
        "out_sf",
        "M",
        "token_stride",
        "head_stride",
        "ROWS_PER_DESTINATION",
        "SCALE_STRIDE",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("q", "k", "v", "out_q", "out_sf"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    NVFP4_P1,
    NVFP4_P2,
    NVFP4_P4,
    NVFP4_P8,
    MXFP8_P1,
    MXFP8_P2,
    MXFP8_P4,
    MXFP8_P8,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "NVFP4_P1",
    "NVFP4_P2",
    "NVFP4_P4",
    "NVFP4_P8",
    "MXFP8_P1",
    "MXFP8_P2",
    "MXFP8_P4",
    "MXFP8_P8",
    "PROGRAMS",
    "aot_specs",
]
