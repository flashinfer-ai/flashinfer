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
"""Generated programs of the Blackwell grouped MXFP8 quantization kernels: the row-2D
kernel (BF16, F16), its bulk-copy form (BULK_BF16, BULK_F16), its row-tiled form
(TILED_BF16, TILED_F16) and the row-tiled forms the host routes K = 7168
(TILED_WIDE_BF16, TILED_WIDE_F16: 224-thread CTAs), K = 4096 (TILED_K4096_BF16,
TILED_K4096_F16: 256-thread CTAs) and K = 2048 (TILED_K2048_BF16, TILED_K2048_F16) to,
each shipped for sm_100a and sm_103a. The host module
``flashinfer.quantization.fp8_quantization`` selects the program from the problem's
shape, computes the grid and launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

BF16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_d8aa920df5a6252450bd",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_d8aa920df5a6252450bd",
    sources=(
        "cake_grouped_mxfp8_quantize_d8aa920df5a6252450bd_kernel.cu",
        "cake_grouped_mxfp8_quantize_d8aa920df5a6252450bd_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

F16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_f84d4eab2905894ac4c2",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_f84d4eab2905894ac4c2",
    sources=(
        "cake_grouped_mxfp8_quantize_d8aa920df5a6252450bd_kernel.cu",
        "cake_grouped_mxfp8_quantize_d8aa920df5a6252450bd_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        (
            "sm_100a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_f84d4eab2905894ac4c2",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_f84d4eab2905894ac4c2",
            ),
        ),
        (
            "sm_103a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_f84d4eab2905894ac4c2",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_f84d4eab2905894ac4c2",
            ),
        ),
    ),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

BULK_BF16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b",
    sources=(
        "cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b_kernel.cu",
        "cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=8320,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

BULK_F16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_977b5e967e2415146d6d",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_977b5e967e2415146d6d",
    sources=(
        "cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b_kernel.cu",
        "cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        (
            "sm_100a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_977b5e967e2415146d6d",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_977b5e967e2415146d6d",
            ),
        ),
        (
            "sm_103a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_977b5e967e2415146d6d",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_977b5e967e2415146d6d",
            ),
        ),
    ),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=8320,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_BF16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_17f0a8f073d6b2b82f6e",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_17f0a8f073d6b2b82f6e",
    sources=(
        "cake_grouped_mxfp8_quantize_17f0a8f073d6b2b82f6e_kernel.cu",
        "cake_grouped_mxfp8_quantize_17f0a8f073d6b2b82f6e_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_F16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_b04f03c1a974efa820a8",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_b04f03c1a974efa820a8",
    sources=(
        "cake_grouped_mxfp8_quantize_17f0a8f073d6b2b82f6e_kernel.cu",
        "cake_grouped_mxfp8_quantize_17f0a8f073d6b2b82f6e_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        (
            "sm_100a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_b04f03c1a974efa820a8",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_b04f03c1a974efa820a8",
            ),
        ),
        (
            "sm_103a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_b04f03c1a974efa820a8",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_b04f03c1a974efa820a8",
            ),
        ),
    ),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_WIDE_BF16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_3167620f1b40056f0af4",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_3167620f1b40056f0af4",
    sources=(
        "cake_grouped_mxfp8_quantize_3167620f1b40056f0af4_kernel.cu",
        "cake_grouped_mxfp8_quantize_3167620f1b40056f0af4_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(224, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_WIDE_F16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_54f46d4ff24768c763c2",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_54f46d4ff24768c763c2",
    sources=(
        "cake_grouped_mxfp8_quantize_3167620f1b40056f0af4_kernel.cu",
        "cake_grouped_mxfp8_quantize_3167620f1b40056f0af4_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        (
            "sm_100a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_54f46d4ff24768c763c2",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_54f46d4ff24768c763c2",
            ),
        ),
        (
            "sm_103a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_54f46d4ff24768c763c2",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_54f46d4ff24768c763c2",
            ),
        ),
    ),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(224, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_K4096_BF16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_1e154097781ff977a460",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_1e154097781ff977a460",
    sources=(
        "cake_grouped_mxfp8_quantize_1e154097781ff977a460_kernel.cu",
        "cake_grouped_mxfp8_quantize_1e154097781ff977a460_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_K4096_F16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_ddf7f3acf57105ae7a5c",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_ddf7f3acf57105ae7a5c",
    sources=(
        "cake_grouped_mxfp8_quantize_1e154097781ff977a460_kernel.cu",
        "cake_grouped_mxfp8_quantize_1e154097781ff977a460_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        (
            "sm_100a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_ddf7f3acf57105ae7a5c",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_ddf7f3acf57105ae7a5c",
            ),
        ),
        (
            "sm_103a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_ddf7f3acf57105ae7a5c",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_ddf7f3acf57105ae7a5c",
            ),
        ),
    ),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_K2048_BF16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_edd5526315d0287d959b",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_edd5526315d0287d959b",
    sources=(
        "cake_grouped_mxfp8_quantize_edd5526315d0287d959b_kernel.cu",
        "cake_grouped_mxfp8_quantize_edd5526315d0287d959b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(("sm_100a", ("--fmad=true",)), ("sm_103a", ("--fmad=true",))),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

TILED_K2048_F16 = cake_program.CakeProgram(
    name="cake_grouped_mxfp8_quantize_88d27b762bed063e8bb1",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_88d27b762bed063e8bb1",
    sources=(
        "cake_grouped_mxfp8_quantize_edd5526315d0287d959b_kernel.cu",
        "cake_grouped_mxfp8_quantize_edd5526315d0287d959b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        (
            "sm_100a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_88d27b762bed063e8bb1",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_88d27b762bed063e8bb1",
            ),
        ),
        (
            "sm_103a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_88d27b762bed063e8bb1",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_88d27b762bed063e8bb1",
            ),
        ),
    ),
    args=(
        "x",
        "mask",
        "quantized",
        "scales",
        "M",
        "K",
        "PADDED_K",
        "PM_TILES",
        "PK_TILES",
        "BLOCKS_PER_ROW",
        "TOTAL_TASKS",
        "ROWS",
        "SUBROWS",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("x", "mask", "quantized", "scales"),
    block=(128, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=0,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    BF16,
    F16,
    BULK_BF16,
    BULK_F16,
    TILED_BF16,
    TILED_F16,
    TILED_WIDE_BF16,
    TILED_WIDE_F16,
    TILED_K4096_BF16,
    TILED_K4096_F16,
    TILED_K2048_BF16,
    TILED_K2048_F16,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "BF16",
    "F16",
    "BULK_BF16",
    "BULK_F16",
    "TILED_BF16",
    "TILED_F16",
    "TILED_WIDE_BF16",
    "TILED_WIDE_F16",
    "TILED_K4096_BF16",
    "TILED_K4096_F16",
    "TILED_K2048_BF16",
    "TILED_K2048_F16",
    "PROGRAMS",
    "aot_specs",
]
