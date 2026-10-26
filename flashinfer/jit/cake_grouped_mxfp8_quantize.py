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
kernel (BF16, F16), its bulk-copy form (BULK_BF16, BULK_F16) and its row-tiled form
(TILED_BF16, TILED_F16), each shipped for sm_100a and sm_103a. The host module
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
    name="cake_grouped_mxfp8_quantize_7f9af4718a40428a293f",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_7f9af4718a40428a293f",
    sources=(
        "cake_grouped_mxfp8_quantize_7f9af4718a40428a293f_kernel.cu",
        "cake_grouped_mxfp8_quantize_7f9af4718a40428a293f_binding.cu",
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
    name="cake_grouped_mxfp8_quantize_8ad36e6d17827be6f603",
    operator_dir="cake_grouped_mxfp8_quantize",
    kernel="kernel_cake_grouped_mxfp8_quantize_8ad36e6d17827be6f603",
    sources=(
        "cake_grouped_mxfp8_quantize_7f9af4718a40428a293f_kernel.cu",
        "cake_grouped_mxfp8_quantize_7f9af4718a40428a293f_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        (
            "sm_100a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_8ad36e6d17827be6f603",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_8ad36e6d17827be6f603",
            ),
        ),
        (
            "sm_103a",
            (
                "--fmad=true",
                "-DCAKE_PROGRAM_VARIANT=1",
                "-DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_8ad36e6d17827be6f603",
                "-DCAKE_PROGRAM_SHIM=cake_binding_kernel_cake_grouped_mxfp8_quantize_8ad36e6d17827be6f603",
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
    "PROGRAMS",
    "aot_specs",
]
