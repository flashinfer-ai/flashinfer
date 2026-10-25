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
"""Generated programs of the BF16 small-batch matrix product (out = input @ weight.T +
bias) on tcgen05, bit for bit the output of the mma.sync tinygemm2 kernel: three tiles
(PACKED_NARROW, PACKED, ACCUMULATORS), each with and without the bias and with and
without programmatic dependent launch, for sm_100a, sm_103a and sm_107a. The host module
``flashinfer.gemm.routergemm`` selects the program from the batch, the output features
and the bias, computes the grid and launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

PACKED_NARROW = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_f087c499b2b4bae6d108",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_f087c499b2b4bae6d108",
    sources=(
        "cake_tinygemm2_tcgen05_f087c499b2b4bae6d108_kernel.cu",
        "cake_tinygemm2_tcgen05_f087c499b2b4bae6d108_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=222464,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_NARROW_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_b6c2400b543776c4c013",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_b6c2400b543776c4c013",
    sources=(
        "cake_tinygemm2_tcgen05_b6c2400b543776c4c013_kernel.cu",
        "cake_tinygemm2_tcgen05_b6c2400b543776c4c013_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=222464,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_NARROW_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_79744b1433f1d2c89435",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_79744b1433f1d2c89435",
    sources=(
        "cake_tinygemm2_tcgen05_79744b1433f1d2c89435_kernel.cu",
        "cake_tinygemm2_tcgen05_79744b1433f1d2c89435_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=222464,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_NARROW_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_16287f52e533f957ed85",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_16287f52e533f957ed85",
    sources=(
        "cake_tinygemm2_tcgen05_16287f52e533f957ed85_kernel.cu",
        "cake_tinygemm2_tcgen05_16287f52e533f957ed85_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=222464,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_bf37103a836956cde306",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_bf37103a836956cde306",
    sources=(
        "cake_tinygemm2_tcgen05_bf37103a836956cde306_kernel.cu",
        "cake_tinygemm2_tcgen05_bf37103a836956cde306_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_78e755ccf10655535ea0",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_78e755ccf10655535ea0",
    sources=(
        "cake_tinygemm2_tcgen05_78e755ccf10655535ea0_kernel.cu",
        "cake_tinygemm2_tcgen05_78e755ccf10655535ea0_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_96c37e6e2085293c8af5",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_96c37e6e2085293c8af5",
    sources=(
        "cake_tinygemm2_tcgen05_96c37e6e2085293c8af5_kernel.cu",
        "cake_tinygemm2_tcgen05_96c37e6e2085293c8af5_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_900aff14d86246275df8",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_900aff14d86246275df8",
    sources=(
        "cake_tinygemm2_tcgen05_900aff14d86246275df8_kernel.cu",
        "cake_tinygemm2_tcgen05_900aff14d86246275df8_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_1e984369d617712da373",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_1e984369d617712da373",
    sources=(
        "cake_tinygemm2_tcgen05_1e984369d617712da373_kernel.cu",
        "cake_tinygemm2_tcgen05_1e984369d617712da373_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=226304,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_1eee3f530969845ed3c7",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_1eee3f530969845ed3c7",
    sources=(
        "cake_tinygemm2_tcgen05_1eee3f530969845ed3c7_kernel.cu",
        "cake_tinygemm2_tcgen05_1eee3f530969845ed3c7_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=226304,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_f1168c30e20d13d518cf",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_f1168c30e20d13d518cf",
    sources=(
        "cake_tinygemm2_tcgen05_f1168c30e20d13d518cf_kernel.cu",
        "cake_tinygemm2_tcgen05_f1168c30e20d13d518cf_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=226304,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_46f583fba87d7a0249be",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_46f583fba87d7a0249be",
    sources=(
        "cake_tinygemm2_tcgen05_46f583fba87d7a0249be_kernel.cu",
        "cake_tinygemm2_tcgen05_46f583fba87d7a0249be_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=226304,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    PACKED_NARROW,
    PACKED_NARROW_NOBIAS,
    PACKED_NARROW_PDL,
    PACKED_NARROW_NOBIAS_PDL,
    PACKED,
    PACKED_NOBIAS,
    PACKED_PDL,
    PACKED_NOBIAS_PDL,
    ACCUMULATORS,
    ACCUMULATORS_NOBIAS,
    ACCUMULATORS_PDL,
    ACCUMULATORS_NOBIAS_PDL,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "PACKED_NARROW",
    "PACKED_NARROW_NOBIAS",
    "PACKED_NARROW_PDL",
    "PACKED_NARROW_NOBIAS_PDL",
    "PACKED",
    "PACKED_NOBIAS",
    "PACKED_PDL",
    "PACKED_NOBIAS_PDL",
    "ACCUMULATORS",
    "ACCUMULATORS_NOBIAS",
    "ACCUMULATORS_PDL",
    "ACCUMULATORS_NOBIAS_PDL",
    "PROGRAMS",
    "aot_specs",
]
