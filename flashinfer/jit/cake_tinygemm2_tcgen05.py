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
bias) on tcgen05, bit for bit the output of the mma.sync tinygemm2 kernel: nine tiles
(PACKED_NARROW, PACKED_NARROW_U4, PACKED_NARROW_FULL_STAGES, PACKED_M64_N16,
PACKED_M64_N4, PACKED, PACKED_BOX8, PACKED_M64_N8, ACCUMULATORS), each with and without
the bias and with and without programmatic dependent launch, for sm_100a, sm_103a and
sm_107a (PACKED_M64_N8 for sm_107a only). The host module ``flashinfer.gemm.routergemm``
selects the program from the batch, the output features and the bias, computes the grid
and launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

PACKED_NARROW = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_8b32b037a0fe50155e9a",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_8b32b037a0fe50155e9a",
    sources=(
        "cake_tinygemm2_tcgen05_8b32b037a0fe50155e9a_kernel.cu",
        "cake_tinygemm2_tcgen05_8b32b037a0fe50155e9a_binding.cu",
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
    name="cake_tinygemm2_tcgen05_9b683dd9478a745d4c55",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_9b683dd9478a745d4c55",
    sources=(
        "cake_tinygemm2_tcgen05_9b683dd9478a745d4c55_kernel.cu",
        "cake_tinygemm2_tcgen05_9b683dd9478a745d4c55_binding.cu",
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
    name="cake_tinygemm2_tcgen05_d5ad477acf131a5b9fd7",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_d5ad477acf131a5b9fd7",
    sources=(
        "cake_tinygemm2_tcgen05_d5ad477acf131a5b9fd7_kernel.cu",
        "cake_tinygemm2_tcgen05_d5ad477acf131a5b9fd7_binding.cu",
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
    name="cake_tinygemm2_tcgen05_e440a653fc1301e5caca",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_e440a653fc1301e5caca",
    sources=(
        "cake_tinygemm2_tcgen05_e440a653fc1301e5caca_kernel.cu",
        "cake_tinygemm2_tcgen05_e440a653fc1301e5caca_binding.cu",
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

PACKED_NARROW_U4 = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_3030ca5eed6c47249d52",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_3030ca5eed6c47249d52",
    sources=(
        "cake_tinygemm2_tcgen05_3030ca5eed6c47249d52_kernel.cu",
        "cake_tinygemm2_tcgen05_3030ca5eed6c47249d52_binding.cu",
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

PACKED_NARROW_U4_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_c9066e280e2e908f9dab",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_c9066e280e2e908f9dab",
    sources=(
        "cake_tinygemm2_tcgen05_c9066e280e2e908f9dab_kernel.cu",
        "cake_tinygemm2_tcgen05_c9066e280e2e908f9dab_binding.cu",
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

PACKED_NARROW_U4_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_2ba8985b0a6ebf69a33d",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_2ba8985b0a6ebf69a33d",
    sources=(
        "cake_tinygemm2_tcgen05_2ba8985b0a6ebf69a33d_kernel.cu",
        "cake_tinygemm2_tcgen05_2ba8985b0a6ebf69a33d_binding.cu",
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

PACKED_NARROW_U4_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_165074e6003ca45c0290",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_165074e6003ca45c0290",
    sources=(
        "cake_tinygemm2_tcgen05_165074e6003ca45c0290_kernel.cu",
        "cake_tinygemm2_tcgen05_165074e6003ca45c0290_binding.cu",
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

PACKED_NARROW_FULL_STAGES = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_af0096e77ec6f05fd1bf",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_af0096e77ec6f05fd1bf",
    sources=(
        "cake_tinygemm2_tcgen05_af0096e77ec6f05fd1bf_kernel.cu",
        "cake_tinygemm2_tcgen05_af0096e77ec6f05fd1bf_binding.cu",
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

PACKED_NARROW_FULL_STAGES_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_f33df96b451d7caa6ee5",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_f33df96b451d7caa6ee5",
    sources=(
        "cake_tinygemm2_tcgen05_f33df96b451d7caa6ee5_kernel.cu",
        "cake_tinygemm2_tcgen05_f33df96b451d7caa6ee5_binding.cu",
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

PACKED_NARROW_FULL_STAGES_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_00dda32c0016befaccac",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_00dda32c0016befaccac",
    sources=(
        "cake_tinygemm2_tcgen05_00dda32c0016befaccac_kernel.cu",
        "cake_tinygemm2_tcgen05_00dda32c0016befaccac_binding.cu",
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

PACKED_NARROW_FULL_STAGES_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_0dd0aa0a9bc6fe52beda",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_0dd0aa0a9bc6fe52beda",
    sources=(
        "cake_tinygemm2_tcgen05_0dd0aa0a9bc6fe52beda_kernel.cu",
        "cake_tinygemm2_tcgen05_0dd0aa0a9bc6fe52beda_binding.cu",
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

PACKED_M64_N16 = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_97a01814dc569e35cbd8",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_97a01814dc569e35cbd8",
    sources=(
        "cake_tinygemm2_tcgen05_97a01814dc569e35cbd8_kernel.cu",
        "cake_tinygemm2_tcgen05_97a01814dc569e35cbd8_binding.cu",
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
    dynamic_smem_bytes=201728,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N16_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_811c4b5a0cb7b8b2de97",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_811c4b5a0cb7b8b2de97",
    sources=(
        "cake_tinygemm2_tcgen05_811c4b5a0cb7b8b2de97_kernel.cu",
        "cake_tinygemm2_tcgen05_811c4b5a0cb7b8b2de97_binding.cu",
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
    dynamic_smem_bytes=201728,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N16_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_d2fa98684abba65a9b82",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_d2fa98684abba65a9b82",
    sources=(
        "cake_tinygemm2_tcgen05_d2fa98684abba65a9b82_kernel.cu",
        "cake_tinygemm2_tcgen05_d2fa98684abba65a9b82_binding.cu",
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
    dynamic_smem_bytes=201728,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N16_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_1cf48fce11475e27a2b1",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_1cf48fce11475e27a2b1",
    sources=(
        "cake_tinygemm2_tcgen05_1cf48fce11475e27a2b1_kernel.cu",
        "cake_tinygemm2_tcgen05_1cf48fce11475e27a2b1_binding.cu",
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
    dynamic_smem_bytes=201728,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N4 = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_34830dccba0f4943f78c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_34830dccba0f4943f78c",
    sources=(
        "cake_tinygemm2_tcgen05_34830dccba0f4943f78c_kernel.cu",
        "cake_tinygemm2_tcgen05_34830dccba0f4943f78c_binding.cu",
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
    dynamic_smem_bytes=124928,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N4_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_c111ebb3eaaad3d4b63e",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_c111ebb3eaaad3d4b63e",
    sources=(
        "cake_tinygemm2_tcgen05_c111ebb3eaaad3d4b63e_kernel.cu",
        "cake_tinygemm2_tcgen05_c111ebb3eaaad3d4b63e_binding.cu",
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
    dynamic_smem_bytes=124928,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N4_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_0723e7061398f785193f",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_0723e7061398f785193f",
    sources=(
        "cake_tinygemm2_tcgen05_0723e7061398f785193f_kernel.cu",
        "cake_tinygemm2_tcgen05_0723e7061398f785193f_binding.cu",
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
    dynamic_smem_bytes=124928,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N4_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_6638722cc0c3ce41977b",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_6638722cc0c3ce41977b",
    sources=(
        "cake_tinygemm2_tcgen05_6638722cc0c3ce41977b_kernel.cu",
        "cake_tinygemm2_tcgen05_6638722cc0c3ce41977b_binding.cu",
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
    dynamic_smem_bytes=124928,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_aa21cffe57e4ba888fb3",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_aa21cffe57e4ba888fb3",
    sources=(
        "cake_tinygemm2_tcgen05_aa21cffe57e4ba888fb3_kernel.cu",
        "cake_tinygemm2_tcgen05_aa21cffe57e4ba888fb3_binding.cu",
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
    name="cake_tinygemm2_tcgen05_b105a13823bfdaf262b7",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_b105a13823bfdaf262b7",
    sources=(
        "cake_tinygemm2_tcgen05_b105a13823bfdaf262b7_kernel.cu",
        "cake_tinygemm2_tcgen05_b105a13823bfdaf262b7_binding.cu",
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
    name="cake_tinygemm2_tcgen05_d29459b4440f306156e7",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_d29459b4440f306156e7",
    sources=(
        "cake_tinygemm2_tcgen05_d29459b4440f306156e7_kernel.cu",
        "cake_tinygemm2_tcgen05_d29459b4440f306156e7_binding.cu",
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
    name="cake_tinygemm2_tcgen05_39c26851b9fb3a89e76f",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_39c26851b9fb3a89e76f",
    sources=(
        "cake_tinygemm2_tcgen05_39c26851b9fb3a89e76f_kernel.cu",
        "cake_tinygemm2_tcgen05_39c26851b9fb3a89e76f_binding.cu",
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

PACKED_BOX8 = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_4f4ee208438bbe8532fc",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_4f4ee208438bbe8532fc",
    sources=(
        "cake_tinygemm2_tcgen05_4f4ee208438bbe8532fc_kernel.cu",
        "cake_tinygemm2_tcgen05_4f4ee208438bbe8532fc_binding.cu",
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

PACKED_BOX8_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_eb55bfbe52f5ff00b048",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_eb55bfbe52f5ff00b048",
    sources=(
        "cake_tinygemm2_tcgen05_eb55bfbe52f5ff00b048_kernel.cu",
        "cake_tinygemm2_tcgen05_eb55bfbe52f5ff00b048_binding.cu",
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

PACKED_BOX8_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_0bb51b8202cba55024f1",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_0bb51b8202cba55024f1",
    sources=(
        "cake_tinygemm2_tcgen05_0bb51b8202cba55024f1_kernel.cu",
        "cake_tinygemm2_tcgen05_0bb51b8202cba55024f1_binding.cu",
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

PACKED_BOX8_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_24eef93ea21fbbcb024c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_24eef93ea21fbbcb024c",
    sources=(
        "cake_tinygemm2_tcgen05_24eef93ea21fbbcb024c_kernel.cu",
        "cake_tinygemm2_tcgen05_24eef93ea21fbbcb024c_binding.cu",
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

PACKED_M64_N8 = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_3b3d21550960b5b6b8b2",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_3b3d21550960b5b6b8b2",
    sources=(
        "cake_tinygemm2_tcgen05_3b3d21550960b5b6b8b2_kernel.cu",
        "cake_tinygemm2_tcgen05_3b3d21550960b5b6b8b2_binding.cu",
    ),
    targets=("sm_107a",),
    compile_options=(("sm_107a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N8_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_53e595e7d938f773c823",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_53e595e7d938f773c823",
    sources=(
        "cake_tinygemm2_tcgen05_53e595e7d938f773c823_kernel.cu",
        "cake_tinygemm2_tcgen05_53e595e7d938f773c823_binding.cu",
    ),
    targets=("sm_107a",),
    compile_options=(("sm_107a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N8_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_8e89c0a937878adae26b",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_8e89c0a937878adae26b",
    sources=(
        "cake_tinygemm2_tcgen05_8e89c0a937878adae26b_kernel.cu",
        "cake_tinygemm2_tcgen05_8e89c0a937878adae26b_binding.cu",
    ),
    targets=("sm_107a",),
    compile_options=(("sm_107a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_M64_N8_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_e7244c463ed684737af3",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_e7244c463ed684737af3",
    sources=(
        "cake_tinygemm2_tcgen05_e7244c463ed684737af3_kernel.cu",
        "cake_tinygemm2_tcgen05_e7244c463ed684737af3_binding.cu",
    ),
    targets=("sm_107a",),
    compile_options=(("sm_107a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
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
    dynamic_smem_bytes=222208,
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
    dynamic_smem_bytes=222208,
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
    dynamic_smem_bytes=222208,
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
    dynamic_smem_bytes=222208,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    PACKED_NARROW,
    PACKED_NARROW_NOBIAS,
    PACKED_NARROW_PDL,
    PACKED_NARROW_NOBIAS_PDL,
    PACKED_NARROW_U4,
    PACKED_NARROW_U4_NOBIAS,
    PACKED_NARROW_U4_PDL,
    PACKED_NARROW_U4_NOBIAS_PDL,
    PACKED_NARROW_FULL_STAGES,
    PACKED_NARROW_FULL_STAGES_NOBIAS,
    PACKED_NARROW_FULL_STAGES_PDL,
    PACKED_NARROW_FULL_STAGES_NOBIAS_PDL,
    PACKED_M64_N16,
    PACKED_M64_N16_NOBIAS,
    PACKED_M64_N16_PDL,
    PACKED_M64_N16_NOBIAS_PDL,
    PACKED_M64_N4,
    PACKED_M64_N4_NOBIAS,
    PACKED_M64_N4_PDL,
    PACKED_M64_N4_NOBIAS_PDL,
    PACKED,
    PACKED_NOBIAS,
    PACKED_PDL,
    PACKED_NOBIAS_PDL,
    PACKED_BOX8,
    PACKED_BOX8_NOBIAS,
    PACKED_BOX8_PDL,
    PACKED_BOX8_NOBIAS_PDL,
    PACKED_M64_N8,
    PACKED_M64_N8_NOBIAS,
    PACKED_M64_N8_PDL,
    PACKED_M64_N8_NOBIAS_PDL,
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
    "PACKED_NARROW_U4",
    "PACKED_NARROW_U4_NOBIAS",
    "PACKED_NARROW_U4_PDL",
    "PACKED_NARROW_U4_NOBIAS_PDL",
    "PACKED_NARROW_FULL_STAGES",
    "PACKED_NARROW_FULL_STAGES_NOBIAS",
    "PACKED_NARROW_FULL_STAGES_PDL",
    "PACKED_NARROW_FULL_STAGES_NOBIAS_PDL",
    "PACKED_M64_N16",
    "PACKED_M64_N16_NOBIAS",
    "PACKED_M64_N16_PDL",
    "PACKED_M64_N16_NOBIAS_PDL",
    "PACKED_M64_N4",
    "PACKED_M64_N4_NOBIAS",
    "PACKED_M64_N4_PDL",
    "PACKED_M64_N4_NOBIAS_PDL",
    "PACKED",
    "PACKED_NOBIAS",
    "PACKED_PDL",
    "PACKED_NOBIAS_PDL",
    "PACKED_BOX8",
    "PACKED_BOX8_NOBIAS",
    "PACKED_BOX8_PDL",
    "PACKED_BOX8_NOBIAS_PDL",
    "PACKED_M64_N8",
    "PACKED_M64_N8_NOBIAS",
    "PACKED_M64_N8_PDL",
    "PACKED_M64_N8_NOBIAS_PDL",
    "ACCUMULATORS",
    "ACCUMULATORS_NOBIAS",
    "ACCUMULATORS_PDL",
    "ACCUMULATORS_NOBIAS_PDL",
    "PROGRAMS",
    "aot_specs",
]
