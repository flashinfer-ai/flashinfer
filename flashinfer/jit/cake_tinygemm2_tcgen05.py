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
bias) on tcgen05, bit for bit the output of the mma.sync tinygemm2 kernel: seven tiles
(PACKED_NARROW, PACKED_NARROW_U4, PACKED_NARROW_FULL_STAGES, PACKED_M64_N16,
PACKED_M64_N4, PACKED, ACCUMULATORS), each with and without the bias and with and
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
    name="cake_tinygemm2_tcgen05_dc3b88de3a4d4944043c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_dc3b88de3a4d4944043c",
    sources=(
        "cake_tinygemm2_tcgen05_dc3b88de3a4d4944043c_kernel.cu",
        "cake_tinygemm2_tcgen05_dc3b88de3a4d4944043c_binding.cu",
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
    name="cake_tinygemm2_tcgen05_947366f6f06ce7cf7cbb",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_947366f6f06ce7cf7cbb",
    sources=(
        "cake_tinygemm2_tcgen05_947366f6f06ce7cf7cbb_kernel.cu",
        "cake_tinygemm2_tcgen05_947366f6f06ce7cf7cbb_binding.cu",
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
    name="cake_tinygemm2_tcgen05_59b640e3d44c6e8ea0cc",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_59b640e3d44c6e8ea0cc",
    sources=(
        "cake_tinygemm2_tcgen05_59b640e3d44c6e8ea0cc_kernel.cu",
        "cake_tinygemm2_tcgen05_59b640e3d44c6e8ea0cc_binding.cu",
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
    name="cake_tinygemm2_tcgen05_5a300f71cc68239a7cee",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_5a300f71cc68239a7cee",
    sources=(
        "cake_tinygemm2_tcgen05_5a300f71cc68239a7cee_kernel.cu",
        "cake_tinygemm2_tcgen05_5a300f71cc68239a7cee_binding.cu",
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
    name="cake_tinygemm2_tcgen05_228e796c414a53792808",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_228e796c414a53792808",
    sources=(
        "cake_tinygemm2_tcgen05_228e796c414a53792808_kernel.cu",
        "cake_tinygemm2_tcgen05_228e796c414a53792808_binding.cu",
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
    name="cake_tinygemm2_tcgen05_76a65f2c2dad73247831",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_76a65f2c2dad73247831",
    sources=(
        "cake_tinygemm2_tcgen05_76a65f2c2dad73247831_kernel.cu",
        "cake_tinygemm2_tcgen05_76a65f2c2dad73247831_binding.cu",
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
    name="cake_tinygemm2_tcgen05_6c31d228d01227e52347",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_6c31d228d01227e52347",
    sources=(
        "cake_tinygemm2_tcgen05_6c31d228d01227e52347_kernel.cu",
        "cake_tinygemm2_tcgen05_6c31d228d01227e52347_binding.cu",
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
    name="cake_tinygemm2_tcgen05_924f01e1f2fc585b09af",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_924f01e1f2fc585b09af",
    sources=(
        "cake_tinygemm2_tcgen05_924f01e1f2fc585b09af_kernel.cu",
        "cake_tinygemm2_tcgen05_924f01e1f2fc585b09af_binding.cu",
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
    name="cake_tinygemm2_tcgen05_3beab9eafb78a7970f03",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_3beab9eafb78a7970f03",
    sources=(
        "cake_tinygemm2_tcgen05_3beab9eafb78a7970f03_kernel.cu",
        "cake_tinygemm2_tcgen05_3beab9eafb78a7970f03_binding.cu",
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
    name="cake_tinygemm2_tcgen05_10a3b8764b02e6fb1962",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_10a3b8764b02e6fb1962",
    sources=(
        "cake_tinygemm2_tcgen05_10a3b8764b02e6fb1962_kernel.cu",
        "cake_tinygemm2_tcgen05_10a3b8764b02e6fb1962_binding.cu",
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
    name="cake_tinygemm2_tcgen05_efcd74722f359ecc50c3",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_efcd74722f359ecc50c3",
    sources=(
        "cake_tinygemm2_tcgen05_efcd74722f359ecc50c3_kernel.cu",
        "cake_tinygemm2_tcgen05_efcd74722f359ecc50c3_binding.cu",
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
    name="cake_tinygemm2_tcgen05_f064bf7bf0666643e096",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_f064bf7bf0666643e096",
    sources=(
        "cake_tinygemm2_tcgen05_f064bf7bf0666643e096_kernel.cu",
        "cake_tinygemm2_tcgen05_f064bf7bf0666643e096_binding.cu",
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
    name="cake_tinygemm2_tcgen05_9d4a8935c43139259468",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_9d4a8935c43139259468",
    sources=(
        "cake_tinygemm2_tcgen05_9d4a8935c43139259468_kernel.cu",
        "cake_tinygemm2_tcgen05_9d4a8935c43139259468_binding.cu",
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
    name="cake_tinygemm2_tcgen05_a1dcf6911cfe7767194c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_a1dcf6911cfe7767194c",
    sources=(
        "cake_tinygemm2_tcgen05_a1dcf6911cfe7767194c_kernel.cu",
        "cake_tinygemm2_tcgen05_a1dcf6911cfe7767194c_binding.cu",
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
    name="cake_tinygemm2_tcgen05_542899150d2f0d97f775",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_542899150d2f0d97f775",
    sources=(
        "cake_tinygemm2_tcgen05_542899150d2f0d97f775_kernel.cu",
        "cake_tinygemm2_tcgen05_542899150d2f0d97f775_binding.cu",
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
    name="cake_tinygemm2_tcgen05_289477047cb049fb25ce",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_289477047cb049fb25ce",
    sources=(
        "cake_tinygemm2_tcgen05_289477047cb049fb25ce_kernel.cu",
        "cake_tinygemm2_tcgen05_289477047cb049fb25ce_binding.cu",
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
    name="cake_tinygemm2_tcgen05_834c2faf585d35f92fa4",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_834c2faf585d35f92fa4",
    sources=(
        "cake_tinygemm2_tcgen05_834c2faf585d35f92fa4_kernel.cu",
        "cake_tinygemm2_tcgen05_834c2faf585d35f92fa4_binding.cu",
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
    name="cake_tinygemm2_tcgen05_50f55250900fd8b82aed",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_50f55250900fd8b82aed",
    sources=(
        "cake_tinygemm2_tcgen05_50f55250900fd8b82aed_kernel.cu",
        "cake_tinygemm2_tcgen05_50f55250900fd8b82aed_binding.cu",
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
    name="cake_tinygemm2_tcgen05_0dc3ee5b84452bc4386c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_0dc3ee5b84452bc4386c",
    sources=(
        "cake_tinygemm2_tcgen05_0dc3ee5b84452bc4386c_kernel.cu",
        "cake_tinygemm2_tcgen05_0dc3ee5b84452bc4386c_binding.cu",
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
    name="cake_tinygemm2_tcgen05_19f56f82fac8483fc5c4",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_19f56f82fac8483fc5c4",
    sources=(
        "cake_tinygemm2_tcgen05_19f56f82fac8483fc5c4_kernel.cu",
        "cake_tinygemm2_tcgen05_19f56f82fac8483fc5c4_binding.cu",
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
    name="cake_tinygemm2_tcgen05_2938e05eec2fdb2cc05a",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_2938e05eec2fdb2cc05a",
    sources=(
        "cake_tinygemm2_tcgen05_2938e05eec2fdb2cc05a_kernel.cu",
        "cake_tinygemm2_tcgen05_2938e05eec2fdb2cc05a_binding.cu",
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
    name="cake_tinygemm2_tcgen05_18c3e2e2a10460cf9859",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_18c3e2e2a10460cf9859",
    sources=(
        "cake_tinygemm2_tcgen05_18c3e2e2a10460cf9859_kernel.cu",
        "cake_tinygemm2_tcgen05_18c3e2e2a10460cf9859_binding.cu",
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
    name="cake_tinygemm2_tcgen05_16bd6fe3969667c1f431",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_16bd6fe3969667c1f431",
    sources=(
        "cake_tinygemm2_tcgen05_16bd6fe3969667c1f431_kernel.cu",
        "cake_tinygemm2_tcgen05_16bd6fe3969667c1f431_binding.cu",
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
    name="cake_tinygemm2_tcgen05_7a4cb08583ed083cfd3d",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_7a4cb08583ed083cfd3d",
    sources=(
        "cake_tinygemm2_tcgen05_7a4cb08583ed083cfd3d_kernel.cu",
        "cake_tinygemm2_tcgen05_7a4cb08583ed083cfd3d_binding.cu",
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
    "ACCUMULATORS",
    "ACCUMULATORS_NOBIAS",
    "ACCUMULATORS_PDL",
    "ACCUMULATORS_NOBIAS_PDL",
    "PROGRAMS",
    "aot_specs",
]
