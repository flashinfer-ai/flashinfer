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
    name="cake_tinygemm2_tcgen05_848355ea3e2e0907f41e",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_848355ea3e2e0907f41e",
    sources=(
        "cake_tinygemm2_tcgen05_848355ea3e2e0907f41e_kernel.cu",
        "cake_tinygemm2_tcgen05_848355ea3e2e0907f41e_binding.cu",
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
    name="cake_tinygemm2_tcgen05_8534911391626feea39d",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_8534911391626feea39d",
    sources=(
        "cake_tinygemm2_tcgen05_8534911391626feea39d_kernel.cu",
        "cake_tinygemm2_tcgen05_8534911391626feea39d_binding.cu",
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
    name="cake_tinygemm2_tcgen05_971916e9f4db76eae167",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_971916e9f4db76eae167",
    sources=(
        "cake_tinygemm2_tcgen05_971916e9f4db76eae167_kernel.cu",
        "cake_tinygemm2_tcgen05_971916e9f4db76eae167_binding.cu",
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
    name="cake_tinygemm2_tcgen05_19e715d5c3a5328f7cbb",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_19e715d5c3a5328f7cbb",
    sources=(
        "cake_tinygemm2_tcgen05_19e715d5c3a5328f7cbb_kernel.cu",
        "cake_tinygemm2_tcgen05_19e715d5c3a5328f7cbb_binding.cu",
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
    name="cake_tinygemm2_tcgen05_d73a7cf63897c144631c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_d73a7cf63897c144631c",
    sources=(
        "cake_tinygemm2_tcgen05_d73a7cf63897c144631c_kernel.cu",
        "cake_tinygemm2_tcgen05_d73a7cf63897c144631c_binding.cu",
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
    name="cake_tinygemm2_tcgen05_0a6aae0d7977d59edb54",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_0a6aae0d7977d59edb54",
    sources=(
        "cake_tinygemm2_tcgen05_0a6aae0d7977d59edb54_kernel.cu",
        "cake_tinygemm2_tcgen05_0a6aae0d7977d59edb54_binding.cu",
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
    name="cake_tinygemm2_tcgen05_f4ba279da7a1fdb36fc9",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_f4ba279da7a1fdb36fc9",
    sources=(
        "cake_tinygemm2_tcgen05_f4ba279da7a1fdb36fc9_kernel.cu",
        "cake_tinygemm2_tcgen05_f4ba279da7a1fdb36fc9_binding.cu",
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
    name="cake_tinygemm2_tcgen05_49458d12a9f23c78d75c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_49458d12a9f23c78d75c",
    sources=(
        "cake_tinygemm2_tcgen05_49458d12a9f23c78d75c_kernel.cu",
        "cake_tinygemm2_tcgen05_49458d12a9f23c78d75c_binding.cu",
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
    name="cake_tinygemm2_tcgen05_78ba265ce00c81b4045a",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_78ba265ce00c81b4045a",
    sources=(
        "cake_tinygemm2_tcgen05_78ba265ce00c81b4045a_kernel.cu",
        "cake_tinygemm2_tcgen05_78ba265ce00c81b4045a_binding.cu",
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
    name="cake_tinygemm2_tcgen05_3001e55f6e7bd77e4033",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_3001e55f6e7bd77e4033",
    sources=(
        "cake_tinygemm2_tcgen05_3001e55f6e7bd77e4033_kernel.cu",
        "cake_tinygemm2_tcgen05_3001e55f6e7bd77e4033_binding.cu",
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
    name="cake_tinygemm2_tcgen05_0741e44713065115edda",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_0741e44713065115edda",
    sources=(
        "cake_tinygemm2_tcgen05_0741e44713065115edda_kernel.cu",
        "cake_tinygemm2_tcgen05_0741e44713065115edda_binding.cu",
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
    name="cake_tinygemm2_tcgen05_3be3c056425e1f5354b9",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_3be3c056425e1f5354b9",
    sources=(
        "cake_tinygemm2_tcgen05_3be3c056425e1f5354b9_kernel.cu",
        "cake_tinygemm2_tcgen05_3be3c056425e1f5354b9_binding.cu",
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
    name="cake_tinygemm2_tcgen05_4b7ed5a2958f2a9a5c65",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_4b7ed5a2958f2a9a5c65",
    sources=(
        "cake_tinygemm2_tcgen05_4b7ed5a2958f2a9a5c65_kernel.cu",
        "cake_tinygemm2_tcgen05_4b7ed5a2958f2a9a5c65_binding.cu",
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
    name="cake_tinygemm2_tcgen05_4fa7a2697c2d9439578f",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_4fa7a2697c2d9439578f",
    sources=(
        "cake_tinygemm2_tcgen05_4fa7a2697c2d9439578f_kernel.cu",
        "cake_tinygemm2_tcgen05_4fa7a2697c2d9439578f_binding.cu",
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
    name="cake_tinygemm2_tcgen05_c829d06f681d5af12d74",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_c829d06f681d5af12d74",
    sources=(
        "cake_tinygemm2_tcgen05_c829d06f681d5af12d74_kernel.cu",
        "cake_tinygemm2_tcgen05_c829d06f681d5af12d74_binding.cu",
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
    name="cake_tinygemm2_tcgen05_41497cd705d43e7a3b17",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_41497cd705d43e7a3b17",
    sources=(
        "cake_tinygemm2_tcgen05_41497cd705d43e7a3b17_kernel.cu",
        "cake_tinygemm2_tcgen05_41497cd705d43e7a3b17_binding.cu",
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
    name="cake_tinygemm2_tcgen05_4c43f8c5910745235bfd",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_4c43f8c5910745235bfd",
    sources=(
        "cake_tinygemm2_tcgen05_4c43f8c5910745235bfd_kernel.cu",
        "cake_tinygemm2_tcgen05_4c43f8c5910745235bfd_binding.cu",
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
    name="cake_tinygemm2_tcgen05_2bfd7f2e3a7ce9fecd32",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_2bfd7f2e3a7ce9fecd32",
    sources=(
        "cake_tinygemm2_tcgen05_2bfd7f2e3a7ce9fecd32_kernel.cu",
        "cake_tinygemm2_tcgen05_2bfd7f2e3a7ce9fecd32_binding.cu",
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
    name="cake_tinygemm2_tcgen05_7c03601e6f8deef1ba09",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_7c03601e6f8deef1ba09",
    sources=(
        "cake_tinygemm2_tcgen05_7c03601e6f8deef1ba09_kernel.cu",
        "cake_tinygemm2_tcgen05_7c03601e6f8deef1ba09_binding.cu",
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
    name="cake_tinygemm2_tcgen05_ac4eb7e744422a5ff332",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_ac4eb7e744422a5ff332",
    sources=(
        "cake_tinygemm2_tcgen05_ac4eb7e744422a5ff332_kernel.cu",
        "cake_tinygemm2_tcgen05_ac4eb7e744422a5ff332_binding.cu",
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
    name="cake_tinygemm2_tcgen05_c5ced1a0b84f85fd702d",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_c5ced1a0b84f85fd702d",
    sources=(
        "cake_tinygemm2_tcgen05_c5ced1a0b84f85fd702d_kernel.cu",
        "cake_tinygemm2_tcgen05_c5ced1a0b84f85fd702d_binding.cu",
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
    name="cake_tinygemm2_tcgen05_f10ae9cb82d5680d40ad",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_f10ae9cb82d5680d40ad",
    sources=(
        "cake_tinygemm2_tcgen05_f10ae9cb82d5680d40ad_kernel.cu",
        "cake_tinygemm2_tcgen05_f10ae9cb82d5680d40ad_binding.cu",
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
    name="cake_tinygemm2_tcgen05_8fbddf8a6d390bda1d35",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_8fbddf8a6d390bda1d35",
    sources=(
        "cake_tinygemm2_tcgen05_8fbddf8a6d390bda1d35_kernel.cu",
        "cake_tinygemm2_tcgen05_8fbddf8a6d390bda1d35_binding.cu",
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
    name="cake_tinygemm2_tcgen05_6b4f9bfcf6e73dcd79d0",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_6b4f9bfcf6e73dcd79d0",
    sources=(
        "cake_tinygemm2_tcgen05_6b4f9bfcf6e73dcd79d0_kernel.cu",
        "cake_tinygemm2_tcgen05_6b4f9bfcf6e73dcd79d0_binding.cu",
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
