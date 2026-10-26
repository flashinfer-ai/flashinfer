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
bias) on tcgen05, bit for bit the output of the mma.sync tinygemm2 kernel: five tiles
(PACKED_NARROW, PACKED_NARROW_U4, PACKED_NARROW_FULL_STAGES, PACKED, ACCUMULATORS), each
with and without the bias and with and without programmatic dependent launch, for
sm_100a, sm_103a and sm_107a. The host module ``flashinfer.gemm.routergemm`` selects the
program from the batch, the output features and the bias, computes the grid and launches
it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

PACKED_NARROW = cake_program.CakeProgram(
    name="cake_tinygemm2_tcgen05_2818cbcca783a2c6b7cc",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_2818cbcca783a2c6b7cc",
    sources=(
        "cake_tinygemm2_tcgen05_2818cbcca783a2c6b7cc_kernel.cu",
        "cake_tinygemm2_tcgen05_2818cbcca783a2c6b7cc_binding.cu",
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
    name="cake_tinygemm2_tcgen05_3f1ec85fe6b0e7af7c5b",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_3f1ec85fe6b0e7af7c5b",
    sources=(
        "cake_tinygemm2_tcgen05_3f1ec85fe6b0e7af7c5b_kernel.cu",
        "cake_tinygemm2_tcgen05_3f1ec85fe6b0e7af7c5b_binding.cu",
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
    name="cake_tinygemm2_tcgen05_60161c30f44e9680e7f0",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_60161c30f44e9680e7f0",
    sources=(
        "cake_tinygemm2_tcgen05_60161c30f44e9680e7f0_kernel.cu",
        "cake_tinygemm2_tcgen05_60161c30f44e9680e7f0_binding.cu",
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
    name="cake_tinygemm2_tcgen05_118f93b00270df39d8b2",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_118f93b00270df39d8b2",
    sources=(
        "cake_tinygemm2_tcgen05_118f93b00270df39d8b2_kernel.cu",
        "cake_tinygemm2_tcgen05_118f93b00270df39d8b2_binding.cu",
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
    name="cake_tinygemm2_tcgen05_1b17cc2342cf2c9429b7",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_1b17cc2342cf2c9429b7",
    sources=(
        "cake_tinygemm2_tcgen05_1b17cc2342cf2c9429b7_kernel.cu",
        "cake_tinygemm2_tcgen05_1b17cc2342cf2c9429b7_binding.cu",
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
    name="cake_tinygemm2_tcgen05_25861eb93d1c8b3a205e",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_25861eb93d1c8b3a205e",
    sources=(
        "cake_tinygemm2_tcgen05_25861eb93d1c8b3a205e_kernel.cu",
        "cake_tinygemm2_tcgen05_25861eb93d1c8b3a205e_binding.cu",
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
    name="cake_tinygemm2_tcgen05_33eb1f2cf2a851e05951",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_33eb1f2cf2a851e05951",
    sources=(
        "cake_tinygemm2_tcgen05_33eb1f2cf2a851e05951_kernel.cu",
        "cake_tinygemm2_tcgen05_33eb1f2cf2a851e05951_binding.cu",
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
    name="cake_tinygemm2_tcgen05_437c49524cf22d67e521",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_437c49524cf22d67e521",
    sources=(
        "cake_tinygemm2_tcgen05_437c49524cf22d67e521_kernel.cu",
        "cake_tinygemm2_tcgen05_437c49524cf22d67e521_binding.cu",
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
    name="cake_tinygemm2_tcgen05_814e78944635609d9149",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_814e78944635609d9149",
    sources=(
        "cake_tinygemm2_tcgen05_814e78944635609d9149_kernel.cu",
        "cake_tinygemm2_tcgen05_814e78944635609d9149_binding.cu",
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
    name="cake_tinygemm2_tcgen05_6d7590d789513d603d19",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_6d7590d789513d603d19",
    sources=(
        "cake_tinygemm2_tcgen05_6d7590d789513d603d19_kernel.cu",
        "cake_tinygemm2_tcgen05_6d7590d789513d603d19_binding.cu",
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
    name="cake_tinygemm2_tcgen05_f9b4c759af9cbb36939c",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_f9b4c759af9cbb36939c",
    sources=(
        "cake_tinygemm2_tcgen05_f9b4c759af9cbb36939c_kernel.cu",
        "cake_tinygemm2_tcgen05_f9b4c759af9cbb36939c_binding.cu",
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
    name="cake_tinygemm2_tcgen05_90674936f7677cb73219",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_90674936f7677cb73219",
    sources=(
        "cake_tinygemm2_tcgen05_90674936f7677cb73219_kernel.cu",
        "cake_tinygemm2_tcgen05_90674936f7677cb73219_binding.cu",
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
    name="cake_tinygemm2_tcgen05_9582f330f1fa6b13f849",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_9582f330f1fa6b13f849",
    sources=(
        "cake_tinygemm2_tcgen05_9582f330f1fa6b13f849_kernel.cu",
        "cake_tinygemm2_tcgen05_9582f330f1fa6b13f849_binding.cu",
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
    name="cake_tinygemm2_tcgen05_e610d4eb6c9b0fecb5d9",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_e610d4eb6c9b0fecb5d9",
    sources=(
        "cake_tinygemm2_tcgen05_e610d4eb6c9b0fecb5d9_kernel.cu",
        "cake_tinygemm2_tcgen05_e610d4eb6c9b0fecb5d9_binding.cu",
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
    name="cake_tinygemm2_tcgen05_e1c1df1ebd3a99a37f13",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_e1c1df1ebd3a99a37f13",
    sources=(
        "cake_tinygemm2_tcgen05_e1c1df1ebd3a99a37f13_kernel.cu",
        "cake_tinygemm2_tcgen05_e1c1df1ebd3a99a37f13_binding.cu",
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
    name="cake_tinygemm2_tcgen05_ee4d1de9f9cb3b9e07d7",
    operator_dir="cake_tinygemm2_tcgen05",
    kernel="kernel_cake_tinygemm2_tcgen05_ee4d1de9f9cb3b9e07d7",
    sources=(
        "cake_tinygemm2_tcgen05_ee4d1de9f9cb3b9e07d7_kernel.cu",
        "cake_tinygemm2_tcgen05_ee4d1de9f9cb3b9e07d7_binding.cu",
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
    PACKED_NARROW_U4,
    PACKED_NARROW_U4_NOBIAS,
    PACKED_NARROW_U4_PDL,
    PACKED_NARROW_U4_NOBIAS_PDL,
    PACKED_NARROW_FULL_STAGES,
    PACKED_NARROW_FULL_STAGES_NOBIAS,
    PACKED_NARROW_FULL_STAGES_PDL,
    PACKED_NARROW_FULL_STAGES_NOBIAS_PDL,
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
