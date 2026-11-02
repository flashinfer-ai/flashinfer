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
bias) on sm_90a wgmma, bit for bit the output of the mma.sync tinygemm2 kernel
(csrc/tinygemm2.cu): the reference sums K in four chains of K16 steps (256-element chunk
j to chain j % 4) and adds the chains, then the bias, in a fixed order. Twenty-one
tiles, each with and without the bias and with and without programmatic dependent
launch: the packed tiles (PACKED_U4, PACKED_S8) pack the four chains of 16 output
features into one m64 wgmma, and PACKED_U4_IL, PACKED_S8_IL fill their copy ring stage
by stage, each stage's weights then its activations; the accumulator tiles
(ACCUMULATORS_CN16, ACCUMULATORS_CN32) keep one accumulator per chain for 64 features x
cn batch rows; the cluster tiles (CLUSTER_S2U4_C1 to C4, CLUSTER_S8U8_C4, CLUSTER_R2_C1)
run each chain on its own CTA of a cluster of min(4, ceil(K / 256)) CTAs and sum the
chains in rank 0, reading through a K / 64-block view (K % 64 == 0), and the
CLUSTER_S2U4_2D_C1 to C4 tiles through the plain 2D view. The BOX3D programs
(PACKED_U4_BOX3D, PACKED_S8_BOX3D, PACKED_U4_IL_BOX3D, PACKED_S8_IL_BOX3D,
ACCUMULATORS_CN32_BOX3D, K % 1024 == 0) load the four chains' boxes of a K slice in one
3D copy. The host module ``flashinfer.gemm.routergemm`` selects the program from the
batch, the output features, the input features and the bias, computes the grid and
launches it.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

PACKED_U4 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_55558d6f73b768d9df79",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_55558d6f73b768d9df79",
    sources=(
        "cake_tinygemm2_wgmma_55558d6f73b768d9df79_kernel.cu",
        "cake_tinygemm2_wgmma_55558d6f73b768d9df79_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_7d5d918c9c8c3913b01f",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_7d5d918c9c8c3913b01f",
    sources=(
        "cake_tinygemm2_wgmma_7d5d918c9c8c3913b01f_kernel.cu",
        "cake_tinygemm2_wgmma_7d5d918c9c8c3913b01f_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_8b8bba57d4e24019cbd7",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_8b8bba57d4e24019cbd7",
    sources=(
        "cake_tinygemm2_wgmma_8b8bba57d4e24019cbd7_kernel.cu",
        "cake_tinygemm2_wgmma_8b8bba57d4e24019cbd7_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_4103e3b63ccb81b4f14e",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_4103e3b63ccb81b4f14e",
    sources=(
        "cake_tinygemm2_wgmma_4103e3b63ccb81b4f14e_kernel.cu",
        "cake_tinygemm2_wgmma_4103e3b63ccb81b4f14e_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_BOX3D = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_5e14608ae62d499d2906",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_5e14608ae62d499d2906",
    sources=(
        "cake_tinygemm2_wgmma_5e14608ae62d499d2906_kernel.cu",
        "cake_tinygemm2_wgmma_5e14608ae62d499d2906_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_BOX3D_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_9a9bd6d2e51e96ac7d38",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_9a9bd6d2e51e96ac7d38",
    sources=(
        "cake_tinygemm2_wgmma_9a9bd6d2e51e96ac7d38_kernel.cu",
        "cake_tinygemm2_wgmma_9a9bd6d2e51e96ac7d38_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_BOX3D_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_d3b55f6a2213350ba370",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_d3b55f6a2213350ba370",
    sources=(
        "cake_tinygemm2_wgmma_d3b55f6a2213350ba370_kernel.cu",
        "cake_tinygemm2_wgmma_d3b55f6a2213350ba370_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_BOX3D_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_cef3de4a57586dd5056b",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_cef3de4a57586dd5056b",
    sources=(
        "cake_tinygemm2_wgmma_cef3de4a57586dd5056b_kernel.cu",
        "cake_tinygemm2_wgmma_cef3de4a57586dd5056b_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_21a62b43b0992bd98d24",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_21a62b43b0992bd98d24",
    sources=(
        "cake_tinygemm2_wgmma_21a62b43b0992bd98d24_kernel.cu",
        "cake_tinygemm2_wgmma_21a62b43b0992bd98d24_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_63c8d1d39c4d22c24f7a",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_63c8d1d39c4d22c24f7a",
    sources=(
        "cake_tinygemm2_wgmma_63c8d1d39c4d22c24f7a_kernel.cu",
        "cake_tinygemm2_wgmma_63c8d1d39c4d22c24f7a_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_6f4bcab2ee15726cfb3d",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_6f4bcab2ee15726cfb3d",
    sources=(
        "cake_tinygemm2_wgmma_6f4bcab2ee15726cfb3d_kernel.cu",
        "cake_tinygemm2_wgmma_6f4bcab2ee15726cfb3d_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_067c1cc17c6e04447467",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_067c1cc17c6e04447467",
    sources=(
        "cake_tinygemm2_wgmma_067c1cc17c6e04447467_kernel.cu",
        "cake_tinygemm2_wgmma_067c1cc17c6e04447467_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_BOX3D = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_8d50f4ed64fa2583bb7f",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_8d50f4ed64fa2583bb7f",
    sources=(
        "cake_tinygemm2_wgmma_8d50f4ed64fa2583bb7f_kernel.cu",
        "cake_tinygemm2_wgmma_8d50f4ed64fa2583bb7f_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_BOX3D_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_af3e212787d1712882d4",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_af3e212787d1712882d4",
    sources=(
        "cake_tinygemm2_wgmma_af3e212787d1712882d4_kernel.cu",
        "cake_tinygemm2_wgmma_af3e212787d1712882d4_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_BOX3D_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_e37aaa1a61710c813ef8",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_e37aaa1a61710c813ef8",
    sources=(
        "cake_tinygemm2_wgmma_e37aaa1a61710c813ef8_kernel.cu",
        "cake_tinygemm2_wgmma_e37aaa1a61710c813ef8_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_BOX3D_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_6dc857560b4b3ca90d78",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_6dc857560b4b3ca90d78",
    sources=(
        "cake_tinygemm2_wgmma_6dc857560b4b3ca90d78_kernel.cu",
        "cake_tinygemm2_wgmma_6dc857560b4b3ca90d78_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_45136a898d79c4247c84",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_45136a898d79c4247c84",
    sources=(
        "cake_tinygemm2_wgmma_45136a898d79c4247c84_kernel.cu",
        "cake_tinygemm2_wgmma_45136a898d79c4247c84_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_b267b94c173f22af7d32",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_b267b94c173f22af7d32",
    sources=(
        "cake_tinygemm2_wgmma_b267b94c173f22af7d32_kernel.cu",
        "cake_tinygemm2_wgmma_b267b94c173f22af7d32_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_94d0d18064ac973dee10",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_94d0d18064ac973dee10",
    sources=(
        "cake_tinygemm2_wgmma_94d0d18064ac973dee10_kernel.cu",
        "cake_tinygemm2_wgmma_94d0d18064ac973dee10_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_9a7a8e76c0a8b800a7e3",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_9a7a8e76c0a8b800a7e3",
    sources=(
        "cake_tinygemm2_wgmma_9a7a8e76c0a8b800a7e3_kernel.cu",
        "cake_tinygemm2_wgmma_9a7a8e76c0a8b800a7e3_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL_BOX3D = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_246108c7d41a35251bf3",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_246108c7d41a35251bf3",
    sources=(
        "cake_tinygemm2_wgmma_246108c7d41a35251bf3_kernel.cu",
        "cake_tinygemm2_wgmma_246108c7d41a35251bf3_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL_BOX3D_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_6891ad0f607038e9e04d",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_6891ad0f607038e9e04d",
    sources=(
        "cake_tinygemm2_wgmma_6891ad0f607038e9e04d_kernel.cu",
        "cake_tinygemm2_wgmma_6891ad0f607038e9e04d_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL_BOX3D_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_9937d1ccf065cfdb75c0",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_9937d1ccf065cfdb75c0",
    sources=(
        "cake_tinygemm2_wgmma_9937d1ccf065cfdb75c0_kernel.cu",
        "cake_tinygemm2_wgmma_9937d1ccf065cfdb75c0_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_U4_IL_BOX3D_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_ec3481204cda058c4f52",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_ec3481204cda058c4f52",
    sources=(
        "cake_tinygemm2_wgmma_ec3481204cda058c4f52_kernel.cu",
        "cake_tinygemm2_wgmma_ec3481204cda058c4f52_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=199680,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_8c1f220707ef1a721388",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_8c1f220707ef1a721388",
    sources=(
        "cake_tinygemm2_wgmma_8c1f220707ef1a721388_kernel.cu",
        "cake_tinygemm2_wgmma_8c1f220707ef1a721388_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_c3fe2ca46d1ab26998a3",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_c3fe2ca46d1ab26998a3",
    sources=(
        "cake_tinygemm2_wgmma_c3fe2ca46d1ab26998a3_kernel.cu",
        "cake_tinygemm2_wgmma_c3fe2ca46d1ab26998a3_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_174dc6f781113af91527",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_174dc6f781113af91527",
    sources=(
        "cake_tinygemm2_wgmma_174dc6f781113af91527_kernel.cu",
        "cake_tinygemm2_wgmma_174dc6f781113af91527_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_e89c649fd2e09b488fe6",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_e89c649fd2e09b488fe6",
    sources=(
        "cake_tinygemm2_wgmma_e89c649fd2e09b488fe6_kernel.cu",
        "cake_tinygemm2_wgmma_e89c649fd2e09b488fe6_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL_BOX3D = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_addf4ffd8440df2a39a1",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_addf4ffd8440df2a39a1",
    sources=(
        "cake_tinygemm2_wgmma_addf4ffd8440df2a39a1_kernel.cu",
        "cake_tinygemm2_wgmma_addf4ffd8440df2a39a1_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL_BOX3D_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_c07a654085bfec309a38",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_c07a654085bfec309a38",
    sources=(
        "cake_tinygemm2_wgmma_c07a654085bfec309a38_kernel.cu",
        "cake_tinygemm2_wgmma_c07a654085bfec309a38_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL_BOX3D_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_930d4eec92a49326780b",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_930d4eec92a49326780b",
    sources=(
        "cake_tinygemm2_wgmma_930d4eec92a49326780b_kernel.cu",
        "cake_tinygemm2_wgmma_930d4eec92a49326780b_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PACKED_S8_IL_BOX3D_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_b3ba684069c631005078",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_b3ba684069c631005078",
    sources=(
        "cake_tinygemm2_wgmma_b3ba684069c631005078_kernel.cu",
        "cake_tinygemm2_wgmma_b3ba684069c631005078_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=101376,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN16 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_9168c67ceb948037292c",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_9168c67ceb948037292c",
    sources=(
        "cake_tinygemm2_wgmma_9168c67ceb948037292c_kernel.cu",
        "cake_tinygemm2_wgmma_9168c67ceb948037292c_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN16_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_82819f8de5efaf517448",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_82819f8de5efaf517448",
    sources=(
        "cake_tinygemm2_wgmma_82819f8de5efaf517448_kernel.cu",
        "cake_tinygemm2_wgmma_82819f8de5efaf517448_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN16_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_07e66cb376e56a9528d8",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_07e66cb376e56a9528d8",
    sources=(
        "cake_tinygemm2_wgmma_07e66cb376e56a9528d8_kernel.cu",
        "cake_tinygemm2_wgmma_07e66cb376e56a9528d8_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN16_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_164886d9e1894a3eb342",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_164886d9e1894a3eb342",
    sources=(
        "cake_tinygemm2_wgmma_164886d9e1894a3eb342_kernel.cu",
        "cake_tinygemm2_wgmma_164886d9e1894a3eb342_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=205824,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_e11d51c563dcd217484e",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_e11d51c563dcd217484e",
    sources=(
        "cake_tinygemm2_wgmma_e11d51c563dcd217484e_kernel.cu",
        "cake_tinygemm2_wgmma_e11d51c563dcd217484e_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_5fd0578bff42e2c7aab7",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_5fd0578bff42e2c7aab7",
    sources=(
        "cake_tinygemm2_wgmma_5fd0578bff42e2c7aab7_kernel.cu",
        "cake_tinygemm2_wgmma_5fd0578bff42e2c7aab7_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_bf74a6e29ef5f665fd3b",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_bf74a6e29ef5f665fd3b",
    sources=(
        "cake_tinygemm2_wgmma_bf74a6e29ef5f665fd3b_kernel.cu",
        "cake_tinygemm2_wgmma_bf74a6e29ef5f665fd3b_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_caf480506abffa6184c0",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_caf480506abffa6184c0",
    sources=(
        "cake_tinygemm2_wgmma_caf480506abffa6184c0_kernel.cu",
        "cake_tinygemm2_wgmma_caf480506abffa6184c0_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32_BOX3D = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_f0ba82df9c255b302657",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_f0ba82df9c255b302657",
    sources=(
        "cake_tinygemm2_wgmma_f0ba82df9c255b302657_kernel.cu",
        "cake_tinygemm2_wgmma_f0ba82df9c255b302657_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32_BOX3D_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_b3005f49d5157230da3b",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_b3005f49d5157230da3b",
    sources=(
        "cake_tinygemm2_wgmma_b3005f49d5157230da3b_kernel.cu",
        "cake_tinygemm2_wgmma_b3005f49d5157230da3b_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32_BOX3D_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_04148d587a09906e3c39",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_04148d587a09906e3c39",
    sources=(
        "cake_tinygemm2_wgmma_04148d587a09906e3c39_kernel.cu",
        "cake_tinygemm2_wgmma_04148d587a09906e3c39_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

ACCUMULATORS_CN32_BOX3D_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_851d116d19b75fc6518e",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_851d116d19b75fc6518e",
    sources=(
        "cake_tinygemm2_wgmma_851d116d19b75fc6518e_kernel.cu",
        "cake_tinygemm2_wgmma_851d116d19b75fc6518e_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=197632,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C1 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_068158420af49ce8e7f3",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_068158420af49ce8e7f3",
    sources=(
        "cake_tinygemm2_wgmma_068158420af49ce8e7f3_kernel.cu",
        "cake_tinygemm2_wgmma_068158420af49ce8e7f3_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C1_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_e916e28da1aa93937589",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_e916e28da1aa93937589",
    sources=(
        "cake_tinygemm2_wgmma_e916e28da1aa93937589_kernel.cu",
        "cake_tinygemm2_wgmma_e916e28da1aa93937589_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C1_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_02e1b41be1e0d0d76f19",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_02e1b41be1e0d0d76f19",
    sources=(
        "cake_tinygemm2_wgmma_02e1b41be1e0d0d76f19_kernel.cu",
        "cake_tinygemm2_wgmma_02e1b41be1e0d0d76f19_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C1_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_d4ba8f42e86c3758798c",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_d4ba8f42e86c3758798c",
    sources=(
        "cake_tinygemm2_wgmma_d4ba8f42e86c3758798c_kernel.cu",
        "cake_tinygemm2_wgmma_d4ba8f42e86c3758798c_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C2 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_ea03b844a4e239846148",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_ea03b844a4e239846148",
    sources=(
        "cake_tinygemm2_wgmma_ea03b844a4e239846148_kernel.cu",
        "cake_tinygemm2_wgmma_ea03b844a4e239846148_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C2_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_290e8fc369bcd90e905e",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_290e8fc369bcd90e905e",
    sources=(
        "cake_tinygemm2_wgmma_290e8fc369bcd90e905e_kernel.cu",
        "cake_tinygemm2_wgmma_290e8fc369bcd90e905e_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C2_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_bb24e4fc04061f85b281",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_bb24e4fc04061f85b281",
    sources=(
        "cake_tinygemm2_wgmma_bb24e4fc04061f85b281_kernel.cu",
        "cake_tinygemm2_wgmma_bb24e4fc04061f85b281_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C2_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_ce17f3b0d1ba3078d9b1",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_ce17f3b0d1ba3078d9b1",
    sources=(
        "cake_tinygemm2_wgmma_ce17f3b0d1ba3078d9b1_kernel.cu",
        "cake_tinygemm2_wgmma_ce17f3b0d1ba3078d9b1_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C3 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_2bc61a174dd8f5043809",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_2bc61a174dd8f5043809",
    sources=(
        "cake_tinygemm2_wgmma_2bc61a174dd8f5043809_kernel.cu",
        "cake_tinygemm2_wgmma_2bc61a174dd8f5043809_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C3_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_ed1b8b48dde9243dba48",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_ed1b8b48dde9243dba48",
    sources=(
        "cake_tinygemm2_wgmma_ed1b8b48dde9243dba48_kernel.cu",
        "cake_tinygemm2_wgmma_ed1b8b48dde9243dba48_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C3_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_87dc274fb8111b2a2316",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_87dc274fb8111b2a2316",
    sources=(
        "cake_tinygemm2_wgmma_87dc274fb8111b2a2316_kernel.cu",
        "cake_tinygemm2_wgmma_87dc274fb8111b2a2316_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C3_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_92c9b1e3b50b58411958",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_92c9b1e3b50b58411958",
    sources=(
        "cake_tinygemm2_wgmma_92c9b1e3b50b58411958_kernel.cu",
        "cake_tinygemm2_wgmma_92c9b1e3b50b58411958_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C4 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_a9faa453b9edd63bd2de",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_a9faa453b9edd63bd2de",
    sources=(
        "cake_tinygemm2_wgmma_a9faa453b9edd63bd2de_kernel.cu",
        "cake_tinygemm2_wgmma_a9faa453b9edd63bd2de_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C4_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_9d303906d815db19eef4",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_9d303906d815db19eef4",
    sources=(
        "cake_tinygemm2_wgmma_9d303906d815db19eef4_kernel.cu",
        "cake_tinygemm2_wgmma_9d303906d815db19eef4_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C4_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_4b8e882e59e4a26c9c6a",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_4b8e882e59e4a26c9c6a",
    sources=(
        "cake_tinygemm2_wgmma_4b8e882e59e4a26c9c6a_kernel.cu",
        "cake_tinygemm2_wgmma_4b8e882e59e4a26c9c6a_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_C4_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_d5936c6c320c5daae27b",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_d5936c6c320c5daae27b",
    sources=(
        "cake_tinygemm2_wgmma_d5936c6c320c5daae27b_kernel.cu",
        "cake_tinygemm2_wgmma_d5936c6c320c5daae27b_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C1 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_f29c9d50ac827b8c480d",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_f29c9d50ac827b8c480d",
    sources=(
        "cake_tinygemm2_wgmma_f29c9d50ac827b8c480d_kernel.cu",
        "cake_tinygemm2_wgmma_f29c9d50ac827b8c480d_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C1_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_da679ebc98bce88304bf",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_da679ebc98bce88304bf",
    sources=(
        "cake_tinygemm2_wgmma_da679ebc98bce88304bf_kernel.cu",
        "cake_tinygemm2_wgmma_da679ebc98bce88304bf_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C1_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_3bdda577567a9cacc879",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_3bdda577567a9cacc879",
    sources=(
        "cake_tinygemm2_wgmma_3bdda577567a9cacc879_kernel.cu",
        "cake_tinygemm2_wgmma_3bdda577567a9cacc879_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C1_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_83a3063918fa61098542",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_83a3063918fa61098542",
    sources=(
        "cake_tinygemm2_wgmma_83a3063918fa61098542_kernel.cu",
        "cake_tinygemm2_wgmma_83a3063918fa61098542_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=29696,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C2 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_34caaf722996086546fd",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_34caaf722996086546fd",
    sources=(
        "cake_tinygemm2_wgmma_34caaf722996086546fd_kernel.cu",
        "cake_tinygemm2_wgmma_34caaf722996086546fd_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C2_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_3f16adbcacae8c0136b5",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_3f16adbcacae8c0136b5",
    sources=(
        "cake_tinygemm2_wgmma_3f16adbcacae8c0136b5_kernel.cu",
        "cake_tinygemm2_wgmma_3f16adbcacae8c0136b5_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C2_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_5d37163451f90813e79d",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_5d37163451f90813e79d",
    sources=(
        "cake_tinygemm2_wgmma_5d37163451f90813e79d_kernel.cu",
        "cake_tinygemm2_wgmma_5d37163451f90813e79d_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C2_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_fc7bf3cbebd82a0d39e7",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_fc7bf3cbebd82a0d39e7",
    sources=(
        "cake_tinygemm2_wgmma_fc7bf3cbebd82a0d39e7_kernel.cu",
        "cake_tinygemm2_wgmma_fc7bf3cbebd82a0d39e7_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=30208,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C3 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_3cc23206cf1cc499cdb4",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_3cc23206cf1cc499cdb4",
    sources=(
        "cake_tinygemm2_wgmma_3cc23206cf1cc499cdb4_kernel.cu",
        "cake_tinygemm2_wgmma_3cc23206cf1cc499cdb4_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C3_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_3908dd463b09e6813b98",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_3908dd463b09e6813b98",
    sources=(
        "cake_tinygemm2_wgmma_3908dd463b09e6813b98_kernel.cu",
        "cake_tinygemm2_wgmma_3908dd463b09e6813b98_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C3_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_218f7ee3cce31592eb67",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_218f7ee3cce31592eb67",
    sources=(
        "cake_tinygemm2_wgmma_218f7ee3cce31592eb67_kernel.cu",
        "cake_tinygemm2_wgmma_218f7ee3cce31592eb67_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C3_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_420db229260a698647ff",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_420db229260a698647ff",
    sources=(
        "cake_tinygemm2_wgmma_420db229260a698647ff_kernel.cu",
        "cake_tinygemm2_wgmma_420db229260a698647ff_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(3, 1, 1),
    dynamic_smem_bytes=30720,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C4 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_34189986c1b60867ae29",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_34189986c1b60867ae29",
    sources=(
        "cake_tinygemm2_wgmma_34189986c1b60867ae29_kernel.cu",
        "cake_tinygemm2_wgmma_34189986c1b60867ae29_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C4_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_0dfd60ac8b75e8bcdda9",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_0dfd60ac8b75e8bcdda9",
    sources=(
        "cake_tinygemm2_wgmma_0dfd60ac8b75e8bcdda9_kernel.cu",
        "cake_tinygemm2_wgmma_0dfd60ac8b75e8bcdda9_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C4_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_b066a6dd3f97b107d73e",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_b066a6dd3f97b107d73e",
    sources=(
        "cake_tinygemm2_wgmma_b066a6dd3f97b107d73e_kernel.cu",
        "cake_tinygemm2_wgmma_b066a6dd3f97b107d73e_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S2U4_2D_C4_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_48d8c07ad4b1ca31f4c2",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_48d8c07ad4b1ca31f4c2",
    sources=(
        "cake_tinygemm2_wgmma_48d8c07ad4b1ca31f4c2_kernel.cu",
        "cake_tinygemm2_wgmma_48d8c07ad4b1ca31f4c2_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=31232,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S8U8_C4 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_9f044cc228aca990a2fd",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_9f044cc228aca990a2fd",
    sources=(
        "cake_tinygemm2_wgmma_9f044cc228aca990a2fd_kernel.cu",
        "cake_tinygemm2_wgmma_9f044cc228aca990a2fd_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=199168,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S8U8_C4_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_79f0bb1321452f687f24",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_79f0bb1321452f687f24",
    sources=(
        "cake_tinygemm2_wgmma_79f0bb1321452f687f24_kernel.cu",
        "cake_tinygemm2_wgmma_79f0bb1321452f687f24_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=199168,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S8U8_C4_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_b64aeb3308c17fa81ef2",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_b64aeb3308c17fa81ef2",
    sources=(
        "cake_tinygemm2_wgmma_b64aeb3308c17fa81ef2_kernel.cu",
        "cake_tinygemm2_wgmma_b64aeb3308c17fa81ef2_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=199168,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_S8U8_C4_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_48aa54fc888d7eadd726",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_48aa54fc888d7eadd726",
    sources=(
        "cake_tinygemm2_wgmma_48aa54fc888d7eadd726_kernel.cu",
        "cake_tinygemm2_wgmma_48aa54fc888d7eadd726_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=199168,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_R2_C1 = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_edf279fa874fccf9ce15",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_edf279fa874fccf9ce15",
    sources=(
        "cake_tinygemm2_wgmma_edf279fa874fccf9ce15_kernel.cu",
        "cake_tinygemm2_wgmma_edf279fa874fccf9ce15_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=50176,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_R2_C1_NOBIAS = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_68afbe92e2300b660538",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_68afbe92e2300b660538",
    sources=(
        "cake_tinygemm2_wgmma_68afbe92e2300b660538_kernel.cu",
        "cake_tinygemm2_wgmma_68afbe92e2300b660538_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=50176,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_R2_C1_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_bd263f2fa30c085b45a0",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_bd263f2fa30c085b45a0",
    sources=(
        "cake_tinygemm2_wgmma_bd263f2fa30c085b45a0_kernel.cu",
        "cake_tinygemm2_wgmma_bd263f2fa30c085b45a0_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=50176,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

CLUSTER_R2_C1_NOBIAS_PDL = cake_program.CakeProgram(
    name="cake_tinygemm2_wgmma_02332a0f1e045f569a19",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_02332a0f1e045f569a19",
    sources=(
        "cake_tinygemm2_wgmma_02332a0f1e045f569a19_kernel.cu",
        "cake_tinygemm2_wgmma_02332a0f1e045f569a19_binding.cu",
    ),
    targets=("sm_90a",),
    compile_options=(("sm_90a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "X", "out", "bias", "M", "N", "K", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "X", "out", "bias"),
    block=(160, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=50176,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    PACKED_U4,
    PACKED_U4_NOBIAS,
    PACKED_U4_PDL,
    PACKED_U4_NOBIAS_PDL,
    PACKED_U4_BOX3D,
    PACKED_U4_BOX3D_NOBIAS,
    PACKED_U4_BOX3D_PDL,
    PACKED_U4_BOX3D_NOBIAS_PDL,
    PACKED_S8,
    PACKED_S8_NOBIAS,
    PACKED_S8_PDL,
    PACKED_S8_NOBIAS_PDL,
    PACKED_S8_BOX3D,
    PACKED_S8_BOX3D_NOBIAS,
    PACKED_S8_BOX3D_PDL,
    PACKED_S8_BOX3D_NOBIAS_PDL,
    PACKED_U4_IL,
    PACKED_U4_IL_NOBIAS,
    PACKED_U4_IL_PDL,
    PACKED_U4_IL_NOBIAS_PDL,
    PACKED_U4_IL_BOX3D,
    PACKED_U4_IL_BOX3D_NOBIAS,
    PACKED_U4_IL_BOX3D_PDL,
    PACKED_U4_IL_BOX3D_NOBIAS_PDL,
    PACKED_S8_IL,
    PACKED_S8_IL_NOBIAS,
    PACKED_S8_IL_PDL,
    PACKED_S8_IL_NOBIAS_PDL,
    PACKED_S8_IL_BOX3D,
    PACKED_S8_IL_BOX3D_NOBIAS,
    PACKED_S8_IL_BOX3D_PDL,
    PACKED_S8_IL_BOX3D_NOBIAS_PDL,
    ACCUMULATORS_CN16,
    ACCUMULATORS_CN16_NOBIAS,
    ACCUMULATORS_CN16_PDL,
    ACCUMULATORS_CN16_NOBIAS_PDL,
    ACCUMULATORS_CN32,
    ACCUMULATORS_CN32_NOBIAS,
    ACCUMULATORS_CN32_PDL,
    ACCUMULATORS_CN32_NOBIAS_PDL,
    ACCUMULATORS_CN32_BOX3D,
    ACCUMULATORS_CN32_BOX3D_NOBIAS,
    ACCUMULATORS_CN32_BOX3D_PDL,
    ACCUMULATORS_CN32_BOX3D_NOBIAS_PDL,
    CLUSTER_S2U4_C1,
    CLUSTER_S2U4_C1_NOBIAS,
    CLUSTER_S2U4_C1_PDL,
    CLUSTER_S2U4_C1_NOBIAS_PDL,
    CLUSTER_S2U4_C2,
    CLUSTER_S2U4_C2_NOBIAS,
    CLUSTER_S2U4_C2_PDL,
    CLUSTER_S2U4_C2_NOBIAS_PDL,
    CLUSTER_S2U4_C3,
    CLUSTER_S2U4_C3_NOBIAS,
    CLUSTER_S2U4_C3_PDL,
    CLUSTER_S2U4_C3_NOBIAS_PDL,
    CLUSTER_S2U4_C4,
    CLUSTER_S2U4_C4_NOBIAS,
    CLUSTER_S2U4_C4_PDL,
    CLUSTER_S2U4_C4_NOBIAS_PDL,
    CLUSTER_S2U4_2D_C1,
    CLUSTER_S2U4_2D_C1_NOBIAS,
    CLUSTER_S2U4_2D_C1_PDL,
    CLUSTER_S2U4_2D_C1_NOBIAS_PDL,
    CLUSTER_S2U4_2D_C2,
    CLUSTER_S2U4_2D_C2_NOBIAS,
    CLUSTER_S2U4_2D_C2_PDL,
    CLUSTER_S2U4_2D_C2_NOBIAS_PDL,
    CLUSTER_S2U4_2D_C3,
    CLUSTER_S2U4_2D_C3_NOBIAS,
    CLUSTER_S2U4_2D_C3_PDL,
    CLUSTER_S2U4_2D_C3_NOBIAS_PDL,
    CLUSTER_S2U4_2D_C4,
    CLUSTER_S2U4_2D_C4_NOBIAS,
    CLUSTER_S2U4_2D_C4_PDL,
    CLUSTER_S2U4_2D_C4_NOBIAS_PDL,
    CLUSTER_S8U8_C4,
    CLUSTER_S8U8_C4_NOBIAS,
    CLUSTER_S8U8_C4_PDL,
    CLUSTER_S8U8_C4_NOBIAS_PDL,
    CLUSTER_R2_C1,
    CLUSTER_R2_C1_NOBIAS,
    CLUSTER_R2_C1_PDL,
    CLUSTER_R2_C1_NOBIAS_PDL,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "PACKED_U4",
    "PACKED_U4_NOBIAS",
    "PACKED_U4_PDL",
    "PACKED_U4_NOBIAS_PDL",
    "PACKED_U4_BOX3D",
    "PACKED_U4_BOX3D_NOBIAS",
    "PACKED_U4_BOX3D_PDL",
    "PACKED_U4_BOX3D_NOBIAS_PDL",
    "PACKED_S8",
    "PACKED_S8_NOBIAS",
    "PACKED_S8_PDL",
    "PACKED_S8_NOBIAS_PDL",
    "PACKED_S8_BOX3D",
    "PACKED_S8_BOX3D_NOBIAS",
    "PACKED_S8_BOX3D_PDL",
    "PACKED_S8_BOX3D_NOBIAS_PDL",
    "PACKED_U4_IL",
    "PACKED_U4_IL_NOBIAS",
    "PACKED_U4_IL_PDL",
    "PACKED_U4_IL_NOBIAS_PDL",
    "PACKED_U4_IL_BOX3D",
    "PACKED_U4_IL_BOX3D_NOBIAS",
    "PACKED_U4_IL_BOX3D_PDL",
    "PACKED_U4_IL_BOX3D_NOBIAS_PDL",
    "PACKED_S8_IL",
    "PACKED_S8_IL_NOBIAS",
    "PACKED_S8_IL_PDL",
    "PACKED_S8_IL_NOBIAS_PDL",
    "PACKED_S8_IL_BOX3D",
    "PACKED_S8_IL_BOX3D_NOBIAS",
    "PACKED_S8_IL_BOX3D_PDL",
    "PACKED_S8_IL_BOX3D_NOBIAS_PDL",
    "ACCUMULATORS_CN16",
    "ACCUMULATORS_CN16_NOBIAS",
    "ACCUMULATORS_CN16_PDL",
    "ACCUMULATORS_CN16_NOBIAS_PDL",
    "ACCUMULATORS_CN32",
    "ACCUMULATORS_CN32_NOBIAS",
    "ACCUMULATORS_CN32_PDL",
    "ACCUMULATORS_CN32_NOBIAS_PDL",
    "ACCUMULATORS_CN32_BOX3D",
    "ACCUMULATORS_CN32_BOX3D_NOBIAS",
    "ACCUMULATORS_CN32_BOX3D_PDL",
    "ACCUMULATORS_CN32_BOX3D_NOBIAS_PDL",
    "CLUSTER_S2U4_C1",
    "CLUSTER_S2U4_C1_NOBIAS",
    "CLUSTER_S2U4_C1_PDL",
    "CLUSTER_S2U4_C1_NOBIAS_PDL",
    "CLUSTER_S2U4_C2",
    "CLUSTER_S2U4_C2_NOBIAS",
    "CLUSTER_S2U4_C2_PDL",
    "CLUSTER_S2U4_C2_NOBIAS_PDL",
    "CLUSTER_S2U4_C3",
    "CLUSTER_S2U4_C3_NOBIAS",
    "CLUSTER_S2U4_C3_PDL",
    "CLUSTER_S2U4_C3_NOBIAS_PDL",
    "CLUSTER_S2U4_C4",
    "CLUSTER_S2U4_C4_NOBIAS",
    "CLUSTER_S2U4_C4_PDL",
    "CLUSTER_S2U4_C4_NOBIAS_PDL",
    "CLUSTER_S2U4_2D_C1",
    "CLUSTER_S2U4_2D_C1_NOBIAS",
    "CLUSTER_S2U4_2D_C1_PDL",
    "CLUSTER_S2U4_2D_C1_NOBIAS_PDL",
    "CLUSTER_S2U4_2D_C2",
    "CLUSTER_S2U4_2D_C2_NOBIAS",
    "CLUSTER_S2U4_2D_C2_PDL",
    "CLUSTER_S2U4_2D_C2_NOBIAS_PDL",
    "CLUSTER_S2U4_2D_C3",
    "CLUSTER_S2U4_2D_C3_NOBIAS",
    "CLUSTER_S2U4_2D_C3_PDL",
    "CLUSTER_S2U4_2D_C3_NOBIAS_PDL",
    "CLUSTER_S2U4_2D_C4",
    "CLUSTER_S2U4_2D_C4_NOBIAS",
    "CLUSTER_S2U4_2D_C4_PDL",
    "CLUSTER_S2U4_2D_C4_NOBIAS_PDL",
    "CLUSTER_S8U8_C4",
    "CLUSTER_S8U8_C4_NOBIAS",
    "CLUSTER_S8U8_C4_PDL",
    "CLUSTER_S8U8_C4_NOBIAS_PDL",
    "CLUSTER_R2_C1",
    "CLUSTER_R2_C1_NOBIAS",
    "CLUSTER_R2_C1_PDL",
    "CLUSTER_R2_C1_NOBIAS_PDL",
    "PROGRAMS",
    "aot_specs",
]
