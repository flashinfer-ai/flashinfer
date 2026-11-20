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
    name="cake_tinygemm2_wgmma_208894d17e10b2a1a8e5",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_208894d17e10b2a1a8e5",
    sources=(
        "cake_tinygemm2_wgmma_208894d17e10b2a1a8e5_kernel.cu",
        "cake_tinygemm2_wgmma_208894d17e10b2a1a8e5_binding.cu",
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
    name="cake_tinygemm2_wgmma_cd6d8da8f120c1e862a5",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_cd6d8da8f120c1e862a5",
    sources=(
        "cake_tinygemm2_wgmma_cd6d8da8f120c1e862a5_kernel.cu",
        "cake_tinygemm2_wgmma_cd6d8da8f120c1e862a5_binding.cu",
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
    name="cake_tinygemm2_wgmma_47369a3ab1a5443148fc",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_47369a3ab1a5443148fc",
    sources=(
        "cake_tinygemm2_wgmma_47369a3ab1a5443148fc_kernel.cu",
        "cake_tinygemm2_wgmma_47369a3ab1a5443148fc_binding.cu",
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
    name="cake_tinygemm2_wgmma_c05398f71b9dfbb81623",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_c05398f71b9dfbb81623",
    sources=(
        "cake_tinygemm2_wgmma_c05398f71b9dfbb81623_kernel.cu",
        "cake_tinygemm2_wgmma_c05398f71b9dfbb81623_binding.cu",
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
    name="cake_tinygemm2_wgmma_86fdfb93767b96039f89",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_86fdfb93767b96039f89",
    sources=(
        "cake_tinygemm2_wgmma_86fdfb93767b96039f89_kernel.cu",
        "cake_tinygemm2_wgmma_86fdfb93767b96039f89_binding.cu",
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
    name="cake_tinygemm2_wgmma_599a43a3711c9f5b0e86",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_599a43a3711c9f5b0e86",
    sources=(
        "cake_tinygemm2_wgmma_599a43a3711c9f5b0e86_kernel.cu",
        "cake_tinygemm2_wgmma_599a43a3711c9f5b0e86_binding.cu",
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
    name="cake_tinygemm2_wgmma_2cfb06d3fd8e6ecb7032",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_2cfb06d3fd8e6ecb7032",
    sources=(
        "cake_tinygemm2_wgmma_2cfb06d3fd8e6ecb7032_kernel.cu",
        "cake_tinygemm2_wgmma_2cfb06d3fd8e6ecb7032_binding.cu",
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
    name="cake_tinygemm2_wgmma_c07d12b4e3e49c29b253",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_c07d12b4e3e49c29b253",
    sources=(
        "cake_tinygemm2_wgmma_c07d12b4e3e49c29b253_kernel.cu",
        "cake_tinygemm2_wgmma_c07d12b4e3e49c29b253_binding.cu",
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
    name="cake_tinygemm2_wgmma_5fbb35f122c1c00cd93f",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_5fbb35f122c1c00cd93f",
    sources=(
        "cake_tinygemm2_wgmma_5fbb35f122c1c00cd93f_kernel.cu",
        "cake_tinygemm2_wgmma_5fbb35f122c1c00cd93f_binding.cu",
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
    name="cake_tinygemm2_wgmma_14562b8dcb63ae78fdfa",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_14562b8dcb63ae78fdfa",
    sources=(
        "cake_tinygemm2_wgmma_14562b8dcb63ae78fdfa_kernel.cu",
        "cake_tinygemm2_wgmma_14562b8dcb63ae78fdfa_binding.cu",
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
    name="cake_tinygemm2_wgmma_506502b7e3aed199cae3",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_506502b7e3aed199cae3",
    sources=(
        "cake_tinygemm2_wgmma_506502b7e3aed199cae3_kernel.cu",
        "cake_tinygemm2_wgmma_506502b7e3aed199cae3_binding.cu",
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
    name="cake_tinygemm2_wgmma_39f39683d43a0337397c",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_39f39683d43a0337397c",
    sources=(
        "cake_tinygemm2_wgmma_39f39683d43a0337397c_kernel.cu",
        "cake_tinygemm2_wgmma_39f39683d43a0337397c_binding.cu",
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
    name="cake_tinygemm2_wgmma_8da714660052de22cd90",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_8da714660052de22cd90",
    sources=(
        "cake_tinygemm2_wgmma_8da714660052de22cd90_kernel.cu",
        "cake_tinygemm2_wgmma_8da714660052de22cd90_binding.cu",
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
    name="cake_tinygemm2_wgmma_4dca465b92f5af7be148",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_4dca465b92f5af7be148",
    sources=(
        "cake_tinygemm2_wgmma_4dca465b92f5af7be148_kernel.cu",
        "cake_tinygemm2_wgmma_4dca465b92f5af7be148_binding.cu",
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
    name="cake_tinygemm2_wgmma_99046d156bc195ff6f79",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_99046d156bc195ff6f79",
    sources=(
        "cake_tinygemm2_wgmma_99046d156bc195ff6f79_kernel.cu",
        "cake_tinygemm2_wgmma_99046d156bc195ff6f79_binding.cu",
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
    name="cake_tinygemm2_wgmma_03b0122426f02d2dd57c",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_03b0122426f02d2dd57c",
    sources=(
        "cake_tinygemm2_wgmma_03b0122426f02d2dd57c_kernel.cu",
        "cake_tinygemm2_wgmma_03b0122426f02d2dd57c_binding.cu",
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
    name="cake_tinygemm2_wgmma_63b2eddd07545b3c525a",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_63b2eddd07545b3c525a",
    sources=(
        "cake_tinygemm2_wgmma_63b2eddd07545b3c525a_kernel.cu",
        "cake_tinygemm2_wgmma_63b2eddd07545b3c525a_binding.cu",
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
    name="cake_tinygemm2_wgmma_f9d0bf564a7dd2d79a6d",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_f9d0bf564a7dd2d79a6d",
    sources=(
        "cake_tinygemm2_wgmma_f9d0bf564a7dd2d79a6d_kernel.cu",
        "cake_tinygemm2_wgmma_f9d0bf564a7dd2d79a6d_binding.cu",
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
    name="cake_tinygemm2_wgmma_3ca302a70c1b30603f27",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_3ca302a70c1b30603f27",
    sources=(
        "cake_tinygemm2_wgmma_3ca302a70c1b30603f27_kernel.cu",
        "cake_tinygemm2_wgmma_3ca302a70c1b30603f27_binding.cu",
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
    name="cake_tinygemm2_wgmma_8d71faa43dd1bed46ec2",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_8d71faa43dd1bed46ec2",
    sources=(
        "cake_tinygemm2_wgmma_8d71faa43dd1bed46ec2_kernel.cu",
        "cake_tinygemm2_wgmma_8d71faa43dd1bed46ec2_binding.cu",
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
    name="cake_tinygemm2_wgmma_4d3511b13c96591c7ef5",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_4d3511b13c96591c7ef5",
    sources=(
        "cake_tinygemm2_wgmma_4d3511b13c96591c7ef5_kernel.cu",
        "cake_tinygemm2_wgmma_4d3511b13c96591c7ef5_binding.cu",
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
    name="cake_tinygemm2_wgmma_dc96d608a075076fe599",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_dc96d608a075076fe599",
    sources=(
        "cake_tinygemm2_wgmma_dc96d608a075076fe599_kernel.cu",
        "cake_tinygemm2_wgmma_dc96d608a075076fe599_binding.cu",
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
    name="cake_tinygemm2_wgmma_6c70b76a631e79cb5390",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_6c70b76a631e79cb5390",
    sources=(
        "cake_tinygemm2_wgmma_6c70b76a631e79cb5390_kernel.cu",
        "cake_tinygemm2_wgmma_6c70b76a631e79cb5390_binding.cu",
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
    name="cake_tinygemm2_wgmma_7dab278600d030d4b20e",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_7dab278600d030d4b20e",
    sources=(
        "cake_tinygemm2_wgmma_7dab278600d030d4b20e_kernel.cu",
        "cake_tinygemm2_wgmma_7dab278600d030d4b20e_binding.cu",
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
    name="cake_tinygemm2_wgmma_bed7f1a92af9f6f916eb",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_bed7f1a92af9f6f916eb",
    sources=(
        "cake_tinygemm2_wgmma_bed7f1a92af9f6f916eb_kernel.cu",
        "cake_tinygemm2_wgmma_bed7f1a92af9f6f916eb_binding.cu",
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
    name="cake_tinygemm2_wgmma_326da69b108f763a8d42",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_326da69b108f763a8d42",
    sources=(
        "cake_tinygemm2_wgmma_326da69b108f763a8d42_kernel.cu",
        "cake_tinygemm2_wgmma_326da69b108f763a8d42_binding.cu",
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
    name="cake_tinygemm2_wgmma_80d5113ca610493342db",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_80d5113ca610493342db",
    sources=(
        "cake_tinygemm2_wgmma_80d5113ca610493342db_kernel.cu",
        "cake_tinygemm2_wgmma_80d5113ca610493342db_binding.cu",
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
    name="cake_tinygemm2_wgmma_1ca59ae343c2f5ee7695",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_1ca59ae343c2f5ee7695",
    sources=(
        "cake_tinygemm2_wgmma_1ca59ae343c2f5ee7695_kernel.cu",
        "cake_tinygemm2_wgmma_1ca59ae343c2f5ee7695_binding.cu",
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
    name="cake_tinygemm2_wgmma_585011d5c6651ac2187a",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_585011d5c6651ac2187a",
    sources=(
        "cake_tinygemm2_wgmma_585011d5c6651ac2187a_kernel.cu",
        "cake_tinygemm2_wgmma_585011d5c6651ac2187a_binding.cu",
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
    name="cake_tinygemm2_wgmma_640caf4a56f88f3ecc43",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_640caf4a56f88f3ecc43",
    sources=(
        "cake_tinygemm2_wgmma_640caf4a56f88f3ecc43_kernel.cu",
        "cake_tinygemm2_wgmma_640caf4a56f88f3ecc43_binding.cu",
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
    name="cake_tinygemm2_wgmma_517046eb9b65c3afa008",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_517046eb9b65c3afa008",
    sources=(
        "cake_tinygemm2_wgmma_517046eb9b65c3afa008_kernel.cu",
        "cake_tinygemm2_wgmma_517046eb9b65c3afa008_binding.cu",
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
    name="cake_tinygemm2_wgmma_adac5ac9ba2eea02cf98",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_adac5ac9ba2eea02cf98",
    sources=(
        "cake_tinygemm2_wgmma_adac5ac9ba2eea02cf98_kernel.cu",
        "cake_tinygemm2_wgmma_adac5ac9ba2eea02cf98_binding.cu",
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
    name="cake_tinygemm2_wgmma_589d4a94c2ea00407aa0",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_589d4a94c2ea00407aa0",
    sources=(
        "cake_tinygemm2_wgmma_589d4a94c2ea00407aa0_kernel.cu",
        "cake_tinygemm2_wgmma_589d4a94c2ea00407aa0_binding.cu",
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
    name="cake_tinygemm2_wgmma_19ca7b6897afed5270f4",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_19ca7b6897afed5270f4",
    sources=(
        "cake_tinygemm2_wgmma_19ca7b6897afed5270f4_kernel.cu",
        "cake_tinygemm2_wgmma_19ca7b6897afed5270f4_binding.cu",
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
    name="cake_tinygemm2_wgmma_b6aa25c2b6052cdb249d",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_b6aa25c2b6052cdb249d",
    sources=(
        "cake_tinygemm2_wgmma_b6aa25c2b6052cdb249d_kernel.cu",
        "cake_tinygemm2_wgmma_b6aa25c2b6052cdb249d_binding.cu",
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
    name="cake_tinygemm2_wgmma_0686b2a13ad8ab87aee2",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_0686b2a13ad8ab87aee2",
    sources=(
        "cake_tinygemm2_wgmma_0686b2a13ad8ab87aee2_kernel.cu",
        "cake_tinygemm2_wgmma_0686b2a13ad8ab87aee2_binding.cu",
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
    name="cake_tinygemm2_wgmma_732172d972d5cfcddb6b",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_732172d972d5cfcddb6b",
    sources=(
        "cake_tinygemm2_wgmma_732172d972d5cfcddb6b_kernel.cu",
        "cake_tinygemm2_wgmma_732172d972d5cfcddb6b_binding.cu",
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
    name="cake_tinygemm2_wgmma_708f5ffd0235c67be6f0",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_708f5ffd0235c67be6f0",
    sources=(
        "cake_tinygemm2_wgmma_708f5ffd0235c67be6f0_kernel.cu",
        "cake_tinygemm2_wgmma_708f5ffd0235c67be6f0_binding.cu",
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
    name="cake_tinygemm2_wgmma_818f96168a07652ac2bd",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_818f96168a07652ac2bd",
    sources=(
        "cake_tinygemm2_wgmma_818f96168a07652ac2bd_kernel.cu",
        "cake_tinygemm2_wgmma_818f96168a07652ac2bd_binding.cu",
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
    name="cake_tinygemm2_wgmma_40909a4f47270a313505",
    operator_dir="cake_tinygemm2_wgmma",
    kernel="kernel_cake_tinygemm2_wgmma_40909a4f47270a313505",
    sources=(
        "cake_tinygemm2_wgmma_40909a4f47270a313505_kernel.cu",
        "cake_tinygemm2_wgmma_40909a4f47270a313505_binding.cu",
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
