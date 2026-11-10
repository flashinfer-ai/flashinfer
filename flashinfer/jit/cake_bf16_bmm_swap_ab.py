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
"""Generated programs of the decode-row BF16 batched matrix product out[b] = A[b] @ W[b]^T
on tcgen05 (A the [b, m, k] tokens, W the [b, n, k] weights, both K-major; FP32
accumulation; bfloat16, float16 or float32 output), the weight rows in the MMA's M: one
program per routed build of the weight tile, the token tile, the ring depth, the in-
cluster K split, the loader form and the output dtype, each shipped for the targets
(sm_100a, sm_103a) whose shapes route to it. The host module
``flashinfer.gemm.gemm_base`` routes the shapes of its table of
``bmm_bf16(backend="cake")`` to a program, computes the grid and launches it; every
other shape keeps the module's mma.sync kernels.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

W64_T8_S2_SK8_PF_F32 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_7819f170405f90fd442f",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_7819f170405f90fd442f",
    sources=(
        "cake_bf16_bmm_swap_ab_7819f170405f90fd442f_kernel.cu",
        "cake_bf16_bmm_swap_ab_7819f170405f90fd442f_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(8, 1, 1),
    dynamic_smem_bytes=35840,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T8_S2_SK8_CP_F16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_617a859ff737a7c23b8a",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_617a859ff737a7c23b8a",
    sources=(
        "cake_bf16_bmm_swap_ab_617a859ff737a7c23b8a_kernel.cu",
        "cake_bf16_bmm_swap_ab_617a859ff737a7c23b8a_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(8, 1, 1),
    dynamic_smem_bytes=34816,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T8_S2_SK8_CP_F32 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_fed8d6e6f12ca4144bef",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_fed8d6e6f12ca4144bef",
    sources=(
        "cake_bf16_bmm_swap_ab_fed8d6e6f12ca4144bef_kernel.cu",
        "cake_bf16_bmm_swap_ab_fed8d6e6f12ca4144bef_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(8, 1, 1),
    dynamic_smem_bytes=35840,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T8_S4_SK4_PF_BF16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_438efe471988fa0d42da",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_438efe471988fa0d42da",
    sources=(
        "cake_bf16_bmm_swap_ab_438efe471988fa0d42da_kernel.cu",
        "cake_bf16_bmm_swap_ab_438efe471988fa0d42da_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=45056,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T8_S4_SK4_PF_F16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_d243af829e13fdedde5b",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_d243af829e13fdedde5b",
    sources=(
        "cake_bf16_bmm_swap_ab_d243af829e13fdedde5b_kernel.cu",
        "cake_bf16_bmm_swap_ab_d243af829e13fdedde5b_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=45056,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T8_S4_SK4_PF_F32 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_835bbb152b5da6351f3d",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_835bbb152b5da6351f3d",
    sources=(
        "cake_bf16_bmm_swap_ab_835bbb152b5da6351f3d_kernel.cu",
        "cake_bf16_bmm_swap_ab_835bbb152b5da6351f3d_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=46080,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T8_S4_SK4_PF_L2_BF16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_e0c8fa3e4b9e6897f11d",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_e0c8fa3e4b9e6897f11d",
    sources=(
        "cake_bf16_bmm_swap_ab_e0c8fa3e4b9e6897f11d_kernel.cu",
        "cake_bf16_bmm_swap_ab_e0c8fa3e4b9e6897f11d_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(4, 1, 1),
    dynamic_smem_bytes=45056,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T16_S8_SK2_PF_BF16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_2a71b40e5da2d8876b11",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_2a71b40e5da2d8876b11",
    sources=(
        "cake_bf16_bmm_swap_ab_2a71b40e5da2d8876b11_kernel.cu",
        "cake_bf16_bmm_swap_ab_2a71b40e5da2d8876b11_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=89088,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T16_S8_SK2_PF_F16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_d3b17eab30b5c2a10988",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_d3b17eab30b5c2a10988",
    sources=(
        "cake_bf16_bmm_swap_ab_d3b17eab30b5c2a10988_kernel.cu",
        "cake_bf16_bmm_swap_ab_d3b17eab30b5c2a10988_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=89088,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T16_S8_SK2_PF_F32 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_6923f7a232a5c320d22d",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_6923f7a232a5c320d22d",
    sources=(
        "cake_bf16_bmm_swap_ab_6923f7a232a5c320d22d_kernel.cu",
        "cake_bf16_bmm_swap_ab_6923f7a232a5c320d22d_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=91136,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T16_S8_SK2_PF_L2_BF16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_33e67e35d5af9839d0dc",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_33e67e35d5af9839d0dc",
    sources=(
        "cake_bf16_bmm_swap_ab_33e67e35d5af9839d0dc_kernel.cu",
        "cake_bf16_bmm_swap_ab_33e67e35d5af9839d0dc_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=89088,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T16_S8_SK2_PF_L2_F16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_29e15e70e64d1b0ef1d2",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_29e15e70e64d1b0ef1d2",
    sources=(
        "cake_bf16_bmm_swap_ab_29e15e70e64d1b0ef1d2_kernel.cu",
        "cake_bf16_bmm_swap_ab_29e15e70e64d1b0ef1d2_binding.cu",
    ),
    targets=("sm_100a",),
    compile_options=(("sm_100a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=89088,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

W64_T32_S8_SK2_PF_BF16 = cake_program.CakeProgram(
    name="cake_bf16_bmm_swap_ab_52a41e5b8581c6b0a64c",
    operator_dir="cake_bf16_bmm_swap_ab",
    kernel="kernel_cake_bf16_bmm_swap_ab_52a41e5b8581c6b0a64c",
    sources=(
        "cake_bf16_bmm_swap_ab_52a41e5b8581c6b0a64c_kernel.cu",
        "cake_bf16_bmm_swap_ab_52a41e5b8581c6b0a64c_binding.cu",
    ),
    targets=("sm_103a",),
    compile_options=(("sm_103a", ("--use_fast_math", "--fmad=true")),),
    args=("W", "A", "out", "M", "N", "K", "B", "grid_x", "grid_y", "grid_z"),
    tensors=("W", "A", "out"),
    block=(256, 1, 1),
    cluster=(2, 1, 1),
    dynamic_smem_bytes=111616,
    use_pdl=True,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (
    W64_T8_S2_SK8_PF_F32,
    W64_T8_S2_SK8_CP_F16,
    W64_T8_S2_SK8_CP_F32,
    W64_T8_S4_SK4_PF_BF16,
    W64_T8_S4_SK4_PF_F16,
    W64_T8_S4_SK4_PF_F32,
    W64_T8_S4_SK4_PF_L2_BF16,
    W64_T16_S8_SK2_PF_BF16,
    W64_T16_S8_SK2_PF_F16,
    W64_T16_S8_SK2_PF_F32,
    W64_T16_S8_SK2_PF_L2_BF16,
    W64_T16_S8_SK2_PF_L2_F16,
    W64_T32_S8_SK2_PF_BF16,
)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = [
    "W64_T8_S2_SK8_PF_F32",
    "W64_T8_S2_SK8_CP_F16",
    "W64_T8_S2_SK8_CP_F32",
    "W64_T8_S4_SK4_PF_BF16",
    "W64_T8_S4_SK4_PF_F16",
    "W64_T8_S4_SK4_PF_F32",
    "W64_T8_S4_SK4_PF_L2_BF16",
    "W64_T16_S8_SK2_PF_BF16",
    "W64_T16_S8_SK2_PF_F16",
    "W64_T16_S8_SK2_PF_F32",
    "W64_T16_S8_SK2_PF_L2_BF16",
    "W64_T16_S8_SK2_PF_L2_F16",
    "W64_T32_S8_SK2_PF_BF16",
    "PROGRAMS",
    "aot_specs",
]
