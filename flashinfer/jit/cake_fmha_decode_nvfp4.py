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
"""Generated program of the paged decode with an E4M3 query, NVFP4 K/V pages and an E4M3
output of the cake_fmha module (head dim 128, 16-token HND pages, 128 requests, 32 query
heads over 4 KV heads), shipped for sm_100a and sm_103a and built with the CUDA 13.4
toolchain's native E2M1 conversion. The host module ``flashinfer.cake_fmha`` selects it
from the batch, the head counts, the page size and the split plan of a launch (one
split), computes the grid and launches it; other shapes keep the module's other decode
routes.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

DEC_NVFP4_B128 = cake_program.CakeProgram(
    name="cake_fmha_decode_nvfp4_bcfe92828a999ce45ff4",
    operator_dir="cake_fmha_decode_nvfp4",
    kernel="kernel_cake_fmha_decode_nvfp4_bcfe92828a999ce45ff4",
    sources=(
        "cake_fmha_decode_nvfp4_bcfe92828a999ce45ff4_kernel.cu",
        "cake_fmha_decode_nvfp4_bcfe92828a999ce45ff4_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "Qt",
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
        "pt_batch_stride",
        "pt_v_offset",
        "bmm1_is_log2",
        "num_splits",
        "blocks_per_split",
        "bmm1_scalar",
        "bmm2_scalar",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Qt",
        "Kp",
        "Vp",
        "Ksf",
        "Vsf",
        "O",
        "page_table",
        "seq_lens_kv",
        "bmm1_scale_ptr",
        "bmm2_scale_ptr",
        "partial_O",
        "partial_max",
        "partial_sum",
    ),
    block=(512, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=147456,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (DEC_NVFP4_B128,)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["DEC_NVFP4_B128", "PROGRAMS", "aot_specs"]
