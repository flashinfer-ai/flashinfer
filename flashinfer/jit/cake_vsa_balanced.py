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
"""Generated program of the shared-selection block-64 balanced attention of the Cake block-
sparse attention module: one program, shipped for sm_100a and sm_103a. A persistent CTA
per SM walks the (query block, head) tiles of one BF16 head-dim-128 selection shared by
every head, four selected 64-key blocks per M64N256 tile; a ragged selection of more
than one wave is handed out in descending selection size through one global ticket
counter that the kernel resets for the next launch. The host module
``flashinfer.cake_vsa`` selects the program at plan time (a shared selection of at least
seven blocks per query block on average over at least eight (query block, head) tiles;
on sm_100a, where the per-head plan is the M64N256 WS program, only at least 0.8 tiles
per SM and rows of at most 64 blocks in one group-count bucket or rows in several
buckets over at least 1.05 waves) and launches the grid (min(query blocks x heads, SMs),
1, 1); every other block-64 plan keeps the module's per-head programs.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

VSA_BLK64_BALANCED_BF16 = cake_program.CakeProgram(
    name="cake_vsa_balanced_058db78861db239ba714",
    operator_dir="cake_vsa_balanced",
    kernel="kernel_cake_vsa_balanced_058db78861db239ba714",
    sources=(
        "cake_vsa_balanced_058db78861db239ba714_kernel.cu",
        "cake_vsa_balanced_058db78861db239ba714_binding.cu",
    ),
    targets=("sm_100a", "sm_103a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
    ),
    args=(
        "q",
        "k",
        "v",
        "out",
        "lse",
        "indptr",
        "indices",
        "sequence_q",
        "query_blocks",
        "total_tiles",
        "num_heads",
        "softmax_scale_log2",
        "return_lse",
        "queue_counters",
        "order_debug",
        "debug_order",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "q",
        "k",
        "v",
        "out",
        "lse",
        "indptr",
        "indices",
        "queue_counters",
        "order_debug",
    ),
    block=(384, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=232320,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (VSA_BLK64_BALANCED_BF16,)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["VSA_BLK64_BALANCED_BF16", "PROGRAMS", "aot_specs"]
