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
"""Generated program of the MiniMax-H3 dense BF16 self-attention: the persistent 128-row
Q-tile kernel with the ping-pong schedule between its two warp groups and the in-kernel
merge of the last wave's split K/V ranges (ATTENTION), shipped for sm_90a, sm_100a,
sm_103a, sm_120a and sm_121a. The host module
``flashinfer.diffusion_ops.cake_minimax_h3_dense_attention`` computes the launch plan
and the grid and launches it behind ``minimax_h3_dense_attention``.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

ATTENTION = cake_program.CakeProgram(
    name="cake_minimax_h3_dense_attention_1ec8efce8d0b0be5f54a",
    operator_dir="cake_minimax_h3_dense_attention",
    kernel="kernel_cake_minimax_h3_dense_attention_1ec8efce8d0b0be5f54a",
    sources=(
        "cake_minimax_h3_dense_attention_1ec8efce8d0b0be5f54a_kernel.cu",
        "cake_minimax_h3_dense_attention_1ec8efce8d0b0be5f54a_binding.cu",
    ),
    targets=("sm_90a", "sm_100a", "sm_103a", "sm_120a", "sm_121a"),
    compile_options=(
        ("sm_90a", ()),
        ("sm_100a", ()),
        ("sm_103a", ()),
        ("sm_120a", ()),
        ("sm_121a", ()),
    ),
    args=(
        "Q_map",
        "K_map",
        "V_map",
        "O",
        "partial_ws",
        "split_counters",
        "tokens",
        "num_q_tiles",
        "num_kv_tiles",
        "total_work",
        "split_base",
        "split_k",
        "split_tiles",
        "softmax_scale_log2",
        "q_scale_bf16x2",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=("Q_map", "K_map", "V_map", "O", "partial_ws", "split_counters"),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=99456,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (ATTENTION,)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["ATTENTION", "PROGRAMS", "aot_specs"]
