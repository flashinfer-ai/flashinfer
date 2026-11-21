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
"""Generated programs of the opt-in exact sequence split of the Cake GDN prefill on
sm_100a, sm_103a and sm_107a (BF16 Q/K/V/output, FP32 state, head size 128): pass 1
walks every chunk-aligned segment of a sequence in one wave (the first from the initial
state into the output, every other one from zeros and from the identity, into the local
output, its BF16 transfer and its two segment states), pass 2 composes the carried
states and corrects the outputs of every later segment, one program per compose slice
width (4, 8 and 16 value columns). The host module
``flashinfer.gdn_kernels.blackwell.cake_gdn_affine_split`` picks the segment cap, owns
the workspaces, computes the grids and launches the two programs of a call.

Generated file: every value below is derived from the exported sources and checked
against them; do not edit.
"""

from __future__ import annotations

from . import cake_program
from .core import JitSpec

PASS1 = cake_program.CakeProgram(
    name="cake_gdn_affine_split_03829716e38373da30f9",
    operator_dir="cake_gdn_affine_split",
    kernel="kernel_cake_gdn_affine_split_03829716e38373da30f9",
    sources=(
        "cake_gdn_affine_split_03829716e38373da30f9_kernel.cu",
        "cake_gdn_affine_split_03829716e38373da30f9_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--use_fast_math", "--fmad=true")),
        ("sm_103a", ("--use_fast_math", "--fmad=true")),
        ("sm_107a", ("--use_fast_math",)),
    ),
    args=(
        "Q",
        "K",
        "V",
        "O",
        "gate",
        "beta",
        "cu_seqlens",
        "state_indices",
        "initial_state",
        "output_state",
        "checkpoint_state",
        "cu_checkpoints",
        "tensormap_workspace",
        "initial_state_stride_slot",
        "output_state_stride_slot",
        "checkpoint_every_n_tokens",
        "scale",
        "num_seqs",
        "total_tokens",
        "num_q_heads",
        "num_v_heads",
        "total_tiles",
        "o_loc",
        "c",
        "bm",
        "p_cap",
        "c_min",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "Q",
        "K",
        "V",
        "O",
        "gate",
        "beta",
        "cu_seqlens",
        "state_indices",
        "initial_state",
        "output_state",
        "checkpoint_state",
        "cu_checkpoints",
        "tensormap_workspace",
        "o_loc",
        "c",
        "bm",
    ),
    block=(384, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=226048,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PASS2_W4 = cake_program.CakeProgram(
    name="cake_gdn_affine_split_dc6b2bd2dc435072698e",
    operator_dir="cake_gdn_affine_split",
    kernel="kernel_cake_gdn_affine_split_dc6b2bd2dc435072698e",
    sources=(
        "cake_gdn_affine_split_dc6b2bd2dc435072698e_kernel.cu",
        "cake_gdn_affine_split_dc6b2bd2dc435072698e_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--fmad=true",)),
        ("sm_103a", ("--fmad=true",)),
        ("sm_107a", ("--fmad=true",)),
    ),
    args=(
        "o_loc",
        "c_rows",
        "c_tma",
        "bm",
        "hpub",
        "hpub_tma",
        "output",
        "output_state",
        "flags",
        "ticket",
        "cu_seqlens",
        "output_state_stride_slot",
        "total_tokens",
        "num_seqs",
        "num_heads",
        "p_cap",
        "c_min",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "o_loc",
        "c_rows",
        "c_tma",
        "bm",
        "hpub",
        "hpub_tma",
        "output",
        "output_state",
        "flags",
        "ticket",
        "cu_seqlens",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=99456,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PASS2_W8 = cake_program.CakeProgram(
    name="cake_gdn_affine_split_4cd76cf613d87c68257c",
    operator_dir="cake_gdn_affine_split",
    kernel="kernel_cake_gdn_affine_split_4cd76cf613d87c68257c",
    sources=(
        "cake_gdn_affine_split_4cd76cf613d87c68257c_kernel.cu",
        "cake_gdn_affine_split_4cd76cf613d87c68257c_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--fmad=true",)),
        ("sm_103a", ("--fmad=true",)),
        ("sm_107a", ("--fmad=true",)),
    ),
    args=(
        "o_loc",
        "c_rows",
        "c_tma",
        "bm",
        "hpub",
        "hpub_tma",
        "output",
        "output_state",
        "flags",
        "ticket",
        "cu_seqlens",
        "output_state_stride_slot",
        "total_tokens",
        "num_seqs",
        "num_heads",
        "p_cap",
        "c_min",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "o_loc",
        "c_rows",
        "c_tma",
        "bm",
        "hpub",
        "hpub_tma",
        "output",
        "output_state",
        "flags",
        "ticket",
        "cu_seqlens",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=111744,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PASS2_W16 = cake_program.CakeProgram(
    name="cake_gdn_affine_split_4bb897eeba49c85dd937",
    operator_dir="cake_gdn_affine_split",
    kernel="kernel_cake_gdn_affine_split_4bb897eeba49c85dd937",
    sources=(
        "cake_gdn_affine_split_4bb897eeba49c85dd937_kernel.cu",
        "cake_gdn_affine_split_4bb897eeba49c85dd937_binding.cu",
    ),
    targets=("sm_100a", "sm_103a", "sm_107a"),
    compile_options=(
        ("sm_100a", ("--fmad=true",)),
        ("sm_103a", ("--fmad=true",)),
        ("sm_107a", ("--fmad=true",)),
    ),
    args=(
        "o_loc",
        "c_rows",
        "c_tma",
        "bm",
        "hpub",
        "hpub_tma",
        "output",
        "output_state",
        "flags",
        "ticket",
        "cu_seqlens",
        "output_state_stride_slot",
        "total_tokens",
        "num_seqs",
        "num_heads",
        "p_cap",
        "c_min",
        "grid_x",
        "grid_y",
        "grid_z",
    ),
    tensors=(
        "o_loc",
        "c_rows",
        "c_tma",
        "bm",
        "hpub",
        "hpub_tma",
        "output",
        "output_state",
        "flags",
        "ticket",
        "cu_seqlens",
    ),
    block=(256, 1, 1),
    cluster=(1, 1, 1),
    dynamic_smem_bytes=103552,
    use_pdl=False,
    cooperative=False,
    needs_device_linking=False,
)

PROGRAMS: tuple[cake_program.CakeProgram, ...] = (PASS1, PASS2_W4, PASS2_W8, PASS2_W16)


def aot_specs(target: str) -> tuple[JitSpec, ...]:
    """JIT specs of every program shipping ``target``, for the ahead-of-time build."""
    return cake_program.aot_specs(PROGRAMS, target)


__all__ = ["PASS1", "PASS2_W4", "PASS2_W8", "PASS2_W16", "PROGRAMS", "aot_specs"]
