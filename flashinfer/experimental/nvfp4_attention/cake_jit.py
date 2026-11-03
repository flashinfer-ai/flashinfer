"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Host plan and loader glue of the Cake experimental NVFP4 attention programs
(SM100, SM103, SM120), the ``backend="cake"`` of
:func:`flashinfer.prefill.prepare_nvfp4_attention`.

Contract of the route: ``out = bf16(softmax(q k^T * scale) v)`` over logical
``[B, H, S, 128]`` BF16 ``q`` / ``k`` / ``v`` / ``out`` with any strides (an
``[B, S, H, 128]`` buffer is passed as its ``[B, H, S, 128]`` view and read or
written in place), any ``S >= 1``, dense and noncausal, ``softmax_scale > 0``.
One call is a chain of programs per architecture group, launched in ``stage``
order under programmatic dependent launch: ``stats`` (per-(batch, head) K
means; on SM120 also the per-128-row-block Q means), ``quantize`` (E2M1 codes
with UE4M3 block scales of the centred Q and K and of the transposed V,
written to the caller's workspace; on SM100 / SM103 the two-level residual
form, whose second E2M1 level holds the residual of the first, plus the
TF32-rounded FP32 block means ``qm32`` and centred K rows ``kc32``), on SM100 /
SM103 ``delta`` (the exact FP32 block-mean compensation table ``qm32 . kc32``
as the attention's (hi, lo) BF16 seed-operand image ``tab``) and
``attention`` (the quantized attention, BF16 output).  :mod:`.cake_backend`
validates the tensors, carves the workspace, binds the logical arguments and
launches the chain; quantization happens inside every launch, so tensor
values are free between launches while identities and strides are bound.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the source ``module`` and ``build`` keywords it was
rendered from, its ``role`` and ``stage`` in the chain, its launch record, the
argument plan of its ``run`` entry, the kernel-parameter -> logical-argument
map over the vocabulary ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``, the element
type (``dtypes``: ``bf16``, ``f32``, ``u8``, ``u32`` or ``i32``) the kernel
declares for every buffer parameter -- the host views the carved region as
that type before the launch -- the grid rule :func:`launch_grid` evaluates, the selection rule (one build per
architecture group) and the workspace rule (``prepare_carved``: the regions a
``stats`` program writes and its self-resetting ``counters`` region, which
reads zero after every complete call; on SM100 / SM103 also the attention's
split-tile partials (``part_o``, ``part_ml``) and their self-resetting
``split_counters``; ``None`` elsewhere).  What to compile --
sources, digests, nvcc flags -- is recorded once in ``build.json`` next to
this file; :mod:`.cake_build` serves the library from a prebuilt ``aot/`` file
when a builder compiled it and otherwise through FlashInfer's JIT from the
same sources and flags.  Every ``build.json`` entry records the compile
options the program's kernel was validated with (``--use_fast_math`` on the
SM100 / SM103 delta and attention programs, none on the others); the loader passes
``use_fast_math=False`` so that the manifest, not the loader, states them.

Tensor maps travel by value in the kernel parameters: the binding encodes a
``CUtensorMap`` from each tensor view per call, so a launch is self-contained,
CUDA-graph capturable, and two layers never share descriptor storage.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_nvfp4_attention_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted,
every default spelled>, "module": <dotted source module>}`` of the kernel the
program was rendered from; a changed keyword is a new program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# The exported architectures and the device capabilities that select them.
ARCHES = ("sm_100a", "sm_103a", "sm_120a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (12, 0): "sm_120a",
}

# The head dimension of every record.
HEAD_DIM = 128
# The chain, in launch order; every architecture delivers one record per stage,
# except the optional stages, which an architecture group may leave out.
STAGES = ("stats", "quantize", "delta", "attention")
OPTIONAL_STAGES = ("delta",)

# The host vocabulary: the logical names ``cake_backend`` binds.  A record's
# ``logical`` map names, for every kernel parameter of its ``arg_plan``, the
# tensor (``buffer``) or scalar (``parameter``) it receives; ``num_kblocks``
# and ``num_units`` count the 128-row blocks of the packed (batch x token) axis
# and the (batch, head, Q block) units, ``q_rows_pad`` / ``kv_rows_pad`` the
# per-(batch, head) workspace rows of the Q and K/V sides, ``split_full_tiles`` /
# ``split_chunks`` the SM100 attention's last-wave split plan (``split_plan``);
# ``q4l`` / ``q_sf_l`` / ``k4l`` / ``k_sf_l`` are the SM100 / SM103 second E2M1
# level, ``qm32`` / ``kc32`` the table program's FP32 operands and ``tab`` its
# (hi, lo) BF16 seed-operand image.
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "out",
    "q4",
    "q_sf",
    "k4",
    "k_sf",
    "vt4",
    "v_sf",
    "v_sf_lo",
    "v_sf_hi",
    "q4l",
    "q_sf_l",
    "k4l",
    "k_sf_l",
    "qm32",
    "kc32",
    "tab",
    "mean_k",
    "q_mean",
    "partials",
    "counters",
    "unit_table",
    "q8",
    "k8",
    "qm8",
    "k_minmax",
    "q_amax",
    "qm_amax",
    "k_amax",
    "part_o",
    "part_ml",
    "split_counters",
)
LOGICAL_SCALARS = (
    "batch",
    "heads",
    "seqlen",
    "total_bh",
    "num_qblocks",
    "num_qblocks_pad",
    "num_kblocks",
    "num_stats_tiles",
    "num_items",
    "num_units",
    "split_full_tiles",
    "split_chunks",
    "q_rows_pad",
    "kv_rows_pad",
    "softmax_scale_log2",
    "q_stride_b",
    "q_stride_h",
    "q_stride_s",
    "k_stride_b",
    "k_stride_h",
    "k_stride_s",
    "v_stride_b",
    "v_stride_h",
    "v_stride_s",
    "o_stride_b",
    "o_stride_h",
    "o_stride_s",
)

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "stats_sm100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "K"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["buffer", "mean_k"],
            ["parameter", "seqlen"],
            ["parameter", "k_stride_s"],
            ["parameter", "k_stride_h"],
            ["parameter", "k_stride_b"],
            ["parameter", "heads"],
            ["parameter", "num_tiles"],
            ["parameter", "total_bh"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "early_launch": True,
        },
        "dtypes": {
            "K": "bf16",
            "counters": "u32",
            "mean_k": "f32",
            "partials": "f32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "capped_tiles",
            "pad": 512,
            "per_sm": 4,
            "rows": 512,
        },
        "kernel": "cake_nvfp4_attention_4899aa3c1a4719209562",
        "kernel_symbol": "kernel_cake_nvfp4_attention_4899aa3c1a4719209562",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8320,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_attention_4899aa3c1a4719209562",
        "logical": {
            "K": "k",
            "counters": "counters",
            "heads": "heads",
            "k_stride_b": "k_stride_b",
            "k_stride_h": "k_stride_h",
            "k_stride_s": "k_stride_s",
            "mean_k": "mean_k",
            "num_tiles": "num_stats_tiles",
            "partials": "partials",
            "seqlen": "seqlen",
            "total_bh": "total_bh",
        },
        "module": "tart.sage3_nvfp4_attention_stats.sm100_sm103.nvfp4_sage3_attention_stats",
        "role": "stats",
        "select_rule": None,
        "stage": 0,
        "workspace_rule": {
            "counters": "counters",
            "kind": "prepare_carved",
            "regions": [
                "partials",
                "counters",
                "mean_k",
            ],
        },
    },
    "quantize_sm100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "mean_k"],
            ["buffer", "q4"],
            ["buffer", "sfq"],
            ["buffer", "qm4"],
            ["buffer", "sfqm"],
            ["buffer", "k4"],
            ["buffer", "sfk"],
            ["buffer", "vt4"],
            ["buffer", "sfv_lo"],
            ["buffer", "sfv_hi"],
            ["parameter", "seqlen"],
            ["parameter", "q_stride"],
            ["parameter", "kv_stride"],
            ["parameter", "heads"],
            ["parameter", "num_q_blocks"],
            ["parameter", "total_bh"],
            ["parameter", "q_stride_s"],
            ["parameter", "q_stride_h"],
            ["parameter", "q_stride_b"],
            ["parameter", "k_stride_s"],
            ["parameter", "k_stride_h"],
            ["parameter", "k_stride_b"],
            ["parameter", "v_stride_s"],
            ["parameter", "v_stride_h"],
            ["parameter", "v_stride_b"],
            ["buffer", "q4l"],
            ["buffer", "sfq_l"],
            ["buffer", "k4l"],
            ["buffer", "sfk_l"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cp_stage": True,
            "early_launch": True,
            "qk_format": "nvfp4_res2d",
        },
        "dtypes": {
            "K": "bf16",
            "Q": "bf16",
            "V": "bf16",
            "k4": "u8",
            "k4l": "u8",
            "mean_k": "f32",
            "q4": "u8",
            "q4l": "u8",
            "qm4": "u8",
            "sfk": "u8",
            "sfk_l": "u8",
            "sfq": "u8",
            "sfq_l": "u8",
            "sfqm": "u8",
            "sfv_hi": "u8",
            "sfv_lo": "u8",
            "vt4": "u8",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "capped_tiles",
            "pad": 512,
            "per_sm": 8,
            "rows": 128,
        },
        "kernel": "cake_nvfp4_attention_f0a6fedc1ec9c401610b",
        "kernel_symbol": "kernel_cake_nvfp4_attention_f0a6fedc1ec9c401610b",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 75520,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_attention_f0a6fedc1ec9c401610b",
        "logical": {
            "K": "k",
            "Q": "q",
            "V": "v",
            "heads": "heads",
            "k4": "k4",
            "k4l": "k4l",
            "k_stride_b": "k_stride_b",
            "k_stride_h": "k_stride_h",
            "k_stride_s": "k_stride_s",
            "kv_stride": "kv_rows_pad",
            "mean_k": "mean_k",
            "num_q_blocks": "num_qblocks_pad",
            "q4": "q4",
            "q4l": "q4l",
            "q_stride": "q_rows_pad",
            "q_stride_b": "q_stride_b",
            "q_stride_h": "q_stride_h",
            "q_stride_s": "q_stride_s",
            "qm4": "qm32",
            "seqlen": "seqlen",
            "sfk": "k_sf",
            "sfk_l": "k_sf_l",
            "sfq": "q_sf",
            "sfq_l": "q_sf_l",
            "sfqm": "kc32",
            "sfv_hi": "v_sf_hi",
            "sfv_lo": "v_sf_lo",
            "total_bh": "total_bh",
            "v_stride_b": "v_stride_b",
            "v_stride_h": "v_stride_h",
            "v_stride_s": "v_stride_s",
            "vt4": "vt4",
        },
        "module": "tart.sage3_nvfp4_attention_quantize.sm100_sm103.nvfp4_sage3_attention_quantize",
        "role": "quantize",
        "select_rule": None,
        "stage": 1,
        "workspace_rule": None,
    },
    "delta_sm100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "QM"],
            ["buffer", "KC"],
            ["buffer", "tab"],
            ["parameter", "kv_stride"],
            ["parameter", "num_q_blocks"],
            ["parameter", "total_bh"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "early_launch": True,
        },
        "dtypes": {
            "KC": "f32",
            "QM": "f32",
            "tab": "bf16",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "blocks_per_tile": 128,
            "key_tile": 128,
            "kind": "capped_table_tiles",
            "q_block": 128,
            "q_pad": 512,
        },
        "kernel": "cake_nvfp4_attention_52f4307e421203d6944d",
        "kernel_symbol": "kernel_cake_nvfp4_attention_52f4307e421203d6944d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_attention_52f4307e421203d6944d",
        "logical": {
            "KC": "kc32",
            "QM": "qm32",
            "kv_stride": "kv_rows_pad",
            "num_q_blocks": "num_qblocks_pad",
            "tab": "tab",
            "total_bh": "total_bh",
        },
        "module": "tart.sage3_nvfp4_attention_delta.sm100_sm103.nvfp4_sage3_attention_delta",
        "role": "delta",
        "select_rule": None,
        "stage": 2,
        "workspace_rule": None,
    },
    "attention_sm100": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "SFQ"],
            ["buffer", "SFK"],
            ["buffer", "SFVtLo"],
            ["buffer", "SFVtHi"],
            ["buffer", "QM"],
            ["buffer", "QL"],
            ["buffer", "SFQL"],
            ["buffer", "KL"],
            ["buffer", "SFKL"],
            ["buffer", "O"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_kv"],
            ["parameter", "q_stride"],
            ["parameter", "kv_stride"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "total_bh"],
            ["parameter", "heads"],
            ["parameter", "n_full_tiles"],
            ["parameter", "n_chunks"],
            ["buffer", "part_o"],
            ["buffer", "part_ml"],
            ["buffer", "split_counters"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "exp2_emu_pairs": [
                6,
                7,
            ],
            "qk_format": "nvfp4_res2d",
        },
        "dtypes": {
            "K": "u8",
            "KL": "u8",
            "O": "bf16",
            "Q": "u8",
            "QL": "u8",
            "QM": "bf16",
            "SFK": "u8",
            "SFKL": "u8",
            "SFQ": "u8",
            "SFQL": "u8",
            "SFVtHi": "u8",
            "SFVtLo": "u8",
            "Vt": "u8",
            "k_scale": "f32",
            "part_ml": "f32",
            "part_o": "f32",
            "q_scale": "f32",
            "split_counters": "u32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_tile": 2,
            "key_block": 128,
            "kind": "split_persistent_tiles",
            "max_chunks": 4,
            "tile_rows": 512,
        },
        "kernel": "cake_nvfp4_attention_28a3f583a040bde34bf1",
        "kernel_symbol": "kernel_cake_nvfp4_attention_28a3f583a040bde34bf1",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232448,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_attention_28a3f583a040bde34bf1",
        "logical": {
            "K": "k4",
            "KL": "k4l",
            "O": "out",
            "Q": "q4",
            "QL": "q4l",
            "QM": "tab",
            "SFK": "k_sf",
            "SFKL": "k_sf_l",
            "SFQ": "q_sf",
            "SFQL": "q_sf_l",
            "SFVtHi": "v_sf_hi",
            "SFVtLo": "v_sf_lo",
            "Vt": "vt4",
            "heads": "heads",
            "k_scale": "mean_k",
            "kv_stride": "kv_rows_pad",
            "n_chunks": "split_chunks",
            "n_full_tiles": "split_full_tiles",
            "part_ml": "part_ml",
            "part_o": "part_o",
            "q_scale": "mean_k",
            "q_stride": "q_rows_pad",
            "seqlen_kv": "seqlen",
            "seqlen_q": "seqlen",
            "softmax_scale_log2": "softmax_scale_log2",
            "split_counters": "split_counters",
            "total_bh": "total_bh",
        },
        "module": "tart.attention_fwd_sage3_nvfp4_bf16o.sm100_sm103.nvfp4_sage3_attention",
        "role": "attention",
        "select_rule": {
            "kind": "always",
        },
        "stage": 3,
        "workspace_rule": {
            "counters": "split_counters",
            "kind": "prepare_carved",
            "regions": [
                "part_o",
                "part_ml",
                "split_counters",
            ],
        },
    },
    "attention_sm103": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "SFQ"],
            ["buffer", "SFK"],
            ["buffer", "SFVtLo"],
            ["buffer", "SFVtHi"],
            ["buffer", "QM"],
            ["buffer", "QL"],
            ["buffer", "SFQL"],
            ["buffer", "KL"],
            ["buffer", "SFKL"],
            ["buffer", "O"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_kv"],
            ["parameter", "q_stride"],
            ["parameter", "kv_stride"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "total_bh"],
            ["parameter", "heads"],
            ["parameter", "n_full_tiles"],
            ["parameter", "n_chunks"],
            ["buffer", "part_o"],
            ["buffer", "part_ml"],
            ["buffer", "split_counters"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "exp2_emu_pairs": [],
            "qk_format": "nvfp4_res2d",
        },
        "dtypes": {
            "K": "u8",
            "KL": "u8",
            "O": "bf16",
            "Q": "u8",
            "QL": "u8",
            "QM": "bf16",
            "SFK": "u8",
            "SFKL": "u8",
            "SFQ": "u8",
            "SFQL": "u8",
            "SFVtHi": "u8",
            "SFVtLo": "u8",
            "Vt": "u8",
            "k_scale": "f32",
            "part_ml": "f32",
            "part_o": "f32",
            "q_scale": "f32",
            "split_counters": "u32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_tile": 2,
            "key_block": 128,
            "kind": "split_persistent_tiles",
            "max_chunks": 4,
            "tile_rows": 512,
        },
        "kernel": "cake_nvfp4_attention_4ba55dcb40d27804c77b",
        "kernel_symbol": "kernel_cake_nvfp4_attention_4ba55dcb40d27804c77b",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232448,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_attention_4ba55dcb40d27804c77b",
        "logical": {
            "K": "k4",
            "KL": "k4l",
            "O": "out",
            "Q": "q4",
            "QL": "q4l",
            "QM": "tab",
            "SFK": "k_sf",
            "SFKL": "k_sf_l",
            "SFQ": "q_sf",
            "SFQL": "q_sf_l",
            "SFVtHi": "v_sf_hi",
            "SFVtLo": "v_sf_lo",
            "Vt": "vt4",
            "heads": "heads",
            "k_scale": "mean_k",
            "kv_stride": "kv_rows_pad",
            "n_chunks": "split_chunks",
            "n_full_tiles": "split_full_tiles",
            "part_ml": "part_ml",
            "part_o": "part_o",
            "q_scale": "mean_k",
            "q_stride": "q_rows_pad",
            "seqlen_kv": "seqlen",
            "seqlen_q": "seqlen",
            "softmax_scale_log2": "softmax_scale_log2",
            "split_counters": "split_counters",
            "total_bh": "total_bh",
        },
        "module": "tart.attention_fwd_sage3_nvfp4_bf16o.sm100_sm103.nvfp4_sage3_attention",
        "role": "attention",
        "select_rule": {
            "kind": "always",
        },
        "stage": 3,
        "workspace_rule": {
            "counters": "split_counters",
            "kind": "prepare_carved",
            "regions": [
                "part_o",
                "part_ml",
                "split_counters",
            ],
        },
    },
    "stats_sm120": {
        "arches": [
            "sm_120a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["buffer", "mean_k"],
            ["buffer", "q_mean"],
            ["buffer", "k_minmax"],
            ["buffer", "q_amax"],
            ["buffer", "qm_amax"],
            ["buffer", "k_amax"],
            ["parameter", "seq_len"],
            ["parameter", "num_heads"],
            ["parameter", "num_qblocks"],
            ["parameter", "num_kblocks"],
            ["parameter", "num_items"],
            ["parameter", "q_stride_b"],
            ["parameter", "q_stride_h"],
            ["parameter", "q_stride_s"],
            ["parameter", "k_stride_b"],
            ["parameter", "k_stride_h"],
            ["parameter", "k_stride_s"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "qk_format": "int8_blk",
        },
        "dtypes": {
            "K": "bf16",
            "Q": "bf16",
            "counters": "u32",
            "k_amax": "f32",
            "k_minmax": "f32",
            "mean_k": "f32",
            "partials": "f32",
            "q_amax": "f32",
            "q_mean": "f32",
            "qm_amax": "f32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "capped_tiles",
            "pad": 128,
            "per_sm": 4,
            "rows": 128,
        },
        "kernel": "cake_nvfp4_attention_108da8395c2b15986349",
        "kernel_symbol": "kernel_cake_nvfp4_attention_108da8395c2b15986349",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 49792,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_attention_108da8395c2b15986349",
        "logical": {
            "K": "k",
            "Q": "q",
            "counters": "counters",
            "k_amax": "k_amax",
            "k_minmax": "k_minmax",
            "k_stride_b": "k_stride_b",
            "k_stride_h": "k_stride_h",
            "k_stride_s": "k_stride_s",
            "mean_k": "mean_k",
            "num_heads": "heads",
            "num_items": "num_items",
            "num_kblocks": "num_kblocks",
            "num_qblocks": "num_qblocks",
            "partials": "partials",
            "q_amax": "q_amax",
            "q_mean": "q_mean",
            "q_stride_b": "q_stride_b",
            "q_stride_h": "q_stride_h",
            "q_stride_s": "q_stride_s",
            "qm_amax": "qm_amax",
            "seq_len": "seqlen",
        },
        "module": "tart.sage3_nvfp4_attention_stats.sm120.sm120_nvfp4_sage3_attention_stats",
        "role": "stats",
        "select_rule": None,
        "stage": 0,
        "workspace_rule": {
            "counters": "counters",
            "kind": "prepare_carved",
            "regions": [
                "partials",
                "counters",
                "mean_k",
                "q_mean",
                "k_minmax",
                "q_amax",
                "qm_amax",
                "k_amax",
            ],
        },
    },
    "quantize_sm120": {
        "arches": [
            "sm_120a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "mean_k"],
            ["buffer", "q_mean"],
            ["buffer", "q_amax"],
            ["buffer", "qm_amax"],
            ["buffer", "k_amax"],
            ["buffer", "q8"],
            ["buffer", "k8"],
            ["buffer", "qm8"],
            ["buffer", "vt4"],
            ["buffer", "v_sf"],
            ["parameter", "seq_len"],
            ["parameter", "seq_pad"],
            ["parameter", "num_heads"],
            ["parameter", "num_qblocks"],
            ["parameter", "num_kblocks"],
            ["parameter", "num_items"],
            ["parameter", "q_stride_b"],
            ["parameter", "q_stride_h"],
            ["parameter", "q_stride_s"],
            ["parameter", "k_stride_b"],
            ["parameter", "k_stride_h"],
            ["parameter", "k_stride_s"],
            ["parameter", "v_stride_b"],
            ["parameter", "v_stride_h"],
            ["parameter", "v_stride_s"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "qk_format": "int8_blk",
        },
        "dtypes": {
            "K": "bf16",
            "Q": "bf16",
            "V": "bf16",
            "k8": "u32",
            "k_amax": "f32",
            "mean_k": "f32",
            "q8": "u32",
            "q_amax": "f32",
            "q_mean": "f32",
            "qm8": "u32",
            "qm_amax": "f32",
            "v_sf": "u8",
            "vt4": "u32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "capped_tiles",
            "pad": 128,
            "per_sm": 8,
            "rows": 32,
        },
        "kernel": "cake_nvfp4_attention_d0deb312a52a2ad18638",
        "kernel_symbol": "kernel_cake_nvfp4_attention_d0deb312a52a2ad18638",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_attention_d0deb312a52a2ad18638",
        "logical": {
            "K": "k",
            "Q": "q",
            "V": "v",
            "k8": "k8",
            "k_amax": "k_amax",
            "k_stride_b": "k_stride_b",
            "k_stride_h": "k_stride_h",
            "k_stride_s": "k_stride_s",
            "mean_k": "mean_k",
            "num_heads": "heads",
            "num_items": "num_items",
            "num_kblocks": "num_kblocks",
            "num_qblocks": "num_qblocks",
            "q8": "q8",
            "q_amax": "q_amax",
            "q_mean": "q_mean",
            "q_stride_b": "q_stride_b",
            "q_stride_h": "q_stride_h",
            "q_stride_s": "q_stride_s",
            "qm8": "qm8",
            "qm_amax": "qm_amax",
            "seq_len": "seqlen",
            "seq_pad": "kv_rows_pad",
            "v_sf": "v_sf",
            "v_stride_b": "v_stride_b",
            "v_stride_h": "v_stride_h",
            "v_stride_s": "v_stride_s",
            "vt4": "vt4",
        },
        "module": "tart.sage3_nvfp4_attention_quantize.sm120.sm120_nvfp4_sage3_attention_quantize",
        "role": "quantize",
        "select_rule": None,
        "stage": 1,
        "workspace_rule": None,
    },
    "attention_sm120": {
        "arches": [
            "sm_120a",
        ],
        "arg_plan": [
            ["buffer", "Q_map"],
            ["buffer", "K_map"],
            ["buffer", "VT_map"],
            ["buffer", "VSF_map"],
            ["buffer", "O"],
            ["buffer", "qm8"],
            ["buffer", "q_amax"],
            ["buffer", "qm_amax"],
            ["buffer", "k_amax"],
            ["buffer", "unit_table"],
            ["parameter", "total_units"],
            ["parameter", "num_heads"],
            ["parameter", "num_qblocks"],
            ["parameter", "num_kblocks"],
            ["parameter", "o_stride_b"],
            ["parameter", "o_stride_h"],
            ["parameter", "o_stride_s"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "kv_stages": 2,
            "lazy_rescale": True,
            "pingpong": True,
            "qk_format": "int8_blk",
            "split_refill": True,
        },
        "dtypes": {
            "K_map": "u8",
            "O": "bf16",
            "Q_map": "u8",
            "VSF_map": "u8",
            "VT_map": "u8",
            "k_amax": "f32",
            "q_amax": "f32",
            "qm8": "u32",
            "qm_amax": "f32",
            "unit_table": "i32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "capped_persistent",
            "unit_rows": 128,
        },
        "kernel": "cake_nvfp4_attention_94fb7e2f8ee5c56bd085",
        "kernel_symbol": "kernel_cake_nvfp4_attention_94fb7e2f8ee5c56bd085",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 69632,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_attention_94fb7e2f8ee5c56bd085",
        "logical": {
            "K_map": "k8",
            "O": "out",
            "Q_map": "q8",
            "VSF_map": "v_sf",
            "VT_map": "vt4",
            "k_amax": "k_amax",
            "num_heads": "heads",
            "num_kblocks": "num_kblocks",
            "num_qblocks": "num_qblocks",
            "o_stride_b": "o_stride_b",
            "o_stride_h": "o_stride_h",
            "o_stride_s": "o_stride_s",
            "q_amax": "q_amax",
            "qm8": "qm8",
            "qm_amax": "qm_amax",
            "softmax_scale_log2": "softmax_scale_log2",
            "total_units": "num_units",
            "unit_table": "unit_table",
        },
        "module": "tart.attention_fwd_sage3_nvfp4_bf16o.sm120.sm120_nvfp4_sage3_attention",
        "role": "attention",
        "select_rule": {
            "kind": "always",
        },
        "stage": 3,
        "workspace_rule": None,
    },
}


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target; the capability is immutable per process)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake NVFP4 attention target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def records_for(arch: str) -> dict[str, str]:
    """``stage -> record name`` of the chain delivered for ``arch``, in launch
    order (``RuntimeError`` unless exactly one record per stage ships for it; a
    stage of ``OPTIONAL_STAGES`` may have none and is then absent)."""
    found: dict[str, list[str]] = {stage: [] for stage in STAGES}
    for name, record in MODULES.items():
        if arch in record["arches"]:
            found.setdefault(record["role"], []).append(name)
    chain: dict[str, str] = {}
    for stage in STAGES:
        names = found.get(stage, [])
        if len(names) > 1 or (not names and stage not in OPTIONAL_STAGES):
            raise RuntimeError(
                f"Cake NVFP4 attention delivers {len(names)} {stage!r} programs for {arch}; "
                f"the exported architectures are {ARCHES}"
            )
        if names:
            chain[stage] = names[0]
    return chain


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake NVFP4 attention record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


def split_plan(
    record: dict[str, Any], *, batch: int, heads: int, seqlen: int, sm_count: int
) -> dict[str, int]:
    """The last-wave split plan of a ``split_persistent_tiles`` record for
    ``batch x heads`` heads of ``seqlen`` rows on ``sm_count`` SMs: ``n_clusters``
    persistent clusters of ``ctas_per_tile`` CTAs work through ``total_tiles``
    tiles of ``tile_rows`` rows; the ``n_split`` tiles of the partial last wave
    are split into ``n_chunks`` key chunks each (at most ``max_chunks``, at most
    ``key_blocks`` = ceil(S / key_block)); the work items are the
    ``n_full_tiles`` whole tiles, then ``n_split * n_chunks`` chunk items
    (``n_items``); ``grid`` is the launch grid.  The chunk items' partial
    outputs occupy ``n_split * n_chunks`` rows of the ``part_o`` / ``part_ml``
    workspace regions and ``2 * n_split`` self-resetting ``split_counters``
    (:func:`.cake_backend.workspace_layout` rounds both up to at least one), so
    the workspace is sized for the SM count of the device it was carved for."""
    rule = record["grid_rule"]
    if rule["kind"] != "split_persistent_tiles":
        raise ValueError(
            f"split_plan reads a 'split_persistent_tiles' rule, got {rule['kind']!r}"
        )
    batch, heads, seqlen, sm_count = int(batch), int(heads), int(seqlen), int(sm_count)
    if min(batch, heads, seqlen, sm_count) <= 0:
        raise ValueError(
            f"the grid rule needs positive extents, got batch={batch}, heads={heads}, "
            f"seqlen={seqlen}, sm_count={sm_count}"
        )
    ctas, tile_rows = int(rule["ctas_per_tile"]), int(rule["tile_rows"])
    key_block, max_chunks = int(rule["key_block"]), int(rule["max_chunks"])
    if sm_count < ctas:
        raise ValueError(
            f"the persistent cluster needs at least {ctas} SMs, got {sm_count}"
        )
    n_clusters = max(1, sm_count // ctas)
    total_tiles = batch * heads * -(-seqlen // tile_rows)
    key_blocks = -(-seqlen // key_block)
    n_split = total_tiles % n_clusters
    n_chunks = min(max_chunks, key_blocks, n_clusters // n_split) if n_split else 1
    if n_chunks < 2:
        n_split, n_chunks = 0, 1
    n_full_tiles = total_tiles - n_split
    n_items = n_full_tiles + n_split * n_chunks
    return {
        "n_clusters": n_clusters,
        "total_tiles": total_tiles,
        "key_blocks": key_blocks,
        "n_full_tiles": n_full_tiles,
        "n_split": n_split,
        "n_chunks": n_chunks,
        "n_items": n_items,
        "grid": ctas * min(n_clusters, n_items),
    }


def launch_grid(
    record: dict[str, Any], *, batch: int, heads: int, seqlen: int, sm_count: int
) -> tuple[int, int, int]:
    """Launch grid of one record for ``batch x heads`` heads of ``seqlen`` rows
    on a device of ``sm_count`` SMs under its grid rule (one-dimensional grids):
    ``persistent_tiles`` -- a persistent cluster of ``ctas_per_tile`` CTAs per
    ``tile_rows``-row tile, at most one cluster per ``ctas_per_tile`` SMs;
    ``split_persistent_tiles`` -- the same over the work items of
    :func:`split_plan` (whole tiles, then the key chunks of the last wave);
    ``capped_persistent`` -- one CTA per ``unit_rows``-row unit, at most one per
    SM; ``batch_heads`` -- one CTA per (batch, head); ``batch_heads_tiles`` --
    one CTA per ``rows``-row tile of every ``pad``-padded head; ``capped_tiles``
    -- the same, at most ``per_sm`` per SM; ``capped_table_tiles`` -- one CTA
    per (batch, head, ``blocks_per_tile``-block tile of the ``q_pad``-padded
    ``q_block``-row Q blocks, ``key_tile``-key tile), at most one per SM."""
    rule = record["grid_rule"]
    kind = rule["kind"]
    batch, heads, seqlen, sm_count = int(batch), int(heads), int(seqlen), int(sm_count)
    if min(batch, heads, seqlen, sm_count) <= 0:
        raise ValueError(
            f"the grid rule needs positive extents, got batch={batch}, heads={heads}, "
            f"seqlen={seqlen}, sm_count={sm_count}"
        )
    bh = batch * heads
    if kind == "persistent_tiles":
        ctas, tile_rows = int(rule["ctas_per_tile"]), int(rule["tile_rows"])
        if sm_count < ctas:
            raise ValueError(
                f"the persistent cluster needs at least {ctas} SMs, got {sm_count}"
            )
        return (ctas * min(sm_count // ctas, bh * -(-seqlen // tile_rows)), 1, 1)
    if kind == "split_persistent_tiles":
        plan = split_plan(
            record, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
        )
        return (plan["grid"], 1, 1)
    if kind == "capped_persistent":
        return (max(1, min(sm_count, bh * -(-seqlen // int(rule["unit_rows"])))), 1, 1)
    if kind == "batch_heads":
        return (bh, 1, 1)
    if kind == "capped_table_tiles":
        q_pad, q_block = int(rule["q_pad"]), int(rule["q_block"])
        blocks_per_tile, key_tile = int(rule["blocks_per_tile"]), int(rule["key_tile"])
        q_blocks = -(-seqlen // q_pad) * (q_pad // q_block)
        items = bh * -(-q_blocks // blocks_per_tile) * -(-seqlen // key_tile)
        return (max(1, min(items, sm_count)), 1, 1)
    if kind in ("batch_heads_tiles", "capped_tiles"):
        rows, pad = int(rule["rows"]), int(rule["pad"])
        tiles = bh * -(-seqlen // pad) * (pad // rows)
        if kind == "batch_heads_tiles":
            return (tiles, 1, 1)
        return (max(1, min(tiles, int(rule["per_sm"]) * sm_count)), 1, 1)
    raise ValueError(f"unknown Cake NVFP4 attention grid rule: {kind!r}")


def launch_args(
    record: dict[str, Any], values: dict[str, Any], grid: tuple[int, int, int]
) -> tuple[Any, ...]:
    """The positional arguments of the record's ``run`` entry: ``values`` by the
    logical name of each kernel parameter of ``arg_plan`` and the grid axes."""
    logical = record["logical"]
    args: list[Any] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(int(grid["xyz".index(key[-1])]))
            continue
        if key not in logical:
            raise ValueError(f"kernel parameter {key!r} has no logical name")
        try:
            args.append(values[logical[key]])
        except KeyError:
            raise ValueError(
                f"the launch needs logical argument {logical[key]!r}"
            ) from None
    return tuple(args)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid`` (the
    record's :func:`launch_grid`) on the current torch stream of the device."""
    record = MODULES[name]
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*launch_args(record, values, grid))


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HEAD_DIM",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "OPTIONAL_STAGES",
    "STAGES",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "records_for",
    "run_module",
    "sm_count",
    "specs_for_arch",
    "split_plan",
]
