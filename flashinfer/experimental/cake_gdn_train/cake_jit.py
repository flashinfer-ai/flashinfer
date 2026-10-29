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

Registry of the Cake packed GDN training kernels of the context-parallel (KCP)
route (SM100 / SM103 / SM107).  The loader glue (architecture lookup, manifest
loading, argument binding and launching) is :mod:`.cake_backend`.

The registry holds the program names, the logical argument vocabulary and the
argument tables of the nine programs, and the grid and
selection rules.  The derived leaves of every record (``kernel_symbol``,
``launch``, ``arg_plan``, ``logical``) are written by the exporter from the
rendered binding; the ``library`` names are the export family's frozen
identities (20 hex digits after the package prefix), and the loader refuses
every record until ``build.json`` / ``build.test.json`` is rendered beside this
registry.  ``MODULES`` and ``TEST_MODULES`` are literal tables (the exporter
and the manifest checker read them with ``ast.literal_eval``).  The
fake backend (:mod:`.cake_fake_backend`) implements the same records with the
same logical arguments, so the host swaps backends by one switch.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan: the program's ``library`` stem
(its ``build.json`` entries are ``<library>_<arch>``), the architectures it is
delivered for, the ``program`` it fills and its ``variant``, the ``select_rule``
that picks one record of a program from the host-known scalars, the
``grid_rule`` that shapes its launch, the kernel-parameter -> logical-argument
map and the argument plan of its ``run`` entry.  No rule reads a tensor.

Programs (``*`` = new program package, ``=`` = reused verbatim):

  #1 * ``fwd_prep``          per chunk: gcum, gamma, the WY inverse ``Akk``, ``w``, ``u``, ``Aqk``; grid (NC, H_qk)
  #2 * ``kcp_fwdh_offset``   forward state walk from zero over the listed segments -> ``B_f`` (no tapes)
     * ``kcp_fwdh_tape``     forward state walk from ``S_in`` -> ``h`` / ``v_new`` tapes
                             (each form at value width ``vh`` 128 and, as the ``_vh64`` record with two CTAs per
                             (segment, head), 64: the host picks the ``_vh64`` record per launch when
                             ``n_walk x HV <= sm_count``; both forms write bitwise-identical tapes / pools)
  #3 * ``kcp_transfer_walk`` transfer walk ``M <- gamma_c M - K^T diag(rowscale) (W M)`` (TF32 hi/lo products on
                             eight compute warps, a column half each; test twins ``_three_mma`` / ``_plain_tf32``
                             (four-warp builds, bitwise the same results) in ``build.test.json``; no bf16x3 module)
  #4 = ``kcp_affine_merge``  FP32 FFMA affine chains (``_c8`` state mode / ``_c16`` map mode)
  #5 * ``fwd_o``             ``o = qscale (q h_c) + (Aqk . mask) v_new``; persistent grid min(NC x HV, SMs), the
                             tiles beyond the grid drawn in units from the ``ticket`` word
  #6 * ``bwd_prep``          ``do_g``, ``dv_intra``; persistent grid min(NC x HV, ctas_per_sm x SMs) with two
                             CTAs per SM on sm_100a, sm_103a and sm_107a (the record's per-architecture
                             ``ctas_per_sm``), items beyond the grid drawn from the ``ticket`` word
  #7 * ``kcp_dhu_offset``    adjoint walk from zero -> ``d_f``
     * ``kcp_dhu_tape``      adjoint walk from ``dS_out`` -> ``dh`` / ``dv_new`` tapes
                             (each form at value width ``vh`` 128 and, as the ``_vh64`` record with two CTAs per
                             (segment, head), 64; both forms write bitwise-identical tapes / pools)
  #8 * ``bwd_dqkg``          dq, dk (grouped FP32 reduction, one BF16 rounding), dv, dbeta, dg partials; persistent grid min(NC x H_qk, SMs),
                             the (chunk, Q/K head) groups drawn from the ``ticket`` word
  #9 * ``gate_epilogue``     reverse cumsum of the dg partials + the chunk term; grid ceil(NC x HV / 128)

Provenance of the rules: ``flashinfer/experimental/cake_kda_train/cake_kcp_jit.py`` (records_of_program,
select_record, launch_grid; sha256 c405e7dcd36c9cbe0ff01e3e3194fd760491504655bda28a646c2067a23b4c90) at
FlashInfer 13d7fcf4041635f5b7f573bb0dbab23570ae1034, adapted to the GDN vocabulary; the loader glue moved to
:mod:`.cake_backend`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# One verified build per architecture: B200 (``sm_100a``), B300 / GB300 (``sm_103a``) and R200 (``sm_107a``).
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
CHUNK = 64
HEAD_DIM = 128
TILE = HEAD_DIM * HEAD_DIM  # elements of one FP32 state tile [128, 128]
PACKAGE = "flashinfer.experimental.cake_gdn_train"
LIBRARY_PREFIX = "cake_gdn_train"
MANIFEST = "build.json"
TEST_MANIFEST = "build.test.json"
# Static knobs of the transfer walk (#3): ``precision_mode``
# 0 = the shipped TF32 hi/lo program (bitwise = the 3-MMA TF32x3 class), 1 = ``_three_mma`` (its explicit three-MMA
# twin; test-only), 2 = ``_plain_tf32`` (negative control; test-only); ``transfer_walk`` ``"tf32_hilo"`` = the shipped
# program, ``"bf16x3"`` = the BF16x3 test module of the KDA template (never a default, never a shipped
# lever) -- NOT BUILT in this package (no GDN bf16x3 walk is rendered; the value stays in the
# vocabulary and the validator / ``select_record`` reject it as not built).
# ``offset_class`` (design D-9): ``"fla"`` = the offset walks from zero with the local rounding
# points (default), ``"fp32"`` = the FP32-class ``B_f`` of #3 (``init_mode=1, use_u=1``).  Non-default values of the
# two ``TEST_KNOBS`` name ``build.test.json`` records and are selected only through the options' ``test_knobs``
# (the options validator); no environment variable anywhere.
PRECISION_MODES = (0, 1, 2)
PRECISION_MODE_DEFAULT = 0
TRANSFER_WALKS = ("tf32_hilo", "bf16x3")
TRANSFER_WALK_DEFAULT = "tf32_hilo"
OFFSET_CLASSES = ("fla", "fp32")
OFFSET_CLASS_DEFAULT = "fla"
TEST_KNOBS = ("precision_mode", "transfer_walk")
TRANSFER_WALKS_BUILT = ("tf32_hilo",)  # the only transfer-walk program family rendered

# The programs by the names the host uses (design # in PROGRAM_TABLE).
PROGRAMS = (
    "fwd_prep",
    "kcp_fwdh_offset",
    "kcp_fwdh_tape",
    "kcp_transfer_walk",
    "kcp_affine_merge",
    "fwd_o",
    "bwd_prep",
    "kcp_dhu_offset",
    "kcp_dhu_tape",
    "bwd_dqkg",
    "gate_epilogue",
)
WALK_PROGRAMS = (
    "kcp_fwdh_offset",
    "kcp_fwdh_tape",
    "kcp_transfer_walk",
    "kcp_dhu_offset",
    "kcp_dhu_tape",
)
LOCAL_PROGRAMS = ("fwd_prep", "fwd_o", "bwd_prep", "bwd_dqkg", "gate_epilogue")

# Logical argument names :func:`.cake_backend.run_module` binds; every record maps its kernel parameter names onto this
# vocabulary (``record["logical"]``).  Operands: ``q`` / ``k`` bf16 ``[T, H_qk, 128]``, ``v`` / ``do``
# bf16 ``[T, HV, 128]``, ``g`` / ``beta`` f32 ``[T, HV]`` (token strides are TMA parameters: the packed QKV view is
# zero-copy).  Tapes (frozen set, §7.1): ``w`` / ``u`` / ``v_new`` / ``do_g`` / ``dv_intra`` / ``dv_new`` bf16
# ``[R, HV, 128]``, ``aqk`` / ``akk`` bf16 ``[R, HV, 64]``, ``g_cum`` / ``rowscale`` / ``qscale`` / ``dg_part`` f32
# ``[R, HV]``, ``gamma`` / ``gamma_dot`` f32 ``[NC, HV]``, ``h_tape`` / ``dh_tape`` bf16 ``[NC, HV, 128, 128]``
# (``R = 64 NC_pad`` padded token rows).  Outputs: ``o`` / ``dv`` bf16 ``[T, HV, 128]``, ``dq`` / ``dk`` bf16
# ``[T, H_qk, 128]``, ``dbeta`` / ``dg`` f32 ``[T, HV]``.  Tables (int32; :mod:`.cake_tables`): ``chunk_bos`` /
# ``chunk_len`` ``[NC_pad]``, ``seq_chunk_start`` ``[n_frag + 1]`` of the local programs; ``seg_chunk_start``
# ``[NSEG + 1]`` and ``walk_seg`` ``[n_walk]`` of the walks with the FP32 pools ``h0`` / ``dht`` / ``out`` and their
# int32 slot tables ``*_index`` ``[NSEG]``; ``ticket`` (u32 ``[>= 1]``): the self-resetting work-ticket word of the
# persistent programs #5 / #6 / #8 -- zero before the launch, left at zero by the kernel; one slot per launch site in the
# caller's workspace, zeroed once when the workspace is (re)allocated, never read back by the host; the merge's FP32 tile pools ``m`` / ``c`` / ``x0`` / ``y`` and its chain
# tables ``chain_start`` (int32 ``[n_chains + 1]``), ``elem_m`` / ``elem_c`` / ``elem_y`` (int64 per element),
# ``x0_index`` / ``y_final`` (int64 ``[n_chains]``).
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "do",
    "g",
    "beta",
    "w",
    "u",
    "aqk",
    "akk",
    "g_cum",
    "rowscale",
    "qscale",
    "gamma",
    "h_tape",
    "v_new",
    "do_g",
    "dv_intra",
    "dh_tape",
    "dv_new",
    "o",
    "dq",
    "dk",
    "dq_part",
    "dk_part",
    "dv",
    "dbeta",
    "dg_part",
    "gamma_dot",
    "dg",
    "chunk_bos",
    "chunk_len",
    "seq_chunk_start",
    "ticket",
    "seg_chunk_start",
    "walk_seg",
    "h0",
    "h0_index",
    "dht",
    "dht_index",
    "out",
    "out_index",
    "m",
    "c",
    "x0",
    "y",
    "chain_start",
    "elem_m",
    "elem_c",
    "elem_y",
    "x0_index",
    "y_final",
)
LOGICAL_SCALARS = (
    "num_heads",
    "num_qk_heads",
    "num_seqs",
    "num_chunks",
    "scale",
    "do_tstride",
    "dqk_tstride",
    "dv_tstride",
    "group_reduce",
    "h0_hg",
    "h0_n",
    "dht_hg",
    "dht_n",
    "out_hg",
    "out_n",
    "tile_stride",
    "init_mode",
    "use_u",
    "write_transposed",
    "ncols",
    "transpose_m",
    "write_states",
    "cols_per_cta",
)


# Argument tables (``R`` padded token rows, ``NC`` chunks, ``NSEG`` segments, ``n_walk`` walked
# segments).  The walks of #2 / #3 / #7 keep ``kda_chunk_kcp_*_v1``'s pool contract: ``h0`` / ``dht`` / ``out``
# are flat FP32 pools of ``[128, 128]`` tiles, tile ``((h // hg) * n + slot) * hg + h % hg`` of (slot, head),
# ``tile_stride`` tiles apart (1 dense plane, 2 interleaved pair), ``*_index[seg]`` the slot of a segment or
# ``-1``.  The walks read the token-indexed strided operands (``k`` / ``q`` / ``do_g``) of a chunk through the
# local chunk table ``chunk_bos`` (#7 also takes ``chunk_len``; the tapes are on padded rows ``64 c``).  ``u`` / ``use_u`` of the transfer walk carry the
# FP32-class ``B_f`` knob (``init_mode=1, use_u=1``; the design's §7.1 row lists ``init_mode`` alone).
# --- BEGIN GENERATED REGISTRY (regenerated by tooling; do not edit by hand) ---
# Every record is one mechanically exported program; the ``library`` names are the export
# family's frozen identities; ``kernel_symbol``, ``launch``, ``arg_plan`` and ``logical`` are rewritten by the
# exporter from the rendered binding.  Test-only records carry the option values (``test_knobs``) that select
# them; they are never selected by default (no bf16x3 module).
MODULES: dict[str, dict[str, Any]] = {
    "bwd_dqkg": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q_tma"],
            ["buffer", "k_tma"],
            ["buffer", "v_tma"],
            ["buffer", "do_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dvn_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "beta"],
            ["buffer", "g_cum"],
            ["buffer", "dq"],
            ["buffer", "dk"],
            ["buffer", "dv"],
            ["buffer", "dbeta"],
            ["buffer", "dg_part"],
            ["buffer", "gamma_dot"],
            ["buffer", "dq_part"],
            ["buffer", "dk_part"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["parameter", "dqk_tstride"],
            ["parameter", "dv_tstride"],
            ["parameter", "group_reduce"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks_qk_heads_persistent"},
        "kernel_symbol": "kernel_cake_gdn_train_83eb8234f378bfe68bb1",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 227328,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_83eb8234f378bfe68bb1",
        "logical": {
            "akk_tma": "akk",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dbeta": "dbeta",
            "dg_part": "dg_part",
            "dh_tma": "dh_tape",
            "dk": "dk",
            "dk_part": "dk_part",
            "do_tma": "do",
            "dq": "dq",
            "dq_part": "dq_part",
            "dqk_tstride": "dqk_tstride",
            "dv": "dv",
            "dv_tstride": "dv_tstride",
            "dvn_tma": "dv_new",
            "g_cum": "g_cum",
            "gamma_dot": "gamma_dot",
            "group_reduce": "group_reduce",
            "h_tma": "h_tape",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "ticket": "ticket",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "program": "bwd_dqkg",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "bwd_prep": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "do_ptr"],
            ["buffer", "g_cum"],
            ["buffer", "do_g"],
            ["buffer", "dv_intra"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["parameter", "do_tstride"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 2, "sm_103a": 2, "sm_107a": 2},
            "kind": "chunks_heads_persistent",
        },
        "kernel_symbol": "kernel_cake_gdn_train_3f3dd0a8504039fd93f0",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50304,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_3f3dd0a8504039fd93f0",
        "logical": {
            "aqk_tma": "aqk",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "do_g": "do_g",
            "do_ptr": "do",
            "do_tma": "do",
            "do_tstride": "do_tstride",
            "dv_intra": "dv_intra",
            "g_cum": "g_cum",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "ticket": "ticket",
        },
        "program": "bwd_prep",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "fwd_o": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q_tma"],
            ["buffer", "h_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "qscale"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "ticket"],
            ["parameter", "num_seqs"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks_heads_persistent"},
        "kernel_symbol": "kernel_cake_gdn_train_37364f5831ded8d5c2af",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222336,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_37364f5831ded8d5c2af",
        "logical": {
            "aqk_tma": "aqk",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "h_tma": "h_tape",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "o_out": "o",
            "q_tma": "q",
            "qscale": "qscale",
            "seq_chunk_start": "seq_chunk_start",
            "ticket": "ticket",
            "vn_tma": "v_new",
        },
        "program": "fwd_o",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "fwd_prep": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q_tma"],
            ["buffer", "k_tma"],
            ["buffer", "v_tma"],
            ["buffer", "g"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "w_out"],
            ["buffer", "u_out"],
            ["buffer", "aqk_out"],
            ["buffer", "akk_out"],
            ["buffer", "g_cum"],
            ["buffer", "rowscale"],
            ["buffer", "qscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks_by_qk_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_d045f73faa78a441d79b",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 113664,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_d045f73faa78a441d79b",
        "logical": {
            "akk_out": "akk",
            "aqk_out": "aqk",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "g": "g",
            "g_cum": "g_cum",
            "gamma": "gamma",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "q_tma": "q",
            "qscale": "qscale",
            "rowscale": "rowscale",
            "scale": "scale",
            "u_out": "u",
            "v_tma": "v",
            "w_out": "w",
        },
        "program": "fwd_prep",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "gate_epilogue": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "dg_part"],
            ["buffer", "gamma_dot"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "num_chunks"],
            ["parameter", "num_heads"],
            ["buffer", "dg_out"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"items_per_cta": 128, "kind": "chunk_head_items"},
        "kernel_symbol": "kernel_cake_gdn_train_eb11410b26f9701f723e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_eb11410b26f9701f723e",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dg_out": "dg",
            "dg_part": "dg_part",
            "gamma_dot": "gamma_dot",
            "num_chunks": "num_chunks",
            "num_heads": "num_heads",
        },
        "program": "gate_epilogue",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "kcp_affine_merge_c16": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m"],
            ["buffer", "c"],
            ["buffer", "x0"],
            ["buffer", "y"],
            ["buffer", "chain_start"],
            ["buffer", "elem_m"],
            ["buffer", "elem_c"],
            ["buffer", "elem_y"],
            ["buffer", "x0_index"],
            ["buffer", "y_final"],
            ["parameter", "ncols"],
            ["parameter", "transpose_m"],
            ["parameter", "write_states"],
            ["parameter", "cols_per_cta"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"cols_per_cta": 16, "kind": "kcp_chains"},
        "kernel_symbol": "kernel_cake_gdn_train_89f5ff59f03eb170e3bf",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 41216,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_89f5ff59f03eb170e3bf",
        "logical": {
            "c": "c",
            "chain_start": "chain_start",
            "cols_per_cta": "cols_per_cta",
            "elem_c": "elem_c",
            "elem_m": "elem_m",
            "elem_y": "elem_y",
            "m": "m",
            "ncols": "ncols",
            "transpose_m": "transpose_m",
            "write_states": "write_states",
            "x0": "x0",
            "x0_index": "x0_index",
            "y": "y",
            "y_final": "y_final",
        },
        "program": "kcp_affine_merge",
        "select_rule": {"kind": "columns", "ncols": 256},
        "variant": "c16",
    },
    "kcp_affine_merge_c8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m"],
            ["buffer", "c"],
            ["buffer", "x0"],
            ["buffer", "y"],
            ["buffer", "chain_start"],
            ["buffer", "elem_m"],
            ["buffer", "elem_c"],
            ["buffer", "elem_y"],
            ["buffer", "x0_index"],
            ["buffer", "y_final"],
            ["parameter", "ncols"],
            ["parameter", "transpose_m"],
            ["parameter", "write_states"],
            ["parameter", "cols_per_cta"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"cols_per_cta": 8, "kind": "kcp_chains"},
        "kernel_symbol": "kernel_cake_gdn_train_c6958a0a4ed5cb55a9a1",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 37120,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_c6958a0a4ed5cb55a9a1",
        "logical": {
            "c": "c",
            "chain_start": "chain_start",
            "cols_per_cta": "cols_per_cta",
            "elem_c": "elem_c",
            "elem_m": "elem_m",
            "elem_y": "elem_y",
            "m": "m",
            "ncols": "ncols",
            "transpose_m": "transpose_m",
            "write_states": "write_states",
            "x0": "x0",
            "x0_index": "x0_index",
            "y": "y",
            "y_final": "y_final",
        },
        "program": "kcp_affine_merge",
        "select_rule": {"kind": "columns", "ncols": 128},
        "variant": "c8",
    },
    "kcp_dhu_offset": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q_tma"],
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_g_tma"],
            ["buffer", "dv_intra"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 1, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_f234e13206ecce204b46",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 116736,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_f234e13206ecce204b46",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_tape": "dh_tape",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_g_tma": "do_g",
            "dv_intra": "dv_intra",
            "dv_new": "dv_new",
            "gamma": "gamma",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "q_tma": "q",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_offset",
        "select_rule": {"ctas_per_head": 1, "kind": "ctas_per_head"},
        "variant": None,
    },
    "kcp_dhu_offset_vh64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q_tma"],
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_g_tma"],
            ["buffer", "dv_intra"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_ac5ce25cc84be93bb2a6",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 83968,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_ac5ce25cc84be93bb2a6",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_tape": "dh_tape",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_g_tma": "do_g",
            "dv_intra": "dv_intra",
            "dv_new": "dv_new",
            "gamma": "gamma",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "q_tma": "q",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_offset",
        "select_rule": {"ctas_per_head": 2, "kind": "ctas_per_head"},
        "variant": "vh64",
    },
    "kcp_dhu_tape": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q_tma"],
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_g_tma"],
            ["buffer", "dv_intra"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 1, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_94ff8219a97e6835ca9e",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_94ff8219a97e6835ca9e",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_tape": "dh_tape",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_g_tma": "do_g",
            "dv_intra": "dv_intra",
            "dv_new": "dv_new",
            "gamma": "gamma",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "q_tma": "q",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_tape",
        "select_rule": {"ctas_per_head": 1, "kind": "ctas_per_head"},
        "variant": None,
    },
    "kcp_dhu_tape_vh64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q_tma"],
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_g_tma"],
            ["buffer", "dv_intra"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_49450e259854e03ab17c",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 83968,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_49450e259854e03ab17c",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dh_tape": "dh_tape",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_g_tma": "do_g",
            "dv_intra": "dv_intra",
            "dv_new": "dv_new",
            "gamma": "gamma",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "q_tma": "q",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_tape",
        "select_rule": {"ctas_per_head": 2, "kind": "ctas_per_head"},
        "variant": "vh64",
    },
    "kcp_fwdh_offset": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "u"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "chunk_bos"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 1, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_04ac407490dd3a053e22",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_04ac407490dd3a053e22",
        "logical": {
            "chunk_bos": "chunk_bos",
            "gamma": "gamma",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h_tape",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_offset",
        "select_rule": {"ctas_per_head": 1, "kind": "ctas_per_head"},
        "variant": None,
    },
    "kcp_fwdh_offset_vh64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "u"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "chunk_bos"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_a5f0e9d8b4b48cf758e7",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_a5f0e9d8b4b48cf758e7",
        "logical": {
            "chunk_bos": "chunk_bos",
            "gamma": "gamma",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h_tape",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_offset",
        "select_rule": {"ctas_per_head": 2, "kind": "ctas_per_head"},
        "variant": "vh64",
    },
    "kcp_fwdh_tape": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "u"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "chunk_bos"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 1, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_2a0326b75a49ed13cfde",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_2a0326b75a49ed13cfde",
        "logical": {
            "chunk_bos": "chunk_bos",
            "gamma": "gamma",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h_tape",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_tape",
        "select_rule": {"ctas_per_head": 1, "kind": "ctas_per_head"},
        "variant": None,
    },
    "kcp_fwdh_tape_vh64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "u"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "chunk_bos"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_6e7a1ca02911aa5f1313",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 99328,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_6e7a1ca02911aa5f1313",
        "logical": {
            "chunk_bos": "chunk_bos",
            "gamma": "gamma",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h_tape",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_tape",
        "select_rule": {"ctas_per_head": 2, "kind": "ctas_per_head"},
        "variant": "vh64",
    },
    "kcp_transfer_walk": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "u"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "chunk_bos"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_7c591421af4c01babb73",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198144,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_7c591421af4c01babb73",
        "logical": {
            "chunk_bos": "chunk_bos",
            "gamma": "gamma",
            "init_mode": "init_mode",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
}
TEST_MODULES: dict[str, dict[str, Any]] = {
    "kcp_transfer_walk_plain_tf32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "u"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "chunk_bos"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_154677eea489d8d78555",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198144,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_154677eea489d8d78555",
        "logical": {
            "chunk_bos": "chunk_bos",
            "gamma": "gamma",
            "init_mode": "init_mode",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "test_knobs": {"precision_mode": 2},
        "variant": "plain_tf32",
    },
    "kcp_transfer_walk_three_mma": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "k_tma"],
            ["buffer", "w_tma"],
            ["buffer", "u"],
            ["buffer", "rowscale"],
            ["buffer", "gamma"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["buffer", "chunk_bos"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_gdn_train_092cee18b83fb8f48a43",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198144,
            "use_pdl": False,
        },
        "library": "cake_gdn_train_092cee18b83fb8f48a43",
        "logical": {
            "chunk_bos": "chunk_bos",
            "gamma": "gamma",
            "init_mode": "init_mode",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_qk_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "rowscale": "rowscale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "test_knobs": {"precision_mode": 1},
        "variant": "three_mma",
    },
}
# --- END GENERATED REGISTRY ---


def records_of_program(program: str) -> tuple[str, ...]:
    """The shipped record names of ``program`` in registry order."""
    if program not in PROGRAMS:
        raise ValueError(
            f"unknown GDN KCP program {program!r}; the programs are {PROGRAMS}"
        )
    return tuple(name for name, rec in MODULES.items() if rec["program"] == program)


def _admits(rule: Mapping[str, Any], *, ncols: int | None, ctas_per_head: int) -> bool:
    kind = rule["kind"]
    if kind == "always":
        return True
    if kind == "columns":
        return ncols is not None and int(ncols) == int(rule["ncols"])
    if kind == "ctas_per_head":
        return int(ctas_per_head) == int(rule["ctas_per_head"])
    raise ValueError(f"unknown Cake GDN KCP selection rule: {kind!r}")


def select_record(
    program: str,
    *,
    ncols: int | None = None,
    ctas_per_head: int = 1,
    precision_mode: int = PRECISION_MODE_DEFAULT,
    transfer_walk: str = TRANSFER_WALK_DEFAULT,
) -> str:
    """The one record of ``program`` the launch's scalars select (``always`` for a single-record program;
    ``columns`` picks the merge variant whose column count is the chain's ``ncols``: 128 state, 256 map;
    ``ctas_per_head`` picks the form of a forward state walk: 1 the ``vh = 128`` record, 2 the ``_vh64`` record with
    two CTAs per (segment, head), chosen per launch when ``n_walk x HV <= sm_count``).  Shipped
    records come from ``MODULES`` only; a non-default ``precision_mode`` / ``transfer_walk`` (test knobs the options
    validator admitted) selects the ``TEST_MODULES`` record whose ``test_knobs`` equal them instead (never by
    default; both knobs at once name two programs and are refused).  Exactly one record admits every launch, or the
    registry is inconsistent."""
    mode, walk = int(precision_mode), str(transfer_walk)
    if mode not in PRECISION_MODES:
        raise ValueError(
            f"precision_mode must be one of {PRECISION_MODES}, got {precision_mode!r}"
        )
    if walk not in TRANSFER_WALKS:
        raise ValueError(
            f"transfer_walk must be one of {TRANSFER_WALKS}, got {transfer_walk!r}"
        )
    if walk not in TRANSFER_WALKS_BUILT:
        raise ValueError(
            f"transfer_walk {walk!r} is not built in this package; built: {TRANSFER_WALKS_BUILT}"
        )
    wanted: dict[str, Any] = {}
    if mode != PRECISION_MODE_DEFAULT:
        wanted["precision_mode"] = mode
    if walk != TRANSFER_WALK_DEFAULT:
        wanted["transfer_walk"] = walk
    if len(wanted) > 1:
        raise ValueError(
            f"precision_mode {mode} and transfer_walk {walk!r} name two test-only programs; set one knob"
        )
    if wanted:
        matches = [
            name
            for name, rec in TEST_MODULES.items()
            if rec["program"] == program and rec.get("test_knobs") == wanted
        ]
        if len(matches) != 1:
            raise ValueError(
                f"program {program!r}: {len(matches)} test-only records match {wanted} ({matches}); expected one"
            )
        return matches[0]
    admitted = [
        name
        for name in records_of_program(program)
        if _admits(
            MODULES[name]["select_rule"], ncols=ncols, ctas_per_head=ctas_per_head
        )
    ]
    if len(admitted) != 1:
        raise ValueError(
            f"program {program!r}: {len(admitted)} records admit ncols={ncols} ctas_per_head={ctas_per_head} "
            f"({admitted}); expected one"
        )
    return admitted[0]


def _ctas_per_sm(rule: Mapping[str, Any], arch: str | None) -> int:
    """The persistent CTAs per SM a grid rule launches: its ``ctas_per_sm`` as one int for every architecture,
    or as a mapping architecture -> int that reads 1 for an architecture it does not name and when the launch
    names no architecture; 1 when the rule carries no ``ctas_per_sm``."""
    per_sm = rule.get("ctas_per_sm", 1)
    if isinstance(per_sm, Mapping):
        return int(per_sm.get(arch, 1)) if arch is not None else 1
    return int(per_sm)


def launch_grid(
    name: str,
    *,
    num_chunks: int = 0,
    num_heads: int = 0,
    num_qk_heads: int = 0,
    n_walk: int = 0,
    n_chains: int = 0,
    ncols: int = 0,
    ctas_per_head: int | None = None,
    sm_count: int = 0,
    arch: str | None = None,
) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its grid rule: ``chunks_by_qk_heads`` ``(NC, H_qk, 1)`` (one CTA
    per chunk and Q/K head: the ``HV / H_qk`` value heads of the group share ``k``), ``chunks_heads_persistent`` /
    ``chunks_qk_heads_persistent`` ``(min(NC x HV, k x sm_count), 1, 1)`` / ``(min(NC x H_qk, k x sm_count), 1, 1)``
    (a persistent CTA walks the tiles ``b, b + gridDim.x, ...``; ``k`` is the record's ``ctas_per_sm`` for ``arch``,
    the architecture the launch loads the library for -- 1 without the field, for an architecture the field does
    not name, or without ``arch``; without ``sm_count`` the grid is the tile count),
    ``chunk_head_items`` ``(ceil(NC x HV / items_per_cta), 1, 1)`` (one thread per (chunk, head) item),
    ``kcp_segments_heads`` ``(n_walk x HV x ctas_per_head, 1, 1)`` (``ctas_per_head`` is the record's: 2
    for the half-width ``vh = 64`` ``_vh64`` records of the forward state walk, chosen per launch when
    ``n_walk x HV <= sm_count``, else 1; both forms write bitwise-identical tapes / pools; a passed ``ctas_per_head``
    must agree with the record), ``kcp_chains`` ``(n_chains x ncols / cols_per_cta, 1, 1)``.  An empty
    problem (no chunk, no walked segment, no chain) has no launch: the caller skips it (``ValueError`` here)."""
    record = MODULES.get(name) or TEST_MODULES.get(name)
    if record is None:
        raise KeyError(f"{name!r} is not a GDN KCP record")
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "chunks_by_qk_heads":
        if int(num_chunks) < 1 or int(num_qk_heads) < 1:
            raise ValueError(
                f"record {name}: num_chunks={num_chunks} and num_qk_heads={num_qk_heads} must be >= 1"
            )
        return (int(num_chunks), int(num_qk_heads), 1)
    if kind in ("chunks_heads_persistent", "chunks_qk_heads_persistent"):
        qk = kind == "chunks_qk_heads_persistent"
        heads = num_qk_heads if qk else num_heads
        if int(num_chunks) < 1 or int(heads) < 1:
            raise ValueError(
                f"record {name}: num_chunks={num_chunks} and "
                f"{'num_qk_heads' if qk else 'num_heads'}={heads} must be >= 1"
            )
        tiles = int(num_chunks) * int(heads)
        per_sm = _ctas_per_sm(rule, arch)
        return (
            min(tiles, per_sm * int(sm_count)) if int(sm_count) >= 1 else tiles,
            1,
            1,
        )
    if kind == "chunk_head_items":
        per_cta = int(rule["items_per_cta"])
        if int(num_chunks) < 1 or int(num_heads) < 1:
            raise ValueError(
                f"record {name}: num_chunks={num_chunks} and num_heads={num_heads} must be >= 1"
            )
        return (-(-(int(num_chunks) * int(num_heads)) // per_cta), 1, 1)
    if kind == "kcp_segments_heads":
        if int(n_walk) < 1 or int(num_heads) < 1:
            raise ValueError(
                f"record {name}: n_walk={n_walk} and num_heads={num_heads} must be >= 1"
            )
        declared = int(rule.get("ctas_per_head", 1))
        per_head = declared if ctas_per_head is None else int(ctas_per_head)
        if per_head not in (1, 2):
            raise ValueError(
                f"record {name}: ctas_per_head={ctas_per_head} must be 1 (vh = 128) or 2 (vh = 64)"
            )
        if per_head != declared:
            raise ValueError(
                f"record {name}: ctas_per_head={ctas_per_head} does not match the record's {declared}"
            )
        return (int(n_walk) * int(num_heads) * per_head, 1, 1)
    if kind == "kcp_chains":
        cols = int(rule["cols_per_cta"])
        if int(n_chains) < 1 or int(ncols) < 1 or int(ncols) % cols:
            raise ValueError(
                f"record {name}: n_chains={n_chains} must be >= 1 and ncols={ncols} a positive multiple of {cols}"
            )
        return (int(n_chains) * (int(ncols) // cols), 1, 1)
    raise ValueError(f"unknown Cake GDN KCP grid rule: {kind!r}")


__all__ = [
    "OFFSET_CLASSES",
    "OFFSET_CLASS_DEFAULT",
    "TEST_KNOBS",
    "TRANSFER_WALKS",
    "TRANSFER_WALKS_BUILT",
    "TRANSFER_WALK_DEFAULT",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CHUNK",
    "HEAD_DIM",
    "LIBRARY_PREFIX",
    "LOCAL_PROGRAMS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MANIFEST",
    "MODULES",
    "PACKAGE",
    "PRECISION_MODES",
    "PRECISION_MODE_DEFAULT",
    "PROGRAMS",
    "TEST_MANIFEST",
    "TEST_MODULES",
    "TILE",
    "WALK_PROGRAMS",
    "launch_grid",
    "records_of_program",
    "select_record",
]
