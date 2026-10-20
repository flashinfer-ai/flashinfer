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

Registry and loader glue of the Cake packed KDA chunked training kernels
(SM100 / SM103 / SM107).

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures it
is delivered for, the ``stage`` of the launch order it fills and its
``variant``, the ``select_rule`` that picks one variant of a stage from the
host-known problem scalars, the ``grid_rule`` that shapes its launch, the
kernel-parameter -> logical-argument map and the argument plan of its ``run``
entry.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.

The rules are data, evaluated here (:func:`select_record`, :func:`launch_grid`)
from four host-known scalars: the padded chunk bound ``num_chunks_pad``, the
head count, the document count and the device's SM count.  No rule reads a
tensor.  Adding a program is one record; a stage with one record has the
``always`` rule unless the stage is optional (``OPTIONAL_STAGES``: the two
programs of the segmented state recurrences, which select nothing on the
serial route).

The state recurrences (``fwdh``, ``dhu``) have two forms.  With more than
``SEGMENTS["max_heads"]`` heads the serial walk over each document's chunk
chain runs: ``fwdh_v2`` / ``dhu_v2`` (four CTAs of 32 value columns per
document and head) when their one-wave form fits the device, else ``fwdh_v1``
/ ``dhu_v1`` (two CTAs of 64 columns); each pair is one kernel built at the
two widths.  With at most that many heads the segmented
route runs: every document is cut into ``segments`` chunk-aligned segments
(:func:`segment_plan`; ``seg_table`` writes the segment table), the ``v3``
program walks every segment in parallel (pass 1: the per-segment transfer and
offset), ``seg_carry`` chains the segments' states within each document, and
the ``v3`` program walks again from the carried states (pass 2, writing the
tapes).  ``seg_carry`` is launched by the segmented ``fwdh`` and ``dhu`` stages
between their two passes; it is a stage of this registry for its record, not a
step of the launch orders.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# One verified build per architecture: B200 (``sm_100a``), B300 / GB300
# (``sm_103a``) and R200 (``sm_107a``).  Every architecture is compiled with its
# exact ``-gencode`` flag set (``build.json`` ``arch_flags``); no family fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}

# The launch orders.  On the serial route the forward launches six programs
# and the backward from a saved set eight (nine when the saved set carries no
# tables, which the table stage then rebuilds; the checkpoint-replay backward
# ten: table, prep and the forward intra stage first).  On the segmented route
# ``seg_table`` follows ``table`` and the ``fwdh`` / ``dhu`` stages launch three
# programs each (pass 1, ``seg_carry``, pass 2): nine forward launches and
# twelve backward.  A stage with several variants launches the record its
# ``select_rule`` admits; an optional stage launches nothing when no record
# admits.
STAGES_FWD = ("table", "seg_table", "fwd_prep", "fwd_intra", "wy", "fwdh", "fwd_o")
STAGES_BWD = (
    "bwd_prep",
    "wy",
    "fwdh",
    "dav",
    "dhu",
    "dqkg",
    "glm_intra",
    "gate_epilogue",
)
STAGES = (
    "table",
    "seg_table",
    "fwd_prep",
    "fwd_intra",
    "wy",
    "fwdh",
    "seg_carry",
    "fwd_o",
    "bwd_prep",
    "dav",
    "dhu",
    "dqkg",
    "glm_intra",
    "gate_epilogue",
)
# The stages of the segmented route only: :func:`select_record` returns ``None``
# for them when no record admits the problem (the serial route).
OPTIONAL_STAGES = ("seg_table", "seg_carry")

# The segmented route (the ``heads_at_most`` / ``heads_above`` rules read
# ``max_heads``): with at most ``max_heads`` heads every document is cut into
# ``clamp(ceil(slot_factor * sm_count / (num_seqs * num_heads)), 1,
# max_segments)`` segments, so the segment walks fill about ``slot_factor / 2``
# waves of the device's SMs at two CTAs per SM.  With at most
# ``small_max_chains`` (document, head) chains the count is capped at
# ``small_segments``: the carry chain then takes 37 steps instead of 63, which
# outweighs the lost parallelism of the segment walks.
SEGMENTS = {
    "max_heads": 8,
    "slot_factor": 4,
    "max_segments": 64,
    "small_segments": 38,
    "small_max_chains": 4,
}

# Logical argument names the host backend binds; every record maps its kernel
# parameter names onto this vocabulary (``record["logical"]``).  ``aqk_src`` /
# ``akk_src`` are the saved ``Aqk`` / ``Akk`` token rows (the forward's internal
# copies in the replay path), ``h2d`` / ``dh2d`` the ``[NC * H * 128, 128]`` views
# of the ``h`` / ``dh`` tapes the TMA-described programs read.  The segmented
# route adds the segment table ``seg_chunk_start`` (``[num_segments + 1]``), the
# fp32 ``[128, 128]`` state blocks per (segment, head): ``seg_state``
# (``[2, num_segments, heads, 128, 128]``: the forward's segment offsets then
# transfers), ``seg_m`` (its transfer half as ``[num_segments * heads, 128,
# 128]``, both TMA operands of ``seg_carry``), ``hseg`` / ``dhseg`` (the carried
# forward / adjoint states) and ``dseg`` (the backward's segment offsets);
# ``carry_b`` / ``carry_out`` are ``seg_carry``'s offset input and carried-state
# output (``seg_state[0]`` -> ``hseg`` forward, ``dseg`` -> ``dhseg`` reverse);
# ``eye`` is the fp32 identity the forward's transfer walk starts from.  The
# backward's fused gradient and GLM intra kernel read ``q`` / ``k`` / ``v`` on
# the token rows (no chunk-layout copies); ``num_pairs`` is the chunk-pair
# count ``num_chunks_pad / 2`` the persistent pair kernels walk.
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "cu_seqlens",
    "dout",
    "chunk_bos",
    "chunk_len",
    "seq_chunk_start",
    "seq_order",
    "gk",
    "vb",
    "kb",
    "kg",
    "qg",
    "v_new",
    "aqk_int",
    "akk_int",
    "aqk_ext",
    "akk_ext",
    "aqk_src",
    "akk_src",
    "u",
    "w",
    "h",
    "h2d",
    "out",
    "do_int",
    "beta_int",
    "dv1",
    "dv2",
    "dAqk",
    "dAkk",
    "dh",
    "dh2d",
    "dq_f",
    "dk_f",
    "dg_f",
    "db_f",
    "dg_int",
    "db_int",
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
    "seg_chunk_start",
    "seg_state",
    "seg_m",
    "hseg",
    "dhseg",
    "dseg",
    "carry_b",
    "carry_out",
    "eye",
)
LOGICAL_SCALARS = (
    "num_seqs",
    "num_chunks_pad",
    "num_heads",
    "num_pairs",
    "group",
    "write_ext",
    "write_int",
    "copy_saved",
    "scale",
    "num_segments",
    "segments",
    "reverse",
    "use_h0",
    "use_dht",
    "write_tapes",
    "write_ht",
)

# Filled mechanically by the exporter (the derived leaves ``kernel_symbol``, ``launch``, ``arg_plan``,
# ``logical``); the host plan (``stage``, ``variant``, ``select_rule``, ``grid_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal: FlashInfer's manifest checker and the exporter read this
# file with ``ast.literal_eval``.  Record names are the stages, suffixed with the variant where a stage has
# several programs (``bwd_prep_g8`` keeps its variant name: the backward prep ships its ``g8`` form alone).
MODULES: dict[str, dict[str, Any]] = {
    "table": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "cu_seqlens"],
            ["parameter", "num_seqs"],
            ["parameter", "num_chunks_max"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["buffer", "seq_order"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "fixed", "ctas": [1, 1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_d7656e2b58b9bab49495",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1152,
            "use_pdl": False,
        },
        "library": "cake_kda_train_d7656e2b58b9bab49495",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "cu_seqlens": "cu_seqlens",
            "num_chunks_max": "num_chunks_pad",
            "num_seqs": "num_seqs",
            "seq_chunk_start": "seq_chunk_start",
            "seq_order": "seq_order",
        },
        "select_rule": {"kind": "always"},
        "stage": "table",
        "variant": None,
    },
    "seg_table": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "segments"],
            ["buffer", "seg_chunk_start"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "fixed", "ctas": [1, 1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_7fb7f9adb0bbdc2a3f90",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_kda_train_7fb7f9adb0bbdc2a3f90",
        "logical": {
            "num_seqs": "num_seqs",
            "seg_chunk_start": "seg_chunk_start",
            "segments": "segments",
            "seq_chunk_start": "seq_chunk_start",
        },
        "select_rule": {"kind": "heads_at_most", "max": 8},
        "stage": "seg_table",
        "variant": None,
    },
    "fwd_prep_bulk": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "g"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "gk_out"],
            ["buffer", "vb_out"],
            ["buffer", "kb_out"],
            ["buffer", "kg_out"],
            ["buffer", "qg_out"],
            ["buffer", "v_new"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_1efd16cb6130128a3ece",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 36864,
            "use_pdl": False,
        },
        "library": "cake_kda_train_1efd16cb6130128a3ece",
        "logical": {
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q": "q",
            "qg_out": "qg",
            "v": "v",
            "v_new": "v_new",
            "vb_out": "vb",
        },
        "select_rule": {"kind": "ctas_at_least", "min": 256},
        "stage": "fwd_prep",
        "variant": "bulk",
    },
    "fwd_prep_split": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "g"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "gk_out"],
            ["buffer", "vb_out"],
            ["buffer", "kb_out"],
            ["buffer", "kg_out"],
            ["buffer", "qg_out"],
            ["buffer", "v_new"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 2},
        "kernel_symbol": "kernel_cake_kda_train_2f896a0e5d78ead36559",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 36864,
            "use_pdl": False,
        },
        "library": "cake_kda_train_2f896a0e5d78ead36559",
        "logical": {
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q": "q",
            "qg_out": "qg",
            "v": "v",
            "v_new": "v_new",
            "vb_out": "vb",
        },
        "select_rule": {"kind": "ctas_below", "max": 256},
        "stage": "fwd_prep",
        "variant": "split",
    },
    "fwd_intra": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "g"],
            ["buffer", "beta"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "aqk_ext"],
            ["buffer", "akk_ext"],
            ["buffer", "aqk_int"],
            ["buffer", "akk_int"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "group"],
            ["parameter", "write_ext"],
            ["parameter", "write_int"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_16882c94a3cdf5b4dbe7",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 76160,
            "use_pdl": False,
        },
        "library": "cake_kda_train_16882c94a3cdf5b4dbe7",
        "logical": {
            "akk_ext": "akk_ext",
            "akk_int": "akk_int",
            "aqk_ext": "aqk_ext",
            "aqk_int": "aqk_int",
            "beta": "beta",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "g": "g",
            "group": "group",
            "k": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q": "q",
            "scale": "scale",
            "write_ext": "write_ext",
            "write_int": "write_int",
        },
        "select_rule": {"kind": "always"},
        "stage": "fwd_intra",
        "variant": None,
    },
    "wy": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "akk_tma"],
            ["buffer", "vb_tma"],
            ["buffer", "kb_tma"],
            ["buffer", "u_out"],
            ["buffer", "w_out"],
            ["parameter", "num_heads"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_5b1151fbf29938115cfa",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_5b1151fbf29938115cfa",
        "logical": {
            "akk_tma": "akk_int",
            "kb_tma": "kb",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "u_out": "u",
            "vb_tma": "vb",
            "w_out": "w",
        },
        "select_rule": {"kind": "always"},
        "stage": "wy",
        "variant": None,
    },
    "fwdh_v1": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_8b4be524f84f94dcd14a",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "use_pdl": False,
        },
        "library": "cake_kda_train_8b4be524f84f94dcd14a",
        "logical": {
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
        },
        "select_rule": {"kind": "otherwise"},
        "stage": "fwdh",
        "variant": "v1",
    },
    "fwdh_v2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_a2d6b181af2a69c3ae38",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 78848,
            "use_pdl": False,
        },
        "library": "cake_kda_train_a2d6b181af2a69c3ae38",
        "logical": {
            "gk": "gk",
            "h_out": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "seq_chunk_start": "seq_chunk_start",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "heads_above", "min": 8},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
            ],
        },
        "stage": "fwdh",
        "variant": "v2",
    },
    "fwdh_v3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["buffer", "h_out"],
            ["buffer", "v_new"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "h0"],
            ["buffer", "eye"],
            ["buffer", "ht_out"],
            ["parameter", "use_h0"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [2, 1]},
        "kernel_symbol": "kernel_cake_kda_train_8f0d871858c43a28de6a",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_8f0d871858c43a28de6a",
        "logical": {
            "eye": "eye",
            "gk": "gk",
            "h0": "hseg",
            "h_out": "h",
            "ht_out": "seg_state",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "seq_chunk_start": "seg_chunk_start",
            "u": "u",
            "use_h0": "use_h0",
            "v_new": "v_new",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {"kind": "heads_at_most", "max": 8},
        "stage": "fwdh",
        "variant": "v3",
    },
    "seg_carry": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m_k_tma"],
            ["buffer", "m_mn_tma"],
            ["buffer", "b"],
            ["buffer", "h_out"],
            ["parameter", "num_heads"],
            ["parameter", "segments"],
            ["parameter", "reverse"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "docs_heads"},
        "kernel_symbol": "kernel_cake_kda_train_56de21f4c00e805321dc",
        "launch": {
            "block": (576, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_56de21f4c00e805321dc",
        "logical": {
            "b": "carry_b",
            "h_out": "carry_out",
            "m_k_tma": "seg_m",
            "m_mn_tma": "seg_m",
            "num_heads": "num_heads",
            "reverse": "reverse",
            "segments": "segments",
        },
        "select_rule": {"kind": "heads_at_most", "max": 8},
        "stage": "seg_carry",
        "variant": None,
    },
    "fwd_o": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "qg_tma"],
            ["buffer", "h_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_1214bfc28e50985bfa0c",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_kda_train_1214bfc28e50985bfa0c",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "h_tma": "h2d",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "vn_tma": "v_new",
        },
        "select_rule": {"kind": "always"},
        "stage": "fwd_o",
        "variant": None,
    },
    "bwd_prep_g8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "g"],
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "beta"],
            ["buffer", "aqk"],
            ["buffer", "akk"],
            ["buffer", "do"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "gk_out"],
            ["buffer", "vb_out"],
            ["buffer", "kb_out"],
            ["buffer", "qg_out"],
            ["buffer", "kg_out"],
            ["buffer", "aqk_tril"],
            ["buffer", "akk_int"],
            ["buffer", "do_int"],
            ["buffer", "beta_int"],
            ["buffer", "v_new"],
            ["buffer", "dv2"],
            ["parameter", "num_qk_heads"],
            ["parameter", "num_heads"],
            ["parameter", "group"],
            ["parameter", "copy_saved"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_fc0b31b14ef852100538",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 36864,
            "use_pdl": False,
        },
        "library": "cake_kda_train_fc0b31b14ef852100538",
        "logical": {
            "akk": "akk_src",
            "akk_int": "akk_int",
            "aqk": "aqk_src",
            "aqk_tril": "aqk_int",
            "beta": "beta",
            "beta_int": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "copy_saved": "copy_saved",
            "do": "dout",
            "do_int": "do_int",
            "dv2": "dv2",
            "g": "g",
            "gk_out": "gk",
            "group": "group",
            "k": "k",
            "kb_out": "kb",
            "kg_out": "kg",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q": "q",
            "qg_out": "qg",
            "v": "v",
            "v_new": "v_new",
            "vb_out": "vb",
        },
        "select_rule": {"kind": "always"},
        "stage": "bwd_prep",
        "variant": "g8",
    },
    "dav": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vnew_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "dAqk"],
            ["buffer", "dv1"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["parameter", "num_pairs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "pairs_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_92243066d75ebec025d6",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "cake_kda_train_92243066d75ebec025d6",
        "logical": {
            "aqk_tma": "aqk_int",
            "dAqk": "dAqk",
            "do_tma": "do_int",
            "dv1": "dv1",
            "num_heads": "num_heads",
            "num_pairs": "num_pairs",
            "scale": "scale",
            "vnew_tma": "v_new",
        },
        "select_rule": {"kind": "always"},
        "stage": "dav",
        "variant": None,
    },
    "dhu_v1": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "dv1"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 2},
        "kernel_symbol": "kernel_cake_kda_train_71d4af02c1baa5148600",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_kda_train_71d4af02c1baa5148600",
        "logical": {
            "dh_out": "dh",
            "do_tma": "do_int",
            "dv1": "dv1",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "w_tma": "w",
        },
        "select_rule": {"kind": "otherwise"},
        "stage": "dhu",
        "variant": "v1",
    },
    "dhu_v2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "dv1"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_doc_head", "slices": 4},
        "kernel_symbol": "kernel_cake_kda_train_0ea173162573b1521e36",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_kda_train_0ea173162573b1521e36",
        "logical": {
            "dh_out": "dh",
            "do_tma": "do_int",
            "dv1": "dv1",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "w_tma": "w",
        },
        "select_rule": {
            "kind": "all_of",
            "rules": [
                {"kind": "heads_above", "min": 8},
                {"kind": "slots_at_most", "slices": 4, "ctas_per_sm": 4},
            ],
        },
        "stage": "dhu",
        "variant": "v2",
    },
    "dhu_v3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "dv1"],
            ["buffer", "gk"],
            ["buffer", "dh_out"],
            ["buffer", "dv2"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "scale"],
            ["buffer", "dht"],
            ["buffer", "ht_out"],
            ["parameter", "use_dht"],
            ["parameter", "write_tapes"],
            ["parameter", "write_ht"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "segments_heads", "passes": [1, 1]},
        "kernel_symbol": "kernel_cake_kda_train_340466b2bf3aad16d9a5",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_340466b2bf3aad16d9a5",
        "logical": {
            "dh_out": "dh",
            "dht": "dhseg",
            "do_tma": "do_int",
            "dv1": "dv1",
            "dv2": "dv2",
            "gk": "gk",
            "ht_out": "dseg",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_segments",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seg_chunk_start",
            "use_dht": "use_dht",
            "w_tma": "w",
            "write_ht": "write_ht",
            "write_tapes": "write_tapes",
        },
        "select_rule": {"kind": "heads_at_most", "max": 8},
        "stage": "dhu",
        "variant": "v3",
    },
    "dqkg": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "dv2_tma"],
            ["buffer", "v_tma"],
            ["buffer", "h_tma"],
            ["buffer", "dh_tma"],
            ["buffer", "akk_tma"],
            ["buffer", "gk_tma"],
            ["buffer", "k_tma"],
            ["buffer", "q_tma"],
            ["buffer", "beta"],
            ["buffer", "dq_out"],
            ["buffer", "dk_out"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["buffer", "dAkk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "sm_bounded", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_kda_train_15ba135cad91be1ee5c1",
        "launch": {
            "block": (576, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224000,
            "use_pdl": False,
        },
        "library": "cake_kda_train_15ba135cad91be1ee5c1",
        "logical": {
            "akk_tma": "akk_int",
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk_out": "dAkk",
            "db_out": "db_f",
            "dg_out": "dg_f",
            "dh_tma": "dh2d",
            "dk_out": "dk_f",
            "do_tma": "do_int",
            "dq_out": "dq_f",
            "dv2_tma": "dv2",
            "dv_out": "dv",
            "gk_tma": "gk",
            "h_tma": "h2d",
            "k_tma": "k",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "q_tma": "q",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "v_tma": "v",
            "vn_tma": "v_new",
        },
        "select_rule": {"kind": "always"},
        "stage": "dqkg",
        "variant": None,
    },
    "glm_intra": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "dAqk"],
            ["buffer", "dAkk"],
            ["buffer", "gk"],
            ["buffer", "k_e"],
            ["buffer", "q_e"],
            ["buffer", "beta"],
            ["buffer", "dq_f"],
            ["buffer", "dk_f"],
            ["buffer", "dg_f"],
            ["buffer", "db_f"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dq_ext"],
            ["buffer", "dk_ext"],
            ["buffer", "dg_out"],
            ["buffer", "db_out"],
            ["parameter", "num_heads"],
            ["parameter", "num_qk_heads"],
            ["parameter", "group"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_14df7843d40c9e76f7f9",
        "launch": {
            "block": (288, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115200,
            "use_pdl": False,
        },
        "library": "cake_kda_train_14df7843d40c9e76f7f9",
        "logical": {
            "beta": "beta_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "dAkk": "dAkk",
            "dAqk": "dAqk",
            "db_f": "db_f",
            "db_out": "db_int",
            "dg_f": "dg_f",
            "dg_out": "dg_int",
            "dk_ext": "dk",
            "dk_f": "dk_f",
            "dq_ext": "dq",
            "dq_f": "dq_f",
            "gk": "gk",
            "group": "group",
            "k_e": "k",
            "num_heads": "num_heads",
            "num_qk_heads": "num_heads",
            "q_e": "q",
        },
        "select_rule": {"kind": "always"},
        "stage": "glm_intra",
        "variant": None,
    },
    "gate_epilogue": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "dg_in"],
            ["buffer", "db_in"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "dg_out"],
            ["buffer", "dbeta_out"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "chunks", "x_factor": 1},
        "kernel_symbol": "kernel_cake_kda_train_47a9c058e6d6c78c166d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_kda_train_47a9c058e6d6c78c166d",
        "logical": {
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "db_in": "db_int",
            "dbeta_out": "dbeta",
            "dg_in": "dg_int",
            "dg_out": "dg",
            "num_heads": "num_heads",
        },
        "select_rule": {"kind": "always"},
        "stage": "gate_epilogue",
        "variant": None,
    },
}
# Rows of the internal chunk layout one pair-shaped CTA covers (two 64-token
# chunks), the chunk length and the head dimension every program is built for.
PAIR_ROWS = 128
CHUNK = 64
HEAD_DIM = 128


def records_of_stage(stage: str) -> tuple[str, ...]:
    """The record names of ``stage`` in registry order (the ``otherwise`` record
    last, so a scan picks a positive rule before the fallback)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {STAGES}")
    names = [name for name, rec in MODULES.items() if rec["stage"] == stage]
    names.sort(key=lambda name: MODULES[name]["select_rule"]["kind"] == "otherwise")
    return tuple(names)


def _admits(
    rule: dict[str, Any], *, ctas: int, num_seqs: int, num_heads: int, sm_count: int
) -> bool:
    kind = rule["kind"]
    if kind in ("always", "otherwise"):
        return True
    if kind == "ctas_at_least":
        return ctas >= int(rule["min"])
    if kind == "ctas_below":
        return ctas < int(rule["max"])
    if kind == "ctas_between":
        return int(rule["min"]) <= ctas < int(rule["max"])
    if kind == "slots_at_most":
        return num_seqs * num_heads * int(rule["slices"]) <= sm_count * int(
            rule["ctas_per_sm"]
        )
    if kind == "heads_at_most":
        return num_heads <= int(rule["max"])
    if kind == "heads_above":
        return num_heads > int(rule["min"])
    if kind == "all_of":
        return all(
            _admits(
                sub,
                ctas=ctas,
                num_seqs=num_seqs,
                num_heads=num_heads,
                sm_count=sm_count,
            )
            for sub in rule["rules"]
        )
    raise ValueError(f"unknown Cake KDA training selection rule: {kind!r}")


def segment_plan(num_seqs: int, num_heads: int, sm_count: int) -> tuple[str, int, int]:
    """``(route, segments, num_segments)`` of the state recurrences: ``("seg", P,
    num_seqs * P)`` with ``P = clamp(ceil(slot_factor * sm_count / (num_seqs *
    num_heads)), 1, max_segments)``, capped at ``small_segments`` when
    ``num_seqs * num_heads <= small_max_chains``, when ``num_heads <=
    SEGMENTS["max_heads"]``, else the serial walk ``("serial", 1, 0)``.  A pure
    function of the scalars, the same bound the ``heads_at_most`` rules of the
    segmented records read.
    """
    n, h, sm = int(num_seqs), int(num_heads), int(sm_count)
    if n < 1 or h < 1 or sm < 1:
        raise ValueError(
            f"segment_plan: num_seqs={num_seqs}, num_heads={num_heads}, "
            f"sm_count={sm_count} must all be >= 1"
        )
    if h > int(SEGMENTS["max_heads"]):
        return ("serial", 1, 0)
    segments = -(-int(SEGMENTS["slot_factor"]) * sm // (n * h))
    segments = min(int(SEGMENTS["max_segments"]), max(1, segments))
    if n * h <= int(SEGMENTS["small_max_chains"]):
        segments = min(segments, int(SEGMENTS["small_segments"]))
    return ("seg", segments, n * segments)


def select_record(
    stage: str, *, num_chunks_pad: int, num_seqs: int, num_heads: int, sm_count: int
) -> str | None:
    """The one record of ``stage`` the host scalars select (``None`` for an
    optional stage no record of which admits the problem).

    ``ctas = num_chunks_pad * num_heads`` is the chunk-shaped CTA count the
    preparation stages switch on (``ctas_at_least`` / ``ctas_below`` /
    ``ctas_between``: ``min <= ctas < max``); the recurrence stages switch on
    the head count (``heads_at_most`` / ``heads_above``: the segmented route
    at most ``SEGMENTS["max_heads"]`` heads, the serial walks above it) and on
    whether their one-wave form fits the device (``slots_at_most``:
    ``num_seqs * num_heads * slices <= sm_count * ctas_per_sm``); ``all_of``
    conjoins rules; ``otherwise`` is the fallback of its stage and ``always``
    the rule of a single-program stage.  Exactly one record admits every
    problem (none, for an optional stage on the serial route), or the registry
    is inconsistent.
    """
    ctas = int(num_chunks_pad) * int(num_heads)
    admitted = [
        name
        for name in records_of_stage(stage)
        if MODULES[name]["select_rule"]["kind"] != "otherwise"
        and _admits(
            MODULES[name]["select_rule"],
            ctas=ctas,
            num_seqs=int(num_seqs),
            num_heads=int(num_heads),
            sm_count=int(sm_count),
        )
    ]
    if len(admitted) > 1:
        raise ValueError(
            f"stage {stage!r}: records {admitted} all admit num_chunks_pad="
            f"{num_chunks_pad}, num_seqs={num_seqs}, num_heads={num_heads}, "
            f"sm_count={sm_count}"
        )
    if admitted:
        return admitted[0]
    fallback = [
        name
        for name in records_of_stage(stage)
        if MODULES[name]["select_rule"]["kind"] == "otherwise"
    ]
    if not fallback and stage in OPTIONAL_STAGES:
        return None
    if len(fallback) != 1:
        raise ValueError(
            f"stage {stage!r}: no record admits num_chunks_pad={num_chunks_pad}, "
            f"num_seqs={num_seqs}, num_heads={num_heads}, sm_count={sm_count} and "
            f"the stage has {len(fallback)} fallback records"
        )
    return fallback[0]


def launches(name: str) -> int:
    """How many times its stage launches record ``name``: the passes of a
    ``segments_heads`` grid (the segmented recurrence walks, ``seg_carry``
    between them), else one."""
    rule = MODULES[name]["grid_rule"]
    return len(rule["passes"]) if rule["kind"] == "segments_heads" else 1


def launch_grid(
    name: str,
    *,
    num_chunks_pad: int,
    num_seqs: int,
    num_heads: int,
    sm_count: int,
    num_segments: int = 0,
    launch: int = 0,
) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its recorded grid rule (of its
    ``launch``-th launch, for a record its stage launches more than once).

    ``fixed`` launches the recorded CTAs; ``chunks`` launches
    ``(x_factor * num_chunks_pad, num_heads, 1)`` (one or two CTAs per chunk
    and head, pad chunks included: the kernels treat ``chunk_len == 0`` as a
    no-op); ``pairs`` launches ``(num_chunks_pad / 2, num_heads, 1)`` (one CTA
    per chunk pair and head; ``num_chunks_pad`` is even); ``pairs_bounded`` is
    the persistent pair grid ``min(num_chunks_pad / 2 * num_heads, ctas_per_sm *
    sm_count)``, at least one (the CTAs walk the (pair, head) items;
    ``num_chunks_pad`` is even); ``per_doc_head``
    launches ``(num_seqs * num_heads * slices, 1, 1)`` (the recurrence over the
    chunk chain of one document and head, in ``slices`` value-column slices);
    ``sm_bounded`` is the persistent grid ``min(num_chunks_pad * num_heads,
    ctas_per_sm * sm_count)``, at least one; ``docs_heads`` launches
    ``(num_seqs * num_heads, 1, 1)`` (the carry over one document's segments
    and head); ``segments_heads`` launches ``(passes[launch] * num_segments *
    num_heads, 1, 1)`` (one CTA per segment and head, the forward's first pass
    two: the offset walk and the transfer walk; ``num_segments`` is the
    segmented route's ``segment_plan`` count, at least one).
    """
    rule = MODULES[name]["grid_rule"]
    kind = rule["kind"]
    nc, n, h = int(num_chunks_pad), int(num_seqs), int(num_heads)
    if launch < 0 or launch >= launches(name):
        raise ValueError(f"record {name}: launch {launch} of {launches(name)} launches")
    if kind == "fixed":
        x, y, z = (int(c) for c in rule["ctas"])
        return (x, y, z)
    if kind == "chunks":
        return (int(rule["x_factor"]) * nc, h, 1)
    if kind == "pairs":
        if nc % 2:
            raise ValueError(f"record {name}: num_chunks_pad={nc} is not even")
        return (nc * CHUNK // PAIR_ROWS, h, 1)
    if kind == "pairs_bounded":
        if nc % 2:
            raise ValueError(f"record {name}: num_chunks_pad={nc} is not even")
        pairs = nc * CHUNK // PAIR_ROWS * h
        return (max(1, min(pairs, int(rule["ctas_per_sm"]) * int(sm_count))), 1, 1)
    if kind == "per_doc_head":
        return (n * h * int(rule["slices"]), 1, 1)
    if kind == "sm_bounded":
        return (max(1, min(nc * h, int(rule["ctas_per_sm"]) * int(sm_count))), 1, 1)
    if kind == "docs_heads":
        return (n * h, 1, 1)
    if kind == "segments_heads":
        if int(num_segments) < 1:
            raise ValueError(
                f"record {name}: num_segments={num_segments} is not a segmented "
                "route (it needs at least one segment)"
            )
        return (int(rule["passes"][launch]) * int(num_segments) * h, 1, 1)
    raise ValueError(f"unknown Cake KDA training grid rule: {kind!r}")


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake KDA training target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake KDA training record {name} is not delivered for {arch}; "
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


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter and
    ``(None, axis)`` per grid argument, in ``run`` order; computed once."""
    record = MODULES[name]
    logical = record["logical"]
    plan: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append((None, "xyz".index(key[-1])))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], -1))
    return tuple(plan)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to
    tensors and Python scalars; the record maps its kernel parameters onto that
    vocabulary and ``grid`` is the plan's launch grid (:func:`launch_grid`).
    The kernel runs on the current torch stream of the device.
    """
    record = MODULES[name]
    args = []
    for logical_name, axis in _argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CHUNK",
    "HEAD_DIM",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "OPTIONAL_STAGES",
    "PAIR_ROWS",
    "SEGMENTS",
    "STAGES",
    "STAGES_BWD",
    "STAGES_FWD",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "launches",
    "library_name",
    "load",
    "records_of_stage",
    "run_module",
    "segment_plan",
    "select_record",
    "sm_count",
    "specs_for_arch",
]
