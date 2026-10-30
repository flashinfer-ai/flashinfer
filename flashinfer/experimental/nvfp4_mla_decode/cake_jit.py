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

Registry and loader glue of the Cake NVFP4 MLA decode (SM100 / SM103), the
``backend="cake"`` of
:func:`flashinfer.mla.prepare_nvfp4_batch_decode_with_kv_cache_mla`: the
persistent decode over a host work table and the split-KV combine behind it.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  The records are keyed by name and each names the stage it fills
(``stage``, one of ``STAGES``: ``decode``, the persistent decode, one cluster
of CTAs per work unit; ``reduce``, the split-KV combine, launched only when the
plan has more than one split).  Each stage may have further records, which
the host picks once per plan; the programs of one stage give bitwise
identical outputs.  The decode stage's are ``decode_qk_ahead``, an SM100
decode (the QK MMAs issued two tiles ahead of PV, its row-pair max exchange
gated by a vote on the exchange barrier) that ``select_decode_record`` picks
for long balanced plans, ``decode_xv``, an SM103 decode with that gated
exchange, which ``select_decode_record`` picks for plans whose every unit
streams long pieces, ``decode_short``, a decode that reads its unit range in
the kernel setup and frees its tensor memory once the last epilogue has read
it, which ``select_decode_record`` picks for plans of short units, and
``decode_short_epi_warp``, that decode with its epilogue store path picked
per softmax warp, which ``select_decode_record`` picks in its place when
every request ends in a CTA of at least 32 valid query rows, and
``decode_short_spec_uh`` and ``decode_short_epi_warp_spec_uh``, those
two decodes loading the first item of their unit in the kernel setup, which
``select_decode_record`` picks in place of either when every unit holds one
item pair; the combine's is
``reduce_general``, a combine that reads the first pair of splits of a row
from its top loads, which ``select_reduce_record`` picks for plans of more
than two splits per row.  A renamed program still changes no host code.  A
record carries the host plan only: the
program's ``library`` stem (its ``build.json`` entries are
``<library>_<arch>``), the architectures it is delivered for, the source
``module`` and ``build`` keywords it was rendered from, its launch record, the
argument plan of its ``run`` entry, the kernel-parameter -> logical-argument
map, the grid rule :func:`launch_grid` evaluates and the constants of the host
plan the kernel reads: for the decode the work-item fields
(``item_fields``), the two item flags (``flag_seed_sink``,
``flag_direct_out``), the packed query rows of one item (``rows_per_tile``),
the keys of one pipeline tile (``tile_kv``) and the tokens of one cache page
(``page_size``); for the combine the split capacity of a row
(``max_splits``).  What to compile -- sources, digests, nvcc flags -- is
recorded once in ``build.json`` next to this file; :mod:`.cake_build` serves
the library from a prebuilt ``aot/`` file when a builder compiled it and
otherwise through FlashInfer's JIT from the same sources and flags.  The
decode entries record ``--use_fast_math`` (the flag the kernel was validated
and measured with), the combine entries none; the loader passes
``use_fast_math=False`` so that the manifest, not the loader, states it.

Tensor maps travel by value in the kernel parameters: the decode binding
encodes a ``CUtensorMap`` from each ``tma_buffer`` view of its argument plan
per call, so a launch is self-contained and CUDA-graph capturable.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: the exporter reads this file with ``ast.literal_eval``.

The program name ``cake_nvfp4_mla_decode_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted,
every default spelled>, "module": <dotted source module>}`` of the kernel the
program was rendered from; a changed keyword is a new program.  The exception
is a keyword at its implicit default (``qk_ahead``, ``short_rows``, ``xv``,
``epi_warp``, ``spec_uh`` and ``general_splits`` at ``False``): the name omits
it, so programs named before the keyword existed keep their names.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# B200 / GB200 (``sm_100a``) and B300 / GB300 (``sm_103a``), each compiled with
# its exact ``-gencode`` flag set (``build.json`` ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
# The launch order of the complete call: the decode, then the combine when the
# plan has more than one split.
STAGES = ("decode", "reduce")

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "decode": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "QS"],
            ["tma_buffer", "KV"],
            ["tma_buffer", "KVS"],
            ["tma_buffer", "OT"],
            ["tma_buffer", "PO"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "work_table"],
            ["buffer", "unit_first"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "sinks"],
            ["parameter", "rows_total"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "max_pages"],
            ["parameter", "max_splits"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "lazy_rescale": True,
            "num_p_stages": 2,
            "num_raw_stages": 3,
            "num_v_stages": 2,
        },
        "ffi_entry": "run",
        "flag_direct_out": 2,
        "flag_seed_sink": 1,
        "grid_rule": {
            "ctas": 2,
            "kind": "units_by_cluster",
        },
        "item_fields": 8,
        "kernel": "cake_nvfp4_mla_decode_4cd8124d834b7ab3a055",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_4cd8124d834b7ab3a055",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_4cd8124d834b7ab3a055",
        "logical": {
            "KV": "KV",
            "KVS": "KVS",
            "LSE": "LSE",
            "O": "O",
            "OT": "OT",
            "PO": "PO",
            "Q": "Q",
            "QS": "QS",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_len": "q_len",
            "rows_total": "rows_total",
            "scale_log2": "scale_log2",
            "seq_lens": "seq_lens",
            "sinks": "sinks",
            "unit_first": "unit_first",
            "work_table": "work_table",
        },
        "module": "tart.mla_decode_dsv4_dense_nvfp4.sm100_sm103.dsv4_nvfp4_fmha_decode",
        "page_size": 64,
        "rows_per_tile": 128,
        "stage": "decode",
        "tile_kv": 128,
    },
    "decode_qk_ahead": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "QS"],
            ["tma_buffer", "KV"],
            ["tma_buffer", "KVS"],
            ["tma_buffer", "OT"],
            ["tma_buffer", "PO"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "work_table"],
            ["buffer", "unit_first"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "sinks"],
            ["parameter", "rows_total"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "max_pages"],
            ["parameter", "max_splits"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "lazy_rescale": True,
            "num_p_stages": 2,
            "num_raw_stages": 3,
            "num_v_stages": 2,
            "qk_ahead": True,
            "xv": True,
        },
        "ffi_entry": "run",
        "flag_direct_out": 2,
        "flag_seed_sink": 1,
        "grid_rule": {
            "ctas": 2,
            "kind": "units_by_cluster",
        },
        "item_fields": 8,
        "kernel": "cake_nvfp4_mla_decode_05927afc40e38d96365b",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_05927afc40e38d96365b",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_05927afc40e38d96365b",
        "logical": {
            "KV": "KV",
            "KVS": "KVS",
            "LSE": "LSE",
            "O": "O",
            "OT": "OT",
            "PO": "PO",
            "Q": "Q",
            "QS": "QS",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_len": "q_len",
            "rows_total": "rows_total",
            "scale_log2": "scale_log2",
            "seq_lens": "seq_lens",
            "sinks": "sinks",
            "unit_first": "unit_first",
            "work_table": "work_table",
        },
        "module": "tart.mla_decode_dsv4_dense_nvfp4.sm100_sm103.dsv4_nvfp4_fmha_decode",
        "page_size": 64,
        "rows_per_tile": 128,
        "stage": "decode",
        "tile_kv": 128,
    },
    "decode_short": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "QS"],
            ["tma_buffer", "KV"],
            ["tma_buffer", "KVS"],
            ["tma_buffer", "OT"],
            ["tma_buffer", "PO"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "work_table"],
            ["buffer", "unit_first"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "sinks"],
            ["parameter", "rows_total"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "max_pages"],
            ["parameter", "max_splits"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "lazy_rescale": True,
            "num_p_stages": 2,
            "num_raw_stages": 3,
            "num_v_stages": 2,
            "short_rows": True,
        },
        "ffi_entry": "run",
        "flag_direct_out": 2,
        "flag_seed_sink": 1,
        "grid_rule": {
            "ctas": 2,
            "kind": "units_by_cluster",
        },
        "item_fields": 8,
        "kernel": "cake_nvfp4_mla_decode_7d6b9ae94df533a027ea",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_7d6b9ae94df533a027ea",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_7d6b9ae94df533a027ea",
        "logical": {
            "KV": "KV",
            "KVS": "KVS",
            "LSE": "LSE",
            "O": "O",
            "OT": "OT",
            "PO": "PO",
            "Q": "Q",
            "QS": "QS",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_len": "q_len",
            "rows_total": "rows_total",
            "scale_log2": "scale_log2",
            "seq_lens": "seq_lens",
            "sinks": "sinks",
            "unit_first": "unit_first",
            "work_table": "work_table",
        },
        "module": "tart.mla_decode_dsv4_dense_nvfp4.sm100_sm103.dsv4_nvfp4_fmha_decode",
        "page_size": 64,
        "rows_per_tile": 128,
        "stage": "decode",
        "tile_kv": 128,
    },
    "decode_short_epi_warp": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "QS"],
            ["tma_buffer", "KV"],
            ["tma_buffer", "KVS"],
            ["tma_buffer", "OT"],
            ["tma_buffer", "PO"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "work_table"],
            ["buffer", "unit_first"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "sinks"],
            ["parameter", "rows_total"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "max_pages"],
            ["parameter", "max_splits"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "epi_warp": True,
            "lazy_rescale": True,
            "num_p_stages": 2,
            "num_raw_stages": 3,
            "num_v_stages": 2,
            "short_rows": True,
        },
        "ffi_entry": "run",
        "flag_direct_out": 2,
        "flag_seed_sink": 1,
        "grid_rule": {
            "ctas": 2,
            "kind": "units_by_cluster",
        },
        "item_fields": 8,
        "kernel": "cake_nvfp4_mla_decode_e088eefe4cdbfba026ee",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_e088eefe4cdbfba026ee",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_e088eefe4cdbfba026ee",
        "logical": {
            "KV": "KV",
            "KVS": "KVS",
            "LSE": "LSE",
            "O": "O",
            "OT": "OT",
            "PO": "PO",
            "Q": "Q",
            "QS": "QS",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_len": "q_len",
            "rows_total": "rows_total",
            "scale_log2": "scale_log2",
            "seq_lens": "seq_lens",
            "sinks": "sinks",
            "unit_first": "unit_first",
            "work_table": "work_table",
        },
        "module": "tart.mla_decode_dsv4_dense_nvfp4.sm100_sm103.dsv4_nvfp4_fmha_decode",
        "page_size": 64,
        "rows_per_tile": 128,
        "stage": "decode",
        "tile_kv": 128,
    },
    "decode_short_spec_uh": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "QS"],
            ["tma_buffer", "KV"],
            ["tma_buffer", "KVS"],
            ["tma_buffer", "OT"],
            ["tma_buffer", "PO"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "work_table"],
            ["buffer", "unit_first"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "sinks"],
            ["parameter", "rows_total"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "max_pages"],
            ["parameter", "max_splits"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "lazy_rescale": True,
            "num_p_stages": 2,
            "num_raw_stages": 3,
            "num_v_stages": 2,
            "short_rows": True,
            "spec_uh": True,
        },
        "ffi_entry": "run",
        "flag_direct_out": 2,
        "flag_seed_sink": 1,
        "grid_rule": {
            "ctas": 2,
            "kind": "units_by_cluster",
        },
        "item_fields": 8,
        "kernel": "cake_nvfp4_mla_decode_ee7dbd4113f22166da8c",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_ee7dbd4113f22166da8c",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_ee7dbd4113f22166da8c",
        "logical": {
            "KV": "KV",
            "KVS": "KVS",
            "LSE": "LSE",
            "O": "O",
            "OT": "OT",
            "PO": "PO",
            "Q": "Q",
            "QS": "QS",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_len": "q_len",
            "rows_total": "rows_total",
            "scale_log2": "scale_log2",
            "seq_lens": "seq_lens",
            "sinks": "sinks",
            "unit_first": "unit_first",
            "work_table": "work_table",
        },
        "module": "tart.mla_decode_dsv4_dense_nvfp4.sm100_sm103.dsv4_nvfp4_fmha_decode",
        "page_size": 64,
        "rows_per_tile": 128,
        "stage": "decode",
        "tile_kv": 128,
    },
    "decode_short_epi_warp_spec_uh": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "QS"],
            ["tma_buffer", "KV"],
            ["tma_buffer", "KVS"],
            ["tma_buffer", "OT"],
            ["tma_buffer", "PO"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "work_table"],
            ["buffer", "unit_first"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "sinks"],
            ["parameter", "rows_total"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "max_pages"],
            ["parameter", "max_splits"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "epi_warp": True,
            "lazy_rescale": True,
            "num_p_stages": 2,
            "num_raw_stages": 3,
            "num_v_stages": 2,
            "short_rows": True,
            "spec_uh": True,
        },
        "ffi_entry": "run",
        "flag_direct_out": 2,
        "flag_seed_sink": 1,
        "grid_rule": {
            "ctas": 2,
            "kind": "units_by_cluster",
        },
        "item_fields": 8,
        "kernel": "cake_nvfp4_mla_decode_5644331f54be9d9aceec",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_5644331f54be9d9aceec",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_5644331f54be9d9aceec",
        "logical": {
            "KV": "KV",
            "KVS": "KVS",
            "LSE": "LSE",
            "O": "O",
            "OT": "OT",
            "PO": "PO",
            "Q": "Q",
            "QS": "QS",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_len": "q_len",
            "rows_total": "rows_total",
            "scale_log2": "scale_log2",
            "seq_lens": "seq_lens",
            "sinks": "sinks",
            "unit_first": "unit_first",
            "work_table": "work_table",
        },
        "module": "tart.mla_decode_dsv4_dense_nvfp4.sm100_sm103.dsv4_nvfp4_fmha_decode",
        "page_size": 64,
        "rows_per_tile": 128,
        "stage": "decode",
        "tile_kv": 128,
    },
    "decode_xv": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "QS"],
            ["tma_buffer", "KV"],
            ["tma_buffer", "KVS"],
            ["tma_buffer", "OT"],
            ["tma_buffer", "PO"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "work_table"],
            ["buffer", "unit_first"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "sinks"],
            ["parameter", "rows_total"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "max_pages"],
            ["parameter", "max_splits"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "lazy_rescale": True,
            "num_p_stages": 2,
            "num_raw_stages": 3,
            "num_v_stages": 2,
            "xv": True,
        },
        "ffi_entry": "run",
        "flag_direct_out": 2,
        "flag_seed_sink": 1,
        "grid_rule": {
            "ctas": 2,
            "kind": "units_by_cluster",
        },
        "item_fields": 8,
        "kernel": "cake_nvfp4_mla_decode_0c668e5be310599fc6ee",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_0c668e5be310599fc6ee",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_0c668e5be310599fc6ee",
        "logical": {
            "KV": "KV",
            "KVS": "KVS",
            "LSE": "LSE",
            "O": "O",
            "OT": "OT",
            "PO": "PO",
            "Q": "Q",
            "QS": "QS",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_len": "q_len",
            "rows_total": "rows_total",
            "scale_log2": "scale_log2",
            "seq_lens": "seq_lens",
            "sinks": "sinks",
            "unit_first": "unit_first",
            "work_table": "work_table",
        },
        "module": "tart.mla_decode_dsv4_dense_nvfp4.sm100_sm103.dsv4_nvfp4_fmha_decode",
        "page_size": 64,
        "rows_per_tile": 128,
        "stage": "decode",
        "tile_kv": 128,
    },
    "reduce": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "o"],
            ["buffer", "lse"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "row_splits"],
            ["parameter", "total_q"],
            ["parameter", "num_heads"],
            ["parameter", "max_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "grid_rule": {
            "heads_per_cta": 16,
            "kind": "rows_by_heads",
        },
        "kernel": "cake_nvfp4_mla_decode_32bd3b7e232d00a5d04e",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_32bd3b7e232d00a5d04e",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_32bd3b7e232d00a5d04e",
        "logical": {
            "lse": "lse",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "o": "o",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "row_splits": "row_splits",
            "total_q": "total_q",
        },
        "max_splits": 64,
        "module": "tart.attention_split_kv_merge_bf16_rowsplits_with_lse.sm100_sm103.dsv4_nvfp4_fmha_decode_reduce",
        "stage": "reduce",
    },
    "reduce_general": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "o"],
            ["buffer", "lse"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "row_splits"],
            ["parameter", "total_q"],
            ["parameter", "num_heads"],
            ["parameter", "max_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"general_splits": True},
        "ffi_entry": "run",
        "grid_rule": {
            "heads_per_cta": 16,
            "kind": "rows_by_heads",
        },
        "kernel": "cake_nvfp4_mla_decode_d3e21f744c0427fef73d",
        "kernel_symbol": "kernel_cake_nvfp4_mla_decode_d3e21f744c0427fef73d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_mla_decode_d3e21f744c0427fef73d",
        "logical": {
            "lse": "lse",
            "max_splits": "max_splits",
            "num_heads": "num_heads",
            "o": "o",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "row_splits": "row_splits",
            "total_q": "total_q",
        },
        "max_splits": 64,
        "module": "tart.attention_split_kv_merge_bf16_rowsplits_with_lse.sm100_sm103.dsv4_nvfp4_fmha_decode_reduce",
        "stage": "reduce",
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
            f"Cake NVFP4 MLA decode target; the exported architectures are {ARCHES}"
        )
    return arch


def program_available(arch: str) -> bool:
    """True when the record named after every stage is delivered for ``arch``
    (a stage's further records are optional programs the host selects only
    where they are delivered)."""
    return all(
        stage in MODULES and arch in MODULES[stage]["arches"] for stage in STAGES
    )


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake NVFP4 MLA decode record {name} is not delivered for {arch}; "
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


def launch_grid(
    record: dict[str, Any],
    *,
    units: int | None = None,
    total_q: int | None = None,
    num_heads: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one record under its grid rule: ``units_by_cluster``, one
    cluster of ``ctas`` CTAs per work unit of the plan (``units``);
    ``rows_by_heads``, one CTA per query row (``total_q``) and ``heads_per_cta``
    heads (``num_heads``)."""
    rule = record["grid_rule"]
    if rule["kind"] == "units_by_cluster":
        if units is None or int(units) <= 0:
            raise ValueError(f"the {rule['kind']} grid needs a positive unit count")
        return (int(rule["ctas"]) * int(units), 1, 1)
    if rule["kind"] == "rows_by_heads":
        if (
            total_q is None
            or num_heads is None
            or min(int(total_q), int(num_heads)) <= 0
        ):
            raise ValueError(f"the {rule['kind']} grid needs positive rows and heads")
        heads_per_cta = int(rule["heads_per_cta"])
        return (int(total_q), -(-int(num_heads) // heads_per_cta), 1)
    raise ValueError(f"unknown Cake NVFP4 MLA decode grid rule: {rule['kind']!r}")


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
    "MODULES",
    "STAGES",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "program_available",
    "run_module",
    "specs_for_arch",
]
