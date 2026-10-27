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

JIT registration of the Cake-generated Blackwell MSA programs.  Device sources
are shared by the flat / paged / L2 evict-first programs of a schedule and host
bindings by argument plan; each program is selected by its record's compile
flags and compiled with the exact flag set of the device it runs on; the
architecture is part of the JIT spec name so two architectures never share one
cached library.  ``ARG_PLANS`` lists the host binding's argument order once per
route family; each ``MODULES`` record names its plan.
"""

import functools
import re
from pathlib import Path
from typing import Any, Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

BlackwellMSATarget = Literal["sm100a", "sm103a"]
BlackwellMSAVariant = str
_NVCC_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_TARGET_ARCH: dict[BlackwellMSATarget, str] = {"sm100a": "sm_100a", "sm103a": "sm_103a"}

_DEFAULT_ARCHES = ["sm_100a", "sm_103a"]
_DEFAULT_COMPILE_FLAGS = ["--use_fast_math"]
_HEX20 = re.compile(r"^[0-9a-f]{20}$")

# One row per program.  Columns: ``id`` (the 20-hex program digest), ``plan`` (an ``ARG_PLANS`` key) and
# the optional overrides ``arches`` (default both targets), ``compile_flags`` (default
# ``--use_fast_math``), ``kernel`` / ``binding`` (default the program's own
# ``cake_blackwell_msa_<id>_{kernel,binding}.cu``; a 20-hex value names another program's file, any
# other value a file stem under ``blackwell_msa/``), ``defines`` (extra ``-D`` flags appended after the
# compile flags) and ``target_compile_flags`` (``{target: [flags]}`` appended for that target only).  A
# row that shares a kernel or binding source compiles with ``-DCAKE_PROGRAM_KERNEL=kernel_<its own
# name>`` (derived here, never written in a row), so every program keeps its own kernel symbol.
# ``_expand_modules`` turns the rows into the ``MODULES`` records the dispatcher reads.
# fmt: off
_MODULE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "0e41f3f76fefcd451bc1", "plan": "decode_m16", "kernel": "4c104ef3b4ee8c2fff55", "binding": "cake_blackwell_msa_decode_m16", "defines": ["-DCAKE_BIND_SMEM=216704", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_L2_EVICT_FIRST=1"]},
    {"id": "1a404b555f04cdde1d2c", "plan": "prefill_union", "kernel": "9d11f8cbc9df656a767b", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=201728", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=0", "-DCAKE_FLAT_KV=1"]},
    {"id": "25c43f61f501a6d4c567", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=212992", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=16"]},
    {"id": "2a87e650f646a2de9df4", "plan": "decode_m16", "binding": "cake_blackwell_msa_decode_m16", "defines": ["-DCAKE_BIND_SMEM=216704", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=1"]},
    {"id": "37190e3efb7c5f267e04", "plan": "long_bf16_reverse_reduce", "binding": "cake_blackwell_msa_long_bf16_reverse_reduce", "defines": ["-DCAKE_BIND_SMEM=40960"]},
    {"id": "37f16ecc96a0bc9d5335", "plan": "long_bf16_reverse_main", "binding": "cake_blackwell_msa_long_bf16_reverse_main", "defines": ["-DCAKE_BIND_SMEM=148480", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_GQA=16"]},
    {"id": "3cb239103137becab6db", "plan": "topk_select", "compile_flags": [], "binding": "cake_blackwell_msa_topk_select", "defines": ["-DCAKE_BIND_SMEM=85632"]},
    {"id": "42c6eced172576d57c92", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=218112", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=1", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=0"]},
    {"id": "4c104ef3b4ee8c2fff55", "plan": "decode_m16", "binding": "cake_blackwell_msa_decode_m16", "defines": ["-DCAKE_BIND_SMEM=216704", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0"]},
    {"id": "4c5c6494c5ff2334ae2d", "plan": "decode_uniform_fp8"},
    {"id": "5a7c4a2da9b6e9c14215", "plan": "long_bf16_reverse_reduce", "binding": "cake_blackwell_msa_long_bf16_reverse_reduce", "defines": ["-DCAKE_BIND_SMEM=40960"]},
    {"id": "5c05cfb4c595d36a27f3", "plan": "long_bf16_reverse_main", "kernel": "37f16ecc96a0bc9d5335", "binding": "cake_blackwell_msa_long_bf16_reverse_main", "defines": ["-DCAKE_BIND_SMEM=148480", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_GQA=16", "-DCAKE_FLAT_KV=1"]},
    {"id": "5ca8b7d363b9c6866923", "plan": "prefill_union", "kernel": "c1141bb98fe0a1e6789a", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=205824", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=8", "-DCAKE_FLAT_KV=1"]},
    {"id": "616f5bbf2b76960e1e8c", "plan": "decode_m16", "kernel": "991dc4bce7a340d2008d", "binding": "cake_blackwell_msa_decode_m16", "defines": ["-DCAKE_BIND_SMEM=215680", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=1", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_FLAT_KV=1"]},
    {"id": "64a3ba42bd5dc74b92f5", "plan": "prefill_union", "kernel": "9c09f71e5846b7d0ce99", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=201728", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=1", "-DCAKE_BIND_GQA=0", "-DCAKE_FLAT_KV=1"]},
    {"id": "6ed84157fc9c633d8a00", "plan": "long_bf16_reverse_main", "binding": "cake_blackwell_msa_long_bf16_reverse_main", "defines": ["-DCAKE_BIND_SMEM=148480", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_GQA=8"]},
    {"id": "7e5235eb5efbcabc3637", "plan": "decode_fp8_q1_flat_xform2"},
    {"id": "82e6a9cf66fe33160b74", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=205824", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=16"]},
    {"id": "89bcd02bb77ea339d039", "plan": "kmaj_p_reduce"},
    {"id": "9457a50f3f40d584c6c7", "plan": "topk_select_streaming", "compile_flags": []},
    {"id": "96eda75a396e3001f22e", "plan": "prefill_union", "kernel": "42c6eced172576d57c92", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=218112", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=1", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=0", "-DCAKE_FLAT_KV=1"]},
    {"id": "991dc4bce7a340d2008d", "plan": "decode_m16", "binding": "cake_blackwell_msa_decode_m16", "defines": ["-DCAKE_BIND_SMEM=215680", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=1", "-DCAKE_BIND_Q_FP16=0"]},
    {"id": "9c09f71e5846b7d0ce99", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=201728", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=1", "-DCAKE_BIND_GQA=0"]},
    {"id": "9d11f8cbc9df656a767b", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=201728", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=0"]},
    {"id": "b5e804da75b24237502b", "plan": "topk_select", "compile_flags": [], "binding": "cake_blackwell_msa_topk_select", "defines": ["-DCAKE_BIND_SMEM=75648"]},
    {"id": "bae868b83abdfe541640", "plan": "kmaj_p_main", "binding": "cake_blackwell_msa_kmaj_p_main", "defines": ["-DCAKE_BIND_SMEM=197632", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_GQA=8"]},
    {"id": "c1141bb98fe0a1e6789a", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=205824", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=8"]},
    {"id": "cc9f8c16164bac3958eb", "plan": "decode_m16", "kernel": "4c104ef3b4ee8c2fff55", "binding": "cake_blackwell_msa_decode_m16", "defines": ["-DCAKE_BIND_SMEM=216704", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_FLAT_KV=1"]},
    {"id": "cd19e83ad952065bc779", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=205824", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=16"]},
    {"id": "d28ef1a700e18456846d", "plan": "prefill_union", "binding": "cake_blackwell_msa_prefill_union", "defines": ["-DCAKE_BIND_SMEM=224256", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=0", "-DCAKE_BIND_GQA=16"]},
    {"id": "dce0ec63212385c16c81", "plan": "topk_select", "compile_flags": [], "binding": "cake_blackwell_msa_topk_select", "target_compile_flags": {"sm100a": ["-DCAKE_BIND_SMEM=75520"], "sm103a": ["-DCAKE_BIND_SMEM=75648"]}},
    {"id": "e272f321506e704d7720", "plan": "kmaj_p_main", "kernel": "ec5682e5fca1b9f91612", "binding": "cake_blackwell_msa_kmaj_p_main", "defines": ["-DCAKE_BIND_SMEM=197632", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_GQA=16", "-DCAKE_FLAT_KV=1"]},
    {"id": "e635b4b314bb670500b5", "plan": "decode_fp8_q1_paged_xform2"},
    {"id": "ec5682e5fca1b9f91612", "plan": "kmaj_p_main", "binding": "cake_blackwell_msa_kmaj_p_main", "defines": ["-DCAKE_BIND_SMEM=197632", "-DCAKE_BIND_PAGED=1", "-DCAKE_BIND_GQA=16"]},
    {"id": "f15033e922adf9476885", "plan": "decode_m16", "kernel": "2a87e650f646a2de9df4", "binding": "cake_blackwell_msa_decode_m16", "defines": ["-DCAKE_BIND_SMEM=216704", "-DCAKE_BIND_PAGED=0", "-DCAKE_BIND_KV_FP8=0", "-DCAKE_BIND_Q_FP16=1", "-DCAKE_FLAT_KV=1"]},
)
# fmt: on


def _source_path(row_id: str, kind: str, override: str | None) -> str:
    stem = row_id if override is None else override
    if _HEX20.match(stem):
        stem = f"cake_blackwell_msa_{stem}"
    return f"blackwell_msa/{stem}_{kind}.cu"


def _expand_modules(rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row["id"]
        compile_flags = list(row.get("compile_flags", _DEFAULT_COMPILE_FLAGS))
        if "kernel" in row or "binding" in row:
            compile_flags.append(
                f"-DCAKE_PROGRAM_KERNEL=kernel_cake_blackwell_msa_{row_id}"
            )
        compile_flags += list(row.get("defines", ()))
        record: dict[str, Any] = {
            "sources": [
                _source_path(row_id, "kernel", row.get("kernel")),
                _source_path(row_id, "binding", row.get("binding")),
            ],
            "compile_flags": compile_flags,
            "ffi_entry": "run",
            "arg_plan": row["plan"],
            "arches": list(row.get("arches", _DEFAULT_ARCHES)),
        }
        if "target_compile_flags" in row:
            record["target_compile_flags"] = {
                target: list(flags)
                for target, flags in row["target_compile_flags"].items()
            }
        modules[f"cake_blackwell_msa_{row_id}"] = record
    return modules


MODULES: dict[str, dict[str, Any]] = _expand_modules(_MODULE_ROWS)

ARG_PLANS: dict[str, list[list[str]]] = {
    "decode_fp8_q1_flat_xform2": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "V"],
        ["buffer", "O"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "kv_indptr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["parameter", "num_requests"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "msa_max_pages"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "decode_fp8_q1_paged_xform2": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "V"],
        ["buffer", "O"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "kv_indptr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["parameter", "num_requests"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "msa_max_pages"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "decode_m16": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "Q_prefill"],
        ["buffer", "Q_prefill_raw"],
        ["tma_buffer", "K"],
        ["tma_buffer", "K_scale"],
        ["tma_buffer", "K_prefill_pair"],
        ["tma_buffer", "V"],
        ["tma_buffer", "V_scale"],
        ["tma_buffer", "V_prefill_pair"],
        ["tma_buffer", "KV"],
        ["buffer", "O"],
        ["buffer", "partial_O"],
        ["buffer", "partial_M"],
        ["buffer", "partial_D"],
        ["buffer", "split_completion"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "qo_indptr"],
        ["buffer", "kv_indptr"],
        ["buffer", "kv_len_arr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["buffer", "task_q_tile"],
        ["buffer", "task_split"],
        ["buffer", "task_kv_tile_begin"],
        ["buffer", "task_kv_tile_end"],
        ["buffer", "task_qo_begin"],
        ["buffer", "task_qo_end"],
        ["buffer", "task_page_begin"],
        ["buffer", "task_page_end"],
        ["buffer", "status"],
        ["parameter", "num_requests"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "max_kv_tiles"],
        ["parameter", "max_splits"],
        ["parameter", "max_task_claims"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "k_global_scale"],
        ["parameter", "v_global_scale"],
        ["parameter", "attention_mode"],
        ["parameter", "is_causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "record_tasks"],
        ["parameter", "msa_max_pages"],
        ["parameter", "msa_split_policy"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "decode_uniform_fp8": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "K_scale"],
        ["tma_buffer", "V"],
        ["tma_buffer", "V_scale"],
        ["buffer", "O"],
        ["buffer", "partial_O"],
        ["buffer", "partial_M"],
        ["buffer", "partial_D"],
        ["buffer", "split_completion"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "kv_indptr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["parameter", "total_q"],
        ["parameter", "seqlen_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "output_scale"],
        ["parameter", "msa_max_pages"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "kmaj_p_main": [
        ["tma_buffer", "q"],
        ["tma_buffer", "k"],
        ["tma_buffer", "v"],
        ["buffer", "scheduler_metadata"],
        ["buffer", "k2q_row_ptr"],
        ["buffer", "k2q_qsplit_indices"],
        ["buffer", "partial_o"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["buffer", "q_offsets"],
        ["buffer", "kv_lens"],
        ["buffer", "page_table"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "total_rows"],
        ["parameter", "nnz_per_head"],
        ["parameter", "num_work_items"],
        ["parameter", "topk"],
        ["parameter", "max_pages"],
        ["parameter", "causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "lse_temperature_scale"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "kmaj_p_reduce": [
        ["buffer", "partial_o"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "split_counts"],
        ["buffer", "out"],
        ["buffer", "lse"],
        ["buffer", "temperature_lse"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "qhead_per_kv"],
        ["parameter", "topk"],
        ["parameter", "return_softmax_lse"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "long_bf16_reverse_main": [
        ["tma_buffer", "q"],
        ["tma_buffer", "k"],
        ["tma_buffer", "v"],
        ["buffer", "scheduler_metadata"],
        ["buffer", "k2q_row_ptr"],
        ["buffer", "k2q_qsplit_indices"],
        ["buffer", "partial_o"],
        ["buffer", "partial_scale"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "out"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["buffer", "q_offsets"],
        ["buffer", "kv_lens"],
        ["buffer", "page_table"],
        ["parameter", "q_group_segment_end_128"],
        ["parameter", "q_group_segment_end_64"],
        ["parameter", "q_group_segment_end_32"],
        ["parameter", "q_group_segment_end_16"],
        ["parameter", "q_group_segment_end_8"],
        ["parameter", "q_group_segment_end_4"],
        ["parameter", "q_group_segment_end_2"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "total_rows"],
        ["parameter", "nnz_per_head"],
        ["parameter", "work_capacity"],
        ["parameter", "num_work_items"],
        ["parameter", "topk"],
        ["parameter", "max_pages"],
        ["parameter", "causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "lse_temperature_scale"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "long_bf16_reverse_reduce": [
        ["buffer", "partial_o"],
        ["buffer", "partial_scale"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "split_counts"],
        ["buffer", "out"],
        ["buffer", "lse"],
        ["buffer", "temperature_lse"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "qhead_per_kv"],
        ["parameter", "topk"],
        ["parameter", "return_softmax_lse"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "prefill_union": [
        ["tma_buffer", "q"],
        ["tma_buffer", "k"],
        ["tma_buffer", "k_scale"],
        ["tma_buffer", "v"],
        ["tma_buffer", "v_scale"],
        ["buffer", "out"],
        ["buffer", "lse"],
        ["buffer", "temperature_lse"],
        ["buffer", "q2k_indices"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["buffer", "q_offsets"],
        ["buffer", "kv_lens"],
        ["buffer", "page_table"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "topk"],
        ["parameter", "batch_size"],
        ["parameter", "uniform_q_len"],
        ["parameter", "max_pages"],
        ["parameter", "causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "k_global_scale"],
        ["parameter", "v_global_scale"],
        ["parameter", "lse_temperature_scale"],
        ["parameter", "return_softmax_lse"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "topk_select": [
        ["buffer", "max_score"],
        ["buffer", "output"],
        ["buffer", "partials"],
        ["buffer", "counters"],
        ["buffer", "block_max"],
        ["parameter", "num_heads"],
        ["parameter", "max_k_tiles"],
        ["parameter", "total_q"],
        ["parameter", "num_valid_pages"],
        ["parameter", "force_begin_blocks"],
        ["parameter", "force_end_blocks"],
        ["parameter", "num_qtiles"],
        ["parameter", "num_slices"],
        ["parameter", "slice_len"],
        ["parameter", "index_bits"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "topk_select_streaming": [
        ["buffer", "max_score"],
        ["buffer", "output"],
        ["parameter", "num_heads"],
        ["parameter", "max_k_tiles"],
        ["parameter", "total_q"],
        ["parameter", "num_valid_pages"],
        ["parameter", "force_begin_blocks"],
        ["parameter", "force_end_blocks"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

# Logical route -> program, the same on both targets except the per-target overrides below.
_ROUTES_COMMON: dict[str, str] = {
    "decode_fp8_q1:q1_flat_xform2:main": "cake_blackwell_msa_7e5235eb5efbcabc3637",
    "decode_fp8_q1:q1_paged_xform2:main": "cake_blackwell_msa_e635b4b314bb670500b5",
    "decode_m16:bfloat16:bfloat16:flat:main": "cake_blackwell_msa_cc9f8c16164bac3958eb",
    "decode_m16:bfloat16:bfloat16:paged:l2ef:main": "cake_blackwell_msa_0e41f3f76fefcd451bc1",
    "decode_m16:bfloat16:bfloat16:paged:main": "cake_blackwell_msa_4c104ef3b4ee8c2fff55",
    "decode_m16:bfloat16:float8_e4m3fn:flat:main": "cake_blackwell_msa_616f5bbf2b76960e1e8c",
    "decode_m16:bfloat16:float8_e4m3fn:paged:main": "cake_blackwell_msa_991dc4bce7a340d2008d",
    "decode_m16:float16:float16:flat:main": "cake_blackwell_msa_f15033e922adf9476885",
    "decode_m16:float16:float16:paged:main": "cake_blackwell_msa_2a87e650f646a2de9df4",
    "decode_uniform_fp8:paged:main": "cake_blackwell_msa_4c5c6494c5ff2334ae2d",
    "kmaj_p:flat:gqa16:main": "cake_blackwell_msa_e272f321506e704d7720",
    "kmaj_p:flat:gqa16:reduce": "cake_blackwell_msa_89bcd02bb77ea339d039",
    "kmaj_p:paged:gqa16:main": "cake_blackwell_msa_ec5682e5fca1b9f91612",
    "kmaj_p:paged:gqa16:reduce": "cake_blackwell_msa_89bcd02bb77ea339d039",
    "kmaj_p:paged:gqa8:main": "cake_blackwell_msa_bae868b83abdfe541640",
    "kmaj_p:paged:gqa8:reduce": "cake_blackwell_msa_89bcd02bb77ea339d039",
    "long_bf16_reverse:flat:gqa16:main": "cake_blackwell_msa_5c05cfb4c595d36a27f3",
    "long_bf16_reverse:flat:gqa16:reduce": "cake_blackwell_msa_5a7c4a2da9b6e9c14215",
    "long_bf16_reverse:paged:gqa16:main": "cake_blackwell_msa_37f16ecc96a0bc9d5335",
    "long_bf16_reverse:paged:gqa16:reduce": "cake_blackwell_msa_5a7c4a2da9b6e9c14215",
    "long_bf16_reverse:paged:gqa8:main": "cake_blackwell_msa_6ed84157fc9c633d8a00",
    "long_bf16_reverse:paged:gqa8:reduce": "cake_blackwell_msa_37190e3efb7c5f267e04",
    "prefill_union:bfloat16:bfloat16:flat:gqa0:any:main": "cake_blackwell_msa_1a404b555f04cdde1d2c",
    "prefill_union:bfloat16:bfloat16:flat:gqa16:any:main": "cake_blackwell_msa_cd19e83ad952065bc779",
    "prefill_union:bfloat16:bfloat16:flat:gqa8:any:main": "cake_blackwell_msa_5ca8b7d363b9c6866923",
    "prefill_union:bfloat16:bfloat16:paged:gqa0:any:main": "cake_blackwell_msa_9d11f8cbc9df656a767b",
    "prefill_union:bfloat16:bfloat16:paged:gqa16:any:main": "cake_blackwell_msa_82e6a9cf66fe33160b74",
    "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_large:main": "cake_blackwell_msa_25c43f61f501a6d4c567",
    "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_mask64:main": "cake_blackwell_msa_d28ef1a700e18456846d",
    "prefill_union:bfloat16:bfloat16:paged:gqa8:any:main": "cake_blackwell_msa_c1141bb98fe0a1e6789a",
    "prefill_union:bfloat16:float8_e4m3fn:flat:gqa0:any:main": "cake_blackwell_msa_96eda75a396e3001f22e",
    "prefill_union:bfloat16:float8_e4m3fn:paged:gqa0:any:main": "cake_blackwell_msa_42c6eced172576d57c92",
    "prefill_union:float16:float16:flat:gqa0:any:main": "cake_blackwell_msa_64a3ba42bd5dc74b92f5",
    "prefill_union:float16:float16:paged:gqa0:any:main": "cake_blackwell_msa_9c09f71e5846b7d0ce99",
    "topk_select:main": "cake_blackwell_msa_dce0ec63212385c16c81",
    "topk_select_keq:main": "cake_blackwell_msa_b5e804da75b24237502b",
    "topk_select_q32:main": "cake_blackwell_msa_3cb239103137becab6db",
    "topk_select_streaming:main": "cake_blackwell_msa_9457a50f3f40d584c6c7",
}
_ROUTE_OVERRIDES: dict[str, dict[str, str]] = {
    "sm100a": {},
    "sm103a": {},
}
ROUTES: dict[str, dict[str, str]] = {
    target: {**_ROUTES_COMMON, **_ROUTE_OVERRIDES[target]} for target in _TARGET_ARCH
}

# Programs built for each target (ahead-of-time build list).
BLACKWELL_MSA_VARIANTS_BY_TARGET: dict[
    BlackwellMSATarget, tuple[BlackwellMSAVariant, ...]
] = {
    target: tuple(name for name, record in MODULES.items() if arch in record["arches"])
    for target, arch in _TARGET_ARCH.items()
}
BLACKWELL_MSA_VARIANTS: tuple[BlackwellMSAVariant, ...] = tuple(MODULES)


def _csrc_dir() -> Path:
    for root in (
        jit_env.FLASHINFER_CSRC_DIR,
        Path(__file__).resolve().parents[2] / "csrc",
    ):
        if (root / "blackwell_msa").exists():
            return root
    raise FileNotFoundError(
        "Blackwell MSA CUDA sources were not found under csrc/blackwell_msa"
    )


def _include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    return Path(__file__).resolve().parents[2] / "include"


def route_program(key: str, target: BlackwellMSATarget) -> str:
    """Program name of one logical route (``<route key>:<stage>``) on ``target``."""
    try:
        return ROUTES[target][key]
    except KeyError:
        raise KeyError(
            f"no Blackwell MSA program for route {key!r} on {target}"
        ) from None


def get_blackwell_msa_uri(name: BlackwellMSAVariant, target: BlackwellMSATarget) -> str:
    """Target-specific JIT / AOT key of one program (the two targets never share a library)."""
    if target not in _TARGET_ARCH:
        raise ValueError(f"unsupported Blackwell MSA target: {target}")
    record = MODULES.get(name)
    if record is None or _TARGET_ARCH[target] not in record["arches"]:
        raise ValueError(f"unsupported Blackwell MSA program/target: {name}/{target}")
    return f"blackwell_msa_{name}_{target}"


@functools.cache
def gen_blackwell_msa_module(
    name: BlackwellMSAVariant, target: BlackwellMSATarget
) -> JitSpec:
    uri = get_blackwell_msa_uri(name, target)
    record = MODULES[name]
    csrc = _csrc_dir()
    # The record's ``compile_flags`` carry the numerics the program was built and
    # validated with; the JIT default (``use_fast_math=True``) would compile the
    # programs whose record has no ``--use_fast_math`` with a different one.
    return gen_jit_spec(
        name=uri,
        sources=[csrc / path for path in record["sources"]],
        extra_cuda_cflags=[
            *_NVCC_FLAGS[target],
            *record["compile_flags"],
            *record.get("target_compile_flags", {}).get(target, []),
        ],
        extra_include_paths=[csrc / "blackwell_msa", csrc, _include_dir()],
        use_fast_math=False,
    )


@functools.cache
def load_blackwell_msa_module(name: BlackwellMSAVariant, target: BlackwellMSATarget):
    return gen_blackwell_msa_module(name, target).build_and_load()


def get_blackwell_msa_module(name: BlackwellMSAVariant, target: BlackwellMSATarget):
    """The loaded module of one program (the entry point ``flashinfer.jit`` re-exports)."""
    return load_blackwell_msa_module(name, target)


__all__ = [
    "ARG_PLANS",
    "BLACKWELL_MSA_VARIANTS",
    "BLACKWELL_MSA_VARIANTS_BY_TARGET",
    "MODULES",
    "ROUTES",
    "BlackwellMSATarget",
    "BlackwellMSAVariant",
    "gen_blackwell_msa_module",
    "get_blackwell_msa_module",
    "get_blackwell_msa_uri",
    "load_blackwell_msa_module",
    "route_program",
]
