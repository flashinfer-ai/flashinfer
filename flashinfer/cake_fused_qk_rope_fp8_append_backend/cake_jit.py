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

Host plan and loader glue of the Cake fused QK RMSNorm + NeoX RoPE + FP8 E4M3
quantization + paged KV append kernel (SM90 / SM100 / SM103).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` carries the host plan only: the kernel's
``library`` stem, the architectures it is delivered for, its launch record,
its stage name and knobs, the grid rule and the kernel-parameter ->
logical-argument map.  What to compile -- sources, digests, nvcc flags -- is
recorded once in ``build.json`` next to this file; :mod:`.cake_build` serves
the library ``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder
compiled it and otherwise through FlashInfer's JIT from the same sources and
flags, verifying every source digest before the first build.

A record is one *stage*: a build of the kernel with one set of knobs (the
launch bound, the scan rounds of the request lookup, the early request guess,
the page-row prefetch, the fold of the clearing work into the row CTAs, the
dispatch order of the grid axes, the KV heads per clearing CTA, the width of
its zero stores and whether it reads its request length after the vote); the
head configuration, the quantization policy
(dynamic per-token scale or the static ``q_scale_inv``), the norm policy,
prefill versus decode and the caller-owned K/V route are launch parameters.
Every stage writes bit-identical outputs; the host picks one per call from
the architecture, the SM count and the problem (``STAGE_POLICY``, mirrored by
:func:`policy_knobs` / :func:`stage_for`).  The host plan is
:func:`launch_grid`: one CTA per
(packed row, KV group) plus one per (request, KV head) for the last-page
zero-fill and the request's ``split_k_flag``, or one per (row, KV group)
alone for a ``fold`` build.

The program name ``cake_fused_qk_rope_fp8_append_<20 hex>`` is the first
twenty hex digits of the SHA-256 of the compact JSON ``{"build": <build
keywords, sorted>, "module": <dotted kernel module>}`` of the kernel the
program was rendered from; the knobs of a record are those build keywords
resolved with the kernel file's defaults.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "fp8_e4m3": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 1,
            "early_rt2": False,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 0,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": False,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_ddc461f6feaaa3960c3c",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_ddc461f6feaaa3960c3c",
        # The default build where no lever gains (B200 / B300 one-wave (64, 8)
        # prefill of one request, decode the prefetch does not serve outside the
        # clearing class; GB300 rows whose clearing block exceeds half a wave).
        # 96 threads: eight Q heads, K, V and two idle eight-lane slots.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_ddc461f6feaaa3960c3c",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3",
        "workspace_rule": None,
    },
    "fp8_e4m3_early_rt21_fold1_clear_tie1": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 1,
            "early_rt2": True,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": True,
            "grid_order": 0,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": True,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_1c960609dcf4891544ae",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_1c960609dcf4891544ae",
        # Decode of at least eight KV heads between one and two waves of row CTAs:
        # the early request guess and the fold of the clearing work into the row
        # CTAs (no clearing CTAs: grid rule ``rows_by_kv_heads``).
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_1c960609dcf4891544ae",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_early_rt21_fold1_clear_tie1",
        "workspace_rule": None,
    },
    "fp8_e4m3_min_blocks10": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 10,
            "scan_rounds": 1,
            "early_rt2": False,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 0,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": False,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_4bf37605f76680f21880",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_4bf37605f76680f21880",
        # Prefill beyond one wave of row CTAs (B200 / B300; H100 / H200 when the
        # clearing block exceeds half a wave): ten CTAs per SM.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_4bf37605f76680f21880",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_min_blocks10",
        "workspace_rule": None,
    },
    "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_clear_tie1": {
        "arches": [
            "sm_90a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 9,
            "scan_rounds": 1,
            "early_rt2": True,
            "page_prefetch": True,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 0,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": True,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_ff22734cc105d1f0f779",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_ff22734cc105d1f0f779",
        # H100 / H200 decode of at least eight KV heads beyond one wave, and
        # single-KV-head decode with a short page row or the static Q scale; GB300
        # decode or single-KV-head prefill within 48 pages per request and without
        # caller-owned K/V; B300 single-KV-head decode from 512 CTAs up to two
        # waves on the same page rows: the early guess with the request's page row
        # prefetched at nine CTAs per SM.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_ff22734cc105d1f0f779",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_clear_tie1",
        "workspace_rule": None,
    },
    "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_grid_order2_clear_tie1": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "kv_heads_by_rows_plus_requests",
        },
        "knobs": {
            "min_blocks": 9,
            "scan_rounds": 1,
            "early_rt2": True,
            "page_prefetch": True,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 2,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": True,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_dc08ab78468329e60f27",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_dc08ab78468329e60f27",
        # H100 / H200 one-wave decode of at least eight KV heads with more row CTAs
        # than SMs: the prefetching guess at nine CTAs per SM with the KV group axis
        # first and the clearing CTAs dispatched before the rows (grid rule
        # ``kv_heads_by_rows_plus_requests``).
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_dc08ab78468329e60f27",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_grid_order2_clear_tie1",
        "workspace_rule": None,
    },
    "fp8_e4m3_grid_order2": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "kv_heads_by_rows_plus_requests",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 1,
            "early_rt2": False,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 2,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": False,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_c0f0b47d168812a2c23f",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_c0f0b47d168812a2c23f",
        # B200 and B300 one-wave prefill, and GB300 prefill, of at least eight KV
        # heads and two requests whose clearing block is at most half a wave: the
        # KV group axis first and the clearing CTAs dispatched before the rows.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_c0f0b47d168812a2c23f",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_grid_order2",
        "workspace_rule": None,
    },
    "fp8_e4m3_grid_order3": {
        "arches": [
            "sm_90a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 1,
            "early_rt2": False,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 3,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": False,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_343d43791fe5ef7ca3fe",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_343d43791fe5ef7ca3fe",
        # The default axes with each KV head's clearing CTAs dispatched first: H100 /
        # H200 prefill beyond one wave; GB300 rows no other lever serves -- while
        # the clearing block is at most half a wave.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_343d43791fe5ef7ca3fe",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_grid_order3",
        "workspace_rule": None,
    },
    "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 1,
            "early_rt2": True,
            "page_prefetch": True,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 0,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": True,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_9a2e2447dc736f790b38",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_9a2e2447dc736f790b38",
        # B200 and B300 decode (B300 single-KV-head decode below 512 CTAs), and
        # single-KV-head prefill, within one wave, 48 pages per request and
        # without caller-owned K/V: the early guess with the page-row prefetch.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_9a2e2447dc736f790b38",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
        "workspace_rule": None,
    },
    "fp8_e4m3_scan_rounds3_clear_tie1": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 3,
            "early_rt2": False,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 0,
            "clear_heads_per_cta": 1,
            "zero_store_width": 128,
            "clear_tie": True,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_1479818c8e3a9ce444c7",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_1479818c8e3a9ce444c7",
        # Three scan rounds of the request lookup: H100 / H200 single-KV-head decode
        # the prefetch does not serve, one-wave prefill and tiny (64, 8) decode
        # grids; B200 and B300 decode beyond two waves from 97 requests.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_1479818c8e3a9ce444c7",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_scan_rounds3_clear_tie1",
        "workspace_rule": None,
    },
    "fp8_e4m3_clear_heads_per_cta2_zero_store_width256": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 1,
            "early_rt2": False,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 0,
            "clear_heads_per_cta": 2,
            "zero_store_width": 256,
            "clear_tie": False,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_d2029163a92958e54025",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_d2029163a92958e54025",
        # GB300 single-KV-head decode of 8 to 64 requests (16 to 128 CTAs), and B300
        # the same when the page row is short (at most 24 entries) or the prefetch
        # does not serve it (a longer row, caller-owned K/V): two KV heads per
        # clearing CTA and 256-bit zero stores.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_d2029163a92958e54025",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_clear_heads_per_cta2_zero_store_width256",
        "workspace_rule": None,
    },
    "fp8_e4m3_grid_order3_zero_store_width256": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "page_indices"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["buffer", "q_scale_inv"],
            ["buffer", "out_q"],
            ["buffer", "key_cache"],
            ["buffer", "value_cache"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_scale"],
            ["buffer", "split_k_flag"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_q_heads"],
            ["parameter", "num_kv_heads"],
            ["parameter", "page_size"],
            ["parameter", "max_pages_per_request"],
            ["parameter", "max_seqlen"],
            ["parameter", "max_seqlen_aligned"],
            ["parameter", "quant_policy"],
            ["parameter", "norm_policy"],
            ["parameter", "is_prefill"],
            ["parameter", "has_out_kv"],
            ["parameter", "upper_max"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "rows_plus_requests_by_kv_heads",
        },
        "knobs": {
            "min_blocks": 0,
            "scan_rounds": 1,
            "early_rt2": False,
            "page_prefetch": False,
            "kv_groups_per_cta": 1,
            "fold": False,
            "grid_order": 3,
            "clear_heads_per_cta": 1,
            "zero_store_width": 256,
            "clear_tie": False,
        },
        "kernel": "cake_fused_qk_rope_fp8_append_79aed55694e2358f81b9",
        "kernel_symbol": "kernel_cake_fused_qk_rope_fp8_append_79aed55694e2358f81b9",
        # B200 single-KV-head decode of 8 to 64 requests (16 to 128 CTAs) the
        # prefetch does not serve (a page row over 48 entries or caller-owned
        # K/V): each KV head's clearing CTAs dispatched first and 256-bit zero
        # stores.
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_fp8_append_79aed55694e2358f81b9",
        "logical": {
            "cos_sin": "cos_sin",
            "has_out_kv": "has_out_kv",
            "is_prefill": "is_prefill",
            "k_norm_weight": "k_norm_weight",
            "k_scale": "k_scale",
            "key_cache": "key_cache",
            "max_pages_per_request": "max_pages_per_request",
            "max_seqlen": "max_seqlen",
            "max_seqlen_aligned": "max_seqlen_aligned",
            "norm_policy": "norm_policy",
            "num_kv_heads": "num_kv_heads",
            "num_q_heads": "num_q_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_indices": "page_indices",
            "page_size": "page_size",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "q_scale": "q_scale",
            "q_scale_inv": "q_scale_inv",
            "qkv": "qkv",
            "quant_policy": "quant_policy",
            "seq_lens": "seq_lens",
            "split_k_flag": "split_k_flag",
            "upper_max": "upper_max",
            "v_scale": "v_scale",
            "value_cache": "value_cache",
        },
        "stage": "fp8_e4m3_grid_order3_zero_store_width256",
        "workspace_rule": None,
    },
}

# One verified build per architecture: H100 / H200 (``sm_90a``), B200
# (``sm_100a``) and B300 / GB300 (``sm_103a``).  Every architecture is compiled
# with its exact ``-gencode`` flag set (``build.json`` ``arch_flags``); there is
# no family fallback.
ARCHES = ("sm_90a", "sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
# The default build's record; every other record is this name plus the knob
# suffix of its kernel symbol (``stage_name``).
STAGE = "fp8_e4m3"
HEAD_DIM = 128
# Eight Q heads, K, V and two idle eight-lane slots: three warps per CTA.
THREADS = 96
FP8_MAX = 448.0
SUPPORTED_HEADS = ((8, 1), (64, 8))
# The build keywords a record's ``knobs`` carry, in the kernel file's ``build``
# signature order: what the policy, the launch grid and the kernel's own symbol
# are derived from.
KNOB_KEYS = (
    "min_blocks",
    "scan_rounds",
    "early_rt2",
    "page_prefetch",
    "kv_groups_per_cta",
    "fold",
    "grid_order",
    "clear_heads_per_cta",
    "zero_store_width",
    "clear_tie",
)
# The Cake stage policy: the per-card constants and the rule of this round
# (``version`` is the policy table's).  ``policy_knobs`` is the executable form.
STAGE_POLICY: dict[str, Any] = {
    "constants": {
        "ctas_per_sm": 9,
        "fold_waves": (1.0, 2.0),
        "fold_min_kv_heads": 8,
        "prefetch_max_pages": 48,
        "scan_rounds3_min_requests": 97,
        "order_max_clear_frac": 0.5,
        "prefetch_short_row": 24,
        "prefetch_quant_policy": 2,
        "swap_min_requests": 2,
        "gb300_sms": 152,
        "u2z_ctas": (16, 128),
        "b300_er2pp9_min_ctas": 512,
    },
    "rule": (
        "num_requests == 0 -> the default build (every card: the early_rt2 "
        "builds require a request). card = h100 (sm_90*), b200 (sm_100a), gb300 "
        "(sm_103a and num_sms == gb300_sms) else b300; cap = ctas_per_sm * "
        "num_sms; waves = (num_rows + "
        "num_requests) * Hkv / cap; clear_frac = num_requests * Hkv / cap; decode "
        "= num_rows == num_requests and not is_prefill; pages_ok = "
        "max_pages_per_request <= prefetch_max_pages; prefetch_ok = pages_ok and "
        "not has_out_kv; order_ok = clear_frac <= order_max_clear_frac; er2pp9 = "
        "min_blocks=9, early_rt2, page_prefetch, clear_tie; s3 = scan_rounds=3, "
        "clear_tie. decode and Hkv >= fold_min_kv_heads and fold_waves[0] < waves "
        "<= fold_waves[1] -> early_rt2, fold, clear_tie. h100: decode and Hkv >= "
        "fold_min_kv_heads -> er2pp9 beyond one wave, s3 when (num_rows + "
        "num_requests) * Hkv <= num_sms, else er2pp9 + grid_order=2; decode "
        "otherwise -> er2pp9 when prefetch_ok and (max_pages_per_request <= "
        "prefetch_short_row or quant_policy == prefetch_quant_policy) else s3; "
        "prefill beyond one wave -> grid_order=3 if order_ok else min_blocks=10; "
        "one-wave prefill -> s3. u2z = clear_heads_per_cta=2, "
        "zero_store_width=256; u2z_class = decode and Hkv == 1 and u2z_ctas[0] <= "
        "(num_rows + num_requests) * Hkv <= u2z_ctas[1]. gb300: u2z_class -> u2z; "
        "prefetch_ok and (decode or Hkv == 1) -> er2pp9; not order_ok -> the "
        "default build; is_prefill and Hkv >= fold_min_kv_heads and num_requests "
        ">= swap_min_requests and num_rows + num_requests <= 65535 -> "
        "grid_order=2; else grid_order=3. b300 first: u2z_class and (not "
        "prefetch_ok or max_pages_per_request <= prefetch_short_row) -> u2z; "
        "decode and Hkv == 1 and prefetch_ok and (num_rows + num_requests) * "
        "Hkv >= b300_er2pp9_min_ctas and waves <= fold_waves[1] -> er2pp9; "
        "is_prefill and Hkv >= fold_min_kv_heads and num_requests >= "
        "swap_min_requests and waves <= 1 and order_ok -> grid_order=2; then the "
        "shared table. b200 first: is_prefill and Hkv >= fold_min_kv_heads and "
        "num_requests >= swap_min_requests and waves <= 1 and order_ok -> "
        "grid_order=2; u2z_class and not prefetch_ok -> grid_order=3, "
        "zero_store_width=256; then the shared table. shared (b200 / b300): "
        "decode and waves > fold_waves[1] -> s3 from scan_rounds3_min_requests "
        "requests else the default build; prefill and waves > 1 -> min_blocks=10; "
        "prefetch_ok and (decode or Hkv == 1) -> early_rt2, page_prefetch, "
        "clear_tie; else the default build"
    ),
    "version": "round3-v11-2026-10-28",
}
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "qkv",
    "cos_sin",
    "seq_lens",
    "q_indptr",
    "page_indices",
    "q_norm_weight",
    "k_norm_weight",
    "k_scale",
    "v_scale",
    "q_scale_inv",
    "out_q",
    "key_cache",
    "value_cache",
    "out_k",
    "out_v",
    "q_scale",
    "split_k_flag",
)
LOGICAL_SCALARS = (
    "num_rows",
    "num_requests",
    "num_q_heads",
    "num_kv_heads",
    "page_size",
    "max_pages_per_request",
    "max_seqlen",
    "max_seqlen_aligned",
    "quant_policy",
    "norm_policy",
    "is_prefill",
    "has_out_kv",
    "upper_max",
)


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def stages_for(arch: str) -> tuple[str, ...]:
    """Stage names (module records) shipped for ``arch``, in record order."""

    return tuple(name for name, record in MODULES.items() if arch in record["arches"])


def card(arch: str, num_sms: int) -> str:
    """The policy's card: ``h100`` (sm_90a; H200 follows it), ``b200``, ``b300`` or
    ``gb300`` (sm_103a with ``gb300_sms`` SMs)."""

    if arch.startswith("sm_90"):
        return "h100"
    if arch == "sm_100a":
        return "b200"
    if arch == "sm_103a":
        return (
            "gb300"
            if int(num_sms) == STAGE_POLICY["constants"]["gb300_sms"]
            else "b300"
        )
    raise ValueError(f"no Cake fused QK RoPE FP8 append stage table for {arch!r}")


def policy_knobs(
    num_q_heads: int,
    num_kv_heads: int,
    *,
    arch: str,
    num_rows: int,
    num_requests: int,
    page_size: int,
    num_sms: int,
    is_prefill: bool,
    max_pages_per_request: int,
    has_out_kv: bool,
    quant_policy: int | None = None,
) -> dict[str, int | bool]:
    """The knobs the Cake stage policy (``STAGE_POLICY``) selects beyond the default
    build for one problem.

    Keys are build keywords (``KNOB_KEYS``); an empty result is the default
    build.  ``max_pages_per_request`` is the page table's row length
    (``page_indices.shape[1]``), ``has_out_kv`` whether the caller owns K/V
    (``out_k is not None``), ``quant_policy`` the quantization policy of the
    call (``None`` when unknown); all are host scalars.
    """

    constants = STAGE_POLICY["constants"]
    heads = (int(num_q_heads), int(num_kv_heads))
    if heads not in SUPPORTED_HEADS:
        raise ValueError(
            f"(num_q_heads, num_kv_heads) = {heads} is not supported; "
            f"expected one of {SUPPORTED_HEADS}"
        )
    which = card(arch, num_sms)
    rows, requests, kv_heads = int(num_rows), int(num_requests), int(num_kv_heads)
    if requests < 1:
        # No request (with rows, a malformed table): the early_rt2 builds load their
        # guessed request's table words before the verdict and require one.
        return {}
    cap = int(constants["ctas_per_sm"]) * int(num_sms)
    waves = (rows + requests) * kv_heads / cap
    clear_frac = requests * kv_heads / cap
    prefill = bool(is_prefill)
    decode = rows == requests and not prefill
    pages = int(max_pages_per_request)
    pages_ok = pages <= int(constants["prefetch_max_pages"])
    prefetch_ok = pages_ok and not bool(has_out_kv)
    order_ok = clear_frac <= constants["order_max_clear_frac"]
    low, high = constants["fold_waves"]
    many_heads = kv_heads >= int(constants["fold_min_kv_heads"])
    if decode and many_heads and low < waves <= high:
        return {"early_rt2": True, "fold": True, "clear_tie": True}
    if which == "h100":
        er2pp9: dict[str, int | bool] = {
            "min_blocks": 9,
            "early_rt2": True,
            "page_prefetch": True,
            "clear_tie": True,
        }
        scan3: dict[str, int | bool] = {"scan_rounds": 3, "clear_tie": True}
        if decode and many_heads:
            if waves > 1.0:
                return er2pp9
            if (rows + requests) * kv_heads <= int(num_sms):
                return scan3
            return {**er2pp9, "grid_order": 2}
        if decode:
            few_pages = pages <= int(constants["prefetch_short_row"])
            static_scale = quant_policy is not None and int(quant_policy) == int(
                constants["prefetch_quant_policy"]
            )
            if prefetch_ok and (few_pages or static_scale):
                return er2pp9
            return scan3
        if waves > 1.0:
            return {"grid_order": 3} if order_ok else {"min_blocks": 10}
        return scan3
    u2z: dict[str, int | bool] = {"clear_heads_per_cta": 2, "zero_store_width": 256}
    low_ctas, high_ctas = constants["u2z_ctas"]
    u2z_class = (
        decode
        and kv_heads == 1
        and int(low_ctas) <= (rows + requests) * kv_heads <= int(high_ctas)
    )
    if which == "gb300":
        if u2z_class:
            return u2z
        if prefetch_ok and (decode or kv_heads == 1):
            return {
                "min_blocks": 9,
                "early_rt2": True,
                "page_prefetch": True,
                "clear_tie": True,
            }
        if not order_ok:
            return {}
        if (
            prefill
            and many_heads
            and requests >= int(constants["swap_min_requests"])
            and rows + requests <= 65535
        ):
            return {"grid_order": 2}
        return {"grid_order": 3}
    if which == "b300":
        short_row = pages <= int(constants["prefetch_short_row"])
        if u2z_class and (not prefetch_ok or short_row):
            return u2z
        if (
            decode
            and kv_heads == 1
            and prefetch_ok
            and (rows + requests) * kv_heads >= int(constants["b300_er2pp9_min_ctas"])
            and waves <= high
        ):
            # Nine CTAs per SM pay off on single-KV-head decode from 512 CTAs
            # up to two waves.
            return {
                "min_blocks": 9,
                "early_rt2": True,
                "page_prefetch": True,
                "clear_tie": True,
            }
        if (
            prefill
            and many_heads
            and requests >= int(constants["swap_min_requests"])
            and waves <= 1.0
            and order_ok
        ):
            return {"grid_order": 2}
    if which == "b200":
        # Round two: one-wave (64, 8) prefill of two requests or more takes the
        # swapped order (the B300 rule); the single-KV-head decode rows of the
        # clearing class the prefetch does not serve (a page row over 48 entries
        # or caller-owned K/V) take the clear-first order with 256-bit zero
        # stores.
        if (
            prefill
            and many_heads
            and requests >= int(constants["swap_min_requests"])
            and waves <= 1.0
            and order_ok
        ):
            return {"grid_order": 2}
        if u2z_class and not prefetch_ok:
            return {"grid_order": 3, "zero_store_width": 256}
    if decode and waves > high:
        if requests >= int(constants["scan_rounds3_min_requests"]):
            return {"scan_rounds": 3, "clear_tie": True}
        return {}
    if prefill and waves > 1.0:
        return {"min_blocks": 10}
    if prefetch_ok and (decode or kv_heads == 1):
        return {"early_rt2": True, "page_prefetch": True, "clear_tie": True}
    return {}


def stage_name(knobs: dict[str, int | bool]) -> str:
    """Stage name of one policy outcome: ``fp8_e4m3`` (the default build) plus
    ``_<knob><int>`` per knob set, in ``KNOB_KEYS`` order."""

    return STAGE + "".join(
        f"_{key}{int(knobs[key])}" for key in KNOB_KEYS if key in knobs
    )


def stage_for(
    num_q_heads: int,
    num_kv_heads: int,
    *,
    arch: str,
    num_rows: int,
    num_requests: int,
    page_size: int,
    num_sms: int,
    is_prefill: bool,
    max_pages_per_request: int,
    has_out_kv: bool,
    quant_policy: int | None = None,
) -> str:
    """The module record the policy selects for one problem on ``arch`` with
    ``num_sms`` SMs; a stage the package does not ship for the architecture is an
    error, never a silent fallback."""

    knobs = policy_knobs(
        num_q_heads,
        num_kv_heads,
        arch=arch,
        num_rows=num_rows,
        num_requests=num_requests,
        page_size=page_size,
        num_sms=num_sms,
        is_prefill=is_prefill,
        max_pages_per_request=max_pages_per_request,
        has_out_kv=has_out_kv,
        quant_policy=quant_policy,
    )
    name = stage_name(knobs)
    shipped = stages_for(arch)
    if name not in shipped:
        raise RuntimeError(
            f"the Cake stage policy selected {name!r} for {arch}, which the package "
            f"does not ship ({shipped})"
        )
    return name


def launch_grid(
    record: dict[str, Any], *, num_rows: int, num_requests: int, num_kv_heads: int
) -> tuple[int, int, int]:
    """Launch grid of one stage for one problem under its grid rule.

    ``rows_plus_requests_by_kv_heads``: ``(num_rows + num_requests, num_kv_heads
    // kv_groups_per_cta, 1)`` -- a CTA per (packed row, KV group) and one per
    (request, KV head); ``rows_by_kv_heads`` (a ``fold`` build, decode only):
    ``(num_rows, num_kv_heads // kv_groups_per_cta, 1)``; the ``kv_heads_by_*``
    rules (``grid_order`` 1 / 2) put the KV group axis first.  An ``early_rt2``
    build requires a request (its launcher refuses ``num_requests == 0``).
    """

    knobs = record["knobs"]
    kind = record["grid_rule"]["kind"]
    rows, requests, kv_heads = int(num_rows), int(num_requests), int(num_kv_heads)
    groups = int(knobs["kv_groups_per_cta"])
    if kv_heads % groups:
        raise ValueError(
            f"{groups} KV groups per CTA do not divide {kv_heads} KV heads"
        )
    if knobs.get("early_rt2", False) and requests < 1:
        raise ValueError("an early_rt2 build requires a request (num_requests >= 1)")
    if kind in ("rows_by_kv_heads", "kv_heads_by_rows"):
        if rows != requests:
            raise ValueError(
                "a fold build serves decode only (num_rows == num_requests)"
            )
        grid = (rows, kv_heads // groups)
    elif kind in ("rows_plus_requests_by_kv_heads", "kv_heads_by_rows_plus_requests"):
        grid = (rows + requests, kv_heads // groups)
    else:
        raise ValueError(f"unknown Cake fused QK RoPE FP8 append grid rule: {kind!r}")
    if kind.startswith("kv_heads_by_"):
        if rows + requests > 65535:
            raise ValueError("a swapped grid serves at most 65535 rows plus requests")
        grid = (grid[1], grid[0])
    return (grid[0], grid[1], 1)


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an
    exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            "Cake fused QK RoPE FP8 append target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``:
    ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake fused QK RoPE FP8 append module {name} is not delivered for "
            f"{arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch`` (prebuilt
    ``aot/`` library, else the JIT build of the same sources and flags through
    ``cake_build.load_library``)."""

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file
    (ahead-of-time warm-up)."""

    return cake_build.jit_specs(arch)


def run_module(name: str, values: dict[str, Any], *, device_index: int) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``
    to tensors and Python scalars; the grid is :func:`launch_grid` of the record
    for the problem the scalars describe, and the kernel runs on the current
    torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(
        record,
        num_rows=int(values["num_rows"]),
        num_requests=int(values["num_requests"]),
        num_kv_heads=int(values["num_kv_heads"]),
    )
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "FP8_MAX",
    "HEAD_DIM",
    "KNOB_KEYS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "STAGE",
    "STAGE_POLICY",
    "SUPPORTED_HEADS",
    "THREADS",
    "arch_for_capability",
    "card",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "policy_knobs",
    "run_module",
    "sm_count",
    "specs_for_arch",
    "stage_for",
    "stage_name",
    "stages_for",
]
