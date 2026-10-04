# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""Prepared fused BF16 routing GEMM and normalized top-k weights on SM100a/SM103a.

One compiled program per physical schedule template (block tokens, MMA CTA
pair, split-K, expert groups, pipeline stages, gate warpgroups, K-block
merge, the single-token cluster-reduction route, the small-M idle L2 touch).
The token count, the SM-count-derived worker stride and the physical-map /
logical-output flags are kernel arguments, so any token count is served: the
DeepGEMM configuration for ``M`` is snapped onto the exported template set.
"""

from __future__ import annotations

import functools

EXPORTED = dict(K=5120, E=384, num_topk=6, scoring_func="sqrtsoftplus")
"""The problem constants the exported programs were generated for."""

ROUTE_FLAG_PHYSICAL_MAP = 1
ROUTE_FLAG_UNMAPPED_OUTPUT = 2

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

PROGRAMS = {
    "cake_deepgemm_mega_gate_03ec269c991bda293e41": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_03ec269c991bda293e41_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_03ec269c991bda293e41_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_0efed871dc36b6202b4b": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_0efed871dc36b6202b4b_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_0efed871dc36b6202b4b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_1ba724ffb65f5c8a34ac": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_1ba724ffb65f5c8a34ac_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_1ba724ffb65f5c8a34ac_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_344eaed5f3cbdb26700d": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_344eaed5f3cbdb26700d_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_344eaed5f3cbdb26700d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_405f6008eb19d6cf4dca": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_405f6008eb19d6cf4dca_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_405f6008eb19d6cf4dca_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_6443777451076c4eaabb": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_6443777451076c4eaabb_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_6443777451076c4eaabb_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_72863ee40f5199dd111d": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_72863ee40f5199dd111d_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_72863ee40f5199dd111d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_9b3753d5dbc609dc49da": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_9b3753d5dbc609dc49da_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_9b3753d5dbc609dc49da_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_9eb0496be67a6aae1388": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_9eb0496be67a6aae1388_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_9eb0496be67a6aae1388_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_af3cc36d7d3167cc2d8b": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_af3cc36d7d3167cc2d8b_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_af3cc36d7d3167cc2d8b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_b5a62fb00848690ade55": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_b5a62fb00848690ade55_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_b5a62fb00848690ade55_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_bbf3dd7af63a05d6c24a": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_bbf3dd7af63a05d6c24a_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_bbf3dd7af63a05d6c24a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
    "cake_deepgemm_mega_gate_cd6cb9119b40e8edd2fc": {
        "sources": [
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_cd6cb9119b40e8edd2fc_kernel.cu",
            "csrc/experimental/deepgemm_mega_gate/generated/cake_deepgemm_mega_gate_cd6cb9119b40e8edd2fc_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "X"],
            ["tma_buffer", "W"],
            ["buffer", "bias"],
            ["buffer", "image_bias"],
            ["buffer", "image_mask"],
            ["buffer", "mask"],
            ["buffer", "physical_map"],
            ["buffer", "logical_count"],
            ["buffer", "topk_idx"],
            ["buffer", "unmapped_idx"],
            ["buffer", "topk_weights"],
            ["buffer", "scratch"],
            ["buffer", "score_barriers"],
            ["buffer", "fixed_mask"],
            ["buffer", "random_mask"],
            ["parameter", "num_tokens"],
            ["parameter", "num_shared"],
            ["parameter", "map_width"],
            ["parameter", "ep_rank"],
            ["parameter", "routed_scale"],
            ["parameter", "unmapped_stride"],
            ["parameter", "num_workers"],
            ["parameter", "route_flags"],
            ["parameter", "num_split_k"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
    },
}
"""program name -> sources (csrc-relative), compile flags, FFI entry and argument plan."""

TEMPLATES = (
    {
        "block_tokens": 16,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 12,
        "num_gate_warpgroups": 2,
        "k_block_merge": 1,
        "single_token": True,
        "idle_touch": True,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_bbf3dd7af63a05d6c24a",
    },
    {
        "block_tokens": 16,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 12,
        "num_gate_warpgroups": 2,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": True,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_b5a62fb00848690ade55",
    },
    {
        "block_tokens": 16,
        "num_mma_ctas": 1,
        "num_split_k": 1,
        "num_expert_groups": 3,
        "num_stages": 12,
        "num_gate_warpgroups": 2,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": True,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_03ec269c991bda293e41",
    },
    {
        "block_tokens": 32,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 11,
        "num_gate_warpgroups": 4,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_9b3753d5dbc609dc49da",
    },
    {
        "block_tokens": 96,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 7,
        "num_gate_warpgroups": 6,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_405f6008eb19d6cf4dca",
    },
    {
        "block_tokens": 176,
        "num_mma_ctas": 2,
        "num_split_k": 4,
        "num_expert_groups": 3,
        "num_stages": 11,
        "num_gate_warpgroups": 6,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_af3cc36d7d3167cc2d8b",
    },
    {
        "block_tokens": 176,
        "num_mma_ctas": 2,
        "num_split_k": 2,
        "num_expert_groups": 3,
        "num_stages": 11,
        "num_gate_warpgroups": 6,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_af3cc36d7d3167cc2d8b",
    },
    {
        "block_tokens": 176,
        "num_mma_ctas": 2,
        "num_split_k": 2,
        "num_expert_groups": 3,
        "num_stages": 11,
        "num_gate_warpgroups": 7,
        "k_block_merge": 2,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_1ba724ffb65f5c8a34ac",
    },
    {
        "block_tokens": 176,
        "num_mma_ctas": 2,
        "num_split_k": 1,
        "num_expert_groups": 3,
        "num_stages": 11,
        "num_gate_warpgroups": 7,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 0,
        "static_route": -1,
        "static_workers": 0,
        "program": "cake_deepgemm_mega_gate_cd6cb9119b40e8edd2fc",
    },
    {
        "block_tokens": 16,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 12,
        "num_gate_warpgroups": 2,
        "k_block_merge": 1,
        "single_token": True,
        "idle_touch": True,
        "static_tokens": 1,
        "static_route": 1,
        "static_workers": 6,
        "program": "cake_deepgemm_mega_gate_6443777451076c4eaabb",
    },
    {
        "block_tokens": 16,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 12,
        "num_gate_warpgroups": 2,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": True,
        "static_tokens": 16,
        "static_route": 1,
        "static_workers": 6,
        "program": "cake_deepgemm_mega_gate_9eb0496be67a6aae1388",
    },
    {
        "block_tokens": 16,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 12,
        "num_gate_warpgroups": 2,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": True,
        "static_tokens": 16,
        "static_route": 2,
        "static_workers": 6,
        "program": "cake_deepgemm_mega_gate_344eaed5f3cbdb26700d",
    },
    {
        "block_tokens": 32,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 11,
        "num_gate_warpgroups": 4,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 128,
        "static_route": 1,
        "static_workers": 6,
        "program": "cake_deepgemm_mega_gate_0efed871dc36b6202b4b",
    },
    {
        "block_tokens": 96,
        "num_mma_ctas": 1,
        "num_split_k": 8,
        "num_expert_groups": 3,
        "num_stages": 7,
        "num_gate_warpgroups": 6,
        "k_block_merge": 1,
        "single_token": False,
        "idle_touch": False,
        "static_tokens": 512,
        "static_route": 1,
        "static_workers": 6,
        "program": "cake_deepgemm_mega_gate_72863ee40f5199dd111d",
    },
)
"""Exported schedule templates (``block_tokens``, ``num_mma_ctas``, ``num_split_k``, ``num_expert_groups``,
``num_stages``, ``num_gate_warpgroups``, ``k_block_merge``, ``single_token``, ``idle_touch``) and their program."""


def _ceildiv(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _ceildiv(a, b) * b


def deepgemm_config(
    M,
    K,
    E,
    num_sms,
    *,
    has_bias=True,
    has_image_bias=False,
    has_physical_map=True,
    deterministic=False,
):
    """DeepGEMM's wave/tile/split-K selection and SMEM budget.

    Returns ``(block, ctas, split, groups, stages, warpgroups, launch_ctas, workers)``.
    """
    if not (
        0 < M <= 1 << 20 and K > 0 and K % 256 == 0 and 0 < E <= 512 and E % 4 == 0
    ):
        raise ValueError(
            "Mega Gate requires 0<M<=2^20, K divisible by 256, E divisible by 4 and <=512"
        )
    aligned_e = _align(E, 128)

    def wave_tile(workers):
        waves = _ceildiv(_ceildiv(M, 256), workers)
        return min(256, _align(_ceildiv(M, waves * workers), 16))

    def num_waves(workers):
        return _ceildiv(_ceildiv(M, wave_tile(workers)), workers)

    candidates = []
    for groups in range(1, aligned_e // 128 + 1):
        if aligned_e % (128 * groups) or aligned_e // groups > 256:
            continue
        experts = aligned_e // groups
        for ctas in (1, 2):
            if experts > 128 and ctas != 2:
                continue
            for split in (1, 2, 4, 8):
                task_ctas = groups * ctas * split
                if (
                    K % (split * 64)
                    or (ctas == 2 and split == 8)
                    or task_ctas > num_sms
                    or task_ctas >= 64
                ):
                    continue
                workers = num_sms // task_ctas
                block = wave_tile(workers)
                waves = num_waves(workers)
                doubled = task_ctas * 2
                if doubled <= num_sms and doubled < 64:
                    doubled_workers = num_sms // doubled
                    if (
                        ctas == 1
                        and split != 8
                        and num_waves(doubled_workers) <= waves + 2
                    ):
                        continue
                    if (
                        ctas == 2
                        and split == 1
                        and waves == 1
                        and wave_tile(doubled_workers) == block
                    ):
                        continue
                token_cols = block // ctas if experts == 128 else block
                launch_ctas = min(workers, _ceildiv(M, block)) * task_ctas
                candidates.append(
                    (
                        (-min(token_cols, 7 * 8), -launch_ctas, -ctas, split, waves),
                        groups,
                        ctas,
                        split,
                        block,
                        launch_ctas,
                        workers,
                        waves,
                    )
                )
    if not candidates:
        raise ValueError("Mega Gate has no configuration for the supplied SM count")
    _, groups, ctas, split, block, launch_ctas, workers, waves = min(candidates)
    if deterministic:
        split = 1
        workers = launch_ctas // (groups * ctas)
    token_cols = block // ctas if aligned_e // groups == 128 else block
    transpose_units = _ceildiv(token_cols, 8)
    topk_units = _ceildiv(block, 4 * groups * ctas * split)
    warpgroups = (
        7
        if waves > 1
        else max(
            _ceildiv(transpose_units, _ceildiv(transpose_units, 7)),
            _ceildiv(topk_units, _ceildiv(topk_units, 7)),
        )
    )
    bias_bytes = aligned_e * 4 * (int(has_bias) + int(has_image_bias))
    cache_counts = has_physical_map and bias_bytes + aligned_e * 4 <= 4096
    fixed = _align(44 + bias_bytes + aligned_e * 4 * cache_counts, 1024)
    stage_bytes = ((block + aligned_e // groups) // ctas) * 64 * 2 + 16
    stages = min(32, (232448 - fixed) // stage_bytes)
    return block, ctas, split, groups, stages, warpgroups, launch_ctas, workers


def _exact_template(cfg, M, K, E):
    """The schedule levers of DeepGEMM configuration ``cfg`` resolved for ``M``: ``(merge, single_token, idle_touch)``."""
    block, ctas, split, groups, stages, _warpgroups, _launch, workers = cfg
    em = _align(E, 128) // groups
    merge = (
        3
        if stages // 3 >= 8 and (K // 64 // split) % 3 == 0
        else 2
        if stages // 2 >= 8 and (K // 64 // split) % 2 == 0
        else 1
    )
    if (
        merge == 1
        and ctas == 2
        and split == 2
        and _ceildiv(M, block) > workers
        and stages // 2 >= 4
        and (K // 64 // split) % 2 == 0
    ):
        merge = 2
    single_token = ctas == 1 and 1 < split <= 8 and M == 1 and split * em * 4 <= 8192
    return merge, single_token, M <= 64


def route_flags_for(has_physical_map, unmapped_output):
    return (ROUTE_FLAG_PHYSICAL_MAP if has_physical_map else 0) | (
        ROUTE_FLAG_UNMAPPED_OUTPUT if unmapped_output else 0
    )


def _tile_geometry(t, M):
    """``(block count, 16-aligned tail rows)`` of ``M`` tokens on template ``t``'s block."""
    block = t["block_tokens"]
    return _ceildiv(M, block), (_align(M % block, 16) if M % block else block)


def _schedule_axes(t):
    return tuple(
        t[k]
        for k in (
            "block_tokens",
            "num_mma_ctas",
            "num_split_k",
            "num_expert_groups",
            "num_stages",
            "num_gate_warpgroups",
            "k_block_merge",
            "single_token",
            "idle_touch",
        )
    )


def select_template(
    M, num_sms, *, has_physical_map=True, unmapped_output=False, deterministic=False
):
    """``(template index, launch_ctas, num_workers)`` for ``M`` tokens on ``num_sms`` SMs.

    The DeepGEMM block snaps to the smallest exported block >= it within the same (ctas, split, groups)
    family (the largest exported block when none is larger); the closest exported instance of that
    block is taken; the launch grid and worker stride follow DeepGEMM's rules at the snapped block.
    An exact-shape program of the chosen schedule (the tile geometry of ``static_tokens`` equal to that
    of ``M``, the same route and worker stride) is preferred when one is exported.
    """
    K, E = EXPORTED["K"], EXPORTED["E"]
    cfg = deepgemm_config(
        M, K, E, num_sms, has_physical_map=has_physical_map, deterministic=deterministic
    )
    block, ctas, split, groups, exact_stages, exact_warpgroups, _launch, _workers = cfg
    exact_merge, single_token, exact_touch = _exact_template(cfg, M, K, E)
    family = [
        i
        for i, t in enumerate(TEMPLATES)
        if not t["static_tokens"]
        and (t["num_mma_ctas"], t["num_split_k"], t["num_expert_groups"])
        == (ctas, split, groups)
        and t["single_token"] == single_token
    ]
    if not family:
        raise NotImplementedError(
            f"no exported Mega Gate template for M={M} (DeepGEMM family ctas={ctas}, split={split}, "
            f"groups={groups}, single_token={single_token})"
        )
    larger = [
        TEMPLATES[i]["block_tokens"]
        for i in family
        if TEMPLATES[i]["block_tokens"] >= block
    ]
    BM = min(larger) if larger else max(TEMPLATES[i]["block_tokens"] for i in family)
    pre = (
        cfg
        if not deterministic
        else deepgemm_config(M, K, E, num_sms, has_physical_map=has_physical_map)
    )
    task_ctas = pre[1] * groups * pre[2]
    workers = num_sms // task_ctas
    launch_ctas = min(workers, _ceildiv(M, BM)) * task_ctas
    if deterministic:
        workers = launch_ctas // (groups * ctas)
    waves = _ceildiv(_ceildiv(M, BM), workers)
    merge_two_tile = ctas == 2 and split == 2 and waves > 1

    def distance(i):
        t = TEMPLATES[i]
        return (
            t["k_block_merge"]
            != (2 if merge_two_tile and exact_merge == 1 else exact_merge),
            t["num_gate_warpgroups"] != (7 if waves > 1 else exact_warpgroups),
            abs(t["num_stages"] - exact_stages),
            t["idle_touch"] != exact_touch,
        )

    index = min((i for i in family if TEMPLATES[i]["block_tokens"] == BM), key=distance)
    route = route_flags_for(has_physical_map, unmapped_output)
    axes = _schedule_axes(TEMPLATES[index])
    for i, t in enumerate(TEMPLATES):
        if (
            t["static_tokens"]
            and _tile_geometry(t, t["static_tokens"]) == _tile_geometry(t, M)
            and t["static_route"] == route
            and t["static_workers"] in (0, workers)
            and _schedule_axes(t) == axes
        ):
            return i, launch_ctas, workers
    return index, launch_ctas, workers


@functools.cache
def device_facts(device_index):
    """``(arch, sm_count)`` of CUDA device ``device_index``: one device query per process and device."""
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    arch = _ARCHES.get((properties.major, properties.minor))
    if arch is None:
        raise RuntimeError(
            f"Mega Gate has no exported programs for compute capability {(properties.major, properties.minor)}; "
            f"exported architectures: {sorted(_ARCHES.values())}"
        )
    return arch, int(properties.multi_processor_count)


def _nvcc_flags(arch):
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


@functools.cache
def load_program(arch, name):
    """Build (once per process) and load the exported program ``name`` for ``arch``."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = PROGRAMS[name]
    spec = gen_jit_spec(
        name=f"{name}_{arch}",
        sources=[
            env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]
        ],
        extra_cuda_cflags=[*_nvcc_flags(arch), *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            env.FLASHINFER_CSRC_DIR,
            env.FLASHINFER_INCLUDE_DIR,
            env.FLASHINFER_CSRC_DIR
            / record["sources"][0].removeprefix("csrc/").rsplit("/", 1)[0],
        ],
        use_fast_math=False,
    )
    return spec.build_and_load(), spec


class MegaGatePlan:
    """Bind a fused routing operation; run() returns (expert_indices, weights).

    x is BF16[M,K], weight is BF16[E,K]. The bias is FP32[E]. Physical mapping
    uses int32[E+shared,width] and logical_count int32[E+shared]. The optional
    logical top-k output is int64[M,topk] with unit column stride. The selected
    rows emit int64 indices and FP32 normalized unbiased scores scaled by
    routed_scale. Deterministic mode selects the split-K-free template.

    Optional scratch and score_barriers are caller-owned; fresh barriers must
    be zeroed once before first use. A plan retains all buffers and submits on
    the current PyTorch stream without allocating. Do not use one plan
    concurrently across streams. Changing tensor addresses or layouts requires
    a new plan; contents may change in place between runs.
    """

    def __init__(
        self,
        x,
        weight,
        num_topk=6,
        *,
        scoring_func="sqrtsoftplus",
        bias=None,
        image_bias=None,
        image_token_mask=None,
        mask=None,
        to_physical_map=None,
        logical_count=None,
        fix_routing_mask=None,
        force_random=None,
        unmapped_topk_idx=None,
        use_shared_as_routed=False,
        num_shared_experts=1,
        routed_scaling_factor=1.5,
        ep_rank=0,
        out=None,
        deterministic=False,
        scratch=None,
        score_barriers=None,
        descriptor_workspace=None,
    ):
        import torch

        if x.device.type != "cuda":
            raise RuntimeError("Mega Gate requires a CUDA device")
        device_index = (
            x.device.index
            if x.device.index is not None
            else torch.cuda.current_device()
        )
        arch, sms = device_facts(device_index)
        if (
            x.ndim != 2
            or weight.ndim != 2
            or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
        ):
            raise ValueError("x and weight must be contiguous BF16 matrices")
        M, K = x.shape
        E, wk = weight.shape
        if wk != K or not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("x and weight must be contiguous and share K")
        if (
            (K, E, num_topk, scoring_func)
            != (
                EXPORTED["K"],
                EXPORTED["E"],
                EXPORTED["num_topk"],
                EXPORTED["scoring_func"],
            )
            or bias is None
            or any(
                t is not None
                for t in (
                    image_bias,
                    image_token_mask,
                    mask,
                    fix_routing_mask,
                    force_random,
                )
            )
        ):
            raise NotImplementedError(
                f"the exported Mega Gate programs cover K={EXPORTED['K']}, E={EXPORTED['E']}, "
                f"top-k {EXPORTED['num_topk']}, {EXPORTED['scoring_func']} scoring with an expert bias and "
                "no image bias, token mask, fixed or random routing"
            )
        if descriptor_workspace is not None:
            raise ValueError(
                "the exported programs take their TMA descriptors by value; no descriptor workspace applies"
            )
        if (to_physical_map is None) != (logical_count is None):
            raise ValueError(
                "to_physical_map and logical_count must be supplied together"
            )
        if not 0 <= ep_rank <= 0x7FFFFFFF:
            raise ValueError("ep_rank must fit nonnegative int32")
        shared = num_shared_experts if use_shared_as_routed else 0
        if shared and (
            shared not in (1, 2) or num_topk % shared or E % (num_topk // shared)
        ):
            raise ValueError("shared experts require the shared/top-k divisibility")
        if num_topk + shared > 32:
            raise ValueError("routed and shared output slots must fit one warp")
        index, launch_ctas, num_workers = select_template(
            M,
            sms,
            has_physical_map=to_physical_map is not None,
            unmapped_output=unmapped_topk_idx is not None,
            deterministic=bool(deterministic),
        )
        template = TEMPLATES[index]
        block_tokens, num_split_k = template["block_tokens"], template["num_split_k"]
        blocks, aligned_e = _ceildiv(M, block_tokens), _align(E, 128)
        if out is None:
            out = (
                torch.empty((M, num_topk + shared), dtype=torch.int64, device=x.device),
                torch.empty(
                    (M, num_topk + shared), dtype=torch.float32, device=x.device
                ),
            )
        if (
            len(out) != 2
            or any(tuple(t.shape) != (M, num_topk + shared) for t in out)
            or out[0].dtype != torch.int64
            or out[1].dtype != torch.float32
        ):
            raise ValueError("out must be int64/FP32 tensors of shape [M,topk+shared]")
        scratch_shape = (blocks, num_split_k, block_tokens, aligned_e)
        if scratch is None:
            scratch = torch.empty(scratch_shape, dtype=torch.float32, device=x.device)
        if score_barriers is None:
            score_barriers = torch.zeros(
                (blocks, 16), dtype=torch.uint64, device=x.device
            )
        if tuple(scratch.shape) != scratch_shape or scratch.dtype != torch.float32:
            raise ValueError(f"scratch must be FP32{scratch_shape}")
        if (
            tuple(score_barriers.shape) != (blocks, 16)
            or score_barriers.dtype != torch.uint64
        ):
            raise ValueError("score_barriers must be uint64[ceil(M/block_tokens),16]")
        if bias.dtype != torch.float32 or tuple(bias.shape) != (E,):
            raise ValueError(f"bias must be FP32[{E}]")
        if logical_count is not None and (
            logical_count.dtype != torch.int32
            or tuple(logical_count.shape) != (E + shared,)
        ):
            raise ValueError(f"logical_count must be int32[{E + shared}]")
        if to_physical_map is not None and (
            to_physical_map.dtype != torch.int32
            or to_physical_map.ndim != 2
            or to_physical_map.shape[0] != E + shared
        ):
            raise ValueError("to_physical_map must be int32[E+shared,width]")
        if unmapped_topk_idx is not None and (
            unmapped_topk_idx.dtype != torch.int64
            or tuple(unmapped_topk_idx.shape) != (M, num_topk)
            or unmapped_topk_idx.stride(1) != 1
        ):
            raise ValueError(
                "unmapped_topk_idx must be int64[M,topk] with unit column stride"
            )
        tensors = (
            x,
            weight,
            bias,
            to_physical_map,
            logical_count,
            *out,
            scratch,
            score_barriers,
        )
        if any(
            t.device != x.device or not t.is_contiguous()
            for t in tensors
            if t is not None
        ):
            raise ValueError(
                "inputs and workspace must be contiguous on one CUDA device"
            )
        if unmapped_topk_idx is not None and unmapped_topk_idx.device != x.device:
            raise ValueError("unmapped_topk_idx must be on the input device")
        dummy_f32 = torch.empty(1, dtype=torch.float32, device=x.device)
        dummy_i32 = torch.empty(1, dtype=torch.int32, device=x.device)
        dummy_i64 = torch.empty(1, dtype=torch.int64, device=x.device)
        dummy_u8 = torch.empty(1, dtype=torch.uint8, device=x.device)
        route_flags = route_flags_for(
            to_physical_map is not None, unmapped_topk_idx is not None
        )
        bindings = dict(
            X=x,
            W=weight,
            bias=bias,
            image_bias=dummy_f32,
            image_mask=dummy_u8,
            mask=dummy_u8,
            physical_map=to_physical_map if to_physical_map is not None else dummy_i32,
            logical_count=logical_count if logical_count is not None else dummy_i32,
            topk_idx=out[0],
            topk_weights=out[1],
            unmapped_idx=unmapped_topk_idx
            if unmapped_topk_idx is not None
            else dummy_i64,
            scratch=scratch,
            score_barriers=score_barriers,
            fixed_mask=dummy_u8,
            random_mask=dummy_u8,
            num_tokens=M,
            num_shared=shared,
            map_width=to_physical_map.shape[1] if to_physical_map is not None else 1,
            ep_rank=ep_rank,
            routed_scale=routed_scaling_factor,
            unmapped_stride=unmapped_topk_idx.stride(0)
            if unmapped_topk_idx is not None
            else num_topk,
            num_workers=num_workers,
            route_flags=route_flags,
            num_split_k=num_split_k,
            grid_x=launch_ctas,
            grid_y=1,
            grid_z=1,
        )
        module, _spec = load_program(arch, template["program"])
        record = PROGRAMS[template["program"]]
        args = tuple(bindings[name] for _kind, name in record["arg_plan"])
        self.route = dict(
            arch=arch,
            template=index,
            program=template["program"],
            launch_ctas=launch_ctas,
            num_workers=num_workers,
            block_tokens=block_tokens,
            num_split_k=num_split_k,
            num_sms=sms,
            route_flags=route_flags,
        )
        self._submission = (module[record["ffi_entry"]], args)
        self._retained = (
            module,
            tensors,
            unmapped_topk_idx,
            dummy_f32,
            dummy_i32,
            dummy_i64,
            dummy_u8,
        )
        self.outputs, self.scratch, self.score_barriers = out, scratch, score_barriers

    def run(self):
        import tvm_ffi

        entry, args = self._submission
        with tvm_ffi.use_torch_stream():
            entry(*args)
        return self.outputs
