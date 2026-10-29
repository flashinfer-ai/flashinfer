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
"""

import ast
import contextlib
import dataclasses
import re
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

# Backend suite: the package is named once, through ``cake_package``; nothing else
# of flashinfer is imported here.
cake_backend = cake_package("dense_projection_gemm", "cake_backend")
cake_build = cake_package("dense_projection_gemm", "cake_build")
cake_jit = cake_package("dense_projection_gemm", "cake_jit")
ARCH_RULE_DEFAULTS = cake_backend.ARCH_RULE_DEFAULTS
BLOCK_K = cake_backend.BLOCK_K
BLOCK_N_CHOICES = cake_backend.BLOCK_N_CHOICES
CONVENIENCE_CACHE_SIZE = cake_backend.CONVENIENCE_CACHE_SIZE
CTA_GROUP = cake_backend.CTA_GROUP
CTA_ROWS_CHOICES = cake_backend.CTA_ROWS_CHOICES
EPI_WARPS = cake_backend.EPI_WARPS
ROUTER_SPLITS = cake_backend.ROUTER_SPLITS
ROW_RULES = cake_backend.ROW_RULES
SK_DUMMY_BASE = cake_backend.SK_DUMMY_BASE
SK_MIN_ITERS = cake_backend.SK_MIN_ITERS
SMEM_OPT_IN = cake_backend.SMEM_OPT_IN
arch_for = cake_backend.arch_for
bind_launch = cake_backend.bind_launch
convenience_key = cake_backend.convenience_key
default_block_n = cake_backend.default_block_n
default_group_m = cake_backend.default_group_m
default_hints = cake_backend.default_hints
default_hints_cgrp = cake_backend.default_hints_cgrp
b_stage_bytes = cake_backend.b_stage_bytes
box_rows_of = cake_backend.box_rows_of
default_stages = cake_backend.default_stages
device_l2_bytes = cake_backend.device_l2_bytes
epi_cols = cake_backend.epi_cols
epi_mode = cake_backend.epi_mode
exact_waves = cake_backend.exact_waves
epi_slots = cake_backend.epi_slots
generated_program_available = cake_backend.generated_program_available
instance_key = cake_backend.instance_key
instance_symbol = cake_backend.instance_symbol
sk_exact_plan = cake_backend.sk_exact_plan
sk_sync_plan = cake_backend.sk_sync_plan
check_stream_k_plan = cake_backend.check_stream_k_plan
operand_view = cake_backend.operand_view
plan_dense_projection_gemm = cake_backend.plan_dense_projection_gemm
plan_router_fp32_gemm = cake_backend.plan_router_fp32_gemm
prepare_dense_projection_gemm = cake_backend.prepare_dense_projection_gemm
prepare_projection_wgrad = cake_backend.prepare_projection_wgrad
prepare_router_fp32_gemm = cake_backend.prepare_router_fp32_gemm
router_layout_class = cake_backend.router_layout_class
row_rule = cake_backend.row_rule
sk_parts_plan = cake_backend.sk_parts_plan
smem_limit_for = cake_backend.smem_limit_for
split_fp32_to_bf16x3_ = cake_backend.split_fp32_to_bf16x3_
stream_k_plan = cake_backend.stream_k_plan
swap_small_m = cake_backend.swap_small_m
wave_working_set = cake_backend.wave_working_set
wave_working_set_cgrp = cake_backend.wave_working_set_cgrp
wgrad_swapped = cake_backend.wgrad_swapped
wgrad_views = cake_backend.wgrad_views

# Accuracy gates against the FP64 reference of the exact BF16 operands.
# BF16 output: elementwise |out - ref| <= atol + rtol * |ref|, zero violations.
BF16_ATOL = BF16_RTOL = 1e-2
# FP32 output of K1: the tensor-core accumulation error scales with the reference, so both
# bounds are scale-relative (never an absolute atol).
F32_REL_FRO_MAX = 5e-5
F32_MAX_ABS_OVER_REF_ABSMAX = 1e-3
# K2 (FP32 router through split-BF16x3 emulation): FP32-class, bit-exact reruns.
ROUTER_REL_FRO_MAX = 2e-6

SEED = 20260930

# GLM-5.2 projection rows: label -> (K = in features, N = out features).
PROJECTION_ROWS = {
    "o_proj": (16384, 6144),
    "q_a": (6144, 2048),
    "q_b": (2048, 16384),
    "kv_a": (6144, 576),
    "shared_gate_up": (6144, 2048),
    "shared_down": (2048, 6144),
    "dense_gate_up": (6144, 12288),
    "dense_down": (12288, 6144),
    "indexer_q": (2048, 4096),
    "indexer_k": (6144, 128),
    "indexer_hw": (6144, 32),
}
# The bounded GPU subset: one row per instance class, at small T.
GPU_ROWS = [
    (
        "kv_a",
        "fwd",
        "bf16",
        257,
    ),  # kk, K = 6144 > 1024 -> register epilogue: dense_proj_gemm_kk_n256
    (
        "indexer_k",
        "fwd",
        "bf16",
        1001,
    ),  # N = 128 -> the 128-column instance (dense_proj_gemm_kk_n128*)
    (
        "indexer_hw",
        "fwd",
        "bf16",
        129,
    ),  # N = 32: masked columns of the 128-column instance
    ("q_a", "dgrad", "bf16", 257),  # kn, K = N_feat = 2048 > 1024 -> register epilogue
    (
        "kv_a",
        "dgrad",
        "bf16",
        1001,
    ),  # kn, K = N_feat = 576 <= 1024 -> TMA-store epilogue
    ("q_a", "dgrad", "f32", 257),  # kn, fp32 output: dense_proj_gemm_kn_n256_f32_tma1
    (
        "shared_down",
        "wgrad",
        "bf16",
        1001,
    ),  # nn, K = T = 1001 <= 1024 -> dense_proj_gemm_nn_n256_tma1
    ("kv_a", "wgrad", "f32", 257),  # nn, fp32 output: dense_proj_gemm_nn_n256_f32_tma1
    (
        "indexer_hw",
        "wgrad",
        "bf16",
        1001,
    ),  # swapped X.T @ G, transposed store, stream-K "auto" split (dense_proj_gemm_nn_n128*_t)
    (
        "indexer_k",
        "wgrad",
        "f32",
        257,
    ),  # swapped, transposed fp32 store (dense_proj_gemm_nn_n128*_f32_t)
]
MLA_ROWS = {
    "qabs": dict(
        H=64, D_in=192, D_out=512, act_stride_head=256, weight_layout="in_out"
    ),
    "vproj": dict(
        H=64, D_in=512, D_out=256, act_stride_head=512, weight_layout="out_in"
    ),
}
# CPU planner inputs: the two device facts the plan depends on are the SM count (stream-K split)
# and the L2 size (TMA eviction-hint working-set gate); B200 SXM (148 SMs) and Rubin R200 (212 SMs)
# both report a 126 MiB L2.
SM_COUNTS = (148, 212)
# The instance every GLM-5.2 row resolves to on those devices.
# GENERATED from the Cake kernel host (adapter _plan_k1 with the mirrored sm_100a rule table) by the Cake
# workspace tool stage/r2/gen_fi_test_tables.py. Do not hand-edit.
# Table values are a template, or {(T, sm_count): template} when the plan depends on T / the device.
# fmt: off
# --- BEGIN GENERATED TABLES ---
L2_BYTES = 132120576
EXPECTED_TEMPLATES = {
    ('o_proj', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_m256_ov_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n256_m256_ht_cg_or_pro7',
        (2049, 148): 'dense_proj_gemm_kk_n256_m256_ov_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n256_m256_ht_cg_or_pro7',
    },
    ('o_proj', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_m256_ov_ht_pro15',
        (2049, 148): 'dense_proj_gemm_kn_n256_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_m256_ov_ht_pro15',
    },
    ('o_proj', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_m256_f32_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_m256_f32_tma1_pro7',
    },
    ('o_proj', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_ht_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_m256_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_ht_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_m256_pro7',
    },
    ('o_proj', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_f32_ht_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_f32_ht_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
    },
    ('q_a', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_cg_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n256_cg_pro15',
        (2049, 148): 'dense_proj_gemm_kk_n256_cg_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n256_cg_pro15',
    },
    ('q_a', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_pro15',
        (2049, 148): 'dense_proj_gemm_kn_n256_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_ht_pro15',
    },
    ('q_a', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_f32_v8_ht_pro7',
    },
    ('q_a', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_pro7',
    },
    ('q_a', 'wgrad', 'f32'): 'dense_proj_gemm_nn_n256_f32_v8_ef_pro7',
    ('q_b', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_cg_pro15',
        (1001, 212): 'dense_proj_gemm_kk_n256_pro15',
        (2049, 148): 'dense_proj_gemm_kk_n256_cg_pro15',
        (2049, 212): 'dense_proj_gemm_kk_n256_pro15',
    },
    ('q_b', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_m256_ov_ht_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_cg_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_m256_ov_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_cg_pro7',
    },
    ('q_b', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_m256_f32_v8_ef_ov_ht_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_m256_f32_v8_ef_ov_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
    },
    ('q_b', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_ov_ht_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_tma1_pro15',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_ov_ht_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_pro15',
    },
    ('q_b', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_f32_v8_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_f32_v8_pro15',
        (2049, 148): 'dense_proj_gemm_nn_n256_f32_v8_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_f32_v8_pro15',
    },
    ('kv_a', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n192_hee_s8_pro15',
        (1001, 212): 'dense_proj_gemm_kk_n192_m256_hen_s5_pro15',
        (2049, 148): 'dense_proj_gemm_kk_n192_hee_s8_pro15',
        (2049, 212): 'dense_proj_gemm_kk_n192_m256_hen_s5_pro15',
    },
    ('kv_a', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_tma1_pd2_pro15',
        (1001, 212): 'dense_proj_gemm_kn_n256_q_pd1_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_tma1_pd2_pro15',
        (2049, 212): 'dense_proj_gemm_kn_n256_q_pd1_pro7',
    },
    ('kv_a', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_f32_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_f32_tma1_pro15',
        (2049, 148): 'dense_proj_gemm_kn_n256_f32_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_f32_tma1_pro15',
    },
    ('kv_a', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n192_hee_t_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n192_bz64_hee_t_stt_pro15',
        (2049, 148): 'dense_proj_gemm_nn_n192_hee_t_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n192_bz64_hee_t_stt_pro15',
    },
    ('kv_a', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n192_hee_f32_t_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n192_bz64_hee_f32_t_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n192_hee_f32_t_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n192_bz64_hee_f32_t_pro7',
    },
    ('shared_gate_up', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_cg_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n256_cg_pro15',
        (2049, 148): 'dense_proj_gemm_kk_n256_cg_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n256_cg_pro15',
    },
    ('shared_gate_up', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_pro15',
        (2049, 148): 'dense_proj_gemm_kn_n256_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_ht_pro15',
    },
    ('shared_gate_up', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_f32_v8_ht_pro7',
    },
    ('shared_gate_up', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_pro7',
    },
    ('shared_gate_up', 'wgrad', 'f32'): 'dense_proj_gemm_nn_n256_f32_v8_ef_pro7',
    ('shared_down', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n256_pro15',
        (2049, 148): 'dense_proj_gemm_kk_n256_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n256_ht_pro15',
    },
    ('shared_down', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256',
        (1001, 212): 'dense_proj_gemm_kn_n256_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256',
        (2049, 212): 'dense_proj_gemm_kn_n256_pro7',
    },
    ('shared_down', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_f32_v8_pro7',
    },
    ('shared_down', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_pro7',
    },
    ('shared_down', 'wgrad', 'f32'): 'dense_proj_gemm_nn_n256_f32_v8_ef_pro7',
    ('dense_gate_up', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n256_m256_cg_or_pro7',
        (2049, 148): 'dense_proj_gemm_kk_n256_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n256_m256_ht_cg_or_pro7',
    },
    ('dense_gate_up', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_m256_ov_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_m256_cg_or_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_m256_ov_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_m256_cg_or_pro7',
    },
    ('dense_gate_up', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_m256_f32_v8_ef_ov_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_m256_f32_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_m256_f32_v8_ef_ov_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_m256_f32_tma1_pro7',
    },
    ('dense_gate_up', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_m256_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_m256_pro7',
    },
    ('dense_gate_up', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_m256_f32_ht_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_m256_f32_ht_tma1_pro7',
    },
    ('dense_down', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_m256_ov_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n256_m256_cg_or_pro7',
        (2049, 148): 'dense_proj_gemm_kk_n256_m256_ov_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n256_m256_cg_or_pro7',
    },
    ('dense_down', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_hne_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_hne_pro7',
    },
    ('dense_down', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_m256_f32_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_m256_f32_tma1_pro7',
    },
    ('dense_down', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_m256_ht_tma1_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_m256_ht_pro7',
    },
    ('dense_down', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_m256_f32_v8_ef_ht_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_m256_f32_v8_ef_ht_pro7',
    },
    ('indexer_q', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n256_cg_pro15',
        (1001, 212): 'dense_proj_gemm_kk_n256_pro15',
        (2049, 148): 'dense_proj_gemm_kk_n256_cg_pro15',
        (2049, 212): 'dense_proj_gemm_kk_n256_pro15',
    },
    ('indexer_q', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_cg_pro15',
        (1001, 212): 'dense_proj_gemm_kn_n256_pro15',
        (2049, 148): 'dense_proj_gemm_kn_n256_cg_pro15',
        (2049, 212): 'dense_proj_gemm_kn_n256_pro15',
    },
    ('indexer_q', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_f32_v8_ef_s9_pro15',
        (2049, 148): 'dense_proj_gemm_kn_n256_f32_v8_ef_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_f32_v8_ef_s9_pro15',
    },
    ('indexer_q', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n160_m256_bz64_pro15',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n160_m256_bz64_pro15',
    },
    ('indexer_q', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n160_m256_bz64_f32_v8_ef_pro15',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n160_m256_bz64_f32_v8_ef_pro15',
    },
    ('indexer_k', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n128_c1',
        (1001, 212): 'dense_proj_gemm_kk_n128_pro15',
        (2049, 148): 'dense_proj_gemm_kk_n128_c1',
        (2049, 212): 'dense_proj_gemm_kk_n128_pro15',
    },
    ('indexer_k', 'dgrad', 'bf16'): 'dense_proj_gemm_kn_n256_tma2_pd2_pro15',
    ('indexer_k', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n128_f32_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n128_f32_tma1_pd2_c1',
        (2049, 148): 'dense_proj_gemm_kn_n128_f32_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n128_f32_tma1_pd2_c1',
    },
    ('indexer_k', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n128_m64_t_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n128_m64_t_skx_pro15',
        (2049, 148): 'dense_proj_gemm_nn_n128_m64_t_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n128_m64_t_skx_pro15',
    },
    ('indexer_k', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n128_m64_f32_t_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n128_m64_f32_t_skx_pro15',
        (2049, 148): 'dense_proj_gemm_nn_n128_m64_f32_t_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n128_m64_f32_t_skx_pro15',
    },
    ('indexer_hw', 'fwd', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kk_n128_c1',
        (1001, 212): 'dense_proj_gemm_kk_n128_pro7',
        (2049, 148): 'dense_proj_gemm_kk_n128_c1',
        (2049, 212): 'dense_proj_gemm_kk_n128_pro7',
    },
    ('indexer_hw', 'dgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_kn_n256_tma2_pd2_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_tma2_pd3_c1',
        (2049, 148): 'dense_proj_gemm_kn_n256_tma2_pd2_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_tma2_pd3_c1',
    },
    ('indexer_hw', 'dgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_kn_n128_f32_tma2_s4_c1',
        (1001, 212): 'dense_proj_gemm_kn_n128_f32_tma1_c1',
        (2049, 148): 'dense_proj_gemm_kn_n128_f32_tma2_s4_c1',
        (2049, 212): 'dense_proj_gemm_kn_n128_f32_tma1_c1',
    },
    ('indexer_hw', 'wgrad', 'bf16'): {
        (1001, 148): 'dense_proj_gemm_nn_n128_m64_t_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n128_t_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n128_m64_t_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n128_t_skx_pro7',
    },
    ('indexer_hw', 'wgrad', 'f32'): {
        (1001, 148): 'dense_proj_gemm_nn_n128_m64_f32_t_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n128_f32_t_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n128_m64_f32_t_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n128_f32_t_skx_pro7',
    },
}
MLA_TEMPLATES = {
    ('qabs', 'fwd'): {
        (33, 148): 'dense_proj_gemm_nk_n128_t_pro7',
        (33, 212): 'dense_proj_gemm_nk_n128_t_pro7',
        (1001, 148): 'dense_proj_gemm_kn_n256_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_tma1_bg8_sol_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_tma1_bg8_sol_pro7',
    },
    ('qabs', 'dgrad'): {
        (33, 148): 'dense_proj_gemm_kk_n128_t_pro7',
        (33, 212): 'dense_proj_gemm_kk_n128_t_pro7',
        (1001, 148): 'dense_proj_gemm_kk_n192_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n192_pro7',
        (2049, 148): 'dense_proj_gemm_kk_n192_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n192_pro7',
    },
    ('qabs', 'wgrad'): {
        (33, 148): 'dense_proj_gemm_nn_n192_m256_t_tma1_pro7',
        (33, 212): 'dense_proj_gemm_nn_n192_m256_bz64_hee_t_tma1_s6_pro7',
        (1001, 148): 'dense_proj_gemm_nn_n192_m256_t_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n192_m256_bz64_hee_t_tma1_s6_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n192_m256_t_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n192_m256_bz64_hee_t_tma1_s6_pro7',
    },
    ('vproj', 'fwd'): {
        (33, 148): 'dense_proj_gemm_kk_n128_t_pro7',
        (33, 212): 'dense_proj_gemm_kk_n128_t_pro7',
        (1001, 148): 'dense_proj_gemm_kk_n256_q_pro7',
        (1001, 212): 'dense_proj_gemm_kk_n256_q_pro7',
        (2049, 148): 'dense_proj_gemm_kk_n256_q_pro7',
        (2049, 212): 'dense_proj_gemm_kk_n256_q_pro7',
    },
    ('vproj', 'dgrad'): {
        (33, 148): 'dense_proj_gemm_nk_n128_t_pro7',
        (33, 212): 'dense_proj_gemm_nk_n128_t_pro7',
        (1001, 148): 'dense_proj_gemm_kn_n256_m256_tma1_bg4_pro7',
        (1001, 212): 'dense_proj_gemm_kn_n256_tma1_bg8_pro7',
        (2049, 148): 'dense_proj_gemm_kn_n256_m256_tma1_bg4_pro7',
        (2049, 212): 'dense_proj_gemm_kn_n256_tma1_bg8_pro7',
    },
    ('vproj', 'wgrad'): {
        (33, 148): 'dense_proj_gemm_nn_n256_m256_t_tma1_pro7',
        (33, 212): 'dense_proj_gemm_nn_n256_m256_hee_t_tma1_s6_pro7',
        (1001, 148): 'dense_proj_gemm_nn_n256_m256_t_tma1_pro7',
        (1001, 212): 'dense_proj_gemm_nn_n256_m256_hee_t_tma1_s6_pro7',
        (2049, 148): 'dense_proj_gemm_nn_n256_m256_t_tma1_pro7',
        (2049, 212): 'dense_proj_gemm_nn_n256_m256_hee_t_tma1_s6_pro7',
    },
}
# --- END GENERATED TABLES ---
# fmt: on
# The K1 templates the export registers: the union over the architectures of
# ``cake_jit.KERNELS`` (the per-architecture filter of ``_registered`` below).
# The registry and the build manifest are checked against each other here, at
# import, before any test runs: every program a registry table names ships a
# build for that architecture, and every build entry belongs to a registered
# program.  Which templates the export holds is left to the tests that name
# them (the planner tables above, the symbol and record pins below).

EXPORTED_TEMPLATES = frozenset(
    template
    for arch in cake_jit.KERNELS
    for template in cake_jit.KERNELS[arch]
    if template.startswith("dense_proj_gemm_")
)
_REGISTRY_BUILDS = frozenset(
    f"{cake_jit.MODULES[name]['library']}_{arch}"
    for arch, table in cake_jit.KERNELS.items()
    for name in set(table.values())
)
_MANIFEST_BUILDS = frozenset(
    entry["name"] for entry in cake_build.manifest()["libraries"]
)
assert _REGISTRY_BUILDS == _MANIFEST_BUILDS, sorted(_REGISTRY_BUILDS ^ _MANIFEST_BUILDS)


def _expected(table, key, T, sm_count):
    value = table[key]
    return value if isinstance(value, str) else value[(T, sm_count)]


ARCH_OF_SM = {148: "sm_100a", 212: "sm_107a"}
# the SM count of a device of each architecture the planner can target (the two export architectures and
# compute capability 10.3, whose programs register when the export ships them)
SM_COUNT_OF_ARCH = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 212}


_TRANSPOSED_STEM = re.compile(
    r"_t(?:_tma[12])?(?:_s\d+)?(?:_box\d+)?(?:_skx)?(?:_sks)?(?:_sb[123])?(?:_b(?:f|g\d+))?(?:_so[flnd])?(?:_pd\d+)?(?:_sh\d+)?(?:_c1)?(?:_cg)?(?:_or)?(?:_pro\d+)?$"
)


def _transposed_template(template: str) -> bool:
    """``_t`` (transposed register epilogue) or ``_t_tma1`` / ``_t_tma2`` (round-14 transposed TMA-store epilogue),
    followed by the knob suffixes ``instance_symbol`` emits after the epilogue term (``_s<stages>``, ``_box<rows>``,
    ``_skx``, ``_sks`` / ``_sb<n>`` (rounds 18 / 19), ``_bf`` / ``_bg<n>``, ``_so<x>``, ``_pd<n>``, ``_sh<n>``, ``_c1``
    (round 19), ``_cg`` (round 21), ``_or`` (round 24), ``_pro<n>`` (round 33)) - e.g. the round-16 six-stage swapped MLA weight-gradient templates ``..._hee_t_tma1_s6``."""
    return _TRANSPOSED_STEM.search(template) is not None


def _nopro(template: str) -> str:
    """``template`` without the round-33 pair prologue / cleanup sync suffix (``_pro<n>``, the LAST knob suffix of every
    planned pair program since round 33): the hand-written tests of rounds 15..27 assert their own round's program modulo that
    suffix; the round-33 tests assert the suffix itself."""
    return re.sub(r"_pro\d+$", "", template)


def _rule_of(plan, sm_count):
    """The measured per-row rule the planner applied to ``plan`` (empty when the row has none or the
    registry fallback dropped it)."""
    if plan.rule_fallback:
        return {}
    return row_rule(
        ARCH_OF_SM[sm_count],
        plan.a_mn,
        plan.b_mn,
        plan.out_f32,
        plan.transposed_out,
        plan.L > 1,
        plan.N,
        plan.K,
        plan.M,
    )


def _registered(arch):
    """The K1 templates that are generated programs of ``arch`` in this tree (the export registers per architecture)."""
    return frozenset(
        t for t in cake_jit.KERNELS.get(arch, {}) if t.startswith("dense_proj_gemm_")
    )


# the knob tokens ``instance_symbol`` emits after ``_ht`` (the transposed store, the epilogue, the tail policies, the
# raster forms, the load-warp prelude, the whole-tile tile resolve, the paced output stores and the pair prologue form)
_AFTER_HT = re.compile(
    r"^(t|tma\d+|s\d+|box\d+|skx|skf|sks|stt|sb\d|bf|bg\d+|so[a-z]|pd\d+|sh\d+|c1|cg|or|lw|mg|sw\d+|pro\d+)$"
)


def _rule_program(cake_template, rule):
    """The program the planner names for a row whose rule pins ``htail``: the Cake launcher's template with the
    ``_ht`` knob where the launcher's half-tail fit dropped it.  The planner's fit selects only the plan
    (``plan.htail_plan``); the program a rule names stays (the round-2 shapes tranche measured the K 5120 thin input
    gradients on that program with the whole-tile plan).  The knob is inserted at its ``instance_symbol`` position and
    checked through the backend's decoder: every other knob of the launcher's template is unchanged."""
    if not (rule and rule.get("htail")):
        return cake_template
    before = cake_backend._template_knobs(cake_template)
    if before is None or before["htail"]:
        return cake_template
    tokens = cake_template.split("_")
    at = next(
        (i for i in range(4, len(tokens)) if _AFTER_HT.match(tokens[i])), len(tokens)
    )
    with_ht = "_".join(tokens[:at] + ["ht"] + tokens[at:])
    after = cake_backend._template_knobs(with_ht)
    assert (
        after is not None and after["htail"] and {**after, "htail": False} == before
    ), (
        cake_template,
        with_ht,
    )
    return with_ht


def _assert_mirrors_cake(plan, cake_template, where, arch, rule=None):
    """The plan is the Cake launcher's plan when that instance is a generated program of ``arch``; otherwise (the
    batched small-M swap on a tiny T outside the export contract, a measured rule whose template exists only for
    the contract's T, or a default tail-T instance no contract row of this architecture produced -- all of which
    the Cake host JIT-compiles) the FlashInfer planner must have fallen back onto a generated program.  A rule that
    pins ``htail`` names its ``_ht`` program on every device and T (``_rule_program``)."""
    cake_template = _rule_program(cake_template, rule)
    registered = _registered(arch)
    assert registered <= EXPORTED_TEMPLATES, (
        arch,
        sorted(registered - EXPORTED_TEMPLATES),
    )
    if cake_template in registered:
        assert plan.template == cake_template, (*where, plan.template)
        assert not (plan.swap_fallback or plan.rule_fallback or plan.knob_fallback), (
            where
        )
    else:
        assert plan.swap_fallback or plan.rule_fallback or plan.knob_fallback, (
            *where,
            cake_template,
            plan.template,
        )
        assert plan.template in registered, (*where, cake_template, plan.template)


def _registered_capabilities() -> str:
    """The compute capabilities this checkout registers generated programs for (the architectures of
    ``cake_jit.KERNELS``), spelled for the skip messages: ``10.0 or 10.7`` while ``build.json`` ships ``sm_100a`` and
    ``sm_107a`` builds, ``10.0, 10.3 or 10.7`` once the ``sm_103a`` programs register (see README)."""
    caps = [
        f"{major}.{minor}"
        for (major, minor), arch in sorted(
            cake_backend.SUPPORTED_COMPUTE_CAPABILITIES.items()
        )
        if arch in cake_jit.KERNELS
    ]
    return (
        " or ".join(caps)
        if len(caps) <= 2
        else ", ".join(caps[:-1]) + " or " + caps[-1]
    )


def _device_supported() -> bool:
    """A device this checkout registers generated programs for (an architecture of ``cake_jit.KERNELS``: compute
    capability ``_registered_capabilities()``; see README)."""
    return torch.cuda.is_available() and generated_program_available(
        torch.device("cuda")
    )


def _require_program(template: str):
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    if not generated_program_available(torch.device("cuda"), template):
        pytest.skip(f"generated instance {template!r} not registered for this device")


def _registered_knobs(v, *candidates):
    """The first planner knob set (in order) whose planned instance the export registered for this device.
    The per-arch export only ships the instances its measured rules route to, so a test that pins a tile
    family takes whichever family's instance this device has; skips when none is registered."""
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    templates = [_plan_template(v, **knobs) for knobs in candidates]
    for knobs, template in zip(candidates, templates, strict=True):
        if generated_program_available(torch.device("cuda"), template):
            return knobs
    pytest.skip(
        f"none of the generated instances {templates!r} is registered for this device"
    )


def _sm_count() -> int:
    return int(torch.cuda.get_device_properties(0).multi_processor_count)


def _l2_bytes() -> int:
    return device_l2_bytes(torch.device("cuda", 0))


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


ARG_KINDS = {"buffer", "tma_buffer", "workspace", "parameter", "grid"}


def test_registry_records_are_well_formed():
    # schema v2: the distinct argument plans are stored once, keyed by the digest of their compact JSON
    assert cake_jit.ARG_PLANS
    for key, plan in cake_jit.ARG_PLANS.items():
        assert re.fullmatch(r"[0-9a-f]{12}", key) and key == cake_jit.arg_plan_key(plan)
        items = [list(item) for item in plan]
        assert items and all(
            len(item) == 2 and all(isinstance(s, str) for s in item) for item in items
        )
        names = [name for _, name in items]
        assert len(set(names)) == len(names)
        assert {kind for kind, _ in items} <= ARG_KINDS
        assert ["workspace", "tma_descriptor_workspace"] not in items
        assert {name for kind, name in items if kind == "grid"} == {
            "grid_x",
            "grid_y",
            "grid_z",
        }
    # every stored plan is referenced by key; a record may instead carry its plan inline (the fused split
    # program: no stored plan spells its signature), and ``arg_plan_of`` resolves both forms to a plan list
    assert {
        record["arg_plan"]
        for record in cake_jit.MODULES.values()
        if isinstance(record["arg_plan"], str)
    } == set(cake_jit.ARG_PLANS)
    for record in cake_jit.MODULES.values():
        plan = cake_jit.arg_plan_of(record)
        assert plan and all(len(item) == 2 for item in plan)
        if not isinstance(record["arg_plan"], str):
            assert plan == [list(item) for item in record["arg_plan"]]
            assert {name for kind, name in plan if kind == "grid"} == {
                "grid_x",
                "grid_y",
                "grid_z",
            }
    # build.json (``cake_build``) is the only record of what to compile; the registry names its
    # entries ``<library>_<arch>``
    manifest = cake_build.manifest()
    libraries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert len(libraries) == len(manifest["libraries"])  # names unique
    assert set(cake_jit.KERNELS) == set(cake_jit.ARCHES) == set(manifest["arch_flags"])
    for arch, table in cake_jit.KERNELS.items():
        for template, name in table.items():
            record = cake_jit.MODULES[name]
            assert arch in record["arches"] and record["template"] == template
            assert cake_jit.select_module(arch, template) == name
    w8_modules: set[str] = set()
    ten_warp_modules: set[str] = set()
    for name, record in cake_jit.MODULES.items():
        # architecture-neutral programs: one source pair, built for every arch it serves
        assert record["arches"] and record["arches"] == sorted(record["arches"])
        assert record["library"] == name
        for arch in record["arches"]:
            assert arch in cake_jit.ARCHES
            assert cake_jit.KERNELS[arch][record["template"]] == name
            entry = libraries[cake_jit.library_name(name, arch)]
            assert entry["arch"] == arch and entry["entry"] == record["ffi_entry"]
            assert len(entry["sources"]) == 2
            assert all(
                s.startswith("csrc/cake_dense_projection_gemm/") and "/sm_" not in s
                for s in entry["sources"]
            )
            assert all(
                (Path(cake_jit.__file__).parent / s).is_file() for s in entry["sources"]
            )
            assert len(entry["closure_sha256"]) == 64
        assert (
            "tma_workspace_bytes" not in record
        )  # by-value TMA ABI: no descriptor workspace
        plan_items = [list(item) for item in cake_jit.arg_plan_of(record)]
        if isinstance(record["arg_plan"], str):
            assert plan_items == [
                list(item) for item in cake_jit.ARG_PLANS[record["arg_plan"]]
            ]
        else:
            # the fused split program is the only record with an inline plan
            assert record["template"] == cake_backend.SPLIT_TEMPLATE, record["template"]
            assert plan_items == [list(item) for item in record["arg_plan"]]
        if record["template"].startswith("dense_proj_gemm_"):
            # the raster group width is a launch parameter since round 11; the stream-K slabs and the
            # slice counters are host-owned buffers
            assert ["parameter", "group_m"] in plan_items
            assert ["buffer", "ws"] in plan_items and [
                "buffer",
                "counters",
            ] in plan_items
            if cake_backend._template_form(
                cake_backend._template_tokens(record["template"])
            ):
                # the eight-warp CTA programs take the raster's four fast divisors, run 256 threads per CTA and
                # serve sm_103a alone; their generator module is not the ten-warp programs'
                for name_fd in ("fd_tiles_l", "fd_tiles_g", "fd_group", "fd_last"):
                    assert ["parameter", name_fd] in plan_items, (name, name_fd)
                assert list(record["launch"]["block"]) == [256, 1, 1], name
                assert record["arches"] == ["sm_103a"], name
                w8_modules.add(record["build"]["module"])
            elif "build" in record:
                ten_warp_modules.add(record["build"]["module"])
        elif record["template"] == cake_backend.SPLIT_TEMPLATE:
            # the fp32 source, the [3, ...] BF16 stack and the element count
            assert ["buffer", "src"] in plan_items and ["buffer", "parts"] in plan_items
            assert ["parameter", "n"] in plan_items
        else:
            assert ["parameter", "k_iters_split"] in plan_items
        launch = record["launch"]
        # round 19 (Cake W3): the single-CTA form (``_c1`` templates) has no cluster, nor has the fused split
        # program (an elementwise kernel); every other program is a two-CTA cluster (the exporter's registry check)
        want_cluster = (
            [1, 1, 1]
            if record["template"].endswith("_c1")
            or record["template"] == cake_backend.SPLIT_TEMPLATE
            else [CTA_GROUP, 1, 1]
        )
        assert list(launch["cluster"]) == want_cluster, (name, launch)
        assert len(launch["block"]) == 3 and all(int(b) >= 1 for b in launch["block"])
        assert int(launch["block"][0]) % 32 == 0  # whole warps
    assert len(w8_modules) <= 1 and w8_modules.isdisjoint(ten_warp_modules)
    with pytest.raises(NotImplementedError, match="not registered"):
        cake_jit.select_module("sm_100a", "dense_proj_gemm_not_a_template")
    # every manifest entry belongs to a registered (program, arch) pair
    assert set(libraries) == {
        cake_jit.library_name(name, arch)
        for name, record in cake_jit.MODULES.items()
        for arch in record["arches"]
    }


def test_registry_literals_follow_the_export_contract():
    """The three registry literals are the text the Cake export replaces: each wrapped in ``# fmt: off`` /
    ``# fmt: on`` behind its exact head line, one record per line, in the order ARG_PLANS, MODULES, KERNELS,
    and each evaluates to the object this module imports."""
    text = Path(cake_jit.__file__).read_text()
    heads = (
        ("ARG_PLANS: dict[str, list[list[str]]] = {", cake_jit.ARG_PLANS),
        ("MODULES: dict[str, dict[str, Any]] = {", cake_jit.MODULES),
        ("KERNELS: dict[str, dict[str, str]] = {", cake_jit.KERNELS),
    )
    positions = []
    for head, value in heads:
        assert text.count("\n" + head + "\n") == 1, head
        start = text.index("\n" + head + "\n") + 1
        assert text[:start].endswith("# fmt: off\n"), head
        end = text.index("\n}\n# fmt: on\n", start)
        body = text[start + len(head) + 1 : end]
        lines = body.split("\n")
        assert lines and all(
            re.fullmatch(r'    "[^"]+": \S.*,', line) for line in lines
        ), head
        assert ast.literal_eval("{\n" + body + "\n}") == value, head
        positions.append(start)
    assert positions == sorted(positions)
    assert cake_jit.arg_plan_key([("grid", "grid_x")]) == cake_jit.arg_plan_key(
        [["grid", "grid_x"]]
    )


def test_epilogue_rules():
    assert epi_mode(False, True) == "reg"
    assert epi_mode(True, True) == "reg"
    assert (
        epi_mode(True, False) == "tma"
    )  # row-major fp32: TMA-store epilogue by default
    assert epi_mode(True, False, K=64) == "tma"
    assert (
        epi_mode(True, False, epi="reg") == "reg"
    )  # the float4 register path is an opt-in
    assert epi_mode(False, False, K=1024) == "tma"
    assert epi_mode(False, False, K=1025) == "reg"
    assert epi_mode(False, False) == "reg"
    # round 14: the transposed TMA-store epilogue is an opt-in whose warp slice must be whole CH_T_COLS-column chunks
    assert epi_mode(False, True, epi="tma", block_n=192, cta_rows=256) == "tma"
    assert epi_mode(False, True, epi="tma", block_n=256, cta_rows=256) == "tma"
    with pytest.raises(ValueError, match="transposed"):
        epi_mode(False, True, epi="tma", block_n=160, cta_rows=256)
    with pytest.raises(ValueError, match="epi must be"):
        epi_mode(False, False, epi="bad")
    assert epi_slots("reg", False, 256) == 0
    assert epi_slots("tma", False, 256, K=6144) == 1
    assert epi_slots("tma", True, 256, K=6144) == 1
    assert (
        epi_slots("tma", True, 128) == 2
    )  # K unknown: two slots when the chunk count allows
    assert (
        epi_slots("tma", True, 128, K=64) == 1
    )  # any K above K_TWO_SLOTS = 0: one slot
    with pytest.raises(ValueError, match="slots"):
        epi_slots("tma", False, 128, slots=3)
    # round 9: the warp slice is BLOCK_N / 2 columns, BLOCK_N / 4 on the 64-row Layout-B family
    assert BLOCK_N_CHOICES == (128, 160, 192, 224, 256) and CTA_ROWS_CHOICES == (
        64,
        128,
        256,
    )
    assert [epi_cols(n) for n in BLOCK_N_CHOICES] == [64, 80, 96, 112, 128]
    assert [epi_cols(n, 64) for n in BLOCK_N_CHOICES] == [32, 40, 48, 56, 64]
    assert epi_cols(256, 256) == 128
    # fp32: TMA stores only for whole 32-column slices (160 / 224 and the 64-row 192 fall back to registers)
    assert epi_mode(True, False, block_n=192) == "tma"
    assert epi_mode(True, False, block_n=160) == "reg"
    assert epi_mode(True, False, block_n=224) == "reg"
    assert epi_mode(True, False, block_n=192, cta_rows=64) == "reg"
    assert epi_mode(True, False, block_n=128, cta_rows=64) == "tma"
    # bf16: whole 64-column slices only (192 / 160 / 224 and the 64-row 128 have none)
    assert epi_mode(False, False, K=1024, block_n=192) == "reg"
    assert epi_mode(False, False, K=1024, block_n=160) == "reg"
    assert epi_mode(False, False, K=1024, block_n=256, cta_rows=64) == "tma"
    assert epi_mode(False, False, K=1024, block_n=128, cta_rows=64) == "reg"
    assert epi_mode(False, False, K=1024, block_n=128, cta_rows=256) == "tma"
    assert (
        epi_slots("tma", False, 256, K=64, cta_rows=64) == 1
    )  # one 64-column chunk per 64-row slice
    assert (
        epi_slots("tma", True, 128, cta_rows=64) == 1
    )  # one 32-column chunk: the two-slot default falls back to one
    with pytest.raises(ValueError, match="slots"):
        epi_slots("tma", False, 256, slots=2, cta_rows=64)


@pytest.mark.parametrize(
    "kwargs, symbol",
    [
        (dict(a_mn=False, b_mn=False), "dense_proj_gemm_kk_n256"),
        (dict(a_mn=False, b_mn=False, block_n=128), "dense_proj_gemm_kk_n128"),
        # the launcher resolves the slot count from K (1 for every K > K_TWO_SLOTS = 0) and passes it
        (
            dict(a_mn=False, b_mn=False, epi="tma", slots=1),
            "dense_proj_gemm_kk_n256_tma1",
        ),
        (dict(a_mn=False, b_mn=False, epi="tma"), "dense_proj_gemm_kk_n256_tma2"),
        (
            dict(a_mn=False, b_mn=True, out_f32=True, slots=1),
            "dense_proj_gemm_kn_n256_f32_tma1",
        ),
        (
            dict(a_mn=False, b_mn=True, out_f32=True, epi="reg"),
            "dense_proj_gemm_kn_n256_f32",
        ),
        (dict(a_mn=True, b_mn=True), "dense_proj_gemm_nn_n256"),
        (
            dict(a_mn=True, b_mn=True, epi="tma", slots=1),
            "dense_proj_gemm_nn_n256_tma1",
        ),
        (
            dict(a_mn=True, b_mn=True, out_f32=True, slots=1),
            "dense_proj_gemm_nn_n256_f32_tma1",
        ),
        (
            dict(a_mn=True, b_mn=True, out_t=True, block_n=128),
            "dense_proj_gemm_nn_n128_t",
        ),
        (
            dict(a_mn=True, b_mn=True, out_t=True, out_f32=True, block_n=128),
            "dense_proj_gemm_nn_n128_f32_t",
        ),
        (dict(a_mn=False, b_mn=False, cta_rows=256), "dense_proj_gemm_kk_n256_m256"),
        (dict(a_mn=False, b_mn=False, stages=5), "dense_proj_gemm_kk_n256_s5"),
        # BLOCK_N = 128 defaults to 9 stages (8 with one staging slot): 9 carries no suffix, 7 does
        (
            dict(a_mn=False, b_mn=False, block_n=128, stages=9),
            "dense_proj_gemm_kk_n128",
        ),
        (
            dict(a_mn=False, b_mn=False, block_n=128, stages=7),
            "dense_proj_gemm_kk_n128_s7",
        ),
        (
            dict(a_mn=False, b_mn=False, block_n=128, epi="tma", slots=1, stages=8),
            "dense_proj_gemm_kk_n128_tma1",
        ),
        # TMA L2 eviction hints (A, B): first letters after ``_h``; L2 promotion and prefetch distance
        (
            dict(a_mn=False, b_mn=False, block_n=128, hints=("evict_first", "none")),
            "dense_proj_gemm_kk_n128_hen",
        ),
        # first letters only, as the Cake host writes them: evict_first / evict_last -> ``_hee``
        (
            dict(a_mn=False, b_mn=False, hints=("evict_first", "evict_last")),
            "dense_proj_gemm_kk_n256_hee",
        ),
        (
            dict(a_mn=False, b_mn=False, hints=("none", "evict_normal")),
            "dense_proj_gemm_kk_n256_hne",
        ),
        # the TMA L2 promotion is a launch parameter since round 11: no symbol suffix, not a key field
        (dict(a_mn=False, b_mn=False, pf=4), "dense_proj_gemm_kk_n256_pf4"),
        # the raster group width is a launch parameter since round 11: no symbol suffix, not a key field
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_f32=True,
                slots=1,
                hints=("evict_first", "none"),
            ),
            "dense_proj_gemm_nn_n256_hen_f32_tma1",
        ),
        (
            dict(
                a_mn=False,
                b_mn=False,
                pf=2,
                hints=("evict_first", "none"),
                epi="tma",
                slots=1,
            ),
            "dense_proj_gemm_kk_n256_pf2_hen_tma1",
        ),
        (
            dict(
                a_mn=False,
                b_mn=False,
                epi="tma",
                slots=1,
                hints=("evict_first", "none"),
            ),
            "dense_proj_gemm_kk_n256_hen_tma1",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_t=True,
                block_n=128,
                hints=("evict_first", "none"),
            ),
            "dense_proj_gemm_nn_n128_hen_t",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_t=True,
                out_f32=True,
                block_n=128,
                hints=("evict_first", "none"),
            ),
            "dense_proj_gemm_nn_n128_hen_f32_t",
        ),
        # round 7: the quad-transposed bf16 register epilogue (``_q``); the pinned stage count keeps its suffix
        (
            dict(a_mn=False, b_mn=True, epi="reg", quad_store=True),
            "dense_proj_gemm_kn_n256_q",
        ),
        (
            dict(a_mn=False, b_mn=True, stages=6, epi="reg", quad_store=True),
            "dense_proj_gemm_kn_n256_q_s6",
        ),
        # round 9: the 64-row Layout-B family (``_m64``); its default is the deepest pipeline that fits
        # (9 at BLOCK_N = 256): 9 carries no suffix, 6 does
        (dict(a_mn=False, b_mn=False, cta_rows=64), "dense_proj_gemm_kk_n256_m64"),
        (dict(a_mn=False, b_mn=True, cta_rows=64), "dense_proj_gemm_kn_n256_m64"),
        (
            dict(a_mn=False, b_mn=False, cta_rows=64, stages=9),
            "dense_proj_gemm_kk_n256_m64",
        ),
        (
            dict(a_mn=False, b_mn=False, cta_rows=64, stages=6),
            "dense_proj_gemm_kk_n256_m64_s6",
        ),
        (
            dict(a_mn=False, b_mn=False, cta_rows=64, epi="tma", slots=1),
            "dense_proj_gemm_kk_n256_m64_tma1",
        ),
        (
            dict(a_mn=False, b_mn=False, block_n=128, cta_rows=64, out_f32=True),
            "dense_proj_gemm_kk_n128_m64_f32_tma1",
        ),
        # BLOCK_N = 160 / 224 and the 64-row BLOCK_N = 192: register epilogues (no whole 32 / 64-column chunks)
        (dict(a_mn=False, b_mn=False, block_n=160), "dense_proj_gemm_kk_n160"),
        (
            dict(a_mn=False, b_mn=True, block_n=224, out_f32=True),
            "dense_proj_gemm_kn_n224_f32",
        ),
        (
            dict(a_mn=False, b_mn=True, block_n=192, cta_rows=64, out_f32=True),
            "dense_proj_gemm_kn_n192_m64_f32",
        ),
        # round 13: the tall family's overlapped single-TMEM-buffer epilogue (``_ov``), the deterministic
        # half-height tail wave (``_ht``) and the parked epilogue (``_pk``); suffix order after ``_q``
        (
            dict(a_mn=False, b_mn=False, cta_rows=256, ovl=True),
            "dense_proj_gemm_kk_n256_m256_ov",
        ),
        (
            dict(a_mn=False, b_mn=True, cta_rows=256, ovl=True, htail=True),
            "dense_proj_gemm_kn_n256_m256_ov_ht",
        ),
        (
            dict(a_mn=False, b_mn=False, cta_rows=256, htail=True),
            "dense_proj_gemm_kk_n256_m256_ht",
        ),
        (
            dict(a_mn=False, b_mn=False, cta_rows=256, park=True),
            "dense_proj_gemm_kk_n256_m256_pk",
        ),
        # round 13 (W3): narrow MN-major B panels (``_bz64`` right after the tile family) and the exact p-way
        # stream-K split (trailing ``_skx``); K-major B instances always keep the 128-byte panel (no suffix)
        (
            dict(a_mn=True, b_mn=True, block_n=160, cta_rows=256, b_swz=64),
            "dense_proj_gemm_nn_n160_m256_bz64",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                block_n=192,
                out_t=True,
                b_swz=64,
                hints=("evict_first", "evict_first"),
            ),
            "dense_proj_gemm_nn_n192_bz64_hee_t",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                block_n=128,
                out_t=True,
                hints=("evict_first", "none"),
                sk_exact=True,
            ),
            "dense_proj_gemm_nn_n128_hen_t_skx",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                block_n=128,
                cta_rows=64,
                out_t=True,
                out_f32=True,
                hints=("evict_first", "none"),
                sk_exact=True,
            ),
            "dense_proj_gemm_nn_n128_m64_hen_f32_t_skx",
        ),
        (
            dict(a_mn=False, b_mn=False, block_n=256, b_swz=64),
            "dense_proj_gemm_kk_n256",
        ),
        # round 13 (W3, L38b): the batch-entry raster of the batched MLA rows (trailing ``_bg{G}`` / ``_bf``)
        (
            dict(a_mn=False, b_mn=True, epi="tma", slots=1, batch_group=8),
            "dense_proj_gemm_kn_n256_tma1_bg8",
        ),
        (
            dict(
                a_mn=False, b_mn=True, cta_rows=256, epi="tma", slots=1, batch_group=4
            ),
            "dense_proj_gemm_kn_n256_m256_tma1_bg4",
        ),
        (
            dict(a_mn=False, b_mn=True, epi="tma", slots=1, batch_group=1),
            "dense_proj_gemm_kn_n256_tma1_bf",
        ),
        # round 13 (W4): the fp32 v8 register stores with L1::no_allocate.L2::evict_first (``_ef`` right after ``_v8``)
        (
            dict(
                a_mn=False,
                b_mn=True,
                out_f32=True,
                epi="reg",
                f32_v8=True,
                store_ef=True,
            ),
            "dense_proj_gemm_kn_n256_f32_v8_ef",
        ),
        (
            dict(
                a_mn=False,
                b_mn=True,
                out_f32=True,
                epi="reg",
                f32_v8=True,
                store_ef=True,
                stages=9,
            ),
            "dense_proj_gemm_kn_n256_f32_v8_ef_s9",
        ),
        # round 14 (W3): the transposed TMA-store epilogue of the swapped MLA weight gradients (``_t_tma1``)
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_t=True,
                block_n=192,
                cta_rows=256,
                epi="tma",
                slots=1,
            ),
            "dense_proj_gemm_nn_n192_m256_t_tma1",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_t=True,
                block_n=256,
                cta_rows=256,
                epi="tma",
                slots=1,
            ),
            "dense_proj_gemm_nn_n256_m256_t_tma1",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_t=True,
                block_n=192,
                cta_rows=256,
                b_swz=64,
                hints=("evict_first", "evict_first"),
                epi="tma",
                slots=1,
            ),
            "dense_proj_gemm_nn_n192_m256_bz64_hee_t_tma1",
        ),
        (
            dict(
                a_mn=False,
                b_mn=True,
                epi="tma",
                slots=1,
                batch_group=8,
                store_hint="evict_last",
            ),
            "dense_proj_gemm_kn_n256_tma1_bg8_sol",
        ),
        (
            dict(a_mn=False, b_mn=True, epi="tma", slots=2, pd=2),
            "dense_proj_gemm_kn_n256_tma2_pd2",
        ),
        (
            dict(a_mn=False, b_mn=True, epi="reg", quad_store=True, pd=1),
            "dense_proj_gemm_kn_n256_q_pd1",
        ),
        # round 19 (Cake W2): the slab path of the synchronised stream-K fixup (``_sb<n>`` right after ``_sks``)
        (
            dict(a_mn=True, b_mn=True, cta_rows=256, sk_sync=True, sk_slab=2),
            "dense_proj_gemm_nn_n256_m256_sks_sb2",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                cta_rows=256,
                out_f32=True,
                slots=1,
                sk_sync=True,
                sk_slab=3,
            ),
            "dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3",
        ),
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_t=True,
                block_n=192,
                b_swz=64,
                sk_sync=True,
                sk_slab=2,
            ),
            "dense_proj_gemm_nn_n192_bz64_t_sks_sb2",
        ),
        # round 19 (Cake W3): the single-CTA form (trailing ``_c1``; ``_s<n>`` relative to its own stage default)
        (
            dict(a_mn=False, b_mn=False, block_n=128, cta1=True),
            "dense_proj_gemm_kk_n128_c1",
        ),
        (
            dict(
                a_mn=False,
                b_mn=True,
                out_f32=True,
                block_n=128,
                slots=2,
                stages=4,
                cta1=True,
            ),
            "dense_proj_gemm_kn_n128_f32_tma2_s4_c1",
        ),
    ],
)
def test_instance_symbols(kwargs, symbol):
    assert instance_symbol(instance_key(**kwargs)) == symbol


def test_round14_transposed_tma_store_instances():
    """Round 14 (W3): a transposed box stages CH_T_COLS output rows per chunk, so the swapped MLA weight-gradient tiles
    (BLOCK_N 192 / 256 x 256 rows) take ``epi="tma"`` while the row-major bf16 192-column tile has no whole 64-column
    chunk path (the round-14 export run died on this mirror check).  [Cake ``instance_key`` L1259-L1261]"""
    key = instance_key(
        a_mn=True, b_mn=True, out_t=True, block_n=192, cta_rows=256, epi="tma", slots=1
    )
    assert key[3] is True and key[7] == "tma" and key[8] == 1
    assert instance_symbol(key) == "dense_proj_gemm_nn_n192_m256_t_tma1"
    with pytest.raises(ValueError, match="128-byte column chunks"):
        instance_key(
            a_mn=True, b_mn=True, block_n=192, cta_rows=256, epi="tma", slots=1
        )
    with pytest.raises(ValueError, match="transposed TMA-store"):
        instance_key(
            a_mn=True,
            b_mn=True,
            out_t=True,
            block_n=160,
            cta_rows=256,
            epi="tma",
            slots=1,
        )


def test_round13_w3_knobs():
    # fields 18 / 19 of the 20-field key: b_swz (128 for K-major B whatever the caller asks), sk_exact; the narrow panel
    # shrinks the B stage (160 columns: 3 x 32-column panels = 12 KiB instead of 2 x 64-column = 16 KiB) and the
    # default stage count follows the deepest fit; ovl needs the 128-byte panel; htail and sk_exact exclude each other
    key = instance_key(a_mn=True, b_mn=True, block_n=160, cta_rows=256, b_swz=64)
    assert (
        len(key) == 37
        and key[18] == 64
        and key[19] is False
        and key[20] == 0
        and key[21] is False
    )
    assert instance_key(a_mn=False, b_mn=False, b_swz=64)[18] == 128
    assert (
        b_stage_bytes(True, 160, 64) == 3 * 64 * 64
        and b_stage_bytes(True, 160, 128) == 2 * 64 * 128
    )
    assert default_stages(0, 256, 160, True, 64) >= default_stages(
        0, 256, 160, True, 128
    )
    assert (
        instance_key(a_mn=True, b_mn=True, block_n=128, out_t=True, sk_exact=True)[19]
        is True
    )
    with pytest.raises(ValueError):
        instance_key(a_mn=True, b_mn=True, block_n=160, cta_rows=256, b_swz=48)
    with pytest.raises(ValueError):
        instance_key(a_mn=False, b_mn=True, cta_rows=256, ovl=True, b_swz=64)
    with pytest.raises(ValueError):
        instance_key(a_mn=False, b_mn=False, cta_rows=256, htail=True, sk_exact=True)


def test_round13_batch_group_knob():
    # field 20 of the 21-field key: the batch-entry raster (0 off, 1 every entry, G > 1 groups); negative values raise
    assert instance_key(a_mn=False, b_mn=True, batch_group=8)[20] == 8
    assert instance_key(a_mn=False, b_mn=True)[20] == 0
    with pytest.raises(ValueError, match="batch_group"):
        instance_key(a_mn=False, b_mn=True, batch_group=-1)


def test_round13_store_ef_knob():
    # field 21 of the 22-field key: the hint exists only on the fp32 v8 register store form
    assert (
        instance_key(
            a_mn=False, b_mn=True, out_f32=True, epi="reg", f32_v8=True, store_ef=True
        )[21]
        is True
    )
    assert (
        instance_key(a_mn=False, b_mn=True, out_f32=True, epi="reg", store_ef=True)[21]
        is False
    )
    assert (
        instance_key(a_mn=False, b_mn=True, out_f32=True, f32_v8=True, store_ef=True)[
            21
        ]
        is False
    )  # TMA-store fp32 epilogue: no v8 stores
    assert instance_key(a_mn=False, b_mn=True, store_ef=True)[21] is False


def test_round15_store_hint_knob():
    # field 22 of the 25-field key (round 15, Cake W3): the L2 eviction policy of the TMA-store epilogue's bulk
    # tensor stores (symbol ``_so<f|l|n>``); the register epilogues carry no operand, so the field is forced to
    # "none" there (one key per register-epilogue instance) and an unknown policy raises
    key = instance_key(
        a_mn=False, b_mn=True, epi="tma", slots=1, store_hint="evict_last"
    )
    assert len(key) == 37 and key[22] == "evict_last"
    assert instance_symbol(key) == "dense_proj_gemm_kn_n256_tma1_sol"
    assert (
        instance_key(a_mn=False, b_mn=True, epi="reg", store_hint="evict_last")[22]
        == "none"
    )
    assert instance_key(a_mn=False, b_mn=True)[22] == "none"
    with pytest.raises(ValueError, match="store_hint"):
        instance_key(
            a_mn=False, b_mn=True, epi="tma", slots=1, store_hint="evict_sometimes"
        )
    # the round-15 sm_107a rule: the batched MLA qabs forward (N = 512, K = 192) plans the 8-head interleave with
    # evict_last stores; sm_100a keeps the round-13 plan without the operand
    v = _views("mla", "qabs", "fwd", "bf16", 1001)
    r200, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=212,
        l2_bytes=L2_BYTES,
        arch="sm_107a",
        _fallback=False,
    )
    assert r200.store_hint == "evict_last" and r200.batch_group == 8
    assert _nopro(r200.template) == "dense_proj_gemm_kn_n256_tma1_bg8_sol"
    b200, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=148,
        l2_bytes=L2_BYTES,
        arch="sm_100a",
        _fallback=False,
    )
    assert b200.store_hint == "none" and "_so" not in _nopro(b200.template)


def test_round15_pd_sh_knobs():
    # fields 23 / 24 of the 26-field key (round 15, Cake W1): the pipelined TMEM drain form of the 128-row single-pass
    # family (1 = in-tile chunk pipelining, 2 = cross-tile prefetch, 3 = both chunks in flight; ``_pd<n>``) and the
    # suspend-time hint of the free-running waits in ns (``_sh<n>``); both 0 when off, validated like the Cake kernel
    key = instance_key(a_mn=False, b_mn=True, epi="tma", slots=2, pd=2)
    assert len(key) == 37 and key[23] == 2 and key[24] == 0
    assert instance_symbol(key) == "dense_proj_gemm_kn_n256_tma2_pd2"
    assert (
        instance_key(a_mn=False, b_mn=True)[23:]
        == (
            0,
            0,
            False,
            0,
            False,
            False,
            False,
            False,
            0,
            False,
            False,
            False,
            0,
            "",
        )
    )  # ... followed by the round-18 sk_sync flag, the round-19 sk_slab / cta1 fields, the round-21 cgrp flag, the round-24 ovr flag, the round-27 stt flag, the round-33 pro knob (0 = off), the round-2 shapes tranche's sk_first / lw flags, the whole-tile tile resolve flag mg and the paced output-store step count sw and the CTA form token ("" = the default form)
    assert (
        instance_symbol(
            instance_key(a_mn=False, b_mn=True, epi="tma", slots=2, pd=2, sh=1000)
        )
        == "dense_proj_gemm_kn_n256_tma2_pd2_sh1000"
    )
    with pytest.raises(ValueError, match="pd must be one of"):
        instance_key(a_mn=False, b_mn=True, pd=4)
    with pytest.raises(ValueError, match="pd needs cta_rows=128"):
        instance_key(a_mn=False, b_mn=True, cta_rows=256, pd=2)
    with pytest.raises(ValueError, match="one chunk per slice"):
        instance_key(a_mn=False, b_mn=True, block_n=128, pd=1)
    with pytest.raises(ValueError, match="suspendTimeHint"):
        instance_key(a_mn=False, b_mn=True, sh=-1)
    # the round-15 rules: the sm_107a indexer_k bf16 input gradient (N = 6144, K = 128) plans the 128-row two-slot
    # TMA-store family with the cross-tile pipelined drain; a caller-forced tall tile drops the family knob
    v = _views("proj", "indexer_k", "dgrad", "bf16", 1001)
    r200, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=212,
        l2_bytes=L2_BYTES,
        arch="sm_107a",
        _fallback=False,
    )
    assert r200.pd == 2 and r200.slots == 2 and r200.cta_rows == 128
    assert _nopro(r200.template) == "dense_proj_gemm_kn_n256_tma2_pd2"
    tall, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=212,
        l2_bytes=L2_BYTES,
        arch="sm_107a",
        _fallback=False,
        cta_rows=256,
    )
    assert tall.pd == 0 and "_pd" not in _nopro(tall.template)


def test_sk_exact_plan_mirrors_cake():
    # 24 tail tiles on 106 pairs, 96 K blocks: 3 parts of 32 steps -> 72 units, one segment each; a 2-part split of 3 K
    # blocks is refused (part under SK_MIN_ITERS), as is a split whose last part would be empty
    assert sk_exact_plan(24, 96, 106, 3) == (0, 24, 72, 32)
    assert sk_exact_plan(130, 96, 106, 2) == (106, 24, 48, 48)
    assert sk_exact_plan(24, 96, 106, 1) is None  # fewer than two parts
    assert sk_exact_plan(212, 96, 106, 3) is None  # no tail
    assert (
        sk_exact_plan(24, SK_MIN_ITERS, 106, 2) is None
    )  # a part under SK_MIN_ITERS steps
    assert (
        sk_exact_plan(24, 72, 106, 10) is None
    )  # ceil(72 / 10) = 8 steps x 9 parts already cover K: empty last part
    assert sk_exact_plan(24, 73, 106, 10) == (
        0,
        24,
        240,
        8,
    )  # ... one more K block and the tenth part is non-empty


def test_sk_sync_plan_mirrors_cake():
    # round 18 (Cake W2): the synchronised stream-K plan (num_full, tail, tail + collectors, s, collectors) with the
    # balanced main length s = ceil(q K / (q + 1)) and the rule's margin m: s = ceil((q K + m) / (q + 1)).  kv_a weight
    # gradient on 106 pairs: 72 tail tiles, 34 collectors of up to q = 3 leftovers over K = 254 steps -> s 191 balanced,
    # 200 with the rule's m = 38; the 1.81-wave 6144 x 2048 rows: 192 tiles -> 86 tails, 20 collectors (q = 5), s 212 /
    # 216 (m = 26); the tall 64-tile B200 indexer_q row on 74 pairs: 10 collectors of 7 leftovers, s 223
    assert sk_sync_plan(72, 254, 106) == (0, 72, 106, 191, 34)
    assert sk_sync_plan(72, 254, 106, None, 38) == (0, 72, 106, 200, 34)
    assert sk_sync_plan(192, 254, 106) == (106, 86, 106, 212, 20)
    assert sk_sync_plan(192, 254, 106, None, 26) == (106, 86, 106, 216, 20)
    assert sk_sync_plan(64, 254, 74) == (0, 64, 74, 223, 10)
    assert sk_sync_plan(72, 254, 106, 173) == (
        0,
        72,
        106,
        173,
        34,
    )  # an explicit s (sweeps only)
    assert sk_sync_plan(212, 254, 106) is None  # no tail
    assert sk_sync_plan(106, 254, 106) is None  # a full wave: no collector
    assert (
        sk_sync_plan(72, 254, 74) is None
    )  # 2 collectors x 36 leftovers: K - s = 6 < SK_MIN_ITERS
    # the short-K rows admit the plan arithmetically (W3's q_a fwd observation: 1536 tiles on 106 pairs, 52 tails,
    # q = 1, s = 16) - the rules never ask for it there (the fixup costs more than the 16-step unit)
    assert sk_sync_plan(1536, 32, 106) == (1484, 52, 104, 16, 52)
    assert sk_sync_plan(52, 32, 74) == (0, 52, 74, 24, 22)
    assert (
        sk_sync_plan(52, 30, 74) is None
    )  # q = 3: s = 23 leaves 7 steps < SK_MIN_ITERS
    with_margin = sk_sync_plan(72, 254, 106, None, 400)
    assert (
        with_margin is None
    )  # a margin that leaves the collectors under SK_MIN_ITERS steps is refused


def test_round13_knob_normalisation():
    # the 20-field key carries park / ovl / htail at fields 15 / 16 / 17; ovl is a 256-row x 256-column tall-tile
    # knob (narrower or shorter tiles raise), htail needs the 256-row family or (round 16, Cake W2) the 128-row standard
    # family with the 256-column row-major tile and the plain drain, park only the bf16 row-major tall store
    key = instance_key(a_mn=False, b_mn=False, cta_rows=256, ovl=True, htail=True)
    assert len(key) == 37 and key[15] is False and key[16] is True and key[17] is True
    assert (
        instance_key(a_mn=False, b_mn=False, cta_rows=256, out_f32=True, park=True)[15]
        is False
    )
    with pytest.raises(ValueError):
        instance_key(a_mn=False, b_mn=False, cta_rows=256, block_n=192, ovl=True)
    key128 = instance_key(a_mn=False, b_mn=True, cta_rows=128, htail=True)
    assert key128[17] is True and instance_symbol(key128).endswith("_ht")
    with pytest.raises(ValueError):
        instance_key(a_mn=False, b_mn=True, cta_rows=128, block_n=192, htail=True)
    with pytest.raises(ValueError):
        instance_key(a_mn=False, b_mn=True, cta_rows=128, htail=True, pd=1)
    with pytest.raises(ValueError):
        instance_key(a_mn=True, b_mn=True, out_t=True, cta_rows=128, htail=True)
    with pytest.raises(ValueError):
        instance_key(a_mn=False, b_mn=False, cta_rows=64, htail=True)
    with pytest.raises(ValueError):
        instance_key(a_mn=False, b_mn=False, cta_rows=256, park=True, ovl=True)


def test_quad_store_knob_normalisation():
    # only the row-major bf16 register epilogue carries the quad-store knob: fp32 output, the TMA-store
    # epilogue and the transposed store drop it from the key (and the symbol); field 14 of the 20-field key
    assert instance_key(a_mn=False, b_mn=True, epi="reg", quad_store=True)[14] is True
    assert (
        instance_key(a_mn=False, b_mn=True, out_f32=True, epi="reg", quad_store=True)[
            14
        ]
        is False
    )
    assert (
        instance_key(a_mn=False, b_mn=True, epi="tma", slots=1, quad_store=True)[14]
        is False
    )
    assert (
        instance_key(a_mn=True, b_mn=True, out_t=True, epi="reg", quad_store=True)[14]
        is False
    )
    assert (
        instance_key(a_mn=False, b_mn=True, block_n=128, epi="reg", quad_store=True)[14]
        is True
    )
    # whole 64-column groups per warp slice: not at BLOCK_N = 160 / 192 / 224 (80 / 96 / 112 columns) nor on
    # 64-row tiles below BLOCK_N = 256
    for block_n in (160, 192, 224):
        assert (
            instance_key(
                a_mn=False, b_mn=True, block_n=block_n, epi="reg", quad_store=True
            )[14]
            is False
        )
    assert (
        instance_key(a_mn=False, b_mn=True, cta_rows=64, epi="reg", quad_store=True)[14]
        is True
    )
    assert (
        instance_key(
            a_mn=False, b_mn=True, block_n=128, cta_rows=64, epi="reg", quad_store=True
        )[14]
        is False
    )
    assert (
        instance_symbol(
            instance_key(
                a_mn=False, b_mn=True, out_f32=True, epi="reg", quad_store=True
            )
        )
        == "dense_proj_gemm_kn_n256_f32"
    )


def test_default_stages_by_tile_shape():
    # 32 KiB stages at BLOCK_N = 256, 24 KiB stages at BLOCK_N = 128 (deeper pipeline for the
    # streaming-bound small-N rows), 48 KiB stages for tall 256-row tiles; one fewer per staging slot
    assert [default_stages(s, 128, 256) for s in (0, 1, 2)] == [7, 6, 5]
    assert [default_stages(s, 128, 128) for s in (0, 1, 2)] == [9, 8, 6]
    assert [default_stages(s, 256, 256) for s in (0, 1, 2)] == [4, 4, 3]
    assert [default_stages(s, 256, 128) for s in (0, 1, 2)] == [4, 4, 3]
    assert default_stages(0) == 7 and default_stages(1) == 6
    assert default_stages(0, 128, 160) == 7 and default_stages(1, 128, 224) == 6
    # 64-row Layout-B tiles (round 9): the deepest pipeline that fits the 227 KiB opt-in beside the staging
    # (32 KiB per slot), at most 12: 24 KiB stages at BLOCK_N = 256, 16 KiB at 128, 20 KiB at K-major 192
    # (MN-major B streams two whole 64-column panels there: 24 KiB)
    assert [default_stages(s, 64, 256) for s in (0, 1, 2)] == [9, 8, 6]
    assert [default_stages(s, 64, 128) for s in (0, 1, 2)] == [12, 12, 10]
    assert [default_stages(s, 64, 192) for s in (0, 1, 2)] == [11, 9, 8]
    assert [default_stages(s, 64, 192, b_mn=True) for s in (0, 1, 2)] == [9, 8, 6]
    assert [default_stages(s, 64, 256, b_mn=True) for s in (0, 1, 2)] == [9, 8, 6]
    # the default stage count carries no symbol suffix; anything else does
    assert instance_key(a_mn=False, b_mn=False, block_n=128)[5] == 9
    assert instance_key(a_mn=False, b_mn=False, block_n=128, epi="tma", slots=1)[5] == 8
    assert instance_key(a_mn=False, b_mn=False, cta_rows=256)[5] == 4
    assert instance_key(a_mn=False, b_mn=False, cta_rows=64)[5] == 9
    assert instance_key(a_mn=False, b_mn=False, block_n=128, cta_rows=64)[5] == 12
    assert instance_key(a_mn=False, b_mn=True, block_n=192, cta_rows=64)[5] == 9


def test_sk_sync_rule_yields_to_caller_forced_forms():
    # round 18 (Cake launcher parity): the sm_107a 6144 x 2048 weight-gradient rule (q_a / shared_gate_up) plans the
    # synchronised stream-K tail of the 128-row family (192 pair tiles on 106 pairs -> 86 tails, 20 collectors, s = 216
    # with the rule's margin 26).  The plan is admitted per family and needs the single-pass / serialised tall epilogue,
    # so a caller-forced tall tile or epilogue form on the row gets that form with the plain plan (the Cake e2e
    # registrations of the ``_m256`` / ``_ov`` programs), while a caller-forced ``sk_sync`` keeps raising on the conflict
    v = _views("proj", "q_a", "wgrad", "bf16", 16231)
    kw = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    rule, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert (
        rule.sk_sync and rule.cta_rows == 128 and "_sks" in _nopro(rule.template)
    )  # round 25: the row's rule adds the slab path (`_sks_sb2`)
    assert (rule.num_full, rule.tail_tiles, rule.sk_units, rule.iters_per_unit) == (
        106,
        86,
        106,
        216,
    )
    assert rule.sk_iters == 86 and rule.sk_collectors == 20
    tall, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], cta_rows=256, **kw)
    assert (
        not tall.sk_sync
        and tall.cta_rows == 256
        and "_sks" not in _nopro(tall.template)
    )
    assert tall.sk_iters == tall.tail_tiles and tall.sk_collectors == 0
    ov, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], cta_rows=256, ovl=True, **kw
    )
    assert not ov.sk_sync and ov.ovl and "_ov" in _nopro(ov.template)
    with pytest.raises(ValueError, match="sk_sync needs the single-pass"):
        plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cta_rows=256, ovl=True, sk_sync=True, **kw
        )


def test_round19_sk_slab_knob():
    # field 27 of the 30-field key (round 19, Cake W2): the slab path of the synchronised stream-K fixup (symbol
    # ``_sb<n>``: 1 = early arrival, 2 = + the bulk slab read through the dead mainloop stages, 3 = + the fp32 output as
    # the slab).  It exists only on the ``_sks`` programs (forced to 0 without ``sk_sync``, so every other instance keeps
    # one key); 3 needs an fp32 output through the TMA-store or the transposed register epilogue; 2 needs the eight warp
    # slabs (EPI_WARPS x epi_cols x 128 B) to fit the mainloop stage ring
    key = instance_key(a_mn=True, b_mn=True, cta_rows=256, sk_sync=True, sk_slab=2)
    assert len(key) == 37 and key[25] is True and key[26] == 2 and key[27] is False
    assert instance_symbol(key) == "dense_proj_gemm_nn_n256_m256_sks_sb2"
    assert instance_key(a_mn=True, b_mn=True, cta_rows=256, sk_slab=2)[26] == 0
    assert instance_key(a_mn=True, b_mn=True, cta_rows=256, sk_slab=2) == instance_key(
        a_mn=True, b_mn=True, cta_rows=256
    )
    f32 = instance_key(
        a_mn=True,
        b_mn=True,
        cta_rows=256,
        out_f32=True,
        slots=1,
        sk_sync=True,
        sk_slab=3,
    )
    assert f32[7] == "tma" and f32[26] == 3
    assert instance_symbol(f32) == "dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3"
    # the transposed register path of an fp32 output carries the output-as-slab form too (red.global.add.f32)
    assert (
        instance_key(
            a_mn=True,
            b_mn=True,
            out_f32=True,
            out_t=True,
            block_n=192,
            b_swz=64,
            sk_sync=True,
            sk_slab=3,
        )[26]
        == 3
    )
    assert (
        instance_symbol(
            instance_key(
                a_mn=True,
                b_mn=True,
                out_t=True,
                block_n=192,
                b_swz=64,
                sk_sync=True,
                sk_slab=1,
            )
        )
        == "dense_proj_gemm_nn_n192_bz64_t_sks_sb1"
    )
    with pytest.raises(ValueError, match="sk_slab must be one of"):
        instance_key(a_mn=True, b_mn=True, sk_sync=True, sk_slab=4)
    with pytest.raises(ValueError, match="sk_slab=3"):
        instance_key(
            a_mn=True, b_mn=True, cta_rows=256, sk_sync=True, sk_slab=3
        )  # bf16 output
    with pytest.raises(ValueError, match="sk_slab=3"):
        # the row-major fp32 register epilogue has no reduce-add store path
        instance_key(
            a_mn=True, b_mn=True, out_f32=True, epi="reg", sk_sync=True, sk_slab=3
        )
    with pytest.raises(ValueError, match="too small"):
        # 8 x 128 x 128 B of warp slabs do not fit two 32 KiB stages
        instance_key(a_mn=True, b_mn=True, sk_sync=True, sk_slab=2, stages=2)
    # without ``sk_sync`` the knob is inert, so the sk_slab=3 / stage-ring conflicts above do not raise either
    assert instance_key(a_mn=True, b_mn=True, cta_rows=256, sk_slab=3)[26] == 0
    assert instance_key(a_mn=True, b_mn=True, sk_slab=2, stages=2)[26] == 0


def test_round19_cta1_knob():
    # field 28 of the 30-field key (LAST until round 21's ``cgrp``; round 19, Cake W3): the single-CTA form of the 128-row single-pass family
    # (symbol ``_c1``, cluster dims 1, cta_group::1 MMA into private TMEM, CLC per CTA).  Its B stage streams the whole
    # BLOCK_N columns per CTA (32 KiB stages at BLOCK_N 128, 48 KiB at 256), so the default stage count is the deepest
    # single-CTA fit and ``_s<n>`` is relative to it; it excludes the tall / 64-row families, the pair-level tail
    # policies / probes and the batched raster knob
    key = instance_key(a_mn=False, b_mn=False, block_n=128, cta1=True)
    assert len(key) == 37 and key[27] is True and key[26] == 0 and key[5] == 7
    assert instance_symbol(key) == "dense_proj_gemm_kk_n128_c1"
    assert b_stage_bytes(False, 128, 128, 1) == 2 * b_stage_bytes(False, 128) == 16384
    assert b_stage_bytes(True, 256, 128, 1) == 2 * b_stage_bytes(True, 256) == 32768
    assert [default_stages(s, 128, 128, False, 128, 1) for s in (0, 1, 2)] == [7, 6, 5]
    assert [default_stages(s, 128, 256, False, 128, 1) for s in (0, 1, 2)] == [4, 4, 3]
    assert box_rows_of(False, False, 128, None, 128, 1) == (128, 128)
    assert box_rows_of(False, False, 128, None, 128) == (128, 64)
    f32 = instance_key(
        a_mn=False, b_mn=True, out_f32=True, block_n=128, slots=2, stages=4, cta1=True
    )
    assert f32[5] == 4 and f32[7] == "tma" and f32[8] == 2 and f32[27] is True
    assert instance_symbol(f32) == "dense_proj_gemm_kn_n128_f32_tma2_s4_c1"
    assert (
        instance_symbol(
            instance_key(
                a_mn=False, b_mn=True, out_f32=True, block_n=128, slots=2, cta1=True
            )
        )
        == "dense_proj_gemm_kn_n128_f32_tma2_c1"
    )  # five stages = the single-CTA default (six on the pair form)
    assert (
        instance_key(a_mn=False, b_mn=True, out_f32=True, block_n=128, slots=2)[5] == 6
    )
    for bad in (
        dict(cta_rows=256),
        dict(cta_rows=64),
        dict(sk_sync=True),
        dict(htail=True),
        dict(sk_exact=True),
        dict(batch_group=8),
    ):
        with pytest.raises(ValueError, match="cta1 needs cta_rows=128"):
            instance_key(a_mn=False, b_mn=True, cta1=True, **bad)
    with pytest.raises(ValueError, match="cta1 needs cta_rows=128"):
        instance_key(a_mn=False, b_mn=False, cta_rows=256, ovl=True, cta1=True)
    with pytest.raises(ValueError, match="cta1 needs cta_rows=128"):
        instance_key(a_mn=False, b_mn=False, cta_rows=256, park=True, cta1=True)
    # a 48 KiB single-CTA stage at BLOCK_N 256: eight of them exceed every SMEM limit
    with pytest.raises(ValueError, match="SMEM"):
        instance_key(a_mn=False, b_mn=False, stages=8, cta1=True)


@pytest.mark.parametrize("T", [16231, 16172])
def test_round19_cta1_plans_one_cta_per_tile(T):
    # the single-CTA form through the planner (caller-forced here; the sm_100a indexer forward rules select it): the
    # tile count is not rounded to pairs, the work items are CTA tiles, the concurrent units are the SMs (the hint
    # working-set gate takes them as the Cake launcher's ``_sm_units``), the grid is one CTA per tile with no tail
    # policy, and the raster group may be any count >= 1  [Cake launcher]
    v = _views("proj", "indexer_hw", "fwd", "bf16", T)
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], cta1=True, **kw)
    assert plan.cta1 and _nopro(plan.template) == "dense_proj_gemm_kk_n128_hen_c1"
    assert (plan.m_tiles, plan.n_tiles, plan.pair_tiles, plan.sm_pairs) == (
        127,
        1,
        127,
        148,
    )
    assert (plan.num_full, plan.tail_tiles, plan.sk_units) == (127, 0, 0)
    assert plan.grid == (plan.m_tiles * plan.n_tiles, 1, 1) == (127, 1, 1)
    assert plan.stages == 7 and plan.ws_f32_elems == 0 and plan.sk_slab == 0
    assert plan.wave_working_set_bytes == wave_working_set(127, 1, 6144, 16, 148)
    pair, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], cta1=False, **kw)
    assert not pair.cta1 and _nopro(pair.template) == "dense_proj_gemm_kk_n128_hen"
    assert (pair.m_tiles, pair.pair_tiles, pair.sm_pairs, pair.stages) == (
        128,
        64,
        74,
        9,
    )
    assert pair.grid == (128, 1, 1)
    # a raster group of one is admitted under cta1 only
    one, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], cta1=True, group_m=1, **kw
    )
    assert one.group_m == 1 and _nopro(one.template) == _nopro(plan.template)
    with pytest.raises(ValueError, match="group_m must be an even number"):
        plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cta1=False, group_m=1, **kw
        )
    # the form carries no tail policy: a caller combining it with one asked for two incompatible forms
    for bad in (
        dict(sk=True),
        dict(sk_sync=True),
        dict(htail=True),
        dict(sk_exact=3),
        dict(sk_parts=2),
        dict(sk_max_units=48),
    ):
        with pytest.raises(ValueError, match="cta1 excludes"):
            plan_dense_projection_gemm(v["A"], v["B"], v["out"], cta1=True, **bad, **kw)
    # a small T: one wave's panels fit the L2, so the form keeps the hint-free program
    small = _views("proj", "indexer_hw", "fwd", "bf16", 2049)
    short, *_ = plan_dense_projection_gemm(
        small["A"], small["B"], small["out"], cta1=True, **kw
    )
    assert _nopro(short.template) == "dense_proj_gemm_kk_n128_c1" and short.grid == (
        17,
        1,
        1,
    )


@pytest.mark.parametrize("T", [16231, 16172])
def test_round19_rules_plan_like_the_cake_launcher(T):
    # the round-19 sm_100a rules (Cake 97c4ca33270): the indexer_hw / indexer_k forwards and the fp32 indexer_hw
    # input gradient take the single-CTA form (Cake W3, lever D), the indexer_q weight gradients the synchronised
    # stream-K tail with the slab path (Cake W2: main-unit margin 62, the bulk slab read for bf16, the fp32 output as
    # the slab for fp32); sm_107a keeps its round-18 rules
    for key in (
        ("sm_100a", False, False, False, False, False, 32, 6144, None),
        ("sm_100a", False, False, False, False, False, 128, 6144, None),
    ):
        assert ROW_RULES[key] == {"cta1": True}
    assert ROW_RULES[("sm_100a", False, True, True, False, False, 6144, 32, None)] == {
        "block_n": 128,
        "slots": 2,
        "stages": 4,
        "cta1": True,
    }
    assert ROW_RULES[
        ("sm_100a", True, True, False, False, False, 2048, None, 4096)
    ] == {
        "cta_rows": 256,
        "sk_sync": True,
        "sk_sync_m": 62,
        "sk_slab": 2,
    }
    assert ROW_RULES[("sm_100a", True, True, True, False, False, 2048, None, 4096)] == {
        "cta_rows": 256,
        "sk_sync": True,
        "sk_sync_m": 62,
        "sk_slab": 3,
    }
    # the table size is asserted by the newest round's test (round 20: 98 = 49 + 49); round 19 shipped 97 = 48 + 49
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    # the 32- / 128-column indexer forwards: one 128 x 128 tile per CTA, 127 tiles on 148 SMs, whole tiles, the
    # streaming A operand evict_first (one column tile, the wave's panels exceed the L2)
    for row in ("indexer_hw", "indexer_k"):
        v = _views("proj", row, "fwd", "bf16", T)
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
        assert (
            plan.cta1 and _nopro(plan.template) == "dense_proj_gemm_kk_n128_hen_c1"
        ), row
        assert (plan.m_tiles, plan.n_tiles, plan.sm_pairs) == (127, 1, 148)
        assert plan.grid == (plan.m_tiles * plan.n_tiles, 1, 1) == (127, 1, 1)
        assert (plan.num_full, plan.sk_units, plan.sk_slab, plan.stages) == (
            127,
            0,
            0,
            7,
        )
    # the fp32 indexer_hw input gradient: the two-slot BLOCK_N 128 TMA-store staging with four 32 KiB single-CTA
    # stages (``_s4``: the single-CTA default is five), 127 x 48 tiles; a caller forcing the pair form drops the
    # rule's stage count (the pair default of that family is six)
    v = _views("proj", "indexer_hw", "dgrad", "f32", T)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert (
        plan.cta1 and _nopro(plan.template) == "dense_proj_gemm_kn_n128_f32_tma2_s4_c1"
    )
    assert (plan.m_tiles, plan.n_tiles, plan.stages, plan.sm_pairs) == (127, 48, 4, 148)
    assert plan.grid == (127 * 48, 1, 1) and plan.sk_units == 0
    pair, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], cta1=False, **kw)
    assert not pair.cta1 and _nopro(pair.template) == "dense_proj_gemm_kn_n128_f32_tma2"
    assert (pair.m_tiles, pair.stages, pair.sm_pairs) == (128, 6, 74)
    # the indexer_q weight gradients (G.T @ X, both MN-major, K = T): 64 tall tiles on 74 pairs -> 64 main units of
    # s = ceil((7 x kb + 62) / 8) = 230 K steps (kb = 253 / 254 K blocks at T 16172 / 16231) + 10 collectors, one slab
    # per tail tile; bf16 reads the slab back
    # through the dead mainloop stages (sb2), fp32 reduce-adds into the output through the TMA-store path (sb3)
    v = _views("proj", "indexer_q", "wgrad", "bf16", T)
    assert not v["transposed"]
    bf16, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert bf16.sk_sync and bf16.sk_slab == 2
    assert _nopro(bf16.template) == "dense_proj_gemm_nn_n256_m256_sks_sb2"
    assert (bf16.pair_tiles, bf16.k_blocks, bf16.sm_pairs) == (64, -(-T // BLOCK_K), 74)
    assert (
        bf16.num_full,
        bf16.tail_tiles,
        bf16.sk_units,
        bf16.iters_per_unit,
        bf16.sk_collectors,
    ) == (0, 64, 74, 230, 10)
    assert bf16.grid == (148, 1, 1) and bf16.sk_iters == 64
    assert bf16.ws_f32_elems == 64 * 2 * 256 * 256
    vf = _views("proj", "indexer_q", "wgrad", "f32", T)
    f32, *_ = plan_dense_projection_gemm(vf["A"], vf["B"], vf["out"], **kw)
    assert f32.sk_sync and f32.sk_slab == 3 and f32.epi == "tma"
    assert _nopro(f32.template) == "dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3"
    assert (f32.num_full, f32.tail_tiles, f32.sk_units, f32.iters_per_unit) == (
        0,
        64,
        74,
        230,
    )
    # the launcher's yield rules for the rule's slab path: the row-major fp32 register epilogue has no reduce-add
    # store, so a caller forcing it keeps the synchronised plan with the bulk slab read; a caller-forced two-stage
    # pipeline cannot stage the eight 16 KiB warp slabs, so the row keeps the plain synchronised program; a
    # caller-forced ``sk_slab`` keeps raising on such a conflict, and a caller-forced tile family drops the rule's
    # synchronised plan with its slab path
    reg, *_ = plan_dense_projection_gemm(vf["A"], vf["B"], vf["out"], epi="reg", **kw)
    assert reg.sk_sync and reg.sk_slab == 2
    assert _nopro(reg.template) == "dense_proj_gemm_nn_n256_m256_f32_sks_sb2"
    narrow, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], stages=2, **kw)
    assert narrow.sk_sync and narrow.sk_slab == 0
    assert _nopro(narrow.template) == "dense_proj_gemm_nn_n256_m256_s2_sks"
    with pytest.raises(ValueError, match="sk_slab=3"):
        plan_dense_projection_gemm(v["A"], v["B"], v["out"], sk_slab=3, **kw)
    with pytest.raises(ValueError, match="too small"):
        plan_dense_projection_gemm(v["A"], v["B"], v["out"], stages=2, sk_slab=2, **kw)
    std, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], cta_rows=128, **kw)
    assert not std.sk_sync and std.sk_slab == 0 and "_sks" not in _nopro(std.template)
    off, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], sk=False, **kw)
    assert not off.sk_sync and _nopro(off.template) == "dense_proj_gemm_nn_n256_m256"
    # at the test tables' T the synchronised plan is not admitted (a part under SK_MIN_ITERS): the plain tall plan
    for short_T in (2049, 1001):
        vs = _views("proj", "indexer_q", "wgrad", "bf16", short_T)
        short, *_ = plan_dense_projection_gemm(vs["A"], vs["B"], vs["out"], **kw)
        assert (
            not short.sk_sync
            and short.sk_slab == 0
            and "_sb" not in _nopro(short.template)
        )


@pytest.mark.parametrize("T", [16172, 16231])
def test_round20_rules_plan_like_the_cake_launcher(T):
    # the round-20 sm_100a rule (Cake e7494f02f70): the shared_down input gradient bf16 (G @ W, 2048 x 6144 x T)
    # takes the raster group 8 like the fp32 shared_down input gradient and the q_a / shared_gate_up input gradients
    # (kn, 6144 x 2048) - a launch parameter of the unchanged ``kn_n256`` program (16 is the row's former default);
    # sm_107a keeps its round-19 rules
    key = ("sm_100a", False, True, False, False, False, 2048, 6144, None)
    assert ROW_RULES[key] == {"group_m": 8, "pro": 0}
    assert ROW_RULES[
        ("sm_100a", False, True, False, False, False, 6144, 2048, None)
    ] == {"group_m": 8}
    # the table size is asserted by the newest round's test (round 21: 98 = 49 + 49); round 20 shipped 98 = 49 + 49
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    v = _views("proj", "shared_down", "dgrad", "bf16", T)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert _nopro(plan.template) == "dense_proj_gemm_kn_n256" and not plan.cta1
    assert plan.group_m == 8
    assert (
        default_group_m(False, True, plan.m_tiles, plan.pair_tiles, plan.sm_pairs) == 16
    )
    # a caller-forced raster group overrides the rule on the same program
    forced, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], group_m=16, **kw)
    assert _nopro(forced.template) == _nopro(plan.template) and forced.group_m == 16
    # the shared_down forward (X @ W^T, 6144 x 2048) has no rule and keeps the 16-row default on ``kk_n256``
    f = _views("proj", "shared_down", "fwd", "bf16", T)
    fwd, *_ = plan_dense_projection_gemm(f["A"], f["B"], f["out"], **kw)
    assert fwd.group_m == 16 and _nopro(fwd.template) == "dense_proj_gemm_kk_n256"


@pytest.mark.parametrize("T", [16172, 16231])
def test_round21_rules_plan_like_the_cake_launcher(T):
    # the round-21 sm_100a rule (Cake e3e42a19d84; W2, lever A): the kv_a input gradient bf16 (G @ W, 6144 x 576 x T) leaves the
    # six-stage 128-row quad register epilogue (``kn_n256_q_s6``: epi reg, stages 6, quad_store) for the one-slot TMA-store epilogue
    # with the two-chunk cross-tile pipelined TMEM drain (``kn_n256_tma1_pd2``); the raster group 8 is kept.  A zero-code instance
    # change: no instance-key field is added; sm_107a keeps its round-20 rules (the kv_a row there stays on the quad stores + pd 1)
    key = ("sm_100a", False, True, False, False, False, 6144, 576, None)
    assert ROW_RULES[key] == {
        "group_m": 8,
        "slots": 1,
        "epi": "tma",
        "pd": 2,
        "pro": 15,
    }
    assert ROW_RULES[
        ("sm_107a", False, True, False, False, False, 6144, 576, None)
    ] == {
        "group_m": 8,
        "epi": "reg",
        "quad_store": True,
        "pd": 1,
    }
    assert ROW_RULES[("sm_100a", False, True, True, False, False, 6144, 576, None)] == {
        "group_m": 8,
        "promo": "l2_256b",
    }
    # the table size is asserted by the newest round's test (round 26: 103 = 49 + 54); round 25 shipped 100 = 49 + 51; round 24 shipped 100 = 49 + 51; round 21 shipped 98 = 49 + 49
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    v = _views("proj", "kv_a", "dgrad", "bf16", T)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert _nopro(plan.template) == "dense_proj_gemm_kn_n256_tma1_pd2" and not plan.cta1
    assert (
        plan.epi,
        plan.slots,
        plan.pd,
        plan.group_m,
        plan.cta_rows,
        plan.block_n,
    ) == ("tma", 1, 2, 8, 128, 256)
    # a caller forcing the former form still plans the round-20 program of the row; the rule's ``pd`` carries over
    # into a forced form unless the caller sets it (``pd=0``), exactly as in the Cake launcher
    old, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], epi="reg", quad_store=True, stages=6, pd=0, **kw
    )
    assert (
        _nopro(old.template) == "dense_proj_gemm_kn_n256_q_s6"
        and old.pd == 0
        and old.group_m == 8
    )
    # the sibling rows keep their programs: the sm_107a kv_a bf16 input gradient (quad stores + in-tile drain), the fp32 kv_a input
    # gradient (one-slot TMA store, no drain) and the indexer_hw bf16 input gradient (two-slot TMA store + the cross-tile drain)
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    p107, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw107)
    assert (
        _nopro(p107.template) == "dense_proj_gemm_kn_n256_q_pd1" and p107.group_m == 8
    )
    f = _views("proj", "kv_a", "dgrad", "f32", T)
    pf, *_ = plan_dense_projection_gemm(f["A"], f["B"], f["out"], **kw)
    assert _nopro(pf.template) == "dense_proj_gemm_kn_n256_f32_tma1" and pf.pd == 0
    h = _views("proj", "indexer_hw", "dgrad", "bf16", T)
    ph, *_ = plan_dense_projection_gemm(h["A"], h["B"], h["out"], **kw)
    assert _nopro(ph.template) == "dense_proj_gemm_kn_n256_tma2_pd2"
    # round 21 (W3, lever B): the sm_100a shared_gate_up forward bf16 (X @ W^T, 2048 x 6144 x T) keeps its raster group 8
    # and takes the column-grouped raster form (``kk_n256_cg``, instance-key field 29: the group counts column tiles);
    # the q_a forward shares the row key (6144 -> 2048) and therefore the rule; sm_107a had no rule on the key in round 21
    # and kept the row-grouped ``kk_n256`` - since round 24 it carries the same rule (test_round24_rules_plan_like_the_cake_launcher
    # row (d)), so the round-21 form is planned here with the caller-forced ``cgrp=False``
    assert ROW_RULES[
        ("sm_100a", False, False, False, False, False, 2048, 6144, None)
    ] == {
        "group_m": 8,
        "cgrp": True,
    }
    g = _views("proj", "shared_gate_up", "fwd", "bf16", T)
    pg, *_ = plan_dense_projection_gemm(g["A"], g["B"], g["out"], **kw)
    assert (
        pg.cgrp
        and _nopro(pg.template) == "dense_proj_gemm_kk_n256_cg"
        and pg.group_m == 8
    )
    pg107, *_ = plan_dense_projection_gemm(
        g["A"], g["B"], g["out"], cgrp=False, **kw107
    )
    assert not pg107.cgrp and _nopro(pg107.template) == "dense_proj_gemm_kk_n256"
    q = _views("proj", "q_a", "fwd", "bf16", T)
    pq, *_ = plan_dense_projection_gemm(q["A"], q["B"], q["out"], **kw)
    assert (
        pq.cgrp
        and _nopro(pq.template) == "dense_proj_gemm_kk_n256_cg"
        and pq.group_m == 8
    )


@pytest.mark.parametrize("T", [16172, 16231])
def test_round24_rules_plan_like_the_cake_launcher(T):
    # the round-24 rule changes (Cake 02c04ba1831): six sm_107a rules - the indexer_hw input gradient bf16 takes the single-CTA
    # form with the pd 3 drain (cta1, slots 2, group_m 2), the indexer_hw input gradient fp32 takes the single-CTA form with one
    # staging slot at BLOCK_N 128 (cta1, group_m 2), the NEW shared_down weight gradient bf16 key takes the synchronised stream-K
    # tail with the bulk-slab fixup (sk_sync, sk_sync_m 38, sk_slab 2), the NEW q_a / shared_gate_up forward bf16 key takes the
    # column-grouped raster (group_m 8, cgrp), the dense_down forward bf16 adds the row-half release and the column-grouped raster
    # (ovr, cgrp, group_m 12) to its cta_rows 256 / sk_parts 3 rule, and the o_proj forward bf16 replaces ovl by ovr and adds the
    # column-grouped raster (cgrp, group_m 8) to its cta_rows 256 / htail rule; no sm_100a rule changes
    # the table size is asserted by the newest round's test (round 26: 103 = 49 + 54); round 25 shipped 100 = 49 + 51; round 24 shipped 100 = 49 + 51
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    # ---- (a) indexer_hw input gradient bf16 (G @ W, 6144 x 32 x T): the single-CTA form with the pd 3 drain replaces the
    #      pair form's pd 2 drain (lock 6f98f5dd: dense_proj_gemm_kn_n256_tma2_pd2); raster group 2 and the two staging slots are kept ----
    hkey = ("sm_107a", False, True, False, False, False, 6144, 32, None)
    assert ROW_RULES[hkey] == {
        "cta_rows": 128,
        "slots": 2,
        "pd": 3,
        "group_m": 2,
        "cta1": True,
    }
    h = _views("proj", "indexer_hw", "dgrad", "bf16", T)
    ph, *_ = plan_dense_projection_gemm(h["A"], h["B"], h["out"], **kw107)
    # TEMPLATES_REACHED_NEW (the kernel's rule comment names kn_n256_tma2_pd3_c1: K = 32 <= K_TMA_BF16 keeps the row's two-slot
    # TMA-store epilogue; the K-less by-kwargs identity render kn_n256_pd3_c1 is the register-epilogue sibling, not this plan)
    assert _nopro(ph.template) == "dense_proj_gemm_kn_n256_tma2_pd3_c1" and ph.cta1
    assert (ph.epi, ph.slots, ph.pd, ph.group_m, ph.cta_rows, ph.block_n) == (
        "tma",
        2,
        3,
        2,
        128,
        256,
    )
    # a caller forcing the former form still plans the round-21 program of the row (pd passed explicitly)
    old_h, *_ = plan_dense_projection_gemm(
        h["A"], h["B"], h["out"], cta1=False, pd=2, **kw107
    )
    assert (
        _nopro(old_h.template) == "dense_proj_gemm_kn_n256_tma2_pd2" and not old_h.cta1
    )
    # the control row indexer_k (K 128 is another key) and the sm_100a sibling keep their round-21 programs
    k_ = _views("proj", "indexer_k", "dgrad", "bf16", T)
    pk, *_ = plan_dense_projection_gemm(k_["A"], k_["B"], k_["out"], **kw107)
    assert _nopro(pk.template) == "dense_proj_gemm_kn_n256_tma2_pd2" and not pk.cta1
    ph100, *_ = plan_dense_projection_gemm(h["A"], h["B"], h["out"], **kw)
    assert (
        _nopro(ph100.template) == "dense_proj_gemm_kn_n256_tma2_pd2" and not ph100.cta1
    )
    # ---- (b) indexer_hw input gradient fp32 (6144 x 32 x T): the single-CTA form with ONE staging slot replaces the pair
    #      form's two-slot TMA store (lock 6f98f5dd: dense_proj_gemm_kn_n128_f32_tma2); BLOCK_N 128 and raster group 2 are kept ----
    fkey = ("sm_107a", False, True, True, False, False, 6144, 32, None)
    assert ROW_RULES[fkey] == {"block_n": 128, "slots": 1, "group_m": 2, "cta1": True}
    f = _views("proj", "indexer_hw", "dgrad", "f32", T)
    pf, *_ = plan_dense_projection_gemm(f["A"], f["B"], f["out"], **kw107)
    assert (
        _nopro(pf.template) == "dense_proj_gemm_kn_n128_f32_tma1_c1" and pf.cta1
    )  # TEMPLATES_REACHED_NEW (the kernel's rule comment names kn_n128_f32_tma1_c1)
    assert (pf.epi, pf.slots, pf.group_m, pf.cta_rows, pf.block_n) == (
        "tma",
        1,
        2,
        128,
        128,
    )
    old_f, *_ = plan_dense_projection_gemm(
        f["A"], f["B"], f["out"], cta1=False, slots=2, **kw107
    )
    assert (
        _nopro(old_f.template) == "dense_proj_gemm_kn_n128_f32_tma2" and not old_f.cta1
    )
    # ---- (c) NEW key: shared_down weight gradient bf16 (nn, 2048 x 6144 over T): the synchronised stream-K tail with the
    #      bulk-slab fixup (main-unit margin 38 K steps) replaces the plain pair program (lock 6f98f5dd: dense_proj_gemm_nn_n256) ----
    wkey = ("sm_107a", True, True, False, False, False, 2048, None, 6144)
    assert ROW_RULES[wkey] == {"sk_sync": True, "sk_sync_m": 38, "sk_slab": 2}
    w = _views("proj", "shared_down", "wgrad", "bf16", T)
    pw, *_ = plan_dense_projection_gemm(w["A"], w["B"], w["out"], **kw107)
    assert (
        _nopro(pw.template) == "dense_proj_gemm_nn_n256_sks_sb2"
        and pw.sk_sync
        and pw.sk_slab == 2
    )  # TEMPLATES_REACHED_NEW (by-kwargs identity: nn_n256_sks_sb2)
    old_w, *_ = plan_dense_projection_gemm(
        w["A"], w["B"], w["out"], sk_sync=False, **kw107
    )
    assert (
        _nopro(old_w.template) == "dense_proj_gemm_nn_n256"
        and not old_w.sk_sync
        and old_w.sk_slab == 0
    )
    # round 25 re-rules the sm_100a shared_down weight gradient onto the synchronised tail as well ({"group_m": 4, "sk_sync": True,
    # "sk_sync_m": 26, "sk_slab": 2}) - asserted by test_round25_rules_plan_like_the_cake_launcher; in round 24 it kept {"group_m": 4, "sk_exact": 3}
    # ---- (d) NEW key: q_a / shared_gate_up forward bf16 (X @ W^T, 2048 x 6144 x T, shared key; the sm_107a twin of the round-21
    #      sm_100a lever B rule): the column-grouped raster with 8 column tiles per group; both sibling rows ----
    gkey = ("sm_107a", False, False, False, False, False, 2048, 6144, None)
    assert ROW_RULES[gkey] == {"group_m": 8, "cgrp": True, "pro": 15}
    assert ROW_RULES[
        ("sm_100a", False, False, False, False, False, 2048, 6144, None)
    ] == {"group_m": 8, "cgrp": True}  # unchanged since round 21
    for row in ("q_a", "shared_gate_up"):
        g = _views("proj", row, "fwd", "bf16", T)
        pg, *_ = plan_dense_projection_gemm(g["A"], g["B"], g["out"], **kw107)
        # TEMPLATES_REACHED_NEW / the figen tables (dense_proj_gemm_kk_n256_cg = the round-21 sm_100a program; the cgrp hint gate stays
        # ('none', 'none') at 106 pairs - wave_working_set_cgrp(128, 8, 6144, 8, 106) = 22 x 256 x 6144 x 2 B = 69 206 016 B <= L2_BYTES)
        assert (
            pg.cgrp
            and _nopro(pg.template) == "dense_proj_gemm_kk_n256_cg"
            and pg.group_m == 8
        ), row
        old_g, *_ = plan_dense_projection_gemm(
            g["A"], g["B"], g["out"], cgrp=False, **kw107
        )
        assert not old_g.cgrp and _nopro(old_g.template) == "dense_proj_gemm_kk_n256", (
            row
        )  # lock 6f98f5dd: the row-grouped kk_n256
    # ---- (e) dense_down forward bf16 (X @ W^T, 6144 x 12288 x T): the tall family released by row half (ovr, instance-key field 30) with the
    #      column-grouped raster of 12 column tiles per group, on top of the row's sk_parts 3 tail (lock 6f98f5dd: dense_proj_gemm_kk_n256_m256) ----
    dkey = ("sm_107a", False, False, False, False, False, 6144, 12288, None)
    assert ROW_RULES[dkey] == {
        "cta_rows": 256,
        "sk_parts": 3,
        "ovr": True,
        "cgrp": True,
        "group_m": 12,
    }
    d = _views("proj", "dense_down", "fwd", "bf16", T)
    pd_, *_ = plan_dense_projection_gemm(d["A"], d["B"], d["out"], **kw107)
    # TEMPLATES_REACHED_NEW: the `_cg_or` tall program WITHOUT `_ht` (sk_parts is the row's tail policy; by-kwargs identity: kk_n256_m256_cg_or)
    assert (
        _nopro(pd_.template) == "dense_proj_gemm_kk_n256_m256_cg_or"
        and pd_.ovr
        and pd_.cgrp
        and not pd_.ovl
        and not pd_.htail
    )
    assert (pd_.cta_rows, pd_.block_n, pd_.group_m) == (256, 256, 12)
    old_d, *_ = plan_dense_projection_gemm(
        d["A"], d["B"], d["out"], ovr=False, cgrp=False, **kw107
    )
    assert (
        _nopro(old_d.template) == "dense_proj_gemm_kk_n256_m256"
        and not old_d.ovr
        and not old_d.cgrp
    )  # the round-21 program (pair form, row raster)
    # ---- (f) o_proj forward bf16 (6144 x 16384 x T): ovr REPLACES ovl on the row (the rule drops ovl) and the column-grouped raster takes 8 column
    #      tiles per group; the half-height tail wave is kept (lock 6f98f5dd: dense_proj_gemm_kk_n256_m256_ov_ht) ----
    okey = ("sm_107a", False, False, False, False, False, 6144, 16384, None)
    assert ROW_RULES[okey] == {
        "cta_rows": 256,
        "htail": True,
        "ovr": True,
        "cgrp": True,
        "group_m": 8,
    }
    o = _views("proj", "o_proj", "fwd", "bf16", T)
    po, *_ = plan_dense_projection_gemm(o["A"], o["B"], o["out"], **kw107)
    # TEMPLATES_REACHED_NEW (the kernel registers dense_projection_gemm_kk_m256_ht_or_cg -> symbol ..._m256_ht_cg_or)
    assert (
        _nopro(po.template) == "dense_proj_gemm_kk_n256_m256_ht_cg_or"
        and po.ovr
        and po.cgrp
        and po.htail
        and not po.ovl
    )
    assert (po.cta_rows, po.block_n, po.group_m) == (256, 256, 8)
    old_o, *_ = plan_dense_projection_gemm(
        o["A"], o["B"], o["out"], ovr=False, ovl=True, cgrp=False, **kw107
    )
    assert (
        _nopro(old_o.template) == "dense_proj_gemm_kk_n256_m256_ov_ht"
        and old_o.ovl
        and not old_o.ovr
    )
    # the sm_100a dense_down / o_proj forwards carry no ovr rule and keep their round-21 programs
    pd100, *_ = plan_dense_projection_gemm(d["A"], d["B"], d["out"], **kw)
    assert (
        _nopro(pd100.template) == "dense_proj_gemm_kk_n256_m256_ov_ht"
        and not pd100.ovr
        and not pd100.cgrp
    )
    po100, *_ = plan_dense_projection_gemm(o["A"], o["B"], o["out"], **kw)
    assert (
        _nopro(po100.template) == "dense_proj_gemm_kk_n256_m256_ov_ht"
        and not po100.ovr
        and not po100.cgrp
        and po100.group_m == 8
    )


@pytest.mark.parametrize("T", [16172, 16231])
def test_round25_rules_plan_like_the_cake_launcher(T):
    # the round-25 rule changes (Cake f397cd4da99): seven weight-gradient rules move to the synchronised stream-K tail with the two-slab
    # fixup - on sm_107a the fp32 shared_down (margin 38) and q_a / shared_gate_up (margin 26) weight gradients leave the round-2 tall tile
    # for the 128-row tail with the register epilogue (f32_v8 + store_ef), the bf16 q_a / shared_gate_up weight gradient adds the slab path
    # to its round-18 synchronised tail; on sm_100a the bf16 and fp32 weight gradients of shared_down / q_a / shared_gate_up replace the
    # round-16 exact 3-way tail split (sk_exact 3) by the synchronised tail with margin 26 and the slab path, keeping their raster groups and
    # fp32 register epilogue.  No key is added or removed and no instance-key field changes.
    # the table size is asserted by the newest round's test (round 26: 103 = 49 + 54); round 25 shipped 100 = 49 + 51
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    # ---- (a) sm_107a shared_down weight gradient fp32 (nn, 2048 x 6144 over T): the 128-row synchronised stream-K tail (main-unit margin
    #      38 K steps, two-slab fixup) with the fp32 v8 evict-first register epilogue replaces the round-2 tall tile (lock e9d8042b:
    #      dense_proj_gemm_nn_n256_m256_f32_tma1 through the rule {"cta_rows": 256}) ----
    sdf_key = ("sm_107a", True, True, True, False, False, 2048, None, 6144)
    assert ROW_RULES[sdf_key] == {
        "sk_sync": True,
        "sk_sync_m": 38,
        "sk_slab": 2,
        "epi": "reg",
        "f32_v8": True,
        "store_ef": True,
    }
    sdf = _views("proj", "shared_down", "wgrad", "f32", T)
    p_sdf, *_ = plan_dense_projection_gemm(sdf["A"], sdf["B"], sdf["out"], **kw107)
    assert (
        _nopro(p_sdf.template) == "dense_proj_gemm_nn_n256_f32_v8_ef_sks_sb2"
    )  # ROUTE_TEMPLATE_NEW of the round-25 templates step = TEMPLATES_REACHED_NEW (nn_n256_f32_v8_ef_sks_sb2, the round-25 sweep symbol)
    assert (
        p_sdf.sk_sync
        and p_sdf.sk_slab == 2
        and p_sdf.sk_exact == 0
        and p_sdf.epi == "reg"
        and p_sdf.store_ef
    )
    assert (p_sdf.cta_rows, p_sdf.block_n) == (128, 256) and not p_sdf.htail
    # a caller forcing the former tall form plans the lock-e9d8042b program of the row (the forced cta_rows leaves the rule's 128-row family,
    # so the planner drops the rule's epilogue / tail knobs; sk_sync and the TMA-store epilogue are passed explicitly all the same)
    old_sdf, *_ = plan_dense_projection_gemm(
        sdf["A"], sdf["B"], sdf["out"], cta_rows=256, sk_sync=False, epi="tma", **kw107
    )
    assert (
        _nopro(old_sdf.template) == "dense_proj_gemm_nn_n256_m256_f32_tma1"
        and not old_sdf.sk_sync
        and old_sdf.cta_rows == 256
    )
    # ---- (b) sm_107a q_a / shared_gate_up weight gradient fp32 (nn, 6144 x 2048 over T; one shared key): the same lever with the bf16
    #      sibling's margin 26 (lock e9d8042b: dense_proj_gemm_nn_n256_m256_f32_tma1 through {"cta_rows": 256}); both sibling rows ----
    qaf_key = ("sm_107a", True, True, True, False, False, 6144, None, 2048)
    assert ROW_RULES[qaf_key] == {
        "sk_sync": True,
        "sk_sync_m": 26,
        "sk_slab": 2,
        "epi": "reg",
        "f32_v8": True,
        "store_ef": True,
    }
    for row in ("q_a", "shared_gate_up"):
        v = _views("proj", row, "wgrad", "f32", T)
        p, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw107)
        assert _nopro(p.template) == "dense_proj_gemm_nn_n256_f32_v8_ef_sks_sb2", (
            row
        )  # ROUTE_TEMPLATE_NEW of the round-25 templates step = TEMPLATES_REACHED_NEW (nn_n256_f32_v8_ef_sks_sb2)
        assert (
            p.sk_sync
            and p.sk_slab == 2
            and p.sk_exact == 0
            and p.epi == "reg"
            and p.store_ef
            and p.cta_rows == 128
        ), row
        old, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cta_rows=256, sk_sync=False, epi="tma", **kw107
        )
        assert (
            _nopro(old.template) == "dense_proj_gemm_nn_n256_m256_f32_tma1"
            and not old.sk_sync
        ), row
    # ---- (c) sm_107a q_a / shared_gate_up weight gradient bf16 (nn, 6144 x 2048 over T; one shared key): the two-slab fixup joins the
    #      round-18 synchronised tail (lock e9d8042b: dense_proj_gemm_nn_n256_sks through {"sk_sync": True, "sk_sync_m": 26}); the plan
    #      arithmetic of the synchronised tail is unchanged by the slab path (192 pair tiles on 106 pairs: 86 tails, 20 collectors, s = 216,
    #      asserted by test_sk_sync_rule_yields_to_caller_forced_forms) ----
    qab_key = ("sm_107a", True, True, False, False, False, 6144, None, 2048)
    assert ROW_RULES[qab_key] == {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2}
    for row in ("q_a", "shared_gate_up"):
        v = _views("proj", row, "wgrad", "bf16", T)
        p, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw107)
        assert _nopro(p.template) == "dense_proj_gemm_nn_n256_sks_sb2", (
            row
        )  # ROUTE_TEMPLATE_NEW of the round-25 templates step (nn_n256_sks_sb2 = the round-24 sm_107a shared_down wgrad bf16 program, no new template)
        assert p.sk_sync and p.sk_slab == 2 and p.cta_rows == 128, row
        old, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], sk_slab=0, **kw107
        )
        assert (
            _nopro(old.template) == "dense_proj_gemm_nn_n256_sks"
            and old.sk_sync
            and old.sk_slab == 0
        ), row  # lock e9d8042b
    # the sm_107a shared_down weight gradient bf16 keeps its round-20 rule ({"sk_sync": True, "sk_sync_m": 38, "sk_slab": 2}) and program
    assert ROW_RULES[
        ("sm_107a", True, True, False, False, False, 2048, None, 6144)
    ] == {"sk_sync": True, "sk_sync_m": 38, "sk_slab": 2}
    sdb107 = _views("proj", "shared_down", "wgrad", "bf16", T)
    p_ctl, *_ = plan_dense_projection_gemm(
        sdb107["A"], sdb107["B"], sdb107["out"], **kw107
    )
    assert (
        _nopro(p_ctl.template) == "dense_proj_gemm_nn_n256_sks_sb2"
        and p_ctl.sk_sync
        and p_ctl.sk_slab == 2
    )
    # ---- (d) sm_100a shared_down weight gradient bf16 (nn, 2048 x 6144 over T): the synchronised tail (margin 26) with the two-slab fixup
    #      replaces the round-16 exact 3-way split of the 44-tile tail wave; the raster group 4 is kept (lock e9d8042b: dense_proj_gemm_nn_n256_skx
    #      through {"group_m": 4, "sk_exact": 3}) ----
    sdb_key = ("sm_100a", True, True, False, False, False, 2048, None, 6144)
    assert ROW_RULES[sdb_key] == {
        "group_m": 4,
        "sk_sync": True,
        "sk_sync_m": 26,
        "sk_slab": 2,
    }
    p_sdb, *_ = plan_dense_projection_gemm(
        sdb107["A"], sdb107["B"], sdb107["out"], **kw
    )
    assert (
        _nopro(p_sdb.template) == "dense_proj_gemm_nn_n256_sks_sb2"
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted nn_n256_sks_sb2 - the same program as the sm_107a shared_down row; confirm per route)
    assert (
        p_sdb.sk_sync
        and p_sdb.sk_slab == 2
        and p_sdb.sk_exact == 0
        and p_sdb.group_m == 4
        and p_sdb.cta_rows == 128
    )
    old_sdb, *_ = plan_dense_projection_gemm(
        sdb107["A"], sdb107["B"], sdb107["out"], sk_sync=False, sk_exact=3, **kw
    )
    assert (
        _nopro(old_sdb.template) == "dense_proj_gemm_nn_n256_skx"
        and not old_sdb.sk_sync
        and old_sdb.sk_exact == 3
    )  # lock e9d8042b
    # ---- (e) sm_100a q_a / shared_gate_up weight gradient bf16 (nn, 6144 x 2048 over T; one shared key): the same lever (lock e9d8042b:
    #      dense_proj_gemm_nn_n256_skx through {"sk_exact": 3}); both sibling rows ----
    qab100_key = ("sm_100a", True, True, False, False, False, 6144, None, 2048)
    assert ROW_RULES[qab100_key] == {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2}
    for row in ("q_a", "shared_gate_up"):
        v = _views("proj", row, "wgrad", "bf16", T)
        p, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
        assert _nopro(p.template) == "dense_proj_gemm_nn_n256_sks_sb2", row
        assert p.sk_sync and p.sk_slab == 2 and p.sk_exact == 0 and p.cta_rows == 128, (
            row
        )
        old, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], sk_sync=False, sk_exact=3, **kw
        )
        assert (
            _nopro(old.template) == "dense_proj_gemm_nn_n256_skx" and old.sk_exact == 3
        ), row  # lock e9d8042b
    # ---- (f) sm_100a shared_down weight gradient fp32 (nn, 2048 x 6144 over T): the synchronised tail (margin 26, two-slab fixup) replaces
    #      the exact 3-way split under the unchanged fp32 v8 evict-first register epilogue and raster group 8 (lock e9d8042b:
    #      dense_proj_gemm_nn_n256_f32_v8_ef_skx through {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_exact": 3}) ----
    sdf100_key = ("sm_100a", True, True, True, False, False, 2048, None, 6144)
    assert ROW_RULES[sdf100_key] == {
        "group_m": 8,
        "epi": "reg",
        "f32_v8": True,
        "store_ef": True,
        "sk_sync": True,
        "sk_sync_m": 26,
        "sk_slab": 2,
    }
    p_sdf100, *_ = plan_dense_projection_gemm(sdf["A"], sdf["B"], sdf["out"], **kw)
    assert (
        _nopro(p_sdf100.template) == "dense_proj_gemm_nn_n256_f32_v8_ef_sks_sb2"
    )  # ROUTE_TEMPLATE_NEW of the templates step = TEMPLATES_REACHED_NEW (predicted nn_n256_f32_v8_ef_sks_sb2 - the same new program as the sm_107a fp32 rows; confirm per route)
    assert (
        p_sdf100.sk_sync
        and p_sdf100.sk_slab == 2
        and p_sdf100.sk_exact == 0
        and p_sdf100.epi == "reg"
        and p_sdf100.store_ef
        and p_sdf100.group_m == 8
    )
    old_sdf100, *_ = plan_dense_projection_gemm(
        sdf["A"], sdf["B"], sdf["out"], sk_sync=False, sk_exact=3, **kw
    )
    assert (
        _nopro(old_sdf100.template) == "dense_proj_gemm_nn_n256_f32_v8_ef_skx"
        and old_sdf100.sk_exact == 3
        and old_sdf100.epi == "reg"
    )  # lock e9d8042b
    # ---- (g) sm_100a q_a / shared_gate_up weight gradient fp32 (nn, 6144 x 2048 over T; one shared key): the same lever under raster group 32
    #      (lock e9d8042b: dense_proj_gemm_nn_n256_f32_v8_ef_skx through {"group_m": 32, "epi": 'reg', "f32_v8": True, "store_ef": True,
    #      "sk_exact": 3}); both sibling rows ----
    qaf100_key = ("sm_100a", True, True, True, False, False, 6144, None, 2048)
    assert ROW_RULES[qaf100_key] == {
        "group_m": 32,
        "epi": "reg",
        "f32_v8": True,
        "store_ef": True,
        "sk_sync": True,
        "sk_sync_m": 26,
        "sk_slab": 2,
    }
    for row in ("q_a", "shared_gate_up"):
        v = _views("proj", row, "wgrad", "f32", T)
        p, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
        assert _nopro(p.template) == "dense_proj_gemm_nn_n256_f32_v8_ef_sks_sb2", row
        assert (
            p.sk_sync
            and p.sk_slab == 2
            and p.sk_exact == 0
            and p.epi == "reg"
            and p.store_ef
            and p.group_m == 32
        ), row
        old, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], sk_sync=False, sk_exact=3, **kw
        )
        assert (
            _nopro(old.template) == "dense_proj_gemm_nn_n256_f32_v8_ef_skx"
            and old.sk_exact == 3
        ), row  # lock e9d8042b
    # the exact-split programs of the seven keys have no other route in lock e9d8042b, so the templates step is expected to RETIRE them
    # (predicted: nn_n256_skx, nn_n256_f32_v8_ef_skx, nn_n256_sks; nn_n256_m256_f32_tma1 stays - dense_down / dense_gate_up wgrad f32 on
    # sm_100a and o_proj wgrad f32 on sm_107a keep it); the EXPORTED_TEMPLATES table of this file is regenerated by the figen step - not asserted here


@pytest.mark.parametrize("T", [16172, 16231])
def test_round26_rules_plan_like_the_cake_launcher(T):
    # the round-26 rule changes (Cake 2f44e6e19f6): six sm_107a rules, no sm_100a change - the dense_gate_up forward bf16 (new key) takes the
    # tall tile with the half-height tail wave, the row-half release and the column-grouped raster of 12 column tiles per group; the dense_down
    # weight gradient fp32-out (new key) takes the tall tile with the half-height tail and the fp32 v8 evict-first register epilogue; the
    # dense_down weight gradient bf16 leaves its 3-way split of the tail wave (refused by the host at the production token counts, so the shipped
    # program was the plain 128-row form) for the tall tile with the half-height tail; the dense_gate_up input gradient bf16 leaves its 2-way
    # split for the tall tile with a 3-way split, the row-half release and the column-grouped raster (12 per group); the q_b input gradient bf16
    # (new key) takes the column-grouped raster with 8 column tiles per group; the dense_gate_up weight gradient fp32-out leaves the register
    # epilogue with the fp32 v8 stores for the tall tile with the half-height tail through the TMA-store epilogue.  No instance-key field changes.
    # the table size is asserted by the newest round's test (round 27: 103 = 49 + 54); round 26 shipped 103 = 49 + 54
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    # ---- (a) dense_gate_up forward bf16 (X @ W^T, 12288 x 6144 x T; new key): the tall family with the half-height tail wave, released by row
    #      half, with the column-grouped raster of 12 column tiles per group (the o_proj forward's form with the dense_down forward's raster
    #      group); the previous lock planned the row-grouped pair form dense_proj_gemm_kk_n256 (no rule) ----
    gkey = ("sm_107a", False, False, False, False, False, 12288, 6144, None)
    assert ROW_RULES[gkey] == {
        "cta_rows": 256,
        "htail": True,
        "ovr": True,
        "cgrp": True,
        "group_m": 12,
    }
    g = _views("proj", "dense_gate_up", "fwd", "bf16", T)
    pg, *_ = plan_dense_projection_gemm(g["A"], g["B"], g["out"], **kw107)
    assert (
        _nopro(pg.template) == "dense_proj_gemm_kk_n256_m256_ht_cg_or"
        and pg.ovr
        and pg.cgrp
        and pg.htail
        and not pg.ovl
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted kk_n256_m256_ht_cg_or - the sweep symbol)
    assert (pg.cta_rows, pg.block_n, pg.group_m) == (256, 256, 12)
    old_g, *_ = plan_dense_projection_gemm(
        g["A"],
        g["B"],
        g["out"],
        cta_rows=128,
        htail=False,
        ovr=False,
        cgrp=False,
        **kw107,
    )
    assert (
        _nopro(old_g.template) == "dense_proj_gemm_kk_n256"
        and not old_g.ovr
        and not old_g.cgrp
        and not old_g.htail
    )  # the previous lock's program
    # ---- (b) dense_down weight gradient fp32-out (G.T @ X, 12288 x 6144 over T; new key): the tall tile with the half-height tail wave and the
    #      fp32 v8 evict-first register epilogue; the previous lock planned the 128-row TMA-store form dense_proj_gemm_nn_n256_f32_tma1 (no rule) ----
    wfkey = ("sm_107a", True, True, True, False, False, 12288, None, 6144)
    assert ROW_RULES[wfkey] == {
        "cta_rows": 256,
        "htail": True,
        "epi": "reg",
        "f32_v8": True,
        "store_ef": True,
    }
    wf = _views("proj", "dense_down", "wgrad", "f32", T)
    pwf, *_ = plan_dense_projection_gemm(wf["A"], wf["B"], wf["out"], **kw107)
    assert (
        _nopro(pwf.template) == "dense_proj_gemm_nn_n256_m256_f32_v8_ef_ht"
        and pwf.htail
        and pwf.epi == "reg"
        and pwf.store_ef
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted nn_n256_m256_f32_v8_ef_ht)
    assert (
        (pwf.cta_rows, pwf.block_n) == (256, 256)
        and not pwf.sk_sync
        and not pwf.sk_exact
    )
    old_wf, *_ = plan_dense_projection_gemm(
        wf["A"], wf["B"], wf["out"], cta_rows=128, htail=False, epi="tma", **kw107
    )
    assert (
        _nopro(old_wf.template) == "dense_proj_gemm_nn_n256_f32_tma1"
        and old_wf.epi == "tma"
        and not old_wf.htail
    )  # the previous lock's program
    # ---- (c) dense_down weight gradient bf16 (same shape): the tall tile with the half-height tail wave replaces the 3-way split rule whose split
    #      the host refuses at these token counts (the previous lock's program is the plain dense_proj_gemm_nn_n256) ----
    wbkey = ("sm_107a", True, True, False, False, False, 12288, None, 6144)
    assert ROW_RULES[wbkey] == {"cta_rows": 256, "htail": True}
    wb = _views("proj", "dense_down", "wgrad", "bf16", T)
    pwb, *_ = plan_dense_projection_gemm(wb["A"], wb["B"], wb["out"], **kw107)
    assert (
        _nopro(pwb.template) == "dense_proj_gemm_nn_n256_m256_ht"
        and pwb.htail
        and not pwb.sk_sync
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted nn_n256_m256_ht - a template the previous lock already declares)
    assert (pwb.cta_rows, pwb.block_n) == (256, 256)
    old_wb, *_ = plan_dense_projection_gemm(
        wb["A"], wb["B"], wb["out"], cta_rows=128, htail=False, **kw107
    )
    assert (
        _nopro(old_wb.template) == "dense_proj_gemm_nn_n256"
        and not old_wb.htail
        and old_wb.cta_rows == 128
    )  # the previous lock's program
    # ---- (d) dense_gate_up input gradient bf16 (G @ W, 6144 x 12288 x T): the tall family released by row half with the column-grouped raster of
    #      12 and a 3-way split of the tail wave replaces the 2-way split rule (sk_parts is the row's tail policy, not a symbol term; the previous
    #      lock's program is dense_proj_gemm_kn_n256) ----
    dkey = ("sm_107a", False, True, False, False, False, 6144, 12288, None)
    assert ROW_RULES[dkey] == {
        "cta_rows": 256,
        "sk_parts": 3,
        "ovr": True,
        "cgrp": True,
        "group_m": 12,
    }
    dg = _views("proj", "dense_gate_up", "dgrad", "bf16", T)
    pdg, *_ = plan_dense_projection_gemm(dg["A"], dg["B"], dg["out"], **kw107)
    assert (
        _nopro(pdg.template) == "dense_proj_gemm_kn_n256_m256_cg_or"
        and pdg.ovr
        and pdg.cgrp
        and not pdg.ovl
        and not pdg.htail
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted kn_n256_m256_cg_or)
    assert (pdg.cta_rows, pdg.block_n, pdg.group_m) == (256, 256, 12)
    old_dg, *_ = plan_dense_projection_gemm(
        dg["A"], dg["B"], dg["out"], cta_rows=128, ovr=False, cgrp=False, **kw107
    )
    assert (
        _nopro(old_dg.template) == "dense_proj_gemm_kn_n256"
        and not old_dg.ovr
        and not old_dg.cgrp
    )  # the previous lock's program
    # ---- (e) q_b input gradient bf16 (G @ W, 2048 x 16384 x T; new key): the column-grouped raster with 8 column tiles per group on the standard
    #      128-row family (the previous lock's program is dense_proj_gemm_kn_n256, no rule) ----
    qkey = ("sm_107a", False, True, False, False, False, 2048, 16384, None)
    assert ROW_RULES[qkey] == {"cgrp": True, "group_m": 8}
    q = _views("proj", "q_b", "dgrad", "bf16", T)
    pq, *_ = plan_dense_projection_gemm(q["A"], q["B"], q["out"], **kw107)
    assert (
        _nopro(pq.template) == "dense_proj_gemm_kn_n256_cg"
        and pq.cgrp
        and not pq.ovr
        and not pq.htail
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted kn_n256_cg)
    assert (pq.cta_rows, pq.block_n, pq.group_m) == (128, 256, 8)
    old_q, *_ = plan_dense_projection_gemm(
        q["A"], q["B"], q["out"], cgrp=False, **kw107
    )
    assert (
        _nopro(old_q.template) == "dense_proj_gemm_kn_n256" and not old_q.cgrp
    )  # the previous lock's program
    # ---- (f) dense_gate_up weight gradient fp32-out (G.T @ X, 6144 x 12288 over T): the tall tile with the half-height tail wave through the
    #      TMA-store epilogue replaces the register-epilogue rule {epi reg, f32_v8} (the previous lock's program is dense_proj_gemm_nn_n256_f32_v8) ----
    gwkey = ("sm_107a", True, True, True, False, False, 6144, None, 12288)
    assert ROW_RULES[gwkey] == {"cta_rows": 256, "htail": True}
    gw = _views("proj", "dense_gate_up", "wgrad", "f32", T)
    pgw, *_ = plan_dense_projection_gemm(gw["A"], gw["B"], gw["out"], **kw107)
    assert (
        _nopro(pgw.template) == "dense_proj_gemm_nn_n256_m256_f32_ht_tma1"
        and pgw.htail
        and pgw.epi == "tma"
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted nn_n256_m256_f32_ht_tma1 - a template the previous lock already declares; the rule names no epi, so the fp32-out tile takes the planner's default TMA-store epilogue like the sm_100a twin's nn_n256_m256_f32_tma1)
    assert (pgw.cta_rows, pgw.block_n) == (256, 256) and not pgw.sk_sync
    old_gw, *_ = plan_dense_projection_gemm(
        gw["A"],
        gw["B"],
        gw["out"],
        cta_rows=128,
        htail=False,
        epi="reg",
        f32_v8=True,
        **kw107,
    )
    assert (
        _nopro(old_gw.template) == "dense_proj_gemm_nn_n256_f32_v8"
        and old_gw.epi == "reg"
        and not old_gw.htail
    )  # the previous lock's program
    # the sm_100a rows of the six keys carried no round-26 change (controls); the shapes tranche later re-valued the three sm_100a weight
    # gradients to the synchronised stream-K tail (slab 2 on bf16, the fp32-output slab 3 on fp32-out), the other three keep the previous
    # lock's programs
    for row, op, dt, tmpl in (
        ("dense_gate_up", "fwd", "bf16", "dense_proj_gemm_kk_n256"),
        ("dense_down", "wgrad", "f32", "dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3"),
        ("dense_down", "wgrad", "bf16", "dense_proj_gemm_nn_n256_m256_sks_sb2"),
        ("dense_gate_up", "dgrad", "bf16", "dense_proj_gemm_kn_n256_m256_ov_ht"),
        ("q_b", "dgrad", "bf16", "dense_proj_gemm_kn_n256_m256_ov_ht"),
        (
            "dense_gate_up",
            "wgrad",
            "f32",
            "dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3",
        ),
    ):
        v = _views("proj", row, op, dt, T)
        p100, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
        assert _nopro(p100.template) == tmpl and not p100.cgrp and not p100.ovr, (
            row,
            op,
            dt,
        )
    # whether the six programs enter the declared template set as new templates or reuse declared ones is the templates step's outcome; the
    # EXPORTED_TEMPLATES table of this file is regenerated by the figen step - not asserted here


@pytest.mark.parametrize("T", [16172, 16231])
def test_round27_rules_plan_like_the_cake_launcher(T):
    # the round-27 rule changes (Cake 20da8851d46): three existing keys re-valued, none inserted or removed - the indexer_k input gradient
    # fp32-out takes the raster group 2 on sm_100a (same program) and, on sm_107a, the single-CTA form with the raster group 2 and the pd 2
    # cross-tile drain; the sm_107a kv_a weight gradient bf16 (the ragged-K rule of the 576-wide latent projection) keeps its synchronised
    # stream-K program and adds the transposed stmatrix / TMA-store staging of its last work item's slices (``stt``, instance-key field 31).
    # the table size is asserted by the newest round's test (round 33: 109 = 51 + 58); round 27 shipped 103 = 49 + 54
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    # ---- (a) indexer_k input gradient fp32-out on sm_100a (G @ W, 6144 x 128 x T): the raster group 2 on the unchanged 128-column TMA-store
    #      program (the previous lock's program; only the launch plan's group_m changes) ----
    akey = ("sm_100a", False, True, True, False, False, 6144, 128, None)
    assert ROW_RULES[akey] == {"block_n": 128, "group_m": 2}
    a = _views("proj", "indexer_k", "dgrad", "f32", T)
    pa, *_ = plan_dense_projection_gemm(a["A"], a["B"], a["out"], **kw)
    assert (
        _nopro(pa.template) == "dense_proj_gemm_kn_n128_f32_tma1"
        and pa.group_m == 2
        and pa.block_n == 128
        and not pa.cta1
        and pa.pd == 0
        and not pa.stt
    )
    # ---- (b) indexer_k input gradient fp32-out on sm_107a (same shape): the single-CTA form with the raster group 2 and the pd 2 cross-tile
    #      drain; the previous lock planned the pair form dense_proj_gemm_kn_n128_f32_tma1 ----
    bkey = ("sm_107a", False, True, True, False, False, 6144, 128, None)
    assert ROW_RULES[bkey] == {"block_n": 128, "cta1": True, "group_m": 2, "pd": 2}
    pb, *_ = plan_dense_projection_gemm(a["A"], a["B"], a["out"], **kw107)
    assert (
        _nopro(pb.template) == "dense_proj_gemm_kn_n128_f32_tma1_pd2_c1"
        and pb.cta1
        and pb.pd == 2
        and pb.group_m == 2
        and pb.block_n == 128
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted kn_n128_f32_tma1_pd2_c1)
    old_b, *_ = plan_dense_projection_gemm(
        a["A"], a["B"], a["out"], cta1=False, pd=0, **kw107
    )
    assert (
        _nopro(old_b.template) == "dense_proj_gemm_kn_n128_f32_tma1"
        and not old_b.cta1
        and old_b.pd == 0
    )  # the previous lock's program
    # ---- (c) kv_a weight gradient bf16 on sm_107a (the swapped orientation: a 576-row transposed output over T, ragged K): the synchronised
    #      stream-K program with the 64-byte B panels and evict-first hints gains the transposed stmatrix / TMA-store staging of the last work
    #      item's slices (``_stt`` directly after ``_sks``); the previous lock's program is dense_proj_gemm_nn_n192_bz64_hee_t_sks ----
    ckey = ("sm_107a", True, True, False, True, False, 576, None, 6144)
    assert ROW_RULES[ckey] == {
        "group_m": 8,
        "hints": ("evict_first", "evict_first"),
        "b_swz": 64,
        "sk_sync": True,
        "sk_sync_m": 38,
        "stt": True,
        "pro": 15,
    }
    c = _views("proj", "kv_a", "wgrad", "bf16", T)
    pc, *_ = plan_dense_projection_gemm(
        c["A"], c["B"], c["out"], transposed_out=c["transposed"], **kw107
    )
    assert (
        _nopro(pc.template) == "dense_proj_gemm_nn_n192_bz64_hee_t_sks_stt"
        and pc.stt
        and pc.sk_sync
        and pc.tma_out
        and pc.transposed_out
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted nn_n192_bz64_hee_t_sks_stt)
    assert (pc.block_n, pc.b_swz, pc.cta_rows, pc.group_m) == (
        192,
        64,
        128,
        8,
    ) and pc.hints == ("evict_first", "evict_first")
    old_c, *_ = plan_dense_projection_gemm(
        c["A"], c["B"], c["out"], transposed_out=c["transposed"], stt=False, **kw107
    )
    assert (
        _nopro(old_c.template) == "dense_proj_gemm_nn_n192_bz64_hee_t_sks"
        and not old_c.stt
        and not old_c.tma_out
        and old_c.sk_sync
    )  # the previous lock's program
    assert old_c.grid == pc.grid and (
        old_c.num_full,
        old_c.tail_tiles,
        old_c.sk_units,
    ) == (
        pc.num_full,
        pc.tail_tiles,
        pc.sk_units,
    )  # the plan arithmetic is the pair form's
    # controls: the sm_100a kv_a weight gradient bf16 keeps its TMA-store epilogue program (no stt rule on sm_100a), and the fp32-out twins of the
    # kv_a weight gradient carry no stt on either architecture (stmatrix is .b16)
    p100, *_ = plan_dense_projection_gemm(
        c["A"], c["B"], c["out"], transposed_out=c["transposed"], **kw
    )
    assert (
        _nopro(p100.template) == "dense_proj_gemm_nn_n192_hee_t_tma1"
        and not p100.stt
        and p100.tma_out
    )
    f = _views("proj", "kv_a", "wgrad", "f32", T)
    for kwx in (kw, kw107):
        pf, *_ = plan_dense_projection_gemm(
            f["A"], f["B"], f["out"], transposed_out=f["transposed"], **kwx
        )
        assert not pf.stt and pf.out_f32
    # whether the two new programs enter the declared template set is the templates step's outcome; the EXPORTED_TEMPLATES table of this file is
    # regenerated by the figen step - not asserted here


@pytest.mark.parametrize("T", [16172, 16231])
def test_round27_stt_knob(T):
    # field 31 of the key - the LAST field through round 32, 32 fields since round 33 (round 27, Cake lane 7 / W-3 variant A1): the transposed stmatrix / TMA-store staging of the bf16
    # transposed register epilogue (symbol ``_stt`` directly after ``_sks``): each epilogue warp writes its 32 x EPI_COLS slice of the CTA's LAST
    # work item with stmatrix.trans into a 64 B-swizzled slot of the dead mainloop stage ring and stores it with one bulk tensor store through
    # OUT16 (the real output view); every other slice keeps the register stores.  bf16 transposed outputs through the single-pass register
    # epilogue of the 128-row pair family only; every other field is unchanged: the OFF form's key is the round-24 key plus a trailing False and
    # its symbol is the round-24 symbol
    on = instance_key(
        a_mn=True,
        b_mn=True,
        out_t=True,
        block_n=192,
        b_swz=64,
        hints=("evict_first", "evict_first"),
        sk_sync=True,
        stt=True,
    )
    assert len(on) == 37 and on[30] is True and on[25] is True and on[29] is False
    assert instance_symbol(on) == "dense_proj_gemm_nn_n192_bz64_hee_t_sks_stt"
    off = instance_key(
        a_mn=True,
        b_mn=True,
        out_t=True,
        block_n=192,
        b_swz=64,
        hints=("evict_first", "evict_first"),
        sk_sync=True,
    )
    assert len(off) == 37 and off[30] is False and off[:30] == on[:30]
    assert (
        instance_symbol(off) == "dense_proj_gemm_nn_n192_bz64_hee_t_sks"
    )  # the round-18 symbol, byte-identical
    plain = instance_key(a_mn=True, b_mn=True, out_t=True, stt=True)
    assert (
        len(plain) == 37
        and plain[30] is True
        and plain[5] == off[5] == on[5]
        or plain[30] is True
    )  # the stage count is the row's default (stt trades no stage)
    assert instance_symbol(plain) == "dense_proj_gemm_nn_n256_t_stt"
    default = instance_key(a_mn=False, b_mn=False)
    assert (
        len(default) == 37
        and default[30] is False
        and instance_symbol(default) == "dense_proj_gemm_kk_n256"
    )
    # validation mirrors the kernel's instance_key: a bf16 TRANSPOSED output through the REGISTER epilogue only - the fp32 twin (stmatrix is .b16),
    # the row-major bf16 output and the TMA-store epilogue of the same transposed row are all valid without stt and refused with it
    for bad in (
        dict(a_mn=True, b_mn=True, out_t=True, out_f32=True),
        dict(a_mn=True, b_mn=True, out_t=False),
        dict(a_mn=True, b_mn=True, out_t=True, epi="tma"),
    ):
        instance_key(**bad)  # valid without stt
        with pytest.raises(ValueError):
            instance_key(stt=True, **bad)
    # through the planner: the sm_107a kv_a weight gradient bf16 rule carries stt (the plan hands the real output view to OUT16 next to the
    # register-store alias: tma_out), a caller-forced stt=False plans the previous program with the same arithmetic, and a caller-forced stt on
    # a row whose output is fp32 raises like the Cake launcher
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    v = _views("proj", "kv_a", "wgrad", "bf16", T)
    p, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], transposed_out=v["transposed"], **kw107
    )
    assert (
        p.stt
        and p.tma_out
        and p.epi == "reg"
        and _nopro(p.template).endswith("_t_sks_stt")
    )
    q, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], transposed_out=v["transposed"], stt=False, **kw107
    )
    assert (
        not q.stt
        and not q.tma_out
        and _nopro(q.template) == _nopro(p.template)[: -len("_stt")]
        and q.grid == p.grid
    )
    f = _views("proj", "kv_a", "wgrad", "f32", T)
    with pytest.raises(ValueError):
        plan_dense_projection_gemm(
            f["A"], f["B"], f["out"], transposed_out=f["transposed"], stt=True, **kw107
        )
    # a caller-forced form outside the staging's admission on the stt row keeps that form WITHOUT the staging (the rule's stt does not carry
    # over, like the rule's sk_sync / sk_slab: the e2e registrations of the _t_tma1 and _sks_sb2 programs at this row), while a caller-forced
    # stt on such a form raises
    # pd (the pipelined TMEM drain) is not a caller form of this row at all: it needs a row-major output and a 128 / 256-column
    # tile, so the planner rejects it here with or without stt, like the Cake launcher
    with pytest.raises(ValueError, match="pd needs"):
        plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], transposed_out=v["transposed"], pd=2, **kw107
        )
    for forced in (
        dict(epi="tma"),
        dict(sk_sync=True, sk_slab=2),
        dict(cta_rows=256),
    ):
        r, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], transposed_out=v["transposed"], **forced, **kw107
        )
        assert (
            not r.stt
            and not _nopro(r.template).endswith("_stt")
            and r.tma_out == (r.epi == "tma")
        ), forced
        with pytest.raises(ValueError):
            plan_dense_projection_gemm(
                v["A"],
                v["B"],
                v["out"],
                transposed_out=v["transposed"],
                stt=True,
                **forced,
                **kw107,
            )


@pytest.mark.parametrize("T", [16172, 16231])
def test_round33_rules_plan_like_the_cake_launcher(T):
    # the round-33 rule changes (Cake 8b34a0b1725): an architecture-wide default under every rule (ARCH_RULE_DEFAULTS: the pair prologue /
    # cleanup sync form ``pro`` 7 on sm_100a and sm_107a - instance-key field 32, symbol ``_pro{n}``), six keys inserted (pro 15 rows: the
    # indexer_q forward / input gradient bf16 on both architectures, the sm_107a indexer_k forward bf16 and q_b weight gradient bf16),
    # 18 existing rules re-valued with pro 15 (24 rules at pro 15 with the six inserted), the sm_100a shared_down input gradient bf16 pinned to pro 0 and the sm_107a kv_a weight
    # gradient fp32-out moved to the cheaper slab path of its synchronised stream-K fixup (sk_slab 2); no key removed
    # round 33 left 109 keys (51 sm_100a, 58 sm_107a; six inserted, none removed); the shapes tranche adds 79 sm_100a and
    # 86 sm_103a keys and re-values seven round-33 sm_100a keys (MEASURED_SHAPE_ROWS); the whole-tile tile resolve
    # adds one sm_103a key (the kv_a input gradient, test_mg_rules_plan_the_whole_tile_programs)
    assert len(ROW_RULES) == 109 + 79 + 86 + 1
    assert sum(k[0] == "sm_100a" for k in ROW_RULES) == 51 + 79
    assert sum(k[0] == "sm_107a" for k in ROW_RULES) == 58
    assert sum(k[0] == "sm_103a" for k in ROW_RULES) == 86 + 1
    assert ARCH_RULE_DEFAULTS == {
        "sm_107a": {"pro": 7},
        "sm_100a": {"pro": 7},
        "sm_103a": {"pro": 7},
    }
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    # ---- (a) the six inserted keys: pro 15 (the full pair sync form + the CTA-local exit), no other knob ----
    n1 = (
        "sm_100a",
        False,
        False,
        False,
        False,
        False,
        4096,
        2048,
        None,
    )  # indexer_q forward bf16 (X @ W.T, 4096 x 2048 x T)
    n2 = (
        "sm_100a",
        False,
        True,
        False,
        False,
        False,
        2048,
        4096,
        None,
    )  # indexer_q input gradient bf16 (G @ W, 2048 x 4096 x T)
    n3 = (
        "sm_107a",
        False,
        False,
        False,
        False,
        False,
        128,
        6144,
        None,
    )  # indexer_k forward bf16 (128 x 6144 x T)
    n4 = (
        "sm_107a",
        False,
        False,
        False,
        False,
        False,
        4096,
        2048,
        None,
    )  # indexer_q forward bf16
    n5 = (
        "sm_107a",
        False,
        True,
        False,
        False,
        False,
        2048,
        4096,
        None,
    )  # indexer_q input gradient bf16
    n6 = (
        "sm_107a",
        True,
        True,
        False,
        False,
        False,
        2048,
        None,
        16384,
    )  # q_b weight gradient bf16 (G.T @ X, 2048 x 16384 over T)
    # n1 / n2 were re-measured in the shapes tranche: the column-grouped raster (16 column tiles per group) over the
    # round-33 pro 15, the forward with the half-height tail wave
    assert ROW_RULES[n1] == {"htail": True, "cgrp": True, "group_m": 16, "pro": 15}
    assert ROW_RULES[n2] == {"cgrp": True, "group_m": 16, "pro": 15}
    assert ROW_RULES[n3] == {"pro": 15}
    assert ROW_RULES[n4] == {"pro": 15}
    assert ROW_RULES[n5] == {"pro": 15}
    assert ROW_RULES[n6] == {"pro": 15}
    # ---- (b) re-valued keys (a sample) and an unchanged single-CTA row ----
    sdb = (
        "sm_100a",
        False,
        True,
        False,
        False,
        False,
        2048,
        6144,
        None,
    )  # shared_down input gradient bf16: the one row pinned to pro 0
    assert ROW_RULES[sdb] == {"group_m": 8, "pro": 0}
    kvf = (
        "sm_107a",
        True,
        True,
        True,
        True,
        False,
        576,
        None,
        6144,
    )  # kv_a weight gradient fp32-out (swapped, ragged K): sk_slab 2, pro from the default
    assert ROW_RULES[kvf] == {
        "group_m": 8,
        "hints": ("evict_first", "evict_first"),
        "b_swz": 64,
        "sk_sync": True,
        "sk_sync_m": 38,
        "sk_slab": 2,
    }
    qbw = (
        "sm_107a",
        True,
        True,
        True,
        False,
        False,
        2048,
        None,
        16384,
    )  # q_b weight gradient fp32-out: pro 15 on the round-27 knobs
    assert ROW_RULES[qbw] == {"group_m": 8, "epi": "reg", "f32_v8": True, "pro": 15}
    kvb = (
        "sm_107a",
        True,
        True,
        False,
        True,
        False,
        576,
        None,
        6144,
    )  # kv_a weight gradient bf16 (the round-27 stt row): + pro 15
    assert ROW_RULES[kvb] == {
        "group_m": 8,
        "hints": ("evict_first", "evict_first"),
        "b_swz": 64,
        "sk_sync": True,
        "sk_sync_m": 38,
        "stt": True,
        "pro": 15,
    }
    hw = (
        "sm_107a",
        False,
        True,
        True,
        False,
        False,
        6144,
        32,
        None,
    )  # indexer_hw input gradient fp32-out: a single-CTA row, no pro
    assert ROW_RULES[hw] == {"block_n": 128, "group_m": 2, "slots": 1, "cta1": True}
    # ---- (c) the ARCH_RULE_DEFAULTS merge of row_rule: a key without a rule takes the architecture default (pro 7) on every supported
    #      architecture (sm_103a included: a pair template planned there without it names the round-27 form, which no measured sm_103a rule
    #      selects) and nothing on an architecture without defaults; a row rule overrides it (pro 0 / 15); a cta1 row carries the default in
    #      the merged rule and the planner zeroes it on the single-CTA program (below) ----
    assert row_rule("sm_100a", False, False, False, False, False, 12288, 6144, T) == {
        "pro": 7
    }  # dense_gate_up forward bf16: no sm_100a rule
    assert row_rule("sm_107a", False, False, False, False, False, 8, 8, 8) == {"pro": 7}
    assert row_rule("sm_103a", False, False, False, False, False, 8, 8, 8) == {"pro": 7}
    assert row_rule("sm_90a", False, False, False, False, False, 8, 8, 8) == {}
    assert row_rule("sm_100a", False, True, False, False, False, 2048, 6144, T) == {
        "group_m": 8,
        "pro": 0,
    }
    assert (
        row_rule("sm_100a", False, False, False, False, False, 4096, 2048, T)
        == {
            "htail": True,
            "cgrp": True,
            "group_m": 16,
            "pro": 15,
        }
    )  # re-valued in the shapes tranche: the row rule's pro 15 still overrides the default
    assert row_rule("sm_107a", False, True, True, False, False, 6144, 32, T) == {
        "block_n": 128,
        "group_m": 2,
        "slots": 1,
        "cta1": True,
        "pro": 7,
    }
    # ---- (d) through the planner: the inserted indexer_q forward bf16 keys plan the round-27 program with the pro-15 suffix on both
    #      architectures; a caller-forced pro 0 is the previous lock's program with the same plan arithmetic ----
    q = _views("proj", "indexer_q", "fwd", "bf16", T)
    for kwx in (kw, kw107):
        p, *_ = plan_dense_projection_gemm(q["A"], q["B"], q["out"], **kwx)
        assert p.pro == 15 and p.template.endswith("_pro15") and not p.cta1, kwx["arch"]
        old, *_ = plan_dense_projection_gemm(q["A"], q["B"], q["out"], pro=0, **kwx)
        assert (
            old.pro == 0
            and old.template == p.template[: -len("_pro15")]
            and old.grid == p.grid
            and old.stages == p.stages
        ), kwx["arch"]
    p107, *_ = plan_dense_projection_gemm(q["A"], q["B"], q["out"], **kw107)
    assert (
        p107.template == "dense_proj_gemm_kk_n256_pro15"
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted kk_n256_pro15)
    # the shared_down input gradient bf16: sm_100a pins pro 0 (the round-27 program, no suffix), sm_107a has no rule (the default pro 7)
    s = _views("proj", "shared_down", "dgrad", "bf16", T)
    ps, *_ = plan_dense_projection_gemm(s["A"], s["B"], s["out"], **kw)
    assert (
        ps.pro == 0 and not re.search(r"_pro\d+$", ps.template) and ps.group_m == 8
    )  # NOT `"_pro" not in`: `dense_proj` contains it
    ps107, *_ = plan_dense_projection_gemm(s["A"], s["B"], s["out"], **kw107)
    assert ps107.pro == 7 and ps107.template.endswith("_pro7")
    # the q_b weight gradient fp32-out on sm_107a: pro 15 on the register epilogue with the 256-bit stores (the round-27 knobs)
    w = _views("proj", "q_b", "wgrad", "f32", T)
    pw, *_ = plan_dense_projection_gemm(
        w["A"], w["B"], w["out"], transposed_out=w["transposed"], **kw107
    )
    assert (
        pw.pro == 15
        and pw.template.endswith("_pro15")
        and pw.epi == "reg"
        and "_f32_v8" in pw.template
        and pw.group_m == 8
    )
    assert (
        pw.template == "dense_proj_gemm_nn_n256_f32_v8_pro15"
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted nn_n256_f32_v8_pro15)
    # a caller-forced tile family drops the rule's pro with the rule's other family-bound knobs (the launcher's filter): pro 0, no suffix
    tall_w, *_ = plan_dense_projection_gemm(
        w["A"], w["B"], w["out"], transposed_out=w["transposed"], cta_rows=256, **kw107
    )
    assert (
        tall_w.cta_rows == 256
        and tall_w.pro == 0
        and not re.search(r"_pro\d+$", tall_w.template)
    )
    # the kv_a weight gradient fp32-out on sm_107a: the synchronised stream-K program takes the bulk slab read (sk_slab 2) and the default
    # pro 7; a caller-forced sk_slab 0 / pro 0 is the previous lock's program with the same plan arithmetic
    f = _views("proj", "kv_a", "wgrad", "f32", T)
    pf, *_ = plan_dense_projection_gemm(
        f["A"], f["B"], f["out"], transposed_out=f["transposed"], **kw107
    )
    assert (
        pf.pro == 7
        and pf.sk_sync
        and pf.sk_slab == 2
        and "_sb2" in pf.template
        and pf.template.endswith("_pro7")
    )
    assert (
        pf.template == "dense_proj_gemm_nn_n192_bz64_hee_f32_t_sks_sb2_pro7"
    )  # ROUTE_TEMPLATE_NEW of the templates step (predicted nn_n192_bz64_hee_f32_t_sks_sb2_pro7)
    old_f, *_ = plan_dense_projection_gemm(
        f["A"],
        f["B"],
        f["out"],
        transposed_out=f["transposed"],
        sk_slab=0,
        pro=0,
        **kw107,
    )
    assert (
        old_f.template == "dense_proj_gemm_nn_n192_bz64_hee_f32_t_sks"
        and old_f.grid == pf.grid
    )  # the previous lock's program
    # a single-CTA row: the indexer_hw input gradient fp32-out on sm_107a keeps its round-19 program (no pro; the merged rule's default is
    # zeroed), a caller-forced pro raises like the Cake launcher, and a caller-forced pair form takes the architecture default
    h = _views("proj", "indexer_hw", "dgrad", "f32", T)
    ph, *_ = plan_dense_projection_gemm(h["A"], h["B"], h["out"], **kw107)
    assert (
        ph.cta1 and ph.pro == 0 and ph.template == "dense_proj_gemm_kn_n128_f32_tma1_c1"
    )
    with pytest.raises(ValueError, match="needs the cluster pair"):
        plan_dense_projection_gemm(h["A"], h["B"], h["out"], pro=7, **kw107)
    pair_h, *_ = plan_dense_projection_gemm(
        h["A"], h["B"], h["out"], cta1=False, **kw107
    )
    assert not pair_h.cta1 and pair_h.pro == 7 and pair_h.template.endswith("_pro7")
    # whether the new programs enter the declared template set is the templates step's outcome; the EXPORTED_TEMPLATES table of this file is
    # regenerated by the figen step - not asserted here


@pytest.mark.parametrize("T", [16172, 16231])
def test_round33_pro_knob(T):
    # field 32 (LAST) of the 32-field key (round 33, Cake unit 2d / 2e): the pair program's prologue / cleanup sync form - a bitmask (1 = no
    # peer-startup cluster rendezvous before the collective tcgen05.alloc, 2 = relaxed arrive on the post-init cluster sync, 4 = the MMA warp
    # allocates TMEM, 8 = CTA-local exit without the cleanup barrier.cluster), symbol ``_pro{n}`` after every other knob suffix (``_or``);
    # pair programs only (the single-CTA form refuses it); every other field is unchanged: the OFF form's key is the round-27 key plus a
    # trailing 0 and its symbol is the round-27 symbol
    on = instance_key(a_mn=False, b_mn=True, pro=7)
    assert len(on) == 37 and on[31] == 7 and on[30] is False and on[27] is False
    assert instance_symbol(on) == "dense_proj_gemm_kn_n256_pro7"
    off = instance_key(a_mn=False, b_mn=True)
    assert len(off) == 37 and off[31] == 0 and off[:31] == on[:31]
    assert (
        instance_symbol(off) == "dense_proj_gemm_kn_n256"
    )  # the round-2 symbol, byte-identical
    full = instance_key(
        a_mn=True, b_mn=True, out_f32=True, epi="reg", f32_v8=True, pro=15
    )
    assert (
        full[31] == 15
        and instance_symbol(full) == "dense_proj_gemm_nn_n256_f32_v8_pro15"
    )
    # the suffix follows the last knob suffixes of the tall, the stt and the slab forms
    tall = instance_key(
        a_mn=False, b_mn=False, cta_rows=256, htail=True, ovr=True, cgrp=True, pro=7
    )
    assert instance_symbol(tall) == "dense_proj_gemm_kk_n256_m256_ht_cg_or_pro7"
    stt = instance_key(
        a_mn=True,
        b_mn=True,
        out_t=True,
        block_n=192,
        b_swz=64,
        hints=("evict_first", "evict_first"),
        sk_sync=True,
        stt=True,
        pro=15,
    )
    assert instance_symbol(stt) == "dense_proj_gemm_nn_n192_bz64_hee_t_sks_stt_pro15"
    slab = instance_key(
        a_mn=True,
        b_mn=True,
        out_f32=True,
        out_t=True,
        block_n=192,
        b_swz=64,
        hints=("evict_first", "evict_first"),
        sk_sync=True,
        sk_slab=2,
        pro=7,
    )
    assert (
        instance_symbol(slab) == "dense_proj_gemm_nn_n192_bz64_hee_f32_t_sks_sb2_pro7"
    )
    default = instance_key(a_mn=False, b_mn=False)
    assert (
        len(default) == 37
        and default[31] == 0
        and instance_symbol(default) == "dense_proj_gemm_kk_n256"
    )
    # validation mirrors the kernel's instance_key: a bitmask in 0..15 on the cluster pair only
    for bad in (16, -1, 32):
        with pytest.raises(ValueError, match="pro must be a bitmask"):
            instance_key(a_mn=False, b_mn=True, pro=bad)
    instance_key(
        a_mn=False, b_mn=True, block_n=128, out_f32=True, cta1=True
    )  # valid without pro
    with pytest.raises(ValueError, match="needs the cluster pair"):
        instance_key(a_mn=False, b_mn=True, block_n=128, out_f32=True, cta1=True, pro=7)
    # through the planner (kw107 = the sm_107a device facts): a pair row without a rule plans the architecture default (pro 7), a cta1 rule
    # zeroes it, a caller-forced value is admitted or refused by instance_key, and the FlashInfer-only registry fallback that drops a row's
    # measured rule keeps the architecture default (a rule-less plan of a pair row is a ``_pro7`` program; the rule's pro 15 goes with the rule)
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    g = _views("proj", "dense_gate_up", "fwd", "bf16", T)
    pg, *_ = plan_dense_projection_gemm(g["A"], g["B"], g["out"], **kw)
    assert (
        pg.pro == 7 and pg.template.endswith("_pro7") and not pg.rule_fallback
    )  # no sm_100a rule: the default alone
    w = _views("proj", "q_b", "wgrad", "f32", T)
    pw, *_ = plan_dense_projection_gemm(
        w["A"], w["B"], w["out"], transposed_out=w["transposed"], **kw107
    )
    assert pw.pro == 15 and pw.template.endswith("_pro15")
    for forced in (0, 7, 8, 15):
        pfx, *_ = plan_dense_projection_gemm(
            w["A"],
            w["B"],
            w["out"],
            transposed_out=w["transposed"],
            pro=forced,
            **kw107,
        )
        assert pfx.pro == forced and pfx.template == pw.template[: -len("_pro15")] + (
            f"_pro{forced}" if forced else ""
        ), forced
        assert pfx.grid == pw.grid and pfx.stages == pw.stages, (
            forced
        )  # the program changes, the plan arithmetic does not
    nr, *_ = plan_dense_projection_gemm(
        w["A"],
        w["B"],
        w["out"],
        transposed_out=w["transposed"],
        _use_rules=False,
        **kw107,
    )
    assert (
        nr.rule_fallback
        and nr.pro == 7
        and nr.template.endswith("_pro7")
        and "_f32_v8" not in nr.template
    )
    h = _views("proj", "indexer_hw", "dgrad", "f32", T)
    ph, *_ = plan_dense_projection_gemm(h["A"], h["B"], h["out"], **kw107)
    assert (
        ph.cta1 and ph.pro == 0 and ph.template == "dense_proj_gemm_kn_n128_f32_tma1_c1"
    )
    with pytest.raises(ValueError, match="needs the cluster pair"):
        plan_dense_projection_gemm(h["A"], h["B"], h["out"], pro=15, **kw107)


@pytest.mark.parametrize("T", [16172, 16231])
def test_round21_cgrp_knob(T):
    # field 29 of the 30-field key (LAST until round 24's ``ovr``; round 21, Cake W3): the column-grouped raster (symbol ``_cg``, after
    # ``_c1``): the ``group_m`` launch parameter counts COLUMN tiles per raster group (any count >= 1; the pair
    # invariant - consecutive CTAs are the two row tiles of one column tile - is unchanged) and consecutive pairs sweep
    # a group's columns of one pair row before the next pair row, so a group's B panels stay L2-resident while A
    # streams once per column group.  Every other field is unchanged: the OFF form's key is the round-20 key plus a
    # trailing False and its symbol is the round-20 symbol
    key = instance_key(
        a_mn=False, b_mn=False, cta_rows=256, ovl=True, htail=True, cgrp=True
    )
    assert len(key) == 37 and key[28] is True and key[27] is False
    assert instance_symbol(key) == "dense_proj_gemm_kk_n256_m256_ov_ht_cg"
    off = instance_key(a_mn=False, b_mn=False, cta_rows=256, ovl=True, htail=True)
    assert len(off) == 37 and off[:28] == key[:28] and off[28] is False
    assert instance_symbol(off) == "dense_proj_gemm_kk_n256_m256_ov_ht"
    assert (
        instance_symbol(instance_key(a_mn=False, b_mn=False, cgrp=True))
        == "dense_proj_gemm_kk_n256_cg"
    )
    default = instance_key(a_mn=False, b_mn=False)
    assert (
        default[28] is False and instance_symbol(default) == "dense_proj_gemm_kk_n256"
    )
    # ``_cg`` is the last knob term, after ``_c1`` (the symbol grammar only: no rule or registered program combines
    # the single-CTA form with the column raster)
    both = instance_key(a_mn=False, b_mn=False, block_n=128, cta1=True, cgrp=True)
    assert both[27] is True and both[28] is True
    assert instance_symbol(both) == "dense_proj_gemm_kk_n128_c1_cg"
    # the hint gate over the column raster's wave working set (Cake ``wave_working_set_cgrp`` / ``default_hints_cgrp``):
    # a group of g column panels is swept by consecutive pairs of one pair row, so a wave covers g column panels and
    # ceil(pairs / g) pair rows while the group's band (g x pair rows) holds the wave, else every pair row and
    # ceil(pairs / band) groups of columns
    assert wave_working_set_cgrp(64, 24, 12288, 12, 74) == (7 + 12) * 256 * 12288 * 2
    assert wave_working_set_cgrp(10, 24, 12288, 12, 74) == (5 + 24) * 256 * 12288 * 2
    assert wave_working_set_cgrp(128, 1, 6144, 16, 74) == (64 + 1) * 256 * 6144 * 2
    assert (
        wave_working_set_cgrp(128, 24, 16384, 16, 74, elt_bytes=1)
        == (5 + 16) * 256 * 16384
    )
    assert default_hints_cgrp(False, False, 128, 24, 16384, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )  # o_proj fwd: no hint on a multi-column row
    assert default_hints_cgrp(False, False, 128, 1, 6144, 16, 74, L2_BYTES) == (
        "evict_first",
        "none",
    )  # a single-use A (one column tile) streams
    assert default_hints_cgrp(False, False, 128, 1, 6144, 16, 74, 1 << 40) == (
        "none",
        "none",
    )  # fits a huge L2
    # the two rasters' waves differ, so the gates differ: 100 pair rows x one column tile at K = 3200 - the row
    # raster's wave spans 10 groups of 8 pair rows (81 panels, over the L2), the column raster's 74 pair rows (75
    # panels, under it)
    assert wave_working_set(200, 1, 3200, 16, 74) == 81 * 256 * 3200 * 2 > L2_BYTES
    assert (
        wave_working_set_cgrp(200, 1, 3200, 16, 74) == 75 * 256 * 3200 * 2 <= L2_BYTES
    )
    assert default_hints(False, False, 200, 1, 3200, 16, 74, L2_BYTES) == (
        "evict_first",
        "none",
    )
    assert default_hints_cgrp(False, False, 200, 1, 3200, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )
    # through the planner on the dense_down forward (X @ W^T: 6144 x 12288, the round-21 W3 candidate row; the knob is
    # caller-forced here - a ``cgrp`` rule arrives through the rule sync): the ``_cg`` program of the row's tall
    # overlapped half-height-tail family with 12 column tiles per raster group; the plan arithmetic (64 x 24 tiles, 768
    # pair tiles on 74 pairs, the half-height tail wave) is the pair form's - only the program and the raster differ
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    v = _views("proj", "dense_down", "fwd", "bf16", T)
    plan, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], cgrp=True, group_m=12, **kw
    )
    assert (
        plan.cgrp and _nopro(plan.template) == "dense_proj_gemm_kk_n256_m256_ov_ht_cg"
    )
    assert (plan.group_m, plan.cta_rows, plan.block_n) == (12, 256, 256)
    assert plan.ovl and plan.htail and not plan.cta1
    assert (plan.m_tiles, plan.n_tiles, plan.pair_tiles, plan.sm_pairs) == (
        64,
        24,
        768,
        74,
    )
    assert (plan.num_full, plan.tail_tiles, plan.sk_units) == (740, 28, 56)
    assert plan.hints == ("none", "none")
    assert (
        plan.wave_working_set_bytes
        == wave_working_set_cgrp(64, 24, 12288, 12, 74)
        == 19 * 256 * 12288 * 2
    )
    base, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], group_m=12, **kw)
    assert (
        not base.cgrp and _nopro(base.template) == "dense_proj_gemm_kk_n256_m256_ov_ht"
    )
    for f in (
        "m_tiles",
        "n_tiles",
        "pair_tiles",
        "sm_pairs",
        "num_full",
        "tail_tiles",
        "sk_units",
        "iters_per_unit",
        "stages",
        "epi",
        "slots",
        "hints",
        "grid",
        "ws_f32_elems",
    ):
        assert getattr(plan, f) == getattr(base, f), f
    # the raster width defaults to the row default (16, ``default_group_m``) under cgrp as well, now as column tiles
    cg16, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], cgrp=True, **kw)
    assert cg16.group_m == 16 and _nopro(cg16.template) == _nopro(plan.template)
    # the pair invariant holds whatever the column count, so the count has no parity constraint under cgrp: 3 (or 1)
    # column tiles per group are admitted where the row raster rejects an odd count (CTA pairs are adjacent row
    # tiles); a count < 1 is rejected on both
    odd, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], cgrp=True, group_m=3, **kw
    )
    assert odd.group_m == 3 and _nopro(odd.template) == _nopro(plan.template)
    one, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], cgrp=True, group_m=1, **kw
    )
    assert one.group_m == 1 and _nopro(one.template) == _nopro(plan.template)
    with pytest.raises(ValueError, match="group_m must be an even number"):
        plan_dense_projection_gemm(v["A"], v["B"], v["out"], group_m=3, **kw)
    with pytest.raises(ValueError, match="group_m must be an even number"):
        plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cgrp=False, group_m=1, **kw
        )
    with pytest.raises(ValueError, match="group_m must be an even number"):
        plan_dense_projection_gemm(v["A"], v["B"], v["out"], cgrp=True, group_m=0, **kw)
    # the planner selects the gate by the raster: one 256-column tile over 25600 rows at K = 3200 (no rule) streams A
    # evict_first on the row raster (``_hen``) and keeps the hint-free program on the column raster
    A = torch.empty(25600, 3200, dtype=torch.bfloat16)
    Bt = torch.empty(256, 3200, dtype=torch.bfloat16).t()
    out = torch.empty(25600, 256, dtype=torch.bfloat16)
    row, *_ = plan_dense_projection_gemm(A, Bt, out, **kw)
    col, *_ = plan_dense_projection_gemm(A, Bt, out, cgrp=True, **kw)
    assert (row.m_tiles, row.n_tiles, row.group_m, row.sm_pairs) == (200, 1, 16, 74)
    assert _nopro(row.template) == "dense_proj_gemm_kk_n256_hen" and row.hints == (
        "evict_first",
        "none",
    )
    assert _nopro(col.template) == "dense_proj_gemm_kk_n256_cg" and col.hints == (
        "none",
        "none",
    )
    assert row.wave_working_set_bytes > L2_BYTES >= col.wave_working_set_bytes
    assert (col.m_tiles, col.n_tiles, col.group_m, col.grid) == (200, 1, 16, row.grid)


@pytest.mark.parametrize("T", [16172, 16231])
def test_round24_ovr_knob(T):
    # field 30 of the key - the LAST field through round 26, 31 fields since round 27 (round 24; the round-20 W3 knob ported in round 23 by W4, Cake 8d837b1ac8): the overlapped tall
    # epilogue released by ROW HALF (symbol ``_or``, after ``_cg``): the eight epilogue warps drain TMEM buffer 0 and release it to the next
    # tile's half-0 MMAs while buffer 1 drains; the B stage keeps the plain tall layout (one N = 256 MMA per row half per K step).  Tall
    # 256-column tiles only; it excludes ovl / park / pf / box_rows / a_mcast, sk_sync and cta1 exclude it, a caller-forced ovr replaces a
    # rule-derived ovl and a caller-forced ovl drops a rule-provided ovr, exactly as the Cake launcher does (62a0f1fb + 95da40d6e8).  Every
    # other field is unchanged: the OFF form's key is the round-21 key plus a
    # trailing False and its symbol is the round-21 symbol
    key = instance_key(
        a_mn=False, b_mn=False, cta_rows=256, htail=True, cgrp=True, ovr=True
    )
    assert len(key) == 37 and key[29] is True and key[28] is True and key[16] is False
    assert instance_symbol(key) == "dense_proj_gemm_kk_n256_m256_ht_cg_or"
    solo = instance_key(a_mn=False, b_mn=False, cta_rows=256, htail=True, ovr=True)
    assert len(solo) == 37 and solo[29] is True and solo[28] is False
    assert instance_symbol(solo) == "dense_proj_gemm_kk_n256_m256_ht_or"
    off = instance_key(a_mn=False, b_mn=False, cta_rows=256, ovl=True, htail=True)
    assert len(off) == 37 and off[29] is False and off[16] is True
    assert (
        instance_symbol(off) == "dense_proj_gemm_kk_n256_m256_ov_ht"
    )  # the round-21 symbol, byte-identical
    cg = instance_key(
        a_mn=False, b_mn=False, cta_rows=256, ovl=True, htail=True, cgrp=True
    )
    assert len(cg) == 37 and cg[:28] == off[:28] and cg[28] is True and cg[29] is False
    assert (
        instance_symbol(cg) == "dense_proj_gemm_kk_n256_m256_ov_ht_cg"
    )  # test_round21_cgrp_knob's symbol, now on a 30-field key
    default = instance_key(a_mn=False, b_mn=False)
    assert (
        len(default) == 37
        and default[29] is False
        and instance_symbol(default) == "dense_proj_gemm_kk_n256"
    )
    # validation mirrors the kernel's instance_key: the tall 256-column tile only, no ovl / park / pf / box_rows; sk_sync and cta1 exclude ovr
    for bad in (
        dict(cta_rows=128),
        dict(cta_rows=256, block_n=128),
        dict(cta_rows=256, ovl=True),
        dict(cta_rows=256, park=True),
        dict(cta_rows=256, pf=2),
        dict(cta_rows=256, box_rows=64),
        dict(cta_rows=256, sk_sync=True),
        dict(cta_rows=128, cta1=True),
    ):
        with pytest.raises(ValueError):
            instance_key(a_mn=False, b_mn=False, ovr=True, **bad)
    # through the planner, both directions of the Cake launcher: (1) a caller-forced ovr on the sm_100a o_proj forward (rule {cta_rows 256,
    # group_m 8, ovl, htail}) REPLACES the rule's ovl (62a0f1fb L2740-L2748: a rule-derived ovl yields to a forced ovr; forcing both raises in
    # instance_key) and keeps the plan arithmetic; (2) a caller-forced ovl on a row whose RULE carries ovr DROPS the rule's ovr (the Cake
    # launcher's round-24 precedence rule, asserted below on the sm_107a o_proj / dense_down forwards)
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    v = _views("proj", "o_proj", "fwd", "bf16", T)
    base, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert (
        base.ovl
        and base.htail
        and not base.ovr
        and _nopro(base.template) == "dense_proj_gemm_kk_n256_m256_ov_ht"
    )
    forced, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], ovr=True, **kw)
    assert (
        forced.ovr
        and not forced.ovl
        and forced.htail
        and _nopro(forced.template) == "dense_proj_gemm_kk_n256_m256_ht_or"
    )
    for f in (
        "m_tiles",
        "n_tiles",
        "pair_tiles",
        "sm_pairs",
        "num_full",
        "tail_tiles",
        "sk_units",
        "iters_per_unit",
        "stages",
        "epi",
        "slots",
        "hints",
        "group_m",
        "grid",
        "ws_f32_elems",
    ):
        assert getattr(forced, f) == getattr(base, f), f
    with pytest.raises(ValueError):
        plan_dense_projection_gemm(v["A"], v["B"], v["out"], ovr=True, ovl=True, **kw)
    # (2) the reverse direction (Cake 95da40d6e8 L2745-L2746, `if ovr and ovl and not rule_ovl: ovr = False` inside the rule read of ovr): a
    # caller-forced ovl=True on the sm_107a o_proj fwd row (rule {cta_rows 256, htail, ovr, cgrp, group_m 8}) drops the rule's ovr - the
    # rule's cgrp / htail / group_m stay and the plan is the `_ov_ht_cg` program of test_round21_cgrp_knob, not a ValueError (ovl + ovr);
    # without this mirror every forced-ovl planner call on these rows raises in FlashInfer while the Cake launcher plans the `_ov` program
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a", _fallback=False)
    o107 = _views("proj", "o_proj", "fwd", "bf16", T)
    forced_ovl, *_ = plan_dense_projection_gemm(
        o107["A"], o107["B"], o107["out"], ovl=True, **kw107
    )
    assert (
        forced_ovl.ovl
        and not forced_ovl.ovr
        and forced_ovl.cgrp
        and forced_ovl.htail
        and forced_ovl.group_m == 8
    )
    assert _nopro(forced_ovl.template) == "dense_proj_gemm_kk_n256_m256_ov_ht_cg"
    # the same on the dense_down fwd row (rule {cta_rows 256, sk_parts 3, ovr, cgrp, group_m 12}: no htail) -> the `_ov_cg` tall program
    d107 = _views("proj", "dense_down", "fwd", "bf16", T)
    forced_ovl_d, *_ = plan_dense_projection_gemm(
        d107["A"], d107["B"], d107["out"], ovl=True, **kw107
    )
    assert (
        forced_ovl_d.ovl
        and not forced_ovl_d.ovr
        and forced_ovl_d.cgrp
        and not forced_ovl_d.htail
        and forced_ovl_d.group_m == 12
    )
    assert _nopro(forced_ovl_d.template) == "dense_proj_gemm_kk_n256_m256_ov_cg"
    with pytest.raises(ValueError):
        plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], ovr=True, cta_rows=128, **kw
        )


def test_wave_working_set_and_hint_rule():
    # o_proj forward at T = 16231 (m_tiles = 128, n_tiles = 24, K = 16384) on 74 pairs: a raster band of
    # 8 pair rows x 24 column tiles exceeds the wave, so the wave covers 8 pair-row panels and
    # ceil(74 / 8) = 10 column panels -> 18 x 256 x 16384 x 2 B
    assert wave_working_set(128, 24, 16384, 16, 74) == 18 * 256 * 16384 * 2
    # a narrow band (8 x 1 < 74 pairs) spans ceil(74 / 8) groups of 8 pair rows, capped by the rows there are
    assert wave_working_set(8, 1, 6144, 16, 74) == (4 + 1) * 256 * 6144 * 2
    assert wave_working_set(1024, 1, 6144, 16, 74) == (80 + 1) * 256 * 6144 * 2
    assert wave_working_set(128, 24, 16384, 16, 74, elt_bytes=1) == 18 * 256 * 16384
    # hints only when the wave does not fit the L2 ...
    assert wave_working_set(128, 24, 16384, 16, 74) > L2_BYTES
    assert default_hints(False, False, 128, 24, 16384, 16, 74, 1 << 40) == (
        "none",
        "none",
    )  # fits a huge L2
    assert default_hints(False, False, 8, 1, 6144, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )  # 15.7 MB working set fits
    # ... then a single-use A (one column tile) streams evict_first on every layout class
    assert default_hints(False, False, 128, 1, 6144, 16, 74, L2_BYTES) == (
        "evict_first",
        "none",
    )  # indexer_k fwd
    assert default_hints(True, True, 48, 1, 16231, 16, 74, L2_BYTES) == (
        "evict_first",
        "none",
    )  # swapped small-N wgrad
    assert default_hints(False, False, 8, 1, 6144, 16, 74, 1 << 20) == (
        "evict_first",
        "none",
    )  # tiny L2
    # ... K-major forward / input-gradient rows get no other hint
    assert default_hints(False, False, 128, 24, 16384, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )  # o_proj fwd
    assert default_hints(False, True, 128, 64, 16384, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )  # o_proj dgrad
    assert default_hints(False, True, 2, 3, 6144, 16, 74, 1 << 20) == ("none", "none")
    # ... and neither does the weight-gradient class (both MN-major): the reuse-ratio hints are not applied
    assert default_hints(True, True, 16, 24, 16231, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )  # q_a wgrad
    assert default_hints(True, True, 48, 8, 16231, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )  # shared_down wgrad
    assert default_hints(True, True, 32, 24, 16231, 16, 74, L2_BYTES) == (
        "none",
        "none",
    )
    assert default_hints(True, True, 2, 3, 6144, 16, 74, 1 << 20) == ("none", "none")


def test_default_group_m_rule():
    # 16 row tiles per raster group on every row: the one-to-two-wave wgrad small-group rule is not applied
    for a_mn, b_mn, m_tiles, pair_tiles, pairs in (
        (True, True, 16, 192, 106),  # q_a wgrad on 212 SMs (one to two waves)
        (True, True, 16, 192, 74),
        (True, True, 32, 192, 106),  # indexer_q wgrad
        (True, True, 6, 148, 74),
        (True, True, 16, 148, 74),
        (True, True, 4, 148, 74),
        (True, True, 2, 128, 74),  # the MLA weight gradients (64 heads x 2 tiles)
        (True, True, 16, 74, 74),  # exactly one wave
        (False, True, 16, 100, 74),  # not the wgrad class
        (False, False, 16, 100, 74),
    ):
        assert default_group_m(a_mn, b_mn, m_tiles, pair_tiles, pairs) == 16


def test_instance_key_rejects_bad_configurations():
    with pytest.raises(ValueError, match="BLOCK_N"):
        instance_key(a_mn=False, b_mn=False, block_n=64)
    with pytest.raises(ValueError, match="cta_rows"):
        instance_key(a_mn=False, b_mn=False, cta_rows=32)
    # a warp slice must be whole 16-column chunks: BLOCK_N = 160 / 224 have none on 64-row tiles (40 / 56 columns)
    with pytest.raises(ValueError, match="16-column"):
        instance_key(a_mn=False, b_mn=False, block_n=160, cta_rows=64)
    with pytest.raises(ValueError, match="16-column"):
        instance_key(a_mn=False, b_mn=False, block_n=224, cta_rows=64)
    # a forced TMA-store epilogue needs whole 128-byte chunks per warp (64 bf16 / 32 fp32 columns)
    with pytest.raises(ValueError, match="epi='reg'"):
        instance_key(a_mn=False, b_mn=False, block_n=192, cta_rows=64, epi="tma")
    with pytest.raises(ValueError, match="epi='reg'"):
        instance_key(a_mn=False, b_mn=False, block_n=160, out_f32=True, epi="tma")
    # the stage bound defaults to the largest architecture limit (sm_107a's 334336 B oversized mode): eight
    # 32 KiB stages fit there, eleven do not; the planner passes its architecture's limit (the 227 KiB opt-in on
    # sm_100a / sm_103a), where eight do not.  The limit is not part of the key.
    assert smem_limit_for(None) == smem_limit_for("sm_107a") == 334848 - 512
    assert (
        smem_limit_for("sm_100a") == smem_limit_for("sm_103a") == SMEM_OPT_IN == 232448
    )
    assert instance_key(a_mn=False, b_mn=False, stages=8)[5] == 8
    assert instance_key(a_mn=False, b_mn=False, stages=8) == instance_key(
        a_mn=False, b_mn=False, stages=8, smem_limit=smem_limit_for("sm_107a")
    )
    with pytest.raises(ValueError, match="SMEM"):
        instance_key(a_mn=False, b_mn=False, stages=11)
    with pytest.raises(ValueError, match="SMEM"):
        instance_key(
            a_mn=False, b_mn=False, stages=8, smem_limit=smem_limit_for("sm_100a")
        )
    with pytest.raises(ValueError, match="SMEM"):
        instance_key(
            a_mn=False, b_mn=False, block_n=128, stages=10, smem_limit=SMEM_OPT_IN
        )  # 9 is the deepest 24 KiB pipeline in the opt-in
    with pytest.raises(ValueError, match="SMEM"):
        instance_key(a_mn=False, b_mn=False, cta_rows=64, stages=14)
    with pytest.raises(ValueError, match="diagnostic"):
        instance_key(a_mn=False, b_mn=False, diag=("no_mma",))
    with pytest.raises(ValueError, match="hints"):
        instance_key(a_mn=False, b_mn=False, hints=("evict_first", "keep"))
    with pytest.raises(ValueError, match="prefetch"):
        instance_key(a_mn=False, b_mn=False, pf=17)
    key = instance_key(a_mn=False, b_mn=False)
    # 20 fields since round 13 (the raster group width and the TMA L2 promotion left the key for the launch
    # arguments in round 11): pf, hints, f32_v8, quad_store, park, ovl, htail, b_swz, sk_exact, batch_group, store_ef,
    # store_hint, pd, sh, sk_sync (26 fields since round 18), sk_slab, cta1 (28 fields since round 19), cgrp (29 fields
    # since round 21), ovr (30 fields since round 24)
    assert len(key) == 37 and key[11:] == (
        0,
        ("none", "none"),
        False,
        False,
        False,
        False,
        False,
        128,
        False,
        0,
        False,
        "none",
        0,
        0,
        False,
        0,
        False,
        False,
        False,
        False,  # stt (31 fields since round 27)
        0,  # pro (32 fields since round 33)
        False,  # sk_first (33 fields: the round-2 shapes tranche)
        False,  # lw (34 fields: the round-2 shapes tranche)
        False,  # mg (35 fields: the whole-tile tile resolve)
        0,  # sw (36 fields: the paced output stores)
        "",  # form (37 fields: the eight-warp CTA form)
    )
    # the promotion is validated where it is resolved, by the planner
    v = _views("proj", "o_proj", "fwd", "bf16", 257)
    with pytest.raises(ValueError, match="promo"):
        plan_dense_projection_gemm(
            v["A"],
            v["B"],
            v["out"],
            sm_count=148,
            l2_bytes=L2_BYTES,
            _fallback=False,
            promo="l2_512b",
        )


@pytest.mark.parametrize("pairs", [74, 106])  # 148 SMs (B200) / 212 SMs (R200)
def test_stream_k_plan_rules(pairs):
    # --- sk="auto" (the launcher default): a two-way split of a single partial wave ---
    # on an even K every tile has exactly two parts (unit u = tile u // 2, K half u % 2), so tiles that share an
    # A / B panel stay K-synchronised and the fixup reads one slab
    assert stream_k_plan(24, 96, pairs, sk="auto") == (0, 24, 48, 48)
    assert stream_k_plan(pairs // 2, 96, pairs, sk="auto") == (
        0,
        pairs // 2,
        2 * (pairs // 2),
        48,
    )
    # odd K steps: units of ceil(k / 2) over the linearised 10 x 17 steps; 2 x 10 units would leave the last one
    # starting at step 19 x 9 = 171 of 170, so the plan has the 19 that hold a K step
    assert stream_k_plan(10, 17, pairs, sk="auto") == (0, 10, 19, 9)
    assert stream_k_plan(16, 17, pairs, sk="auto") == (0, 16, 31, 9)
    # 15 x 9 < 8 x 17: still two units per tile
    assert stream_k_plan(8, 17, pairs, sk="auto") == (0, 8, 16, 9)
    assert stream_k_plan(10, 2 * SK_MIN_ITERS, pairs, sk="auto") == (
        0,
        10,
        20,
        SK_MIN_ITERS,
    )
    # the halves must fit the pairs, K needs 2 * SK_MIN_ITERS steps, and multi-wave problems keep a
    # data-parallel tail: otherwise every work item is a whole tile
    assert stream_k_plan(pairs // 2 + 1, 96, pairs, sk="auto") == (
        pairs // 2 + 1,
        0,
        0,
        96,
    )
    assert stream_k_plan(10, 2 * SK_MIN_ITERS - 1, pairs, sk="auto") == (
        10,
        0,
        0,
        2 * SK_MIN_ITERS - 1,
    )
    assert stream_k_plan(pairs + 1, 256, pairs, sk="auto") == (pairs + 1, 0, 0, 256)
    assert stream_k_plan(pairs + 10, 256, pairs, sk="auto") == (pairs + 10, 0, 0, 256)
    assert stream_k_plan(pairs, 256, pairs, sk="auto") == (
        pairs,
        0,
        0,
        256,
    )  # exactly one full wave
    assert stream_k_plan(pairs * 3, 96, pairs, sk="auto") == (pairs * 3, 0, 0, 96)
    assert stream_k_plan(0, 96, pairs, sk="auto") == (0, 0, 0, 96)
    # --- sk=False: every tile is a work item ---
    assert stream_k_plan(100, 96, pairs, sk=False) == (100, 0, 0, 96)
    assert stream_k_plan(24, 96, pairs, sk=False) == (24, 0, 0, 96)
    # --- sk=True: the tail tiles linearised over their K steps into up to `pairs` (`max_units`) units ---
    # a whole number of waves: no tail, no stream-K
    assert stream_k_plan(pairs * 3, 96, pairs, sk=True) == (pairs * 3, 0, 0, 96)
    # fewer tiles than pairs: every tile is a tail tile shared by up to `pairs` units of >= SK_MIN_ITERS steps
    num_full, tail, units, iters = stream_k_plan(24, 96, pairs, sk=True)
    assert (num_full, tail) == (0, 24) and tail < units <= pairs
    assert (
        iters >= SK_MIN_ITERS
        and units * iters >= tail * 96
        and (units - 1) * iters < tail * 96
    )
    num_full, tail, units, iters = stream_k_plan(24, 96, pairs, sk=True, max_units=30)
    assert (
        (num_full, tail) == (0, 24)
        and tail < units <= 30
        and iters == -(-24 * 96 // 30)
    )
    assert stream_k_plan(24, 96, pairs, sk=True, max_units=24) == (
        24,
        0,
        0,
        96,
    )  # no more units than tiles
    # a partial last wave
    num_full, tail, units, iters = stream_k_plan(pairs + 10, 256, pairs, sk=True)
    assert (num_full, tail) == (pairs, 10) and units > tail
    # short K: the tail cannot be split into more units than tiles -> whole tiles only
    assert stream_k_plan(pairs + 10, 1, pairs, sk=True) == (pairs + 10, 0, 0, 1)
    # --- sk="tiles": the diagnostic whole-tile unit path ---
    assert stream_k_plan(pairs + 10, 256, pairs, sk="tiles") == (pairs, 10, 10, 256)
    assert stream_k_plan(0, 96, pairs, sk=True) == (0, 0, 0, 96)


@pytest.mark.parametrize("pairs", [74, 76, 106])  # 148 / 152 / 212 SMs
def test_stream_k_plans_give_every_unit_a_k_step(pairs):
    # a unit without a K step never finishes in the pd 2 / sk_slab 2 / stt programs (the work handoff runs in an
    # item's last segment): every linearised policy plans the units that hold one, and nothing more
    for tiles in range(1, 2 * pairs + 3):
        for k_blocks in range(1, 80):
            for sk in ("auto", True):
                num_full, tail, units, ipu = stream_k_plan(tiles, k_blocks, pairs, sk)
                if units:
                    assert (
                        ipu >= 1 and (units - 1) * ipu < tail * k_blocks <= units * ipu
                    )


def test_check_stream_k_plan_refuses_a_plan_outside_the_launch_contract():
    # the dgrad of T = 1024, N = 1032, K = 1000 on 76 CTA pairs: 16 tail tiles of 17 K steps, one partial wave
    E = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16)  # noqa: E731
    kw = dict(sm_count=152, l2_bytes=L2_BYTES, _fallback=False)
    plan, *_ = plan_dense_projection_gemm(
        E(1024, 1032), E(1032, 1000), E(1024, 1000), pd=2, slots=2, epi="tma", **kw
    )
    assert plan.pd == 2 and (plan.tail_tiles, plan.k_blocks) == (16, 17)
    assert (plan.sk_units, plan.iters_per_unit, plan.sk_iters) == (31, 9, 272)
    check_stream_k_plan(plan)
    # the 2 x tail units of the earlier plan: unit 31 starts at K step 279 of 272
    with pytest.raises(ValueError, match="every stream-K unit to hold a K step"):
        check_stream_k_plan(dataclasses.replace(plan, sk_units=32))
    with pytest.raises(ValueError, match="every stream-K unit to hold a K step"):
        check_stream_k_plan(dataclasses.replace(plan, iters_per_unit=0))
    # the same empty unit is legal on a program that hands off at the bottom of the item loop
    check_stream_k_plan(dataclasses.replace(plan, pd=0, sk_units=32))
    # synchronised plans: a collector for every program, at most one per tail tile for the sk_slab 2 / stt ones
    sync = dataclasses.replace(
        plan, pd=0, sk_sync=True, sk_slab=2, sk_units=32, iters_per_unit=9
    )
    check_stream_k_plan(sync)
    with pytest.raises(ValueError, match="collectors <= tail_tiles"):
        check_stream_k_plan(dataclasses.replace(sync, sk_units=33))
    check_stream_k_plan(dataclasses.replace(sync, sk_slab=0, sk_units=33))
    for slab in (0, 2):
        with pytest.raises(ValueError, match="1 <= collectors"):
            check_stream_k_plan(dataclasses.replace(sync, sk_slab=slab, sk_units=16))


def test_planner_defaults_to_auto_stream_k_and_guards_the_slice_counters():
    # pure planner mirror (_fallback=False): synthetic shapes may name instances the export never generated
    # the 24-tile swapped weight gradient (indexer_hw, T = 1001): one partial wave, K = 16 steps -> split.  The
    # round-9 sm_100a rule plans this row's 64-row family (48 pair tiles, which 74 pairs cannot split two ways), so
    # the 128-row family is forced here to keep the split under test
    v = _views("proj", "indexer_hw", "wgrad", "bf16", 1001)
    plan, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=148,
        l2_bytes=L2_BYTES,
        _fallback=False,
        transposed_out=v["transposed"],
        cta_rows=128,
    )
    assert (plan.pair_tiles, plan.k_blocks) == (24, 16)
    assert (plan.num_full, plan.tail_tiles, plan.sk_units, plan.iters_per_unit) == (
        0,
        24,
        48,
        8,
    )
    assert plan.grid == (96, 1, 1) and plan.sk_iters == 24 * 16
    assert plan.ws_f32_elems == (48 + 24) * 2 * 128 * 128
    assert plan.counters_u32 == 24 * 16 and plan.counters_alloc_u32 == 8192
    off, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=148,
        l2_bytes=L2_BYTES,
        _fallback=False,
        transposed_out=v["transposed"],
        cta_rows=128,
        sk=False,
    )
    assert (off.num_full, off.tail_tiles, off.sk_units, off.iters_per_unit) == (
        24,
        0,
        0,
        16,
    )
    assert (
        off.ws_f32_elems == 0
        and off.counters_alloc_u32 == 8192
        and _nopro(off.template) == _nopro(plan.template)
    )
    # stride-0 rows keep these planner-only views tiny: 300 tiles of 256 x 256 on a 1000-SM device (500 pairs)
    # with the linearised policy put 300 tail tiles in flight -> 4800 slice counters exceed SK_DUMMY_BASE
    A = torch.empty(1024, dtype=torch.bfloat16).as_strided((300 * 256, 1024), (0, 1))
    W = torch.empty(1024, dtype=torch.bfloat16).as_strided((256, 1024), (0, 1))
    out = torch.empty(256, dtype=torch.bfloat16).as_strided((300 * 256, 256), (0, 1))
    assert SK_DUMMY_BASE == 4096
    with pytest.raises(ValueError, match="slice-counter budget"):
        plan_dense_projection_gemm(
            A, W.t(), out, sm_count=1000, l2_bytes=L2_BYTES, _fallback=False, sk=True
        )
    plan, *_ = plan_dense_projection_gemm(
        A, W.t(), out, sm_count=1000, l2_bytes=L2_BYTES, _fallback=False
    )  # auto: 2 * 300 > 500 pairs -> whole tiles
    assert (plan.num_full, plan.tail_tiles, plan.sk_units) == (300, 0, 0)


def _views(family, row, op, out_dtype, T, device="cpu"):
    """The training views of one row (empty tensors are enough for the planner)."""
    dt = {"bf16": torch.bfloat16, "f32": torch.float32}[out_dtype]
    if family == "proj":
        K, N = PROJECTION_ROWS[row]
        X = torch.empty(T, K, dtype=torch.bfloat16, device=device)
        W = torch.empty(N, K, dtype=torch.bfloat16, device=device)
        G = torch.empty(T, N, dtype=torch.bfloat16, device=device)
        if op == "fwd":
            return dict(
                A=X,
                B=W.t(),
                out=torch.empty(T, N, dtype=dt, device=device),
                X=X,
                W=W,
                G=G,
            )
        if op == "dgrad":
            return dict(
                A=G, B=W, out=torch.empty(T, K, dtype=dt, device=device), X=X, W=W, G=G
            )
        A, B, transposed = wgrad_views(G, X)
        return dict(
            A=A,
            B=B,
            out=torch.empty(N, K, dtype=dt, device=device),
            transposed=transposed,
            X=X,
            W=W,
            G=G,
        )
    spec = MLA_ROWS[row]
    H, Din, Dout, slot = spec["H"], spec["D_in"], spec["D_out"], spec["act_stride_head"]
    act = torch.empty(T, H, slot, dtype=torch.bfloat16, device=device)[..., :Din]
    if spec["weight_layout"] == "in_out":
        weight = torch.empty(H, Din, Dout, dtype=torch.bfloat16, device=device)
        w_in_out = weight
    else:
        weight = torch.empty(H, Dout, Din, dtype=torch.bfloat16, device=device)
        w_in_out = weight.transpose(1, 2)
    d_out = torch.empty(T, H, Dout, dtype=torch.bfloat16, device=device)
    if op == "fwd":
        out = torch.empty(T, H, Dout, dtype=dt, device=device).permute(1, 0, 2)
        return dict(
            A=act.permute(1, 0, 2),
            B=w_in_out,
            out=out,
            act=act,
            weight=weight,
            d_out=d_out,
        )
    if op == "dgrad":
        buf = torch.empty(T, H, slot, dtype=dt, device=device)
        return dict(
            A=d_out.permute(1, 0, 2),
            B=w_in_out.transpose(1, 2),
            out=buf[..., :Din].permute(1, 0, 2),
            buf=buf,
            act=act,
            weight=weight,
            d_out=d_out,
        )
    if spec["weight_layout"] == "in_out":
        A, B = act.permute(1, 2, 0), d_out.permute(1, 0, 2)
    else:
        A, B = d_out.permute(1, 2, 0), act.permute(1, 0, 2)
    return dict(
        A=A,
        B=B,
        out=torch.empty_like(weight, dtype=dt),
        act=act,
        weight=weight,
        d_out=d_out,
    )


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize(
    "T", [2049, 1001]
)  # K = T above / at or below the TMA-store epilogue bound
def test_projection_rows_plan_like_the_cake_launcher(sm_count, T):
    for row, (K, N) in PROJECTION_ROWS.items():
        for op in ("fwd", "dgrad", "wgrad"):
            for out_dtype in ("bf16", "f32") if op != "fwd" else ("bf16",):
                v = _views("proj", row, op, out_dtype, T)
                plan, a_desc, b_desc, out3 = plan_dense_projection_gemm(
                    v["A"],
                    v["B"],
                    v["out"],
                    sm_count=sm_count,
                    l2_bytes=L2_BYTES,
                    transposed_out=v.get("transposed", False),
                    arch=ARCH_OF_SM[sm_count],
                )
                rule = _rule_of(plan, sm_count)
                _assert_mirrors_cake(
                    plan,
                    _expected(EXPECTED_TEMPLATES, (row, op, out_dtype), T, sm_count),
                    (row, op, out_dtype, T, sm_count),
                    ARCH_OF_SM[sm_count],
                    rule,
                )
                # round 19 (Cake W3): under the single-CTA form of a ``cta1`` rule the work items are CTA tiles, the
                # concurrent units are the SMs and the grid is one CTA per work item (cluster dims 1)
                cg = 1 if plan.cta1 else CTA_GROUP
                assert plan.sm_pairs == sm_count // cg
                assert plan.grid == ((plan.num_full + plan.sk_units) * cg, 1, 1)
                assert plan.m_tiles % cg == 0
                assert plan.k_blocks == -(-plan.K // BLOCK_K)
                assert plan.pair_tiles == plan.L * (plan.m_tiles // cg) * plan.n_tiles
                assert a_desc.stride(2) == 1 and b_desc.stride(2) == 1
                # the knob defaults of the launcher (stage depth by tile shape, raster group, working-set-gated
                # hints) unless the row's measured rule (ROW_RULES, round 2) pins a knob; a plan that landed on the
                # nearest generated knob variant (knob_fallback) carries that variant's knobs instead
                mirrored = not plan.knob_fallback
                assert not mirrored or plan.cta_rows == rule.get("cta_rows", 128)
                assert not mirrored or plan.cta1 == bool(rule.get("cta1", False))
                assert _nopro(plan.template).endswith("_c1") == plan.cta1
                # round 21 (Cake W3): the column-grouped raster of a ``cgrp`` rule (``_cg`` templates; the plan
                # arithmetic is the pair form's, the hint gate takes the column raster's wave working set)
                assert not mirrored or plan.cgrp == bool(rule.get("cgrp", False))
                assert _nopro(plan.template).endswith(("_cg", "_cg_or")) == plan.cgrp
                # round 24 (the round-20 Cake W3 knob, ported in round 23): the row-half release of an ``ovr`` rule (trailing
                # ``_or`` templates, after ``_cg``; tall 256-column tiles only, the plan arithmetic is unchanged)
                assert not mirrored or plan.ovr == bool(rule.get("ovr", False))
                assert _nopro(plan.template).endswith("_or") == plan.ovr
                assert not plan.cta1 or plan.cta_rows == 128
                assert not mirrored or (
                    plan.pf == rule.get("pf", 0)
                    and plan.promo == rule.get("promo", "none")
                )
                assert not mirrored or plan.group_m == rule.get(
                    "group_m",
                    default_group_m(
                        plan.a_mn,
                        plan.b_mn,
                        plan.m_tiles,
                        plan.pair_tiles,
                        plan.sm_pairs,
                    ),
                )
                # the raster group width is a launch parameter since round 11: never a template suffix
                # ("_gemm" is not one)
                assert re.search(r"_g\d+", _nopro(plan.template)) is None
                # round 13: the narrow MN-major B panels (``b_swz`` 64 / 32) have smaller stages and take the deepest
                # pipeline that fits the opt-in, like the 64-row family
                assert not mirrored or plan.b_swz == (
                    rule.get("b_swz", 128) if plan.b_mn else 128
                )
                assert plan.stages == rule.get(
                    "stages",
                    default_stages(
                        plan.slots,
                        plan.cta_rows,
                        plan.block_n,
                        plan.b_mn,
                        plan.b_swz,
                        cg,
                    ),
                )
                if (
                    "stages" not in rule
                    and plan.cta_rows == 128
                    and plan.b_swz == 128
                    and not plan.cta1
                ):
                    assert (
                        plan.stages
                        == {
                            (256, 0): 7,
                            (256, 1): 6,
                            (256, 2): 5,
                            (192, 0): 7,
                            (192, 1): 6,
                            (192, 2): 5,
                            (128, 0): 9,
                            (128, 1): 8,
                            (128, 2): 6,
                        }[(plan.block_n, plan.slots)]
                    )
                assert plan.l2_bytes == L2_BYTES
                assert plan.wave_working_set_bytes == (
                    wave_working_set_cgrp if plan.cgrp else wave_working_set
                )(plan.m_tiles, plan.n_tiles, plan.K, plan.group_m, plan.sm_pairs)
                assert not mirrored or plan.hints == tuple(
                    rule.get(
                        "hints",
                        (default_hints_cgrp if plan.cgrp else default_hints)(
                            plan.a_mn,
                            plan.b_mn,
                            plan.m_tiles,
                            plan.n_tiles,
                            plan.K,
                            plan.group_m,
                            plan.sm_pairs,
                            L2_BYTES,
                        ),
                    )
                )
                if mirrored and "hints" not in rule:
                    if plan.wave_working_set_bytes <= L2_BYTES:
                        assert plan.hints == ("none", "none")
                    if not (plan.a_mn and plan.b_mn) and plan.n_tiles > 1:
                        assert plan.hints == ("none", "none")
                assert (plan.n_tiles == 1) == (plan.N <= 256)
                # stream-K tail policies in the Cake launcher's order (round 13): an exact ``sk_exact`` p-way split
                # of every tail tile (one unit per part, identical K ranges on every tile) pre-empts the deterministic
                # half-height tail wave (``htail``: the tail tiles become 2 x tail whole-K items, no partial slabs),
                # which pre-empts a measured ``sk_parts`` rule (round 4, L20: every tail tile split p ways when the
                # p * tail units fit the CTA pairs with SK_MIN_ITERS K steps each); otherwise "auto" = a two-part
                # K-aligned split of a single partial wave, else whole tiles
                tail = (
                    plan.pair_tiles % plan.sm_pairs
                    if plan.pair_tiles > plan.sm_pairs
                    else plan.pair_tiles
                )
                exact = int(rule.get("sk_exact") or 0)
                exact_plan = (
                    sk_exact_plan(plan.pair_tiles, plan.k_blocks, plan.sm_pairs, exact)
                    if exact
                    else None
                )
                # round 18 (Cake W2): the synchronised stream-K plan of an ``sk_sync`` rule (main units + collectors, the
                # rule's margin ``sk_sync_m``) yields to ``sk_exact`` and pre-empts ``htail`` / ``sk_parts``
                sync_plan = (
                    sk_sync_plan(
                        plan.pair_tiles,
                        plan.k_blocks,
                        plan.sm_pairs,
                        None,
                        int(rule.get("sk_sync_m", 0)),
                    )
                    if rule.get("sk_sync") and exact_plan is None
                    else None
                )
                htail = (
                    bool(rule.get("htail"))
                    and (
                        plan.cta_rows == 256
                        or (
                            plan.cta_rows == 128
                            and plan.block_n == 256
                            and not plan.transposed_out
                            and int(rule.get("pd", 0)) == 0
                            and not int(rule.get("pf", 0))
                        )
                    )
                    and exact_plan is None
                    and sync_plan is None
                )
                # the fit of the half items selects the plan, never the program the rule names
                htail_plan = htail and 0 < 2 * tail <= plan.sm_pairs
                parts = rule.get("sk_parts") if sync_plan is None else None
                sk_rule = (
                    sk_parts_plan(plan.pair_tiles, plan.k_blocks, plan.sm_pairs, parts)
                    if parts and exact_plan is None
                    else ("auto", None)
                )
                assert plan.sk_exact == (exact if exact_plan is not None else 0)
                assert plan.htail == htail
                assert plan.htail_plan == htail_plan
                assert plan.sk_sync == (sync_plan is not None)
                assert ("_sks" in _nopro(plan.template)) == plan.sk_sync
                # round 19 (Cake W2): the slab path of a rule's synchronised plan, with the launcher's yield rules for
                # a rule-derived value (3 needs the fp32 TMA-store / transposed register epilogue, 2 needs the eight
                # warp slabs to fit the mainloop stage ring); 0 on every other plan
                sk_slab = int(rule.get("sk_slab", 0)) if sync_plan is not None else 0
                if sk_slab == 3 and (
                    not plan.out_f32 or (plan.epi == "reg" and not plan.transposed_out)
                ):
                    sk_slab = 2
                if sk_slab == 2 and EPI_WARPS * epi_cols(
                    plan.block_n, plan.cta_rows
                ) * 128 > plan.stages * (
                    plan.cta_rows * BLOCK_K * 2
                    + b_stage_bytes(plan.b_mn, plan.block_n, plan.b_swz)
                ):
                    sk_slab = 0
                assert plan.sk_slab == sk_slab
                assert (f"_sb{sk_slab}" in _nopro(plan.template)) == bool(sk_slab)
                if exact_plan is not None:
                    assert (
                        plan.num_full,
                        plan.tail_tiles,
                        plan.sk_units,
                        plan.iters_per_unit,
                    ) == exact_plan
                    # exactly p units of ceil(k_blocks / p) K steps per tail tile, a non-empty last part, no drift;
                    # the unit count may exceed the pairs (the persistent CTAs take several units)
                    assert (plan.num_full, plan.tail_tiles, plan.sk_units) == (
                        plan.pair_tiles - tail,
                        tail,
                        exact * tail,
                    )
                    assert (
                        plan.iters_per_unit
                        == -(-plan.k_blocks // exact)
                        >= SK_MIN_ITERS
                    )
                    assert (exact - 1) * plan.iters_per_unit < plan.k_blocks
                    assert plan.sk_iters == exact
                elif sync_plan is not None:
                    # n_t main units over [0, s) + C = min(pairs - n_t, n_t) collectors over the leftovers [s, K): one
                    # wave of n_t + C units after the whole tiles, both parts at least SK_MIN_ITERS steps, one slab per
                    # tail tile, s in iters_per_unit and n_t in sk_iters
                    assert (
                        plan.num_full,
                        plan.tail_tiles,
                        plan.sk_units,
                        plan.iters_per_unit,
                        plan.sk_collectors,
                    ) == sync_plan
                    assert (plan.num_full, plan.tail_tiles) == (
                        plan.pair_tiles - tail,
                        tail,
                    )
                    assert plan.sk_collectors == min(plan.sm_pairs - tail, tail) >= 1
                    assert plan.sk_units == tail + plan.sk_collectors <= plan.sm_pairs
                    q = -(-tail // plan.sk_collectors)
                    assert plan.iters_per_unit == -(
                        -(q * plan.k_blocks + int(rule.get("sk_sync_m", 0))) // (q + 1)
                    )
                    assert (
                        SK_MIN_ITERS
                        <= plan.iters_per_unit
                        <= plan.k_blocks - SK_MIN_ITERS
                    )
                    assert plan.sk_iters == tail
                    assert plan.ws_f32_elems == tail * 2 * plan.cta_rows * plan.block_n
                elif htail_plan:
                    # 2 x tail half-height items of whole K: no partial slabs, no fixup, no counters in use beyond
                    # the tail tiles; the tall tile may exceed one wave of pairs
                    assert (
                        plan.num_full,
                        plan.tail_tiles,
                        plan.sk_units,
                        plan.iters_per_unit,
                    ) == (plan.pair_tiles - tail, tail, 2 * tail, plan.k_blocks)
                    assert plan.sk_iters == plan.sk_units * plan.k_blocks
                    assert plan.ws_f32_elems == 0
                elif sk_rule[0] is True:
                    assert 0 < parts * tail <= plan.sm_pairs
                    assert plan.k_blocks >= parts * SK_MIN_ITERS
                    assert (
                        plan.num_full,
                        plan.tail_tiles,
                        plan.sk_units,
                        plan.iters_per_unit,
                    ) == stream_k_plan(
                        plan.pair_tiles, plan.k_blocks, plan.sm_pairs, *sk_rule
                    )
                    # p * tail linearised units of ceil(k_blocks / p) K steps; when p does not divide k_blocks
                    # the unit count is re-derived from that step count and lands below p * tail (the kernel's
                    # two-segment units absorb the drift)
                    assert (plan.num_full, plan.tail_tiles) == (
                        plan.pair_tiles - tail,
                        tail,
                    )
                    assert plan.iters_per_unit == -(-plan.k_blocks // parts)
                    assert tail < plan.sk_units <= parts * tail
                    assert plan.sk_units == -(
                        -(tail * plan.k_blocks) // plan.iters_per_unit
                    )
                elif plan.sk_units:
                    assert (
                        plan.pair_tiles <= plan.sm_pairs
                        and 2 * plan.tail_tiles <= plan.sm_pairs
                    )
                    assert plan.k_blocks >= 2 * SK_MIN_ITERS
                    # the units of ceil(k_blocks / 2) K steps that hold one: 2 x tail on an even k_blocks
                    assert plan.iters_per_unit == -(-plan.k_blocks // 2)
                    assert (plan.num_full, plan.tail_tiles, plan.sk_units) == (
                        0,
                        plan.pair_tiles,
                        -(-(plan.pair_tiles * plan.k_blocks) // plan.iters_per_unit),
                    )
                if plan.sk_units and not htail and sync_plan is None:
                    assert (
                        plan.ws_f32_elems
                        == (plan.sk_units + plan.tail_tiles)
                        * 2
                        * plan.cta_rows
                        * plan.block_n
                    )
                    if exact_plan is None:
                        assert plan.sk_iters == plan.tail_tiles * plan.k_blocks
                elif not plan.sk_units:
                    assert (plan.num_full, plan.tail_tiles, plan.iters_per_unit) == (
                        plan.pair_tiles,
                        0,
                        plan.k_blocks,
                    )
                    # (the single-CTA form carries no tail policy: whole tiles whatever the wave count)
                    assert (
                        plan.cta1
                        or plan.pair_tiles > plan.sm_pairs
                        or 2 * plan.pair_tiles > plan.sm_pairs
                        or plan.k_blocks < 2 * SK_MIN_ITERS
                    )
                    assert plan.ws_f32_elems == 0
                assert plan.tail_tiles * 16 <= SK_DUMMY_BASE
                assert (
                    plan.counters_alloc_u32
                    == max(8192, plan.tail_tiles * 16)
                    >= SK_DUMMY_BASE + 16 * 32
                )
                if op == "fwd":
                    assert (
                        (plan.M, plan.N, plan.K) == (T, N, K)
                        and not plan.a_mn
                        and not plan.b_mn
                    )
                elif op == "dgrad":
                    assert (
                        (plan.M, plan.N, plan.K) == (T, K, N)
                        and not plan.a_mn
                        and plan.b_mn
                    )
                elif wgrad_swapped(
                    N, K
                ):  # work-minimising swap (round 3): X.T @ G with the transposed store
                    assert (plan.M, plan.N, plan.K) == (K, N, T) and plan.transposed_out
                else:
                    assert (
                        (plan.M, plan.N, plan.K) == (N, K, T)
                        and plan.a_mn
                        and plan.b_mn
                    )
                assert not mirrored or plan.block_n == rule.get(
                    "block_n", default_block_n(plan.N, plan.b_mn)
                )


# MLA rows: in_out weights are MN-major as B of the forward and K-major as B of the input gradient
# (out_in the other way round); the weight gradient contracts over T with both operands MN-major.
MLA_T = (2049, 1001, 33)


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("T", MLA_T)
def test_mla_rows_plan_as_batched_views(T, sm_count):
    for row in MLA_ROWS:
        for op in ("fwd", "dgrad", "wgrad"):
            v = _views("mla", row, op, "bf16", T)
            plan, a_desc, b_desc, out3 = plan_dense_projection_gemm(
                v["A"],
                v["B"],
                v["out"],
                sm_count=sm_count,
                l2_bytes=L2_BYTES,
                arch=ARCH_OF_SM[sm_count],
            )
            assert plan.L == 64 and out3.shape[0] == 64
            _assert_mirrors_cake(
                plan,
                _expected(MLA_TEMPLATES, (row, op), T, sm_count),
                (row, op, T, sm_count),
                ARCH_OF_SM[sm_count],
            )
            # the swapped plans (weight gradients; the tiny-T forward / input gradient) store transposed: the register
            # epilogue's ``_t`` templates or, since round 14, the transposed TMA-store epilogue's ``_t_tma<slots>`` ones
            assert plan.transposed_out == _transposed_template(_nopro(plan.template))
            rule = _rule_of(plan, sm_count)
            mirrored = not plan.knob_fallback
            assert not mirrored or plan.hints == tuple(
                rule.get(
                    "hints",
                    default_hints(
                        plan.a_mn,
                        plan.b_mn,
                        plan.m_tiles,
                        plan.n_tiles,
                        plan.K,
                        plan.group_m,
                        plan.sm_pairs,
                        L2_BYTES,
                    ),
                )
            )
            assert not mirrored or plan.group_m == rule.get(
                "group_m",
                default_group_m(
                    plan.a_mn, plan.b_mn, plan.m_tiles, plan.pair_tiles, plan.sm_pairs
                ),
            )
    # qabs dgrad writes the first 192 columns of a 256-wide slot of a token-major [T, H, 256] buffer:
    # the [H, T, D] output view has batch stride 256 (heads) and row stride 64 * 256 (tokens)
    v = _views("mla", "qabs", "dgrad", "bf16", 33)
    plan, _, _, out3 = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], sm_count=148, l2_bytes=L2_BYTES
    )
    assert (out3.stride(0), out3.stride(1), out3.stride(2)) == (256, 64 * 256, 1)
    # A = d_out [H, T, 512] K-major, B = the in_out weight transposed ([H, 512, 192], k contiguous) K-major;
    # N = 192: BLOCK_N = 192 (one exact tile); the 96-column CTA halves cannot use the 64-column TMA-store chunks,
    # so the row runs the register epilogue (round 3; K = 512 <= 1024 would otherwise pick the TMA-store epilogue)
    _assert_mirrors_cake(
        plan,
        _expected(MLA_TEMPLATES, ("qabs", "dgrad"), 33, 148),
        ("qabs", "dgrad", 33, 148),
        "sm_100a",
    )
    assert plan.block_n == default_block_n(192, False) == 192
    assert _nopro(plan.template).startswith("dense_proj_gemm_kk_n192") and not _nopro(
        plan.template
    ).endswith("_tma1")


def test_operand_view_classification_and_rejections():
    A = torch.empty(4, 16, 64, dtype=torch.bfloat16)
    mn, desc = operand_view(A, "A", k_axis=2)
    assert not mn and desc.shape == A.shape
    # the transposed view keeps K contiguous: with K on axis 1 it is still K-major, with the
    # contraction on axis 2 (stride 64) and M unit-strided it is MN-major and desc = [L, M, K]
    mn, desc = operand_view(A.transpose(1, 2), "A", k_axis=1)
    assert not mn and tuple(desc.shape) == (4, 16, 64)
    mn, desc = operand_view(A.transpose(1, 2), "A", k_axis=2)
    assert mn and tuple(desc.shape) == (4, 16, 64)
    with pytest.raises(ValueError, match="unit stride"):
        operand_view(
            torch.empty(4, 16, 64, dtype=torch.bfloat16)[:, :, ::2], "A", k_axis=2
        )
    with pytest.raises(ValueError, match="multiples of 8"):
        operand_view(
            torch.empty(1, 16, 68, dtype=torch.bfloat16)[:, :, :64], "A", k_axis=2
        )
    with pytest.raises(ValueError, match="16-byte aligned"):
        operand_view(
            torch.empty(1, 16, 72, dtype=torch.bfloat16)[:, :, 4:68], "A", k_axis=2
        )
    A2 = torch.empty(16, 64, dtype=torch.bfloat16)
    B2 = torch.empty(64, 24, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="bf16"):
        plan_dense_projection_gemm(
            A2.float(), B2, torch.empty(16, 24), sm_count=148, l2_bytes=L2_BYTES
        )
    with pytest.raises(ValueError, match="bf16 or fp32"):
        plan_dense_projection_gemm(
            A2,
            B2,
            torch.empty(16, 24, dtype=torch.float16),
            sm_count=148,
            l2_bytes=L2_BYTES,
        )
    with pytest.raises(ValueError, match=r"B must be \[L=1, K=64, N\]"):
        plan_dense_projection_gemm(
            A2,
            torch.empty(32, 24, dtype=torch.bfloat16),
            torch.empty(16, 24),
            sm_count=148,
            l2_bytes=L2_BYTES,
        )
    with pytest.raises(ValueError, match="out must be"):
        plan_dense_projection_gemm(
            A2, B2, torch.empty(24, 16), sm_count=148, l2_bytes=L2_BYTES
        )
    with pytest.raises(ValueError, match="multiple of 8"):
        plan_dense_projection_gemm(
            A2,
            torch.empty(64, 20, dtype=torch.bfloat16),
            torch.empty(16, 20),
            sm_count=148,
            l2_bytes=L2_BYTES,
        )
    with pytest.raises(ValueError, match="unit inner stride"):
        plan_dense_projection_gemm(
            A2, B2, torch.empty(24, 16).t(), sm_count=148, l2_bytes=L2_BYTES
        )
    # transposed output: [N, M] with unit stride on m
    plan, *_ = plan_dense_projection_gemm(
        A2,
        B2,
        torch.empty(24, 16),
        sm_count=148,
        l2_bytes=L2_BYTES,
        transposed_out=True,
    )
    assert (
        plan.transposed_out
        and plan.epi == "reg"
        and _nopro(plan.template).endswith("_t")
    )


def test_wgrad_swap_rule():
    # work-minimising orientation (round 3): the swapped GEMM X.T @ G with the transposed store runs when its padded
    # tile work is smaller - the small-N rows (N = 32, K = 6144) and the 576-wide kv_a row; equal work keeps G.T @ X
    X = torch.empty(100, 6144, dtype=torch.bfloat16)
    G = torch.empty(100, 32, dtype=torch.bfloat16)
    assert wgrad_swapped(32, 6144)
    A, B, transposed = wgrad_views(G, X)
    assert transposed and A.shape == (6144, 100) and B is G
    G = torch.empty(100, 576, dtype=torch.bfloat16)
    assert wgrad_swapped(576, 6144)  # 3 x 192 columns over K instead of 256 + 256 + 64
    A, B, transposed = wgrad_views(G, X)
    assert transposed and A.shape == (6144, 100) and B is G
    G = torch.empty(100, 256, dtype=torch.bfloat16)
    assert not wgrad_swapped(256, 6144)  # a tie keeps the direct route
    A, B, transposed = wgrad_views(G, X)
    assert not transposed and A.shape == (256, 100) and B is X
    assert not wgrad_swapped(6144, 2048) and not wgrad_swapped(
        32, 64
    )  # large N; a 32 x 64 tie


def test_batched_small_m_swap_and_row_rules():
    # pure planner mirror (_fallback=False): synthetic rules may name instances the export never generated
    # MLA weight gradient dW[h] = A_h^T B_h with M = 192, N = 512 (K = T = 1001): the planner runs the transposed GEMM
    # (A' = B^T [H, 512, T], B' = A^T [H, T, 192], transposed store into the caller's [H, 192, 512] view)
    H, T, M, N = 4, 1001, 192, 512
    Q = torch.empty(T, H, M, dtype=torch.bfloat16)
    dL = torch.empty(T, H, N, dtype=torch.bfloat16)
    out = torch.empty(H, M, N, dtype=torch.bfloat16)
    assert swap_small_m(H, M, N, False) and not swap_small_m(1, M, N, False)
    assert not swap_small_m(H, 512, 192, False) and not swap_small_m(H, M, N, True)
    plan, a_desc, b_desc, out3 = plan_dense_projection_gemm(
        Q.permute(1, 2, 0),
        dL.permute(1, 0, 2),
        out,
        sm_count=148,
        l2_bytes=L2_BYTES,
        _fallback=False,
    )
    assert (plan.L, plan.M, plan.N, plan.K) == (H, N, M, T)
    assert (
        plan.a_mn and plan.b_mn and plan.transposed_out and plan.block_n == 192
    )  # N' = 192: one exact tile
    assert _nopro(plan.template).startswith(
        "dense_proj_gemm_nn_n192"
    ) and _transposed_template(_nopro(plan.template))
    # round 14: the sm_100a qabs weight-gradient rule puts the swapped tile on the tall transposed TMA-store family
    assert _nopro(plan.template) == "dense_proj_gemm_nn_n192_m256_t_tma1"
    assert plan.epi == "tma" and plan.cta_rows == 256 and plan.slots == 1
    assert tuple(out3.shape) == (H, M, N)
    # the rule table: exact (N, K, M) first, then the ragged-M (N, K, None) key, then the ragged-K (N, None, M) key
    ident = ("sm_100a", True, True, False, True, True)
    saved = dict(ROW_RULES)
    try:
        ROW_RULES.clear()
        ROW_RULES[ident + (M, None, N)] = {"cta_rows": 256}
        assert row_rule("sm_100a", True, True, False, True, True, M, T, N) == {
            "pro": 7,
            "cta_rows": 256,
        }
        assert row_rule("sm_107a", True, True, False, True, True, M, T, N) == {"pro": 7}
        ROW_RULES[ident + (M, T, None)] = {"group_m": 8}
        assert row_rule("sm_100a", True, True, False, True, True, M, T, N) == {
            "pro": 7,
            "group_m": 8,
        }
        ROW_RULES[ident + (M, T, N)] = {"group_m": 4}
        assert row_rule("sm_100a", True, True, False, True, True, M, T, N) == {
            "pro": 7,
            "group_m": 4,
        }
        # a rule fills only the knobs the caller left unset
        ROW_RULES.clear()
        ROW_RULES[ident + (M, None, N)] = {"cta_rows": 256, "group_m": 8}
        ruled, *_ = plan_dense_projection_gemm(
            Q.permute(1, 2, 0),
            dL.permute(1, 0, 2),
            out,
            sm_count=148,
            l2_bytes=L2_BYTES,
            _fallback=False,
            arch="sm_100a",
        )
        # group_m is a launch parameter (round 11): it is planned from the rule but is not a template axis
        assert (
            ruled.cta_rows == 256
            and ruled.group_m == 8
            and "_m256" in _nopro(ruled.template)
            and re.search(r"_g\d+", _nopro(ruled.template)) is None
        )
        pinned, *_ = plan_dense_projection_gemm(
            Q.permute(1, 2, 0),
            dL.permute(1, 0, 2),
            out,
            sm_count=148,
            l2_bytes=L2_BYTES,
            _fallback=False,
            arch="sm_100a",
            cta_rows=128,
        )
        assert pinned.cta_rows == 128 and pinned.group_m == 8
        # a caller-forced tile family (round 9) keeps the rule's other knobs but drops the family-bound ones
        # (stages, slots, epi); the 64-row Layout-B family plans like any other
        ROW_RULES.clear()
        ROW_RULES[ident + (M, None, N)] = {
            "cta_rows": 256,
            "group_m": 8,
            "stages": 3,
            "epi": "reg",
        }
        ruled, *_ = plan_dense_projection_gemm(
            Q.permute(1, 2, 0),
            dL.permute(1, 0, 2),
            out,
            sm_count=148,
            l2_bytes=L2_BYTES,
            _fallback=False,
            arch="sm_100a",
        )
        assert (ruled.cta_rows, ruled.stages, ruled.group_m) == (256, 3, 8)
        assert _nopro(ruled.template) == "dense_proj_gemm_nn_n192_m256_t_s3"
        for rows, template in (
            (128, "dense_proj_gemm_nn_n192_t"),
            (64, "dense_proj_gemm_nn_n192_m64_t"),
        ):
            forced, *_ = plan_dense_projection_gemm(
                Q.permute(1, 2, 0),
                dL.permute(1, 0, 2),
                out,
                sm_count=148,
                l2_bytes=L2_BYTES,
                _fallback=False,
                arch="sm_100a",
                cta_rows=rows,
            )
            assert (forced.cta_rows, forced.group_m, forced.epi) == (rows, 8, "reg")
            assert forced.stages == default_stages(
                forced.slots, rows, forced.block_n, forced.b_mn
            )
            assert _nopro(forced.template) == template
    finally:
        ROW_RULES.clear()
        ROW_RULES.update(saved)


def _ht_rows_without_half_fit(sm_count, arch):
    """``(row, op, T, plan)`` of the ``_ht`` rule rows whose half-tail items do not fit ``sm_count``'s pairs, planned
    with a caller-forced ``sk=True`` (the planner's view: empty CPU tensors)."""
    found = []
    for row in PROJECTION_ROWS:
        for op in ("fwd", "dgrad", "wgrad"):
            for T in range(4096, 16385, 1024):
                v = _views("proj", row, op, "bf16", T)
                kw = dict(
                    sm_count=sm_count,
                    l2_bytes=L2_BYTES,
                    arch=arch,
                    transposed_out=v.get("transposed", False),
                    _fallback=False,
                )
                try:
                    plan, *_ = plan_dense_projection_gemm(
                        v["A"], v["B"], v["out"], sk=True, **kw
                    )
                except (ValueError, NotImplementedError):
                    continue
                if "_ht" not in plan.template or plan.htail_plan:
                    continue
                found.append((row, op, T, plan))
    return found


@pytest.mark.parametrize(
    "arch, sm_count", [("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 212)]
)
def test_ht_program_never_plans_linearised_stream_k_units(arch, sm_count):
    # The ``_ht`` program reads every unit past ``num_full`` as a half-height item of full K; a linearised stream-K
    # plan (a caller-forced ``sk=True`` / ``sk="tiles"`` on an ``_ht`` rule row whose half items do not fit the pairs)
    # would be misread.  The planner keeps the program and runs the plain whole-tile plan there instead.
    found = _ht_rows_without_half_fit(sm_count, arch)
    assert found, "no _ht rule row without the half-tail fit in the sweep"
    for row, op, T, plan in found:
        assert (plan.num_full, plan.tail_tiles, plan.sk_units, plan.iters_per_unit) == (
            plan.pair_tiles,
            0,
            0,
            plan.k_blocks,
        ), (row, op, T, plan.template)
        # the same row with the default policy plans identically (auto never splits where the half items do not fit)
        v = _views("proj", row, op, "bf16", T)
        auto, *_ = plan_dense_projection_gemm(
            v["A"],
            v["B"],
            v["out"],
            sm_count=sm_count,
            l2_bytes=L2_BYTES,
            arch=arch,
            transposed_out=v.get("transposed", False),
            _fallback=False,
        )
        assert auto.template == plan.template and auto.sk_units == 0
    # and where the half items fit, a forced ``sk`` still yields the half-tail plan of the program
    fits = [
        p
        for row in PROJECTION_ROWS
        for op in ("fwd", "dgrad", "wgrad")
        for T in (4096, 8192, 12288)
        for p in [_plan_or_none(row, op, T, sm_count, arch, sk=True)]
        if p is not None and "_ht" in p.template and p.htail_plan
    ]
    for p in fits:
        assert p.sk_units == 2 * p.tail_tiles and p.iters_per_unit == p.k_blocks


def _plan_or_none(row, op, T, sm_count, arch, **knobs):
    v = _views("proj", row, op, "bf16", T)
    try:
        plan, *_ = plan_dense_projection_gemm(
            v["A"],
            v["B"],
            v["out"],
            sm_count=sm_count,
            l2_bytes=L2_BYTES,
            arch=arch,
            transposed_out=v.get("transposed", False),
            _fallback=False,
            **knobs,
        )
    except (ValueError, NotImplementedError):
        return None
    return plan


def test_ht_program_with_forced_stream_k_matches_fp64_reference():
    # end to end on the device: the first ``_ht`` rule row without the half-tail fit, launched with ``sk=True``
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    arch = arch_for(torch.device("cuda"))
    found = [f for f in _ht_rows_without_half_fit(_sm_count(), arch) if f[1] != "wgrad"]
    if not found:
        pytest.skip(
            f"no _ht rule row without the half-tail fit on {arch} with {_sm_count()} SMs"
        )
    row, op, T, _ = found[0]
    v = _make_inputs("proj", row, op, "bf16", T, seed=SEED + 7)
    _require_program(_plan_template(v, sk=True))
    prepared = prepare_dense_projection_gemm(v["A"], v["B"], v["out"], sk=True)
    assert "_ht" in prepared.plan.template and prepared.plan.sk_units == 0
    out = prepared.launch()
    torch.cuda.synchronize()
    expected = torch.matmul(v["A"].double(), v["B"].double())
    _check(out, expected)


@pytest.mark.parametrize("pairs", [74, 106])  # 148 SMs (B200) / 212 SMs (R200)
def test_sk_parts_rule(pairs):
    # Cake round 4 (L20): a measured ``sk_parts = p`` row rule splits every tail tile p ways when the units
    # fit the CTA pairs and keep SK_MIN_ITERS K steps each; otherwise the row keeps the ``auto`` policy.
    assert sk_parts_plan(24, 254, pairs, 3) == (True, 72)
    assert sk_parts_plan(24, 254, pairs, 2) == (True, 48)
    assert stream_k_plan(24, 254, pairs, *sk_parts_plan(24, 254, pairs, 3)) == (
        0,
        24,
        72,
        85,
    )
    assert sk_parts_plan(72, 254, pairs, 2) == (
        "auto",
        None,
    )  # 2 * 72 units do not fit either pair count
    assert sk_parts_plan(24, 3 * SK_MIN_ITERS - 1, pairs, 3) == (
        "auto",
        None,
    )  # too few K steps per unit
    assert sk_parts_plan(24, 254, pairs, 1) == ("auto", None)
    assert sk_parts_plan(pairs, 254, pairs, 2) == (
        "auto",
        None,
    )  # a full wave has no tail
    if pairs == 106:
        # indexer_q wgrad on R200: 128-wide tiles -> 256 pair tiles (tail 44, p = 2 -> 88 units);
        # 256-wide tiles -> 128 pair tiles (tail 22, p = 3 -> 66 units)
        assert sk_parts_plan(256, 254, pairs, 2) == (True, 88)
        assert stream_k_plan(256, 254, pairs, True, 88) == (212, 44, 88, 127)
        assert sk_parts_plan(128, 254, pairs, 3) == (True, 66)
        assert stream_k_plan(128, 254, pairs, True, 66) == (106, 22, 66, 85)
    else:
        # on B200 the 128-tile row's tail of 54 admits no split: the rule falls back to auto (whole tiles)
        assert sk_parts_plan(128, 254, pairs, 3) == ("auto", None)
    # the sm_107a rules adopted in round 4 (round 8 adds the 4-tile raster group to the bf16 indexer_q wgrad rule) ...
    # (round 9 moves the indexer_q weight gradients onto 256 x 160 tiles and the N = 128 swapped weight gradient
    # onto the 64-row family; round 13 adds the 64-byte MN-major B panels (b_swz 64) to the indexer_q weight gradients
    # and replaces the sk_parts split of the indexer_k / indexer_hw weight gradients by the exact p-way split (sk_exact))
    assert ROW_RULES[
        ("sm_107a", True, True, False, False, False, 2048, None, 4096)
    ] == {
        "block_n": 160,
        "cta_rows": 256,
        "group_m": 4,
        "sk_parts": 2,
        "b_swz": 64,
        "pro": 15,
    }
    # round 17 (L61): the fp32 weight gradient leaves through the evict-first 8-wide register stores
    assert ROW_RULES[("sm_107a", True, True, True, False, False, 2048, None, 4096)] == {
        "block_n": 160,
        "cta_rows": 256,
        "sk_parts": 3,
        "b_swz": 64,
        "epi": "reg",
        "f32_v8": True,
        "store_ef": True,
        "pro": 15,
    }
    for f32 in (False, True):
        assert ROW_RULES[("sm_107a", True, True, f32, True, False, 32, None, 6144)] == {
            "block_n": 128,
            "sk_exact": 3,
        }
        assert ROW_RULES[
            ("sm_107a", True, True, f32, True, False, 128, None, 6144)
        ] == {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15}
    # ... and their plans through the planner (CPU views; the launch grid is num_full + sk_units pairs)
    if pairs == 106:
        T = 16231
        # round 9: the indexer_q weight gradients plan 256 x 160 tiles (104 pair tiles = one wave on 106 pairs, so
        # the sk_parts split does not fit and the row keeps whole tiles) and the N = 128 swapped weight gradient
        # plans the 64-row family (48 pair tiles; 3 x 48 units do not fit -> the auto two-way split)
        for row, dt, want in (
            ("indexer_q", "f32", (160, 104, 0, 0, 254)),
            ("indexer_q", "bf16", (160, 104, 0, 0, 254)),
            ("indexer_hw", "bf16", (128, 0, 24, 72, 85)),
            ("indexer_k", "f32", (128, 0, 48, 96, 127)),
        ):
            v = _views("proj", row, "wgrad", dt, T)
            plan = plan_dense_projection_gemm(
                v["A"],
                v["B"],
                v["out"],
                sm_count=212,
                l2_bytes=L2_BYTES,
                transposed_out=v.get("transposed", False),
                arch="sm_107a",
                _fallback=False,
            )[0]
            assert (
                plan.block_n,
                plan.num_full,
                plan.tail_tiles,
                plan.sk_units,
                plan.iters_per_unit,
            ) == want, (row, dt)
            assert plan.num_cluster_tiles == want[1] + want[3]
        # an explicit caller split wins over the rule, and sm_100a keeps its own (ruleless) plan
        v = _views("proj", "indexer_hw", "wgrad", "bf16", T)
        pinned = plan_dense_projection_gemm(
            v["A"],
            v["B"],
            v["out"],
            sm_count=212,
            l2_bytes=L2_BYTES,
            transposed_out=v["transposed"],
            arch="sm_107a",
            _fallback=False,
            sk=True,
            sk_max_units=48,
        )[0]
        assert (pinned.tail_tiles, pinned.sk_units, pinned.iters_per_unit) == (
            24,
            48,
            127,
        )
        # (sm_100a has no sk_parts rule; its round-9 rule plans the 64-row family for this row, so the 128-row
        # family is forced to keep the auto two-way split of the 24 tiles under test)
        other = plan_dense_projection_gemm(
            v["A"],
            v["B"],
            v["out"],
            sm_count=148,
            l2_bytes=L2_BYTES,
            transposed_out=v["transposed"],
            arch="sm_100a",
            _fallback=False,
            cta_rows=128,
        )[0]
        assert (
            (other.pair_tiles, other.tail_tiles, other.sk_units)
            == (
                24,
                24,
                48,
            )
        )  # sm_100a has no sk_parts rule: its 128-wide tiles keep the auto two-way split
        forced = plan_dense_projection_gemm(
            v["A"],
            v["B"],
            v["out"],
            sm_count=148,
            l2_bytes=L2_BYTES,
            transposed_out=v["transposed"],
            arch="sm_100a",
            _fallback=False,
            cta_rows=128,
            sk_parts=3,
        )[0]
        assert (
            forced.pair_tiles,
            forced.tail_tiles,
            forced.sk_units,
            forced.iters_per_unit,
        ) == (24, 24, 72, 85)  # ... unless the caller passes ``sk_parts`` (round 9)


def test_f32_v8_symbol_only_on_the_fp32_register_epilogue():
    assert (
        instance_symbol(
            instance_key(a_mn=False, b_mn=True, out_f32=True, epi="reg", f32_v8=True)
        )
        == "dense_proj_gemm_kn_n256_f32_v8"
    )
    # the knob is inert on the TMA-store, bf16 and transposed epilogues (same key and symbol as without it)
    assert instance_key(
        a_mn=False, b_mn=True, out_f32=True, f32_v8=True
    ) == instance_key(a_mn=False, b_mn=True, out_f32=True)
    assert instance_key(a_mn=False, b_mn=True, f32_v8=True) == instance_key(
        a_mn=False, b_mn=True
    )
    assert instance_key(
        a_mn=True, b_mn=True, out_f32=True, out_t=True, f32_v8=True
    ) == instance_key(a_mn=True, b_mn=True, out_f32=True, out_t=True)


def test_router_layout_classification_and_swap():
    T, K, N = 100, 6144, 256
    X = torch.empty(T, K)
    W = torch.empty(N, K)
    G = torch.empty(T, N)
    plan, A_eff, B_eff = plan_router_fp32_gemm(X, W.t(), torch.empty(T, N))
    assert (plan.layout, plan.template, plan.splits) == (
        "kk",
        "router_fp32_gemm_kk",
        ROUTER_SPLITS["kk"],
    )
    assert not plan.swapped and (plan.M, plan.N, plan.K) == (T, N, K)
    plan, A_eff, B_eff = plan_router_fp32_gemm(G, W, torch.empty(T, K))
    assert (plan.layout, plan.template, plan.splits) == ("kn", "router_fp32_gemm_kn", 1)
    assert (plan.M, plan.N, plan.K) == (T, K, N) and plan.b_mn
    plan, A_eff, B_eff = plan_router_fp32_gemm(G.t(), X, torch.empty(N, K))
    assert (plan.layout, plan.template, plan.splits) == (
        "nn_t",
        "router_fp32_gemm_nn_t",
        6,
    )
    assert (
        plan.swapped and plan.transposed_out and (plan.M, plan.N, plan.K) == (K, N, T)
    )
    assert A_eff.data_ptr() == X.data_ptr() and B_eff.data_ptr() == G.data_ptr()
    assert plan.a_mn and plan.b_mn
    assert plan.grid == (
        plan.splits * (plan.m_tiles // CTA_GROUP) * plan.n_tiles * CTA_GROUP,
        1,
        1,
    )
    plan, *_ = plan_router_fp32_gemm(X, W.t(), torch.empty(T, N), splits=2)
    assert plan.splits == 2 and plan.k_iters_split == -(-plan.k_blocks // 2)
    with pytest.raises(NotImplementedError, match="MN-major A"):
        router_layout_class(True, False)
    with pytest.raises(ValueError, match="fp32"):
        plan_router_fp32_gemm(X.to(torch.bfloat16), W.t(), torch.empty(T, N))
    with pytest.raises(ValueError, match="K must agree"):
        plan_router_fp32_gemm(X, torch.empty(K + 32, N), torch.empty(T, N))
    with pytest.raises(ValueError, match="out must be"):
        plan_router_fp32_gemm(X, W.t(), torch.empty(N, T))
    with pytest.raises(ValueError, match="multiple of 8"):
        plan_router_fp32_gemm(
            X, torch.empty(K, 20).t().contiguous().t(), torch.empty(T, 20)
        )
    with pytest.raises(ValueError, match="unit stride"):
        plan_router_fp32_gemm(torch.empty(T, 2 * K)[:, ::2], W.t(), torch.empty(T, N))
    # a ragged batch with T = 0: M = 0 for the forward / input gradient, K = 0 for the
    # weight gradient -- rejected before a zero-sized grid or TMA extent is planned
    with pytest.raises(ValueError, match="must be >= 1"):
        plan_router_fp32_gemm(torch.empty(0, K), W.t(), torch.empty(0, N))
    with pytest.raises(ValueError, match="must be >= 1"):
        plan_router_fp32_gemm(torch.empty(N, 0), torch.empty(0, K), torch.empty(N, K))


def test_split_fp32_to_bf16x3_is_exact_and_allocation_free_in_place():
    g = torch.Generator().manual_seed(SEED)
    src = torch.randn(64, 96, generator=g) * torch.logspace(-3, 3, 96)
    parts = torch.empty(3, 64, 96, dtype=torch.bfloat16)
    resid = torch.empty(64, 96)
    split_fp32_to_bf16x3_(src, parts, resid)
    total = parts[0].double() + parts[1].double() + parts[2].double()
    assert torch.equal(
        total, src.double()
    )  # 3 x 8 significant bits cover the 24 of an fp32
    assert torch.equal(parts[0], src.to(torch.bfloat16))
    # the split of a transposed source view (K-major parts come from B.T): same values, transposed
    parts_t = torch.empty(3, 96, 64, dtype=torch.bfloat16)
    resid_t = torch.empty(96, 64)
    split_fp32_to_bf16x3_(src.t(), parts_t, resid_t)
    assert torch.equal(parts_t, parts.transpose(1, 2))


def test_bind_launch_fails_closed_and_checks_the_cluster(monkeypatch):
    record = {
        "arches": ["sm_100a"],
        "template": "dense_proj_gemm_kk_n256",
        "kernel": "kernel_fake",
        "library": "cake_dense_projection_gemm_fake",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["parameter", "M"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
    }
    monkeypatch.setitem(cake_jit.MODULES, "cake_dense_projection_gemm_fake", record)
    monkeypatch.setitem(
        cake_jit.KERNELS,
        "sm_100a",
        {"dense_proj_gemm_kk_n256": "cake_dense_projection_gemm_fake"},
    )
    calls = []
    fake = SimpleNamespace(run=lambda *args: calls.append(args))
    monkeypatch.setattr(
        cake_backend, "load_cake_dense_projection_gemm_module", lambda name, arch: fake
    )
    A = torch.zeros(1, 8, 64, dtype=torch.bfloat16)
    launch = bind_launch(
        "cake_dense_projection_gemm_fake",
        dict(A=A, M=8),
        (4, 1, 1),
        "sm_100a",
    )
    assert launch.arguments == (A, 8, 4, 1, 1)
    launch()
    assert calls == [(A, 8, 4, 1, 1)]
    with pytest.raises(KeyError, match="'M'"):
        bind_launch(
            "cake_dense_projection_gemm_fake",
            dict(A=A),
            (4, 1, 1),
            "sm_100a",
        )
    with pytest.raises(ValueError, match="cluster"):
        bind_launch(
            "cake_dense_projection_gemm_fake",
            dict(A=A, M=8),
            (3, 1, 1),
            "sm_100a",
        )
    assert (
        cake_jit.select_module("sm_100a", "dense_proj_gemm_kk_n256")
        == "cake_dense_projection_gemm_fake"
    )
    # schema v2: a record names its plan by key and the key resolves through ARG_PLANS; an inline plan
    # (the record above) is still accepted; an unknown key fails closed
    plan = record["arg_plan"]
    key = cake_jit.arg_plan_key(plan)
    assert re.fullmatch(r"[0-9a-f]{12}", key) and key not in cake_jit.ARG_PLANS
    monkeypatch.setitem(cake_jit.ARG_PLANS, key, plan)
    monkeypatch.setitem(
        cake_jit.MODULES,
        "cake_dense_projection_gemm_fake_v2",
        dict(record, arg_plan=key),
    )
    assert (
        cake_jit.arg_plan_of(cake_jit.MODULES["cake_dense_projection_gemm_fake_v2"])
        == plan
    )
    assert cake_jit.arg_plan_of(record) == plan
    launch_v2 = bind_launch(
        "cake_dense_projection_gemm_fake_v2", dict(A=A, M=8), (4, 1, 1), "sm_100a"
    )
    assert launch_v2.arguments == (A, 8, 4, 1, 1)
    monkeypatch.setitem(
        cake_jit.MODULES,
        "cake_dense_projection_gemm_fake_v2",
        dict(record, arg_plan="f" * 12),
    )
    with pytest.raises(KeyError, match="ARG_PLANS"):
        bind_launch(
            "cake_dense_projection_gemm_fake_v2", dict(A=A, M=8), (4, 1, 1), "sm_100a"
        )


# ---------------------------------------------------------------------------
# Convenience path (CPU): cache key, cache bound, per-call view rebuild, device facts
# ---------------------------------------------------------------------------


def test_convenience_key_follows_the_view_geometry_not_the_address():
    A = torch.empty(64, 128, dtype=torch.bfloat16)
    A_again = torch.empty(64, 128, dtype=torch.bfloat16)
    B = torch.empty(32, 128, dtype=torch.bfloat16).t()
    buf = torch.empty(2, 64, 32, dtype=torch.bfloat16)
    out, out_same, out_offset = (
        torch.empty(64, 32, dtype=torch.bfloat16),
        buf[0],
        buf[1],
    )
    key = convenience_key(0, 7, (False,), A, B, out)
    # a new tensor of the same geometry hits (the address is not part of the key)
    assert key == convenience_key(0, 7, (False,), A_again, B, out_same)
    assert hash(key) == hash(convenience_key(0, 7, (False,), A_again, B, out_same))
    others = [
        convenience_key(1, 7, (False,), A, B, out),  # device index
        convenience_key(0, 8, (False,), A, B, out),  # stream
        convenience_key(0, 7, (True,), A, B, out),  # flags (transposed_out)
        convenience_key(0, 7, (False,), A[:32], B, out[:32]),  # shape
        convenience_key(0, 7, (False,), A.t().contiguous().t(), B, out),  # strides
        convenience_key(0, 7, (False,), A, B, out.float()),  # dtype
        convenience_key(0, 7, (False,), A, B, out_offset),  # storage offset
        convenience_key(
            0, 7, (False,), torch.empty(64, 129, dtype=torch.bfloat16)[:, 1:], B, out
        ),  # alignment
    ]
    assert len({key, *others}) == len(others) + 1
    # the entry points put their knobs into the flags (sorted items): a knob change is a new key
    assert convenience_key(
        0, 7, (True, ("cta_rows", 64), ("sk", "auto")), A, B, out
    ) != convenience_key(0, 7, (True, ("cta_rows", None), ("sk", "auto")), A, B, out)


def test_convenience_cache_is_bounded_and_evicts_the_least_recently_used_entry():
    cache: OrderedDict = OrderedDict()
    for i in range(5):
        cake_backend._cache_put(cache, ("k", i), i, bound=4)
    assert list(cache) == [("k", 1), ("k", 2), ("k", 3), ("k", 4)]
    assert cake_backend._cache_get(cache, ("k", 1)) == 1  # a hit refreshes the entry
    assert list(cache)[-1] == ("k", 1)
    assert cake_backend._cache_get(cache, ("k", 0)) is None
    cake_backend._cache_put(cache, ("k", 5), 5, bound=4)
    assert list(cache) == [("k", 3), ("k", 4), ("k", 1), ("k", 5)]
    assert CONVENIENCE_CACHE_SIZE >= 1
    assert isinstance(cake_backend._K1_CACHE, OrderedDict) and isinstance(
        cake_backend._K2_CACHE, OrderedDict
    )
    # the argument template keeps constants and fills the per-call slots in plan order
    template = (cake_backend._Slot("A"), 7, cake_backend._Slot("out3"), 2, 1, 1)
    assert cake_backend._resolve_arguments(template, dict(A="a", out3="o")) == [
        "a",
        7,
        "o",
        2,
        1,
        1,
    ]


def test_convenience_cache_bound_is_read_at_call_time(monkeypatch):
    """``CONVENIENCE_CACHE_SIZE`` is a module knob: ``_cache_put`` reads it on every insertion, so a value set
    after the import bounds the caches; an explicit ``bound=`` still overrides it."""
    monkeypatch.setattr(cake_backend, "CONVENIENCE_CACHE_SIZE", 2)
    cache: OrderedDict = OrderedDict()
    for i in range(3):
        cake_backend._cache_put(cache, ("k", i), i)
    assert list(cache) == [("k", 1), ("k", 2)]
    monkeypatch.setattr(cake_backend, "CONVENIENCE_CACHE_SIZE", 3)
    cake_backend._cache_put(cache, ("k", 3), 3)
    assert list(cache) == [("k", 1), ("k", 2), ("k", 3)]
    cake_backend._cache_put(cache, ("k", 4), 4, bound=1)
    assert list(cache) == [("k", 4)]


def test_convenience_cache_hit_evicted_before_its_refresh_is_a_miss():
    """A hit whose entry another thread evicts between the lookup and the recency refresh is a miss -- the
    caller plans the key again (at worst a key is planned twice) -- not a ``KeyError`` out of the launch."""

    class _EvictedBetween(OrderedDict):
        def get(self, key, default=None):
            entry = super().get(key, default)
            if entry is not None:
                # the other thread's eviction lands between the lookup and the refresh
                del self[key]
            return entry

    cache = _EvictedBetween()
    cake_backend._cache_put(cache, ("k", 0), "first", bound=4)
    assert cake_backend._cache_get(cache, ("k", 0)) is None
    assert ("k", 0) not in cache
    assert cake_backend._cache_put(cache, ("k", 0), "second", bound=4) == "second"
    assert cache[("k", 0)] == "second" and list(cache) == [("k", 0)]
    assert cake_backend._cache_get(OrderedDict(), ("k", 0)) is None


def test_convenience_cache_put_reports_every_entry_that_leaves_the_cache():
    """``on_evict`` sees the entries the bound evicts and an earlier entry a key replaces (the arena
    reference count of the convenience caches hangs on it), each exactly once."""
    gone = []
    cache: OrderedDict = OrderedDict()
    for i in range(3):
        cake_backend._cache_put(cache, ("k", i), i, bound=2, on_evict=gone.append)
    assert gone == [0] and list(cache) == [("k", 1), ("k", 2)]
    cake_backend._cache_put(cache, ("k", 2), 20, bound=2, on_evict=gone.append)
    assert gone == [0, 2] and cache[("k", 2)] == 20 and list(cache)[-1] == ("k", 2)
    cake_backend._cache_put(cache, ("k", 2), 20, bound=2, on_evict=gone.append)
    assert gone == [0, 2]  # re-inserting the same entry is not a replacement
    cake_backend._cache_put(cache, ("k", 3), 3, bound=1, on_evict=gone.append)
    assert gone == [0, 2, 1, 20] and list(cache) == [("k", 3)]


def _cpu_router_entry(arena, op, T, splits=None):
    """A planned K2 convenience entry on CPU views (no module: a no-op entry callable)."""
    v = _router_inputs(op, T, seed=SEED + 8, device="cpu")
    plan, A_eff, B_eff = plan_router_fp32_gemm(v["A"], v["B"], v["out"], splits=splits)
    _, split_src = cake_backend._k2_views(v["A"], v["B"], plan)
    split_shape = tuple(int(s) for s in split_src.shape)
    partials = None
    if plan.splits > 1:
        rows, cols = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
        partials = (plan.splits, rows, cols)
    arena.require("k2_parts", 3 * split_src.numel())
    arena.require("k2_resid", split_src.numel())
    if partials is not None:
        arena.require("k2_partials", partials[0] * partials[1] * partials[2])
    arena.refs += 1
    return cake_backend._CachedRouter(
        "router_fp32_gemm_kk",
        "sm_100a",
        plan,
        lambda *args: None,
        (
            cake_backend._Slot("A"),
            cake_backend._Slot("parts"),
            cake_backend._Slot("out32"),
        ),
        arena,
        split_shape,
        partials,
    ), v


def test_convenience_arena_grows_to_the_largest_requirement_and_carves_views(
    monkeypatch,
):
    """The arena of one (device, stream) holds one buffer per workspace kind, grown to the largest requirement
    its entries asked for (never shrunk; the counters zero-initialised); an entry carves its workspace out of
    the buffers at launch time and memoises the views until the arena grows -- CPU tensors, no module."""
    monkeypatch.setattr(cake_backend, "_K1_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_K2_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_ARENAS", {})
    arena = cake_backend._Arena(torch.device("cpu"), (0, 7))
    assert arena.nbytes() == 0 and arena.version == 0 and arena.refs == 0
    small = arena.require("k1_counters", 5)
    assert (
        small.dtype == torch.uint32 and small.numel() == cake_backend._ARENA_MIN_ELEMS
    )
    assert (
        int(torch.count_nonzero(small.view(torch.int32))) == 0
    )  # bitcast: no u32 CPU kernel
    assert arena.version == 1
    ws = arena.require("k1_ws", 1000)
    assert ws.dtype == torch.float32 and ws.numel() == 1000 and arena.version == 2
    assert (
        arena.require("k1_ws", 400) is ws and arena.version == 2
    )  # a smaller requirement reuses
    assert arena.require("k1_ws", 1000) is ws
    grown = arena.require("k1_ws", 1001)
    assert grown is not ws and grown.numel() == 1001 and arena.version == 3
    assert arena.nbytes() == 1001 * 4 + cake_backend._ARENA_MIN_ELEMS * 4
    carved = cake_backend._carve(torch.arange(24.0), (2, 3, 4))
    assert carved.is_contiguous() and carved.storage_offset() == 0
    assert torch.equal(carved, torch.arange(24.0).view(2, 3, 4))
    assert set(cake_backend._ARENA_KINDS) == {
        "k1_ws",
        "k1_counters",
        "k2_parts",
        "k2_resid",
        "k2_partials",
    }
    # K2 entries of two geometries share the buffers; each carves its own exact shapes at offset 0
    fwd, _ = _cpu_router_entry(arena, "fwd", 257)
    wgrad, _ = _cpu_router_entry(arena, "wgrad", 1001)
    assert fwd.arena is arena and wgrad.arena is arena and arena.refs == 2
    for entry in (fwd, wgrad):
        views = cake_backend._entry_views(entry)
        assert views is cake_backend._entry_views(entry)  # memoised
        assert tuple(views["parts"].shape) == (3, *entry.split_shape)
        assert tuple(views["resid"].shape) == entry.split_shape
        assert views["parts"].data_ptr() == arena.buffers["k2_parts"].data_ptr()
        assert views["resid"].data_ptr() == arena.buffers["k2_resid"].data_ptr()
        assert tuple(views["workspace"].shape) == entry.partials_shape
        assert views["workspace"].data_ptr() == arena.buffers["k2_partials"].data_ptr()
        assert views["parts"].is_contiguous() and views["workspace"].is_contiguous()
    # parts of W [3, 256, 6144] (forward) and of G [3, 1001, 256] (weight gradient): one buffer, the larger
    assert (fwd.split_shape, wgrad.split_shape) == ((256, 6144), (1001, 256))
    assert (
        arena.buffers["k2_parts"].numel()
        == 3 * 256 * 6144
        == max(3 * 256 * 6144, 3 * 1001 * 256)
    )
    assert (fwd.partials_shape, wgrad.partials_shape) == (
        (4, 257, 256),
        (6, 256, 6144),
    )
    assert arena.buffers["k2_partials"].numel() == 6 * 256 * 6144
    # a growth forgets every memo of the arena's cached entries and the next launch carves again
    cake_backend._K2_CACHE[("fwd",)] = fwd
    cake_backend._K2_CACHE[("wgrad",)] = wgrad
    old_parts = cake_backend._entry_views(fwd)["parts"]
    version = arena.version
    arena.require("k2_parts", arena.buffers["k2_parts"].numel() + 1)
    assert arena.version == version + 1 and fwd.views is None and wgrad.views is None
    new_parts = cake_backend._entry_views(fwd)["parts"]
    assert new_parts.data_ptr() != old_parts.data_ptr() and tuple(
        new_parts.shape
    ) == tuple(old_parts.shape)
    assert fwd.views_version == arena.version
    # a K1 entry carves the buffers themselves (pointer parameters)
    v = _views("proj", "kv_a", "fwd", "bf16", 257)
    plan, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a"
    )
    arena.require("k1_ws", plan.ws_f32_elems)
    arena.require("k1_counters", plan.counters_alloc_u32)
    gemm_entry = cake_backend._CachedGemm(
        "m", "sm_100a", plan, False, lambda *args: None, (), arena
    )
    k1 = cake_backend._entry_views(gemm_entry)
    assert (
        k1["ws"] is arena.buffers["k1_ws"]
        and k1["counters"] is arena.buffers["k1_counters"]
    )
    assert k1["counters"].numel() >= plan.counters_alloc_u32 >= SK_DUMMY_BASE + 16 * 32


def test_convenience_arena_is_released_with_its_last_entry(monkeypatch):
    """``_release_entry`` (the caches' ``on_evict``) drops an arena from the registry once its last cached
    entry is evicted; ``clear_convenience_caches`` drops them all."""
    monkeypatch.setattr(cake_backend, "_K1_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_K2_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_ARENAS", {})
    device = torch.device("cpu")
    a7 = cake_backend._arena_for(0, 7, device)
    assert cake_backend._arena_for(0, 7, device) is a7 and a7.key == (0, 7)
    a8 = cake_backend._arena_for(0, 8, device)
    assert a8 is not a7 and len(cake_backend._ARENAS) == 2
    e1, _ = _cpu_router_entry(a7, "fwd", 257)
    e2, _ = _cpu_router_entry(a7, "dgrad", 257)
    e3, _ = _cpu_router_entry(a8, "fwd", 129)
    assert (a7.refs, a8.refs) == (2, 1)
    cache = cake_backend._K2_CACHE
    put = cake_backend._cache_put
    put(cache, ("e1",), e1, bound=2, on_evict=cake_backend._release_entry)
    put(cache, ("e2",), e2, bound=2, on_evict=cake_backend._release_entry)
    put(cache, ("e3",), e3, bound=2, on_evict=cake_backend._release_entry)  # evicts e1
    assert list(cache) == [("e2",), ("e3",)] and a7.refs == 1
    assert cake_backend._ARENAS[(0, 7)] is a7
    put(
        cache, ("e3",), e3, bound=1, on_evict=cake_backend._release_entry
    )  # evicts e2: a7 is released
    assert (
        list(cache) == [("e3",)] and a7.refs == 0 and (0, 7) not in cake_backend._ARENAS
    )
    assert {(0, 8): a8} == cake_backend._ARENAS and a8.refs == 1
    # a stale release never drops a successor arena of the same key
    a7b = cake_backend._arena_for(0, 7, device)
    cake_backend._release_entry(e1)
    assert cake_backend._ARENAS[(0, 7)] is a7b
    cake_backend.clear_convenience_caches()
    assert cake_backend._ARENAS == {} and cache == OrderedDict()


def test_convenience_arena_require_keeps_the_larger_buffer(monkeypatch):
    """``require`` is monotone: when a larger buffer of the kind was stored between its size check and its own
    allocation (another planning thread), the larger one is kept and returned, never the smaller."""
    monkeypatch.setattr(cake_backend, "_K1_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_K2_CACHE", OrderedDict())
    arena = cake_backend._Arena(torch.device("cpu"), (0, 7))
    larger = torch.empty(2000, dtype=torch.float32)

    class _StoredMeanwhile(dict):
        reads = 0

        def get(self, key, default=None):
            self.reads += 1
            if (
                self.reads == 2
            ):  # the re-read after allocating: the other thread's buffer landed
                dict.__setitem__(self, key, larger)
            return dict.get(self, key, default)

    arena.buffers = _StoredMeanwhile()
    got = arena.require("k1_ws", 1000)
    assert got is larger and arena.buffers["k1_ws"] is larger and got.numel() == 2000
    assert arena.version == 1 and arena.buffers.reads == 2
    # the plain case still grows to the requirement and keeps its own allocation
    assert arena.require("k1_ws", 3000).numel() == 3000 and arena.version == 2
    assert isinstance(arena.lock, type(threading.Lock())) and not arena.lock.locked()


def test_convenience_entry_views_memo_is_stamped_before_carving():
    """``_entry_views`` reads the arena version before carving: a growth that interleaves with the carve leaves
    the memo stale (re-carved on the next use) instead of pinning the replaced buffer as current."""
    arena = cake_backend._Arena(torch.device("cpu"), (0, 7))
    entry, _ = _cpu_router_entry(arena, "fwd", 257)
    carve = entry.carve
    before = arena.version
    grown = []

    def carve_with_interleaved_growth(a):
        views = carve(a)
        a.require(
            "k2_parts", a.buffers["k2_parts"].numel() + 1
        )  # another thread grows meanwhile
        grown.append(a.version)
        return views

    entry.carve = carve_with_interleaved_growth
    first = cake_backend._entry_views(entry)
    assert grown == [before + 1] and arena.version == before + 1
    assert (
        entry.views_version == before
    )  # stamped with the version read before the carve
    assert first["parts"].data_ptr() != arena.buffers["k2_parts"].data_ptr()
    entry.carve = carve
    second = cake_backend._entry_views(entry)
    assert second is not first and entry.views_version == arena.version
    assert second["parts"].data_ptr() == arena.buffers["k2_parts"].data_ptr()
    assert cake_backend._entry_views(entry) is second


def test_convenience_plan_entry_rolls_back_a_failed_plan_and_rechecks_the_cache(
    monkeypatch,
):
    """A plan that fails after growing a *shared* arena (bind / JIT / grid error) restores the arena's earlier
    buffers (and forgets memos carved against the dropped growth); a fresh arena it created is removed; a key
    another thread planned meanwhile is returned without planning again."""
    monkeypatch.setattr(cake_backend, "_K1_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_K2_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_ARENAS", {})
    device = torch.device("cpu")
    cache = cake_backend._K2_CACHE
    shared = cake_backend._arena_for(0, 7, device)
    entry, _ = _cpu_router_entry(
        shared, "fwd", 257
    )  # refs 1, buffers grown for the forward
    cache[("fwd",)] = entry
    memo = cake_backend._entry_views(entry)
    saved, version = dict(shared.buffers), shared.version
    calls = []

    def grows_then_fails(arena):
        calls.append(arena)
        have = (
            arena.buffers["k2_parts"].numel() if "k2_parts" in arena.buffers else 1000
        )
        arena.require("k2_parts", 4 * have)
        raise RuntimeError("bind failed")

    with pytest.raises(RuntimeError, match="bind failed"):
        cake_backend._plan_entry(cache, ("huge",), 0, 7, device, grows_then_fails)
    assert calls == [shared] and ("huge",) not in cache and list(cache) == [("fwd",)]
    assert {(0, 7): shared} == cake_backend._ARENAS and shared.refs == 1
    assert shared.buffers.keys() == saved.keys() and all(
        shared.buffers[k] is saved[k] for k in saved
    )  # the same tensors as before the failed plan
    assert (
        shared.version == version + 2 and entry.views is None
    )  # grown, then rolled back: memos forgotten
    assert (
        cake_backend._entry_views(entry)["parts"].data_ptr() == memo["parts"].data_ptr()
    )
    assert not cake_backend._PLAN_LOCK.locked()
    # a fresh arena a failed plan created is removed
    with pytest.raises(RuntimeError):
        cake_backend._plan_entry(cache, ("other",), 0, 8, device, grows_then_fails)
    assert (0, 8) not in cake_backend._ARENAS and len(calls) == 2
    # the cache is re-checked under the lock: a key planned meanwhile is returned with ``views = None``
    got, views = cake_backend._plan_entry(
        cache, ("fwd",), 0, 7, device, grows_then_fails
    )
    assert got is entry and views is None and len(calls) == 2
    assert list(cache)[-1] == ("fwd",)


def test_convenience_router_launch_holds_the_arena_lock(monkeypatch):
    """The K2 hit path holds its arena's lock from the carve of the workspace views through the launch and the
    split-K reduction (CPU stand-ins for the device, the stream and the FFI entry)."""
    monkeypatch.setattr(cake_backend, "_K1_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_K2_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_ARENAS", {})
    device = torch.device("cpu")
    arena = cake_backend._arena_for(0, 7, device)
    cached, v = _cpu_router_entry(arena, "fwd", 257, splits=4)
    seen = []

    def entry_call(*arguments):
        assert arena.lock.locked()
        seen.append(("kernel", len(arguments)))

    cached.entry = entry_call
    key = convenience_key(0, 7, (4,), v["A"], v["B"], v["out"])
    cake_backend._K2_CACHE[key] = cached
    monkeypatch.setattr(cake_backend, "_same_cuda_device", lambda A, B, out: out.device)
    monkeypatch.setattr(cake_backend, "_cuda_index", lambda d: 0)
    monkeypatch.setattr(cake_backend, "_current_raw_stream", lambda index: 7)
    monkeypatch.setattr(
        cake_backend,
        "_ffi_stream_context",
        lambda index, raw=None: contextlib.nullcontext(),
    )
    split = cake_backend.split_fp32_to_bf16x3_

    def locked_split(src, parts, resid):
        assert arena.lock.locked()
        seen.append(("split", tuple(parts.shape)))
        return split(src, parts, resid)

    monkeypatch.setattr(cake_backend, "split_fp32_to_bf16x3_", locked_split)
    out = cake_backend._launch_cached_router(v["A"], v["B"], v["out"], splits=4)
    assert out is v["out"] and not arena.lock.locked()
    assert seen == [("split", (3, 256, 6144)), ("kernel", 3)]
    assert cached.views is not None and cached.views_version == arena.version
    # the memoised workspace views are the arena's buffers, carved under the lock
    assert (
        cached.views["workspace"].data_ptr() == arena.buffers["k2_partials"].data_ptr()
    )


_FAKE_SPLIT_MODULE = "cake_dense_projection_gemm_fake_split"
_FAKE_SPLIT_RECORD = {
    "arches": ["sm_100a"],
    "template": cake_backend.SPLIT_TEMPLATE,
    "kernel": "kernel_fake_split",
    "library": _FAKE_SPLIT_MODULE,
    "ffi_entry": "run",
    "arg_plan": [
        ["buffer", "src"],
        ["buffer", "parts"],
        ["parameter", "n"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
}


def _cpu_router_registry(monkeypatch, *, fused):
    """CPU stand-ins for a K2 plan on ``sm_100a``: the device check and the architecture, the stream and FFI
    context, and a module loader whose entries record ``(module, arguments, an arena lock was held)``; with
    ``fused`` the registry holds a stand-in fused split program, without it none.  Returns the recorded calls."""
    calls = []
    kernels = dict(cake_jit.KERNELS["sm_100a"])
    kernels.pop(cake_backend.SPLIT_TEMPLATE, None)
    if fused:
        kernels[cake_backend.SPLIT_TEMPLATE] = _FAKE_SPLIT_MODULE
        monkeypatch.setitem(cake_jit.MODULES, _FAKE_SPLIT_MODULE, _FAKE_SPLIT_RECORD)
    monkeypatch.setitem(cake_jit.KERNELS, "sm_100a", kernels)
    monkeypatch.setattr(cake_backend, "_same_cuda_device", lambda A, B, out: out.device)
    monkeypatch.setattr(cake_backend, "require_arch", lambda device: "sm_100a")
    monkeypatch.setattr(cake_backend, "_cuda_index", lambda d: 0)
    monkeypatch.setattr(cake_backend, "_device_index", lambda device: 0)
    monkeypatch.setattr(cake_backend, "_current_raw_stream", lambda index: 7)
    monkeypatch.setattr(
        cake_backend,
        "_ffi_stream_context",
        lambda index, raw=None: contextlib.nullcontext(),
    )

    def loader(name, arch):
        def run(*args):
            locked = any(a.lock.locked() for a in cake_backend._ARENAS.values())
            calls.append((name, args, locked))

        return SimpleNamespace(run=run)

    monkeypatch.setattr(cake_backend, "load_cake_dense_projection_gemm_module", loader)
    return calls


def test_fused_split_selection_follows_the_registry(monkeypatch):
    """The K2 paths split with the fused program only when the registry holds it for the architecture and the
    source is a contiguous fp32 view of at most int32 elements; otherwise the torch chain splits."""
    assert cake_backend._split_grid(1) == (1, 1, 1)
    assert cake_backend._split_grid(3 * cake_backend.SPLIT_BLOCK_ELEMS + 1) == (4, 1, 1)
    assert cake_backend._split_values("s", "p", 5) == {"src": "s", "parts": "p", "n": 5}
    W = torch.empty(256, 6144)
    _cpu_router_registry(monkeypatch, fused=False)
    assert cake_backend.fused_split_module("sm_100a") is None
    assert cake_backend._fused_split_for("sm_100a", W) is None
    _cpu_router_registry(monkeypatch, fused=True)
    assert cake_backend.fused_split_module("sm_100a") == _FAKE_SPLIT_MODULE
    assert cake_backend.fused_split_module("sm_90a") is None
    assert cake_backend._fused_split_for("sm_100a", W) == _FAKE_SPLIT_MODULE
    base = torch.empty(W.numel() + 2)
    offset = base[1 : 1 + W.numel()].view(
        W.shape
    )  # contiguous at a storage offset: fine
    assert cake_backend._fused_split_for("sm_100a", offset) == _FAKE_SPLIT_MODULE
    assert cake_backend._fused_split_for("sm_100a", W.t()) is None  # not contiguous
    assert cake_backend._fused_split_for("sm_100a", W[:, 1:]) is None
    monkeypatch.setattr(cake_backend, "_SPLIT_MAX_N", W.numel() - 1)
    assert cake_backend._fused_split_for("sm_100a", W) is None  # beyond the int32 count


@pytest.mark.parametrize("fused", [False, True])
def test_router_paths_split_through_the_registry(monkeypatch, fused):
    """Both K2 paths select the split through the registry: with the fused program registered they bind one
    launch of it over the fp32 source into the BF16x3 stack and allocate (prepared) or carve (convenience) no
    fp32 scratch; otherwise the torch chain splits through the scratch.  The convenience path issues the fused
    split under its arena's lock, before the kernel (CPU stand-ins for the device, the loader and the entries)."""
    monkeypatch.setattr(cake_backend, "_K1_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_K2_CACHE", OrderedDict())
    monkeypatch.setattr(cake_backend, "_ARENAS", {})
    calls = _cpu_router_registry(monkeypatch, fused=fused)
    chain = []
    split = cake_backend.split_fp32_to_bf16x3_

    def chain_split(src, parts, resid):
        chain.append(tuple(parts.shape))
        return split(src, parts, resid)

    monkeypatch.setattr(cake_backend, "split_fp32_to_bf16x3_", chain_split)
    v = _router_inputs("fwd", 257, seed=SEED + 9, device="cpu")
    n = v["W"].numel()
    split_args = (n, -(-n // cake_backend.SPLIT_BLOCK_ELEMS), 1, 1)
    # the prepared path
    prepared = prepare_router_fp32_gemm(v["A"], v["B"], v["out"], splits=4)
    assert prepared.fused_split is fused and ("resid" in prepared.tensors) is not fused
    prepared.launch()
    if fused:
        assert chain == []
        assert [c[0] for c in calls] == [_FAKE_SPLIT_MODULE, prepared.module_name]
        _, args, _ = calls[0]
        assert args[0].data_ptr() == v["W"].data_ptr() and args[0].shape == (256, 6144)
        assert args[1] is prepared.tensors["parts"] and args[2:] == split_args
    else:
        assert chain == [(3, 256, 6144)]
        assert [c[0] for c in calls] == [prepared.module_name]
    # the convenience path
    calls.clear()
    chain.clear()
    out = cake_backend.router_fp32_gemm(v["A"], v["B"], v["out"], splits=4)
    assert out is v["out"]
    cached = next(iter(cake_backend._K2_CACHE.values()))
    arena = cached.arena
    assert (cached.split is None) is not fused
    assert ("k2_resid" in arena.buffers) is not fused
    views = cake_backend._entry_views(cached)
    assert ("resid" in views) is not fused
    assert tuple(views["parts"].shape) == (3, 256, 6144)
    assert not arena.lock.locked()
    if fused:
        assert chain == []
        assert [(c[0], c[2]) for c in calls] == [
            (_FAKE_SPLIT_MODULE, True),
            (cached.module_name, True),
        ]
        _, args, _ = calls[0]
        assert args[0].data_ptr() == v["W"].data_ptr() and args[0].shape == (256, 6144)
        assert args[1].data_ptr() == arena.buffers["k2_parts"].data_ptr()
        assert args[2:] == split_args
    else:
        assert chain == [(3, 256, 6144)]
        assert [(c[0], c[2]) for c in calls] == [(cached.module_name, True)]
        assert torch.equal(views["parts"], prepared.tensors["parts"])


@pytest.mark.parametrize("sm_count", SM_COUNTS)
def test_convenience_views_match_the_planner_views(sm_count):
    """The per-call view rebuild of the convenience path (``_k1_views`` under ``_k1_swapped``) returns the
    planner's operand views for the GPU rows and the MLA rows (swapped, swap-fallback and direct plans) --
    CPU views, the planner needs no device."""
    arch = ARCH_OF_SM[sm_count]
    cases = [("proj", row, op, dt, T) for row, op, dt, T in GPU_ROWS] + [
        ("mla", row, op, "bf16", T)
        for row in MLA_ROWS
        for op in ("fwd", "dgrad", "wgrad")
        for T in MLA_T
    ]
    seen = set()
    # the registry plan (what the convenience path caches) and the pure mirror (``_fallback=False``: the
    # batched small-M swap is always taken, whether or not the swapped instance is registered)
    for fallback in (True, False):
        for family, row, op, dt, T in cases:
            v = _views(family, row, op, dt, T)
            transposed = v.get("transposed", False)
            plan, a_desc, b_desc, out3 = plan_dense_projection_gemm(
                v["A"],
                v["B"],
                v["out"],
                sm_count=sm_count,
                l2_bytes=L2_BYTES,
                transposed_out=transposed,
                arch=arch,
                _fallback=fallback,
            )
            swapped = cake_backend._k1_swapped(
                plan, v["A"], v["B"], transposed_out=transposed
            )
            rebuilt = cake_backend._k1_views(v["A"], v["B"], v["out"], swapped=swapped)
            for got, want in zip(rebuilt, (a_desc, b_desc, out3), strict=True):
                assert cake_backend._same_view(got, want), (family, row, op, dt, T)
            if swapped:
                assert plan.transposed_out and not plan.swap_fallback
            seen.add((swapped, plan.swap_fallback))
            ldo, out_l = cake_backend._out_strides(out3)
            assert ldo == out3.stride(1)
            # the kernel requires out_l >= rows * ldo for every batch size; a batch of one takes its view's batch
            # stride when that spans the matrix (the unsqueezed 2-D outputs of every row here) and the span otherwise;
            # a real batch passes its stride as is (the batched rows interleave their batches inside a row)
            span = out3.shape[1] * ldo
            if out3.shape[0] > 1 or out3.stride(0) >= span:
                assert out_l == out3.stride(0)
            else:
                assert out_l == span
    # a swapped plan and a direct plan were both exercised
    assert (True, False) in seen and (False, False) in seen


def test_out_strides_span_the_matrix_for_a_batch_of_one():
    """The kernel requires ``out_l >= rows * ldo`` for every batch size (it declares its tile grid against the
    output's extents); a batch of one never dereferences its batch stride, so the host passes the view's batch
    stride when it spans the matrix and the matrix span otherwise.  A real batch passes its stride as is."""
    M, N = 48, 64
    plain = cake_backend.as_batched(torch.empty(M, N), "out")
    assert cake_backend._out_strides(plain) == (N, M * N)
    # a padded row stride: the span follows ldo, not the column count
    padded = cake_backend.as_batched(torch.empty(M, 2 * N)[:, :N], "out")
    assert cake_backend._out_strides(padded) == (2 * N, M * 2 * N)
    # a batch axis of stride 0 (expand) or below the span (as_strided): the span
    expanded = torch.empty(M, N).expand(1, M, N)
    assert cake_backend._out_strides(expanded) == (N, M * N)
    narrow = torch.empty(M * N + 8).as_strided((1, M, N), (8, N, 1))
    assert cake_backend._out_strides(narrow) == (N, M * N)
    # a batch axis above the span: as is
    wide = torch.empty(2 * M * N).as_strided((1, M, N), (2 * M * N, N, 1))
    assert cake_backend._out_strides(wide) == (N, 2 * M * N)
    # the transposed store: rows = N
    transposed = cake_backend.as_batched(torch.empty(N, M), "out")
    assert cake_backend._out_strides(transposed) == (M, N * M)
    # a real batch: its strides as is (dense and padded)
    assert cake_backend._out_strides(torch.empty(3, M, N)) == (N, M * N)
    assert cake_backend._out_strides(torch.empty(3, M, 2 * N)[..., :N]) == (
        2 * N,
        M * 2 * N,
    )


def test_template_knobs_invert_instance_symbol():
    """``_template_knobs`` (the registry fallback's symbol decoder) names every generated program back through
    ``instance_key`` / ``instance_symbol``; the batch-raster and tail tokens decode to the knobs that emit them."""
    seen = 0
    for arch, table in cake_jit.KERNELS.items():
        for template in table:
            tokens = cake_backend._template_tokens(template)
            if tokens is None:
                # the router programs and the fused split program carry no knob tokens
                assert (
                    template.startswith("router_fp32_gemm_")
                    or template == cake_backend.SPLIT_TEMPLATE
                ), (
                    arch,
                    template,
                )
                continue
            knobs = cake_backend._template_knobs(template)
            assert knobs is not None, (arch, template)
            layout, out_f32, out_t = cake_backend._template_class(tokens)
            key_knobs = {k: v for k, v in knobs.items() if v is not None}
            key_knobs["sk_exact"] = bool(knobs["sk_exact"])
            key = cake_backend.instance_key(
                a_mn=layout[0] == "n",
                b_mn=layout[1] == "n",
                out_f32=out_f32,
                out_t=out_t,
                **key_knobs,
            )
            assert cake_backend.instance_symbol(key) == template, (arch, template)
            seen += 1
    assert seen == sum(len(_registered(arch)) for arch in cake_jit.KERNELS)
    # tokens the planner never emits, and names outside the family, decode to nothing
    assert cake_backend._template_knobs("dense_proj_gemm_kk_n256_box64_pro7") is None
    assert cake_backend._template_knobs("router_fp32_gemm_kk") is None
    assert cake_backend._template_tokens("router_fp32_gemm_kk") is None
    # the exact split carries no part count: the rule's parts, else the kernel tests' 3 (128 rows) / 2 (64 rows)
    assert (
        cake_backend._template_knobs("dense_proj_gemm_nn_n128_hen_t_skx_pro7")[
            "sk_exact"
        ]
        == 3
    )
    assert (
        cake_backend._template_knobs("dense_proj_gemm_nn_n128_m64_hen_t_skx_pro15")[
            "sk_exact"
        ]
        == 2
    )
    assert (
        cake_backend._template_knobs(
            "dense_proj_gemm_nn_n128_hen_t_skx_pro7", sk_exact_parts=4
        )["sk_exact"]
        == 4
    )
    # a single-CTA program leaves pro to the planner (instance_key refuses a forced one)
    assert cake_backend._template_knobs("dense_proj_gemm_kk_n128_hen_c1")["pro"] is None
    # the hint tokens are the nearest edit; a batch-raster program is no variant of a batch of one
    assert (
        cake_backend._template_distance(
            ("kk", "n256", "hen", "pro7"), ("kk", "n256", "pro7")
        )
        == 0.5
    )
    assert (
        cake_backend._template_distance(("kk", "n256", "pro7"), ("kk", "n256", "pro15"))
        == 2.0
    )
    for arch in cake_jit.KERNELS:
        single = cake_backend._registry_variants(
            "dense_proj_gemm_kn_n256_pro7", arch, False
        )
        assert single and all(not knobs["batch_group"] for knobs in single), arch
        batched = cake_backend._registry_variants(
            "dense_proj_gemm_kn_n256_pro7", arch, True
        )
        assert len(batched) >= len(single), arch
    assert (
        cake_backend._registry_variants("router_fp32_gemm_kk", "sm_100a", False) == ()
    )


def _class_views(layout, out_f32, out_t, M, N, K):
    """Empty views of one (layout, output dtype, transposed store) class: ``kk`` = A rows contiguous along K and B
    contiguous along K (the forward), ``kn`` = B contiguous along N (the input gradient), ``nn`` = both MN-major (the
    weight gradient), ``nk`` = A MN-major only."""
    dt = torch.float32 if out_f32 else torch.bfloat16
    A = (
        torch.empty(M, K, dtype=torch.bfloat16)
        if layout[0] == "k"
        else torch.empty(K, M, dtype=torch.bfloat16).t()
    )
    B = (
        torch.empty(N, K, dtype=torch.bfloat16).t()
        if layout[1] == "k"
        else torch.empty(K, N, dtype=torch.bfloat16)
    )
    out = torch.empty(N, M, dtype=dt) if out_t else torch.empty(M, N, dtype=dt)
    return dict(A=A, B=B, out=out, transposed=out_t)


_CLASS_SWEEP_M = (4096, 16384)
_CLASS_SWEEP_N = (32, 64, 128, 256, 512, 1024, 2048, 4096)
_CLASS_SWEEP_K = (512, 2048, 4096, 5120, 8192, 16384)


@pytest.mark.parametrize("arch", sorted(cake_jit.KERNELS))
def test_registry_fallback_serves_every_registered_class(arch):
    """Every (layout, output dtype, transposed store) class with a generated program on ``arch`` plans a
    registered program for a synthetic row whose launcher plan is unregistered: the planner lands on the nearest
    generated knob variant (``knob_fallback``) - the hand-ordered variants, then the registry's own symbols - instead
    of raising when the launch is bound.  Rows with a measured rule are left out (the rule, not the fallback, names
    their program)."""
    registered = _registered(arch)
    classes = sorted(
        {
            cake_backend._template_class(cake_backend._template_tokens(t))
            for t in registered
        }
    )
    assert classes
    kw = dict(sm_count=SM_COUNT_OF_ARCH[arch], l2_bytes=L2_BYTES, arch=arch)
    exercised = set()
    for layout, out_f32, out_t in classes:
        for M in _CLASS_SWEEP_M:
            for N in _CLASS_SWEEP_N:
                for K in _CLASS_SWEEP_K:
                    v = _class_views(layout, out_f32, out_t, M, N, K)
                    try:
                        mirror, *_ = plan_dense_projection_gemm(
                            v["A"],
                            v["B"],
                            v["out"],
                            transposed_out=out_t,
                            _fallback=False,
                            **kw,
                        )
                    except ValueError:
                        continue  # a shape the class refuses
                    if mirror.template in registered:
                        continue
                    if cake_backend._measured_row_rule(
                        arch, mirror.a_mn, mirror.b_mn, out_f32, out_t, False, N, K, M
                    ):
                        continue
                    plan, *_ = plan_dense_projection_gemm(
                        v["A"], v["B"], v["out"], transposed_out=out_t, **kw
                    )
                    where = (
                        arch,
                        layout,
                        out_f32,
                        out_t,
                        M,
                        N,
                        K,
                        mirror.template,
                        plan.template,
                    )
                    assert plan.template in registered, where
                    assert plan.knob_fallback, where
                    assert not (plan.swap_fallback or plan.rule_fallback), where
                    assert cake_backend._template_class(
                        cake_backend._template_tokens(plan.template)
                    ) == (layout, out_f32, out_t), where
                    exercised.add((layout, out_f32, out_t))
    # a class the sweep cannot push off the registry needs no fallback; the registries of today leave several off it
    assert exercised, arch


def test_registry_fallback_reaches_the_shapes_tranche_programs(monkeypatch):
    """The rows whose launcher plan the sm_100a export does not carry (forward N <= 128 and N = 256 at K 4096 / 5120,
    the swapped weight gradient at N <= 128 with M 4096 / 5120 / 8192): with the measured rules set aside, the
    fallback lands on the first registered program its hand-ordered variants name (``_FALLBACK_KNOB_VARIANTS``: the
    other ``pro`` forms of the launcher's own tile walk, then the plain-hint twin, the single-CTA form and the 64-row
    pair family) and marks the plan ``knob_fallback``.  The expected neighbour is read from the registry: the first
    candidate of the row's hand-ordered list that is registered."""
    monkeypatch.setattr(cake_backend, "ROW_RULES", {})
    P = "dense_proj_gemm_"
    kk_n128 = (
        P + "kk_n128_hen_pro15",
        P + "kk_n128_hen",
        P + "kk_n128_pro7",
        P + "kk_n128_pro15",
        P + "kk_n128",
        P + "kk_n128_hen_c1",
        P + "kk_n128_c1",
        P + "kk_n128_m64_hen_pro7",
        P + "kk_n128_m64_hen_pro15",
        P + "kk_n128_m64_hen",
    )
    kk_n256 = (
        P + "kk_n256_hen_pro15",
        P + "kk_n256_hen",
        P + "kk_n256_pro7",
        P + "kk_n256_pro15",
        P + "kk_n256",
        P + "kk_n256_hen_c1",
        P + "kk_n256_c1",
    )
    nn_n128_t = (
        P + "nn_n128_hen_t_pro15",
        P + "nn_n128_hen_t",
        P + "nn_n128_t_pro7",
        P + "nn_n128_t_pro15",
        P + "nn_n128_t",
        P + "nn_n128_hen_t_c1",
        P + "nn_n128_t_c1",
        P + "nn_n128_m64_hen_t_pro7",
        P + "nn_n128_m64_hen_t_pro15",
        P + "nn_n128_m64_hen_t",
    )
    nn_n128_f32_t = tuple(c.replace("_t", "_f32_t") for c in nn_n128_t)
    cases = (
        # layout, fp32 out, transposed, M, N, K, launcher template, hand-ordered candidates
        ("kk", False, False, 16384, 128, 4096, P + "kk_n128_hen_pro7", kk_n128),
        ("kk", False, False, 16384, 64, 5120, P + "kk_n128_hen_pro7", kk_n128),
        ("kk", False, False, 16384, 256, 4096, P + "kk_n256_hen_pro7", kk_n256),
        ("nn", False, True, 4096, 128, 16384, P + "nn_n128_hen_t_pro7", nn_n128_t),
        (
            "nn",
            True,
            True,
            4096,
            128,
            16384,
            P + "nn_n128_hen_f32_t_pro7",
            nn_n128_f32_t,
        ),
        ("nn", False, True, 8192, 32, 16231, P + "nn_n128_hen_t_pro7", nn_n128_t),
    )
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a")
    registered = _registered("sm_100a")
    for layout, out_f32, out_t, M, N, K, launcher, candidates in cases:
        neighbour = next((c for c in candidates if c in registered), None)
        assert neighbour is not None, (M, N, K, launcher)
        v = _class_views(layout, out_f32, out_t, M, N, K)
        mirror, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], transposed_out=out_t, _fallback=False, **kw
        )
        assert mirror.template == launcher and launcher not in registered, (
            M,
            N,
            K,
            mirror.template,
        )
        plan, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], transposed_out=out_t, **kw
        )
        assert plan.template == neighbour, (M, N, K, plan.template, neighbour)
        assert plan.knob_fallback and not plan.rule_fallback and not plan.swap_fallback
        # the single-CTA neighbour carries no tail policy and no pro; the 64-row neighbour keeps the hints
        if neighbour.endswith("_c1"):
            assert plan.cta1 and plan.sk_units == 0 and not plan.sk_sync
        if "_m64_" in neighbour:
            assert plan.cta_rows == 64 and plan.hints == ("evict_first", "none")
    # a caller-forced tail policy keeps the row on a registered program: a pair form admits it, the single-CTA
    # form (``cta1`` clears the tail knobs) stays reachable when no pair form is registered
    v = _class_views("kk", False, False, 16384, 128, 4096)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], sk=True, **kw)
    assert plan.template in registered and plan.knob_fallback, plan.template
    if plan.template.endswith("_c1"):
        assert plan.cta1 and plan.sk_units == 0 and not plan.sk_sync
    # on sm_107a the same weight-gradient row lands on the first registered candidate of its hand order as well
    kw107 = dict(sm_count=212, l2_bytes=L2_BYTES, arch="sm_107a")
    registered107 = _registered("sm_107a")
    neighbour107 = next(c for c in nn_n128_t if c in registered107)
    v = _class_views("nn", False, True, 4096, 128, 16384)
    mirror, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], transposed_out=True, _fallback=False, **kw107
    )
    plan, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], transposed_out=True, **kw107
    )
    assert (
        mirror.template == P + "nn_n128_hen_t_pro7"
        and mirror.template not in registered107
    )
    assert plan.template == neighbour107 and plan.knob_fallback, (
        plan.template,
        neighbour107,
    )


def test_registry_fallback_decodes_the_nearest_generated_program(monkeypatch):
    """A class whose only generated programs carry knobs outside the hand-ordered variants (a narrow MN-major B
    panel, the synchronised stream-K slab path) still plans: the second pass decodes the registry's symbols into
    knobs (``_registry_variants``) and the plan carries the decoded program's knobs."""
    fake = {
        "sm_100a": {
            "dense_proj_gemm_nn_n256_sks_sb2_pro7": "a",
            "dense_proj_gemm_nn_n160_m256_bz64_pro15": "b",
            "dense_proj_gemm_kk_n256_pro7": "c",
        }
    }
    monkeypatch.setattr(cake_backend, "KERNELS", fake)
    monkeypatch.setattr(cake_backend, "ROW_RULES", {})
    kw = dict(sm_count=148, l2_bytes=L2_BYTES, arch="sm_100a")
    v = _class_views("nn", False, False, 4096, 4096, 16384)
    mirror, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], _fallback=False, **kw
    )
    assert mirror.template not in fake["sm_100a"]
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert plan.template in (
        "dense_proj_gemm_nn_n256_sks_sb2_pro7",
        "dense_proj_gemm_nn_n160_m256_bz64_pro15",
    )
    assert plan.knob_fallback
    knobs = cake_backend._template_knobs(plan.template)
    assert (plan.block_n, plan.cta_rows, plan.b_swz, plan.sk_sync, plan.sk_slab) == (
        knobs["block_n"],
        knobs["cta_rows"],
        knobs["b_swz"],
        knobs["sk_sync"],
        knobs["sk_slab"],
    )
    # a class without a program stays unregistered (the launch binding raises, not the planner) and unflagged
    v = _class_views("kn", False, False, 4096, 4096, 16384)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
    assert plan.template not in fake["sm_100a"]
    assert not (plan.knob_fallback or plan.rule_fallback or plan.swap_fallback)
    # a caller-forced knob wins over a decoded one: the forced 256-row tile never reaches the 128-row sync program
    v = _class_views("nn", False, False, 4096, 4096, 16384)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], cta_rows=256, **kw)
    assert (
        plan.template == "dense_proj_gemm_nn_n160_m256_bz64_pro15"
        and plan.knob_fallback
    )


# ---------------------------------------------------------------------------
# The measured rows of the shapes tranche: (ROW_RULES key, T, SM count of the measuring device, the program the
# measurement selected).  Generated from the measurement records of the tranche; the rule of each key (or, for a row
# whose choice is the planner's default, the default) must plan exactly this program.
# ---------------------------------------------------------------------------
# fmt: off
MEASURED_SHAPE_ROWS = (
    (('sm_100a', False, False, False, False, False, 32, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 32, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 32, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 32, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 48, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 48, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 48, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 48, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 64, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 64, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 64, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 64, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 128, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 128, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 128, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 128, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_100a', False, False, False, False, False, 128, 8192, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_pro15'),
    (('sm_100a', False, False, False, False, False, 128, 8192, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_pro15'),
    (('sm_100a', False, False, False, False, False, 512, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_100a', False, False, False, False, False, 512, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_100a', False, False, False, False, False, 512, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_100a', False, False, False, False, False, 512, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_100a', False, False, False, False, False, 1024, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 1024, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 1024, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 1024, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 4096, 2048, None), 16172, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 4096, 2048, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 4096, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 4096, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 4096, 12288, None), 16231, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_100a', False, False, False, False, False, 4096, 12288, None), 16384, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_100a', False, False, False, False, False, 5120, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 5120, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 5120, 17408, None), 16231, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_100a', False, False, False, False, False, 5120, 17408, None), 16384, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_100a', False, False, False, False, False, 12288, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 12288, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 16384, 2048, None), 16172, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 16384, 2048, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 17408, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, False, False, False, False, 17408, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 2048, 4096, None), 16172, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 2048, 4096, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd3_c1'),
    (('sm_100a', False, True, False, False, False, 4096, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd3_c1'),
    (('sm_100a', False, True, False, False, False, 4096, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 1024, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 1024, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 4096, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 4096, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 4096, 12288, None), 16231, 152, 'dense_proj_gemm_kn_n256_m256_ov_ht_cg_pro7'),
    (('sm_100a', False, True, False, False, False, 4096, 12288, None), 16384, 152, 'dense_proj_gemm_kn_n256_m256_ov_ht_cg_pro7'),
    (('sm_100a', False, True, False, False, False, 5120, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd3_c1'),
    (('sm_100a', False, True, False, False, False, 5120, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd3_c1'),
    (('sm_100a', False, True, False, False, False, 5120, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 1024, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 1024, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 5120, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 5120, None), 16384, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 5120, 17408, None), 16231, 152, 'dense_proj_gemm_kn_n256_m256_ht_or_pro7'),
    (('sm_100a', False, True, False, False, False, 5120, 17408, None), 16384, 152, 'dense_proj_gemm_kn_n256_m256_ht_or_pro7'),
    (('sm_100a', False, True, False, False, False, 8192, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 8192, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_pro15'),
    (('sm_100a', False, True, False, False, False, 12288, 4096, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 12288, 4096, None), 16384, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 17408, 5120, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, False, False, False, 17408, 5120, None), 16384, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 4096, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_100a', False, True, True, False, False, 4096, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_100a', False, True, True, False, False, 5120, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 5120, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_100a', False, True, True, False, False, 5120, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_100a', False, True, True, False, False, 8192, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', False, True, True, False, False, 8192, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_100a', True, True, False, False, False, 4096, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n224_hne_cg_pro15'),
    (('sm_100a', True, True, False, False, False, 4096, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n224_hne_cg_pro15'),
    (('sm_100a', True, True, False, False, False, 4096, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n224_m256_pro7'),
    (('sm_100a', True, True, False, False, False, 4096, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n224_m256_pro7'),
    (('sm_100a', True, True, False, False, False, 4096, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_100a', True, True, False, False, False, 4096, None, 12288), 16384, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_100a', True, True, False, False, False, 5120, None, 512), 16231, 152, 'dense_proj_gemm_nn_n192_bz64_hne_sks_sb2_pro15'),
    (('sm_100a', True, True, False, False, False, 5120, None, 512), 16384, 152, 'dense_proj_gemm_nn_n192_bz64_hne_sks_sb2_pro15'),
    (('sm_100a', True, True, False, False, False, 5120, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_cg_pro15'),
    (('sm_100a', True, True, False, False, False, 5120, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_cg_pro15'),
    (('sm_100a', True, True, False, False, False, 5120, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_100a', True, True, False, False, False, 5120, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_100a', True, True, False, False, False, 6144, None, 12288), 16172, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_100a', True, True, False, False, False, 6144, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_100a', True, True, False, False, False, 12288, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_100a', True, True, False, False, False, 12288, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_100a', True, True, False, False, False, 12288, None, 6144), 16172, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_100a', True, True, False, False, False, 12288, None, 6144), 16231, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_100a', True, True, False, True, False, 32, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 32, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 32, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 32, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 48, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 48, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 48, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 48, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 64, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 64, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 64, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 64, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 128, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 128, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 128, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 128, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 128, None, 8192), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 128, None, 8192), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 512, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m64_hen_t_pro15'),
    (('sm_100a', True, True, False, True, False, 512, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m64_hen_t_pro15'),
    (('sm_100a', True, True, True, False, False, 4096, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n256_f32_tma1_cg_pro15'),
    (('sm_100a', True, True, True, False, False, 4096, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n256_f32_tma1_cg_pro15'),
    (('sm_100a', True, True, True, False, False, 4096, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, False, False, 4096, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, False, False, 4096, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_100a', True, True, True, False, False, 4096, None, 12288), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_100a', True, True, True, False, False, 5120, None, 512), 16231, 152, 'dense_proj_gemm_nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15'),
    (('sm_100a', True, True, True, False, False, 5120, None, 512), 16384, 152, 'dense_proj_gemm_nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15'),
    (('sm_100a', True, True, True, False, False, 5120, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_f32_v8_ef_cg_pro15'),
    (('sm_100a', True, True, True, False, False, 5120, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_f32_v8_ef_cg_pro15'),
    (('sm_100a', True, True, True, False, False, 5120, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, False, False, 5120, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, False, False, 6144, None, 12288), 16172, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, False, False, 6144, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, False, False, 12288, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_100a', True, True, True, False, False, 12288, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_100a', True, True, True, False, False, 12288, None, 6144), 16172, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, False, False, 12288, None, 6144), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_100a', True, True, True, True, False, 32, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 32, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 32, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 32, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 48, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 48, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 48, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 48, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 64, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 64, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 64, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 64, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 128, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 128, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 128, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 128, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 128, None, 8192), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 128, None, 8192), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 512, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m64_hen_f32_t_pro15'),
    (('sm_100a', True, True, True, True, False, 512, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m64_hen_f32_t_pro15'),
    (('sm_103a', False, False, False, False, False, 32, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 32, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 32, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 32, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 48, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 48, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 48, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 48, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 64, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 64, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 64, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 64, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 128, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 128, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 128, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 128, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_c1'),
    (('sm_103a', False, False, False, False, False, 128, 8192, None), 16231, 152, 'dense_proj_gemm_kk_n128_hen_pro15'),
    (('sm_103a', False, False, False, False, False, 128, 8192, None), 16384, 152, 'dense_proj_gemm_kk_n128_hen_pro15'),
    (('sm_103a', False, False, False, False, False, 512, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_103a', False, False, False, False, False, 512, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_103a', False, False, False, False, False, 512, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_103a', False, False, False, False, False, 512, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_pro7'),
    (('sm_103a', False, False, False, False, False, 1024, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 1024, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 1024, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 1024, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 4096, 2048, None), 16172, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 4096, 2048, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 4096, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 4096, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_ht_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 4096, 12288, None), 16231, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_103a', False, False, False, False, False, 4096, 12288, None), 16384, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_103a', False, False, False, False, False, 5120, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 5120, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 5120, 17408, None), 16231, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_103a', False, False, False, False, False, 5120, 17408, None), 16384, 152, 'dense_proj_gemm_kk_n256_m256_ov_ht_cg_pro7'),
    (('sm_103a', False, False, False, False, False, 12288, 4096, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 12288, 4096, None), 16384, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 16384, 2048, None), 16172, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 16384, 2048, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 17408, 5120, None), 16231, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, False, False, False, False, 17408, 5120, None), 16384, 152, 'dense_proj_gemm_kk_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 2048, 4096, None), 16172, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 2048, 4096, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_w8rs'),
    (('sm_103a', False, True, False, False, False, 4096, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_w8rs'),
    (('sm_103a', False, True, False, False, False, 4096, 1024, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 1024, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 4096, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 4096, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 4096, 12288, None), 16231, 152, 'dense_proj_gemm_kn_n256_m256_ov_ht_cg_pro7'),
    (('sm_103a', False, True, False, False, False, 4096, 12288, None), 16384, 152, 'dense_proj_gemm_kn_n256_m256_ov_ht_cg_pro7'),
    (('sm_103a', False, True, False, False, False, 5120, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_ht_tma2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd3_c1'),
    (('sm_103a', False, True, False, False, False, 5120, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd3_c1'),
    (('sm_103a', False, True, False, False, False, 5120, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_w8r'),
    (('sm_103a', False, True, False, False, False, 5120, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_w8r'),
    (('sm_103a', False, True, False, False, False, 5120, 1024, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_sw25_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 1024, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma1_pd2_sw25_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 5120, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 5120, None), 16384, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 5120, 17408, None), 16231, 152, 'dense_proj_gemm_kn_n256_m256_ht_or_pro7'),
    (('sm_103a', False, True, False, False, False, 5120, 17408, None), 16384, 152, 'dense_proj_gemm_kn_n256_m256_ht_or_pro7'),
    (('sm_103a', False, True, False, False, False, 8192, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_pro15'),
    (('sm_103a', False, True, False, False, False, 8192, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_tma2_pd2_pro15'),
    (('sm_103a', False, True, False, False, False, 12288, 4096, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 12288, 4096, None), 16384, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 17408, 5120, None), 16231, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, False, False, False, 17408, 5120, None), 16384, 152, 'dense_proj_gemm_kn_n256_cg_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 4096, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_103a', False, True, True, False, False, 4096, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_103a', False, True, True, False, False, 5120, 32, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 32, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 48, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 48, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 64, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 64, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 5120, 512, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_103a', False, True, True, False, False, 5120, 512, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma1_pro7'),
    (('sm_103a', False, True, True, False, False, 8192, 128, None), 16231, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', False, True, True, False, False, 8192, 128, None), 16384, 152, 'dense_proj_gemm_kn_n256_f32_tma2_pd2_pro15'),
    (('sm_103a', True, True, False, False, False, 4096, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n224_hne_cg_pro15'),
    (('sm_103a', True, True, False, False, False, 4096, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n224_hne_cg_pro15'),
    (('sm_103a', True, True, False, False, False, 4096, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n224_m256_pro7'),
    (('sm_103a', True, True, False, False, False, 4096, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n224_m256_pro7'),
    (('sm_103a', True, True, False, False, False, 4096, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_103a', True, True, False, False, False, 4096, None, 12288), 16384, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_103a', True, True, False, False, False, 5120, None, 512), 16231, 152, 'dense_proj_gemm_nn_n192_bz64_hne_sks_sb2_pro15'),
    (('sm_103a', True, True, False, False, False, 5120, None, 512), 16384, 152, 'dense_proj_gemm_nn_n192_bz64_hne_sks_sb2_pro15'),
    (('sm_103a', True, True, False, False, False, 5120, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_pro15'),
    (('sm_103a', True, True, False, False, False, 5120, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_pro15'),
    (('sm_103a', True, True, False, False, False, 5120, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_103a', True, True, False, False, False, 5120, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_103a', True, True, False, False, False, 6144, None, 12288), 16172, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_103a', True, True, False, False, False, 6144, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_103a', True, True, False, False, False, 12288, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_103a', True, True, False, False, False, 12288, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m256_skx_skf_pro7'),
    (('sm_103a', True, True, False, False, False, 12288, None, 6144), 16172, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_103a', True, True, False, False, False, 12288, None, 6144), 16231, 152, 'dense_proj_gemm_nn_n256_m256_sks_sb2_pro7'),
    (('sm_103a', True, True, False, True, False, 32, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 32, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 32, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 32, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 48, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 48, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 48, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 48, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 64, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 64, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 64, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 64, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 128, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 128, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 128, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 128, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_t_sks_pro7'),
    (('sm_103a', True, True, False, True, False, 128, None, 8192), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 128, None, 8192), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 512, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m64_hen_t_pro15'),
    (('sm_103a', True, True, False, True, False, 512, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m64_hen_t_pro15'),
    (('sm_103a', True, True, True, False, False, 4096, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n224_f32_v8_ef_pro15'),
    (('sm_103a', True, True, True, False, False, 4096, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n224_f32_v8_ef_pro15'),
    (('sm_103a', True, True, True, False, False, 4096, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, False, False, 4096, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, False, False, 4096, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_103a', True, True, True, False, False, 4096, None, 12288), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_103a', True, True, True, False, False, 5120, None, 512), 16231, 152, 'dense_proj_gemm_nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15'),
    (('sm_103a', True, True, True, False, False, 5120, None, 512), 16384, 152, 'dense_proj_gemm_nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15'),
    (('sm_103a', True, True, True, False, False, 5120, None, 1024), 16231, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_f32_v8_ef_pro15'),
    (('sm_103a', True, True, True, False, False, 5120, None, 1024), 16384, 152, 'dense_proj_gemm_nn_n160_m256_bz64_hne_f32_v8_ef_pro15'),
    (('sm_103a', True, True, True, False, False, 5120, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, False, False, 5120, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, False, False, 6144, None, 12288), 16172, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, False, False, 6144, None, 12288), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, False, False, 12288, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_103a', True, True, True, False, False, 12288, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_pro7'),
    (('sm_103a', True, True, True, False, False, 12288, None, 6144), 16172, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, False, False, 12288, None, 6144), 16231, 152, 'dense_proj_gemm_nn_n256_m256_f32_tma1_sks_sb3_pro7'),
    (('sm_103a', True, True, True, True, False, 32, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 32, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 32, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 32, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 48, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 48, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 48, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 48, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 64, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 64, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 64, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 64, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 128, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 128, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 128, None, 5120), 16231, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 128, None, 5120), 16384, 152, 'dense_proj_gemm_nn_n128_hen_f32_t_sks_pro7'),
    (('sm_103a', True, True, True, True, False, 128, None, 8192), 16231, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 128, None, 8192), 16384, 152, 'dense_proj_gemm_nn_n128_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 512, None, 4096), 16231, 152, 'dense_proj_gemm_nn_n256_m64_hen_f32_t_pro15'),
    (('sm_103a', True, True, True, True, False, 512, None, 4096), 16384, 152, 'dense_proj_gemm_nn_n256_m64_hen_f32_t_pro15'),
)
# fmt: on
# Programs the measured rows name that the registry of this tree does not carry yet, per architecture: the consistency
# test tolerates them until the export registers them and fails once one of them lands without leaving this set.  Empty
# sets once the export ships the tranche: then every measured row runs its measured program.
PENDING_PROGRAMS: dict[str, frozenset[str]] = {
    "sm_100a": frozenset(),
    "sm_103a": frozenset(),
}


def _key_views(key, T):
    """The views of a ROW_RULES key at sequence length ``T`` (M for the forward / input gradient keys, K for the
    weight-gradient keys)."""
    arch, a_mn, b_mn, out_f32, out_t, batched, N, K, M = key
    assert not batched  # the tranche has no batched rows
    layout = ("n" if a_mn else "k") + ("n" if b_mn else "k")
    return _class_views(
        layout, out_f32, out_t, T if M is None else M, N, T if K is None else K
    )


@pytest.mark.parametrize("arch", sorted({row[0][0] for row in MEASURED_SHAPE_ROWS}))
def test_measured_shape_rules_plan_their_programs(arch):
    """Every measured row of the shapes tranche plans the program its measurement selected (its rule reproduces the
    measured knobs through the planner, at the measuring device's SM count); where this tree registers the
    architecture, that program is registered or pending (``PENDING_PROGRAMS``), a registered one runs as is (no
    fallback) and a pending one still runs a registered neighbour (the registry fallback)."""
    rows = [row for row in MEASURED_SHAPE_ROWS if row[0][0] == arch]
    assert rows
    registered = _registered(arch)
    for key, T, sm_count, template in rows:
        v = _key_views(key, T)
        kw = dict(
            sm_count=sm_count, l2_bytes=L2_BYTES, transposed_out=key[4], arch=arch
        )
        mirror, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], _fallback=False, **kw
        )
        assert mirror.template == template, (key, T, mirror.template)
        if arch not in cake_jit.KERNELS:
            continue  # the registry of this architecture has not landed: the plans are checked, the programs cannot be
        assert template in registered or template in PENDING_PROGRAMS[arch], (
            key,
            T,
            template,
        )
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
        if template in registered:
            assert plan.template == template, (key, T, plan.template)
            assert not (
                plan.rule_fallback or plan.knob_fallback or plan.swap_fallback
            ), (key, T)
        else:
            assert plan.template in registered, (key, T, plan.template)
    # a program that lands leaves the pending set (the set stays honest; it is empty once the tranche ships)
    assert not (PENDING_PROGRAMS.get(arch, frozenset()) & registered), sorted(
        PENDING_PROGRAMS[arch] & registered
    )


def _owner_rows():
    """Every row the package owns, as planner views with their planner keywords: the GLM-5.2 projection and MLA
    rows at the contract's two T and the measured rows of the shapes tranche at their T."""
    for row in PROJECTION_ROWS:
        for op in ("fwd", "dgrad", "wgrad"):
            for out_dtype in ("bf16",) if op == "fwd" else ("bf16", "f32"):
                for T in (16172, 16231):
                    v = _views("proj", row, op, out_dtype, T)
                    yield (
                        (row, op, out_dtype, T),
                        v,
                        dict(transposed_out=v.get("transposed", False)),
                    )
    for row in MLA_ROWS:
        for op in ("fwd", "dgrad", "wgrad"):
            for T in (16172, 16231):
                yield (row, op, "bf16", T), _views("mla", row, op, "bf16", T), {}
    seen = set()
    for key, T, _sm, _template in MEASURED_SHAPE_ROWS:
        if (key[1:], T) in seen:
            continue
        seen.add((key[1:], T))
        yield ("shape", *key[1:], T), _key_views(key, T), dict(transposed_out=key[4])


def _plan_signature(plan):
    """Every planned field but the per-architecture shared-memory limit: the instance (template) and the plan."""
    fields = dataclasses.asdict(plan)
    fields.pop("smem_limit", None)
    fields.pop("arch", None)
    return fields


def _own_measured_rule(arch, plan):
    """The ``ROW_RULES`` entry ``arch`` itself holds for the planned row (the three ragged lookups, no parent)."""
    ident = (
        arch,
        plan.a_mn,
        plan.b_mn,
        plan.out_f32,
        plan.transposed_out,
        plan.L > 1,
    )
    for tail in (
        (plan.N, plan.K, plan.M),
        (plan.N, plan.K, None),
        (plan.N, None, plan.M),
    ):
        if ident + tail in ROW_RULES:
            return ROW_RULES[ident + tail]
    return None


def test_sm103a_inherits_the_sm100a_rules_where_it_has_none():
    """sm_103a registers every sm_100a program, but its lanes measured only the shapes tranche: a row without an
    sm_103a rule plans with the sm_100a rule (the same instance-key knobs, the same program, the same weight-gradient
    orientation) and runs that program where the sm_103a registry has it - the registry decides, no template is
    named here; a row the GB300 lanes measured keeps its own rule."""
    assert cake_backend.RULE_PARENT == {"sm_103a": "sm_100a"}
    keys = lambda arch: {k[1:] for k in ROW_RULES if k[0] == arch}  # noqa: E731
    own, inherited = keys("sm_103a"), keys("sm_100a") - keys("sm_103a")
    assert inherited and not (own & inherited)
    for arch_of, table in (("sm_100a", inherited), ("sm_103a", own)):
        for key in table:
            a_mn, b_mn, out_f32, out_t, batched, N, K, M = key
            dims = (N, 16231 if K is None else K, 16231 if M is None else M)
            assert (
                cake_backend._measured_row_rule(
                    "sm_103a", a_mn, b_mn, out_f32, out_t, batched, *dims
                )
                == ROW_RULES[(arch_of,) + key]
            ), key
    sm = SM_COUNT_OF_ARCH["sm_103a"]
    registered = _registered("sm_103a")
    direct = fallen = own_rows = 0
    for label, v, extra in _owner_rows():
        kw = dict(sm_count=sm, l2_bytes=L2_BYTES, **extra)
        mirror, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], arch="sm_103a", _fallback=False, **kw
        )
        if _own_measured_rule("sm_103a", mirror) is not None:
            own_rows += 1  # a GB300-measured row: its own rule, checked by the measured-rows test
            continue
        parent, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], arch="sm_100a", _fallback=False, **kw
        )
        assert _plan_signature(mirror) == _plan_signature(parent), label
        plan, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], arch="sm_103a", **kw
        )
        if parent.template in registered:
            assert plan.template == parent.template, (label, plan.template)
            assert not (
                plan.rule_fallback or plan.knob_fallback or plan.swap_fallback
            ), label
            direct += 1
        else:
            assert plan.template in registered, (label, plan.template)
            fallen += 1
    assert own_rows and direct, (own_rows, direct, fallen)


def test_rule_inheritance_leaves_the_other_architectures_unchanged(monkeypatch):
    """Byte for byte: planned with the inheritance map emptied (the planner before it), every owner row plans the
    same instance and the same plan on sm_100a and sm_107a at their devices' SM counts; on sm_103a only rows
    without an own rule change, each to a program the sm_103a registry has."""

    def plans(arch):
        out = {}
        for label, v, extra in _owner_rows():
            try:
                plan, *_ = plan_dense_projection_gemm(
                    v["A"],
                    v["B"],
                    v["out"],
                    arch=arch,
                    sm_count=SM_COUNT_OF_ARCH[arch],
                    l2_bytes=L2_BYTES,
                    **extra,
                )
            except ValueError as e:
                out[label] = ("raises", str(e))
                continue
            out[label] = _plan_signature(plan)
        return out

    archs = ("sm_100a", "sm_103a", "sm_107a")
    after = {arch: plans(arch) for arch in archs}
    with monkeypatch.context() as m:
        m.setattr(cake_backend, "RULE_PARENT", {})
        before = {arch: plans(arch) for arch in archs}
    assert after["sm_100a"] == before["sm_100a"]
    assert after["sm_107a"] == before["sm_107a"]
    changed = [
        k for k in after["sm_103a"] if after["sm_103a"][k] != before["sm_103a"][k]
    ]
    assert (
        changed
    )  # the GLM-5.2 rows the GB300 benchmark found slower on the generic programs
    registered = _registered("sm_103a")
    for label in changed:
        sig = after["sm_103a"][label]
        assert isinstance(sig, dict) and sig["template"] in registered, label
        assert label[0] != "shape", label  # the GB300-measured rows do not move


def test_wgrad_swap_follows_the_measured_rule():
    """The weight-gradient orientation: a measured rule keyed in exactly one orientation (the swapped ``_t`` key
    ``(arch, nn, out_f32, True, False, N, None, K)`` or the direct key ``(arch, nn, out_f32, False, False, K, None, N)``)
    decides; without one - or without an architecture (the planner tests' views) - the launcher's padded-tile-work
    heuristic."""
    # the heuristic: ties keep the direct route
    assert not wgrad_swapped(512, 4096) and not wgrad_swapped(1024, 4096)
    # measured: 4096 -> 512 swaps on both architectures for bf16 and fp32, 5120 -> 512 was measured direct ...
    for arch in ("sm_100a", "sm_103a"):
        for out_f32 in (False, True):
            assert wgrad_swapped(512, 4096, arch=arch, out_f32=out_f32), (arch, out_f32)
            assert not wgrad_swapped(512, 5120, arch=arch, out_f32=out_f32), (
                arch,
                out_f32,
            )
    # ... and 4096 -> 1024 runs direct on both architectures (the round-2 sm_100a bf16 rule moved from the swapped
    # ``nn_n256_hen_t_pro15`` to the direct ``nn_n224_hne_cg_pro15`` the sm_103a rule names)
    for arch in ("sm_100a", "sm_103a"):
        for out_f32 in (False, True):
            assert not wgrad_swapped(1024, 4096, arch=arch, out_f32=out_f32), (
                arch,
                out_f32,
            )
    # rows with a rule in the heuristic's orientation and rows without a rule agree with the heuristic everywhere
    for arch in ("sm_100a", "sm_103a", "sm_107a", "sm_90a"):
        assert wgrad_swapped(32, 6144, arch=arch) and wgrad_swapped(
            576, 6144, arch=arch
        ), arch
        assert not wgrad_swapped(256, 6144, arch=arch) and not wgrad_swapped(
            6144, 2048, arch=arch
        ), arch
    # the views follow the decision and the swapped plan takes the measured program
    T = 16384
    G = torch.empty(T, 512, dtype=torch.bfloat16)
    X = torch.empty(T, 4096, dtype=torch.bfloat16)
    A, B, transposed = wgrad_views(G, X)
    assert not transposed and A.shape == (512, T) and B is X
    A, B, transposed = wgrad_views(G, X, arch="sm_100a")
    assert transposed and A.shape == (4096, T) and B is G
    plan, *_ = plan_dense_projection_gemm(
        A,
        B,
        torch.empty(512, 4096, dtype=torch.bfloat16),
        sm_count=152,
        l2_bytes=L2_BYTES,
        transposed_out=True,
        arch="sm_100a",
        _fallback=False,
    )
    assert plan.template == "dense_proj_gemm_nn_n256_m64_hen_t_pro15"
    # the measured rules never key one weight-gradient row in both orientations (the decision would be undefined)
    for key in ROW_RULES:
        arch, a_mn, b_mn, out_f32, out_t, batched, N, K, M = key
        if a_mn and b_mn and not batched and K is None:
            mirror = (arch, True, True, out_f32, not out_t, False, M, None, N)
            assert mirror not in ROW_RULES, (key, mirror)


def test_round2_knobs_name_their_programs():
    """The round-2 shapes tranche's build knobs: ``sk_first`` (``_skf`` directly after ``_skx``) and ``lw`` (``_lw``
    between ``_or`` and ``_pro``) in ``instance_key`` / ``instance_symbol``, their refusals, and the symbol decoder's
    round trip of the five programs the round-2 rules name (the spelling the export must register)."""
    nn, kn, kk = (
        dict(a_mn=True, b_mn=True),
        dict(a_mn=False, b_mn=True),
        dict(a_mn=False, b_mn=False),
    )
    key = cake_backend.instance_key(
        **nn, cta_rows=256, sk_exact=True, sk_first=True, pro=7
    )
    assert len(key) == 37 and key[32] is True and key[33] is False and key[34] is False
    assert (
        cake_backend.instance_symbol(key) == "dense_proj_gemm_nn_n256_m256_skx_skf_pro7"
    )
    key = cake_backend.instance_key(
        **kn, epi="tma", slots=2, htail=True, lw=True, pro=15
    )
    assert key[32] is False and key[33] is True
    assert (
        cake_backend.instance_symbol(key) == "dense_proj_gemm_kn_n256_ht_tma2_lw_pro15"
    )
    for knobs, template in (
        (
            dict(**kn, epi="tma", slots=2, pd=2, lw=True, pro=15),
            "dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15",
        ),
        (
            dict(**kn, epi="tma", slots=1, pd=2, lw=True, pro=15),
            "dense_proj_gemm_kn_n256_tma1_pd2_lw_pro15",
        ),
        (
            dict(**kn, htail=True, cgrp=True, pro=15),
            "dense_proj_gemm_kn_n256_ht_cg_pro15",
        ),
        (
            dict(**kk, htail=True, cgrp=True, pro=15),
            "dense_proj_gemm_kk_n256_ht_cg_pro15",
        ),
        (
            dict(**kn, cgrp=True, lw=True, ovr=False, pro=12),
            "dense_proj_gemm_kn_n256_cg_lw_pro12",
        ),
    ):
        assert (
            cake_backend.instance_symbol(cake_backend.instance_key(**knobs)) == template
        )
    # the knobs the symbol decoder reads back name the program again (the registry fallback's inverse)
    for template in (
        "dense_proj_gemm_nn_n256_m256_skx_skf_pro7",
        "dense_proj_gemm_kn_n256_ht_tma2_lw_pro15",
        "dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15",
        "dense_proj_gemm_kn_n256_tma1_pd2_lw_pro15",
        "dense_proj_gemm_kn_n256_ht_cg_pro15",
        "dense_proj_gemm_nn_n224_hne_cg_pro15",
    ):
        tokens = cake_backend._template_tokens(template)
        knobs = cake_backend._template_knobs(template, sk_exact_parts=16)
        assert knobs is not None, template
        layout, out_f32, out_t = cake_backend._template_class(tokens)
        key_knobs = {k: v for k, v in knobs.items() if v is not None}
        key_knobs["sk_exact"] = bool(knobs["sk_exact"])
        key = cake_backend.instance_key(
            a_mn=layout[0] == "n",
            b_mn=layout[1] == "n",
            out_f32=out_f32,
            out_t=out_t,
            **key_knobs,
        )
        assert cake_backend.instance_symbol(key) == template
        assert knobs["sk_first"] == ("skf" in tokens) and knobs["lw"] == (
            "lw" in tokens
        )
    # refusals: sk_first needs the exact split and the plain item walk; lw needs the pair with the MMA-warp allocator
    for bad in (
        dict(**nn, cta_rows=256, sk_first=True, pro=7),
        dict(**kn, epi="tma", slots=2, pd=2, sk_exact=True, sk_first=True, pro=15),
        dict(**nn, cta_rows=256, htail=True, sk_first=True, pro=7),
        dict(**nn, cta_rows=256, sk_sync=True, sk_first=True, pro=7),
        dict(**kn, cta1=True, lw=True),
        dict(**kn, lw=True, pro=3),
        dict(**kn, lw=True, pro=0),
        dict(**kn, lw=True),
    ):
        with pytest.raises(ValueError):
            cake_backend.instance_key(**bad)
    for ok in (
        dict(**kn, lw=True, pro=4),
        dict(**kn, lw=True, pro=7),
        dict(**kn, lw=True, pro=12),
    ):
        cake_backend.instance_key(**ok)
    # the single-CTA fallback variant clears both (instance_key refuses them there)
    assert {"sk_first", "lw"} <= set(cake_backend._CTA1_CLEARED)


def test_round2_rules_plan_their_knobs():
    """The round-2 rules through the planner at the measuring devices' SM count (152): the tail-first exact split of
    the two 4096 x 12288 weight gradients, the half-tail column-grouped programs of the 4096 squares, the 12-column
    raster groups of the N 5120 rows, the load-warp programs and the ``l2_256b`` promotion of the thin input gradients,
    and the direct 4096 -> 1024 weight gradient on sm_100a; a rule-derived ``sk_first`` / ``lw`` yields to a caller-forced
    form the program cannot carry."""
    kw = dict(sm_count=152, l2_bytes=L2_BYTES, _fallback=False)
    for arch in ("sm_100a", "sm_103a"):
        a = dict(kw, arch=arch)
        # WGR2 A rows: sk_exact 16 + sk_first, cta_rows 256, pro 7 (units first: items [0, 64) are the K parts)
        for M, N in ((12288, 4096), (4096, 12288)):
            for T, k_blocks in ((16384, 256), (16231, 254)):
                v = _class_views("nn", False, False, M, N, T)
                plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
                assert plan.template == "dense_proj_gemm_nn_n256_m256_skx_skf_pro7", (
                    arch,
                    M,
                    N,
                    T,
                    plan.template,
                )
                assert (plan.sk_exact, plan.sk_first, plan.cta_rows, plan.block_n) == (
                    16,
                    True,
                    256,
                    256,
                )
                assert (
                    plan.num_full,
                    plan.tail_tiles,
                    plan.sk_units,
                    plan.iters_per_unit,
                ) == (380, 4, 64, 16), (arch, M, N, T)
                assert (
                    plan.k_blocks == k_blocks
                    and plan.sk_iters == 16
                    and plan.ws_f32_elems > 0
                )
                # a caller-forced synchronised plan or single-CTA form drops the rule's order with its split
                forced, *_ = plan_dense_projection_gemm(
                    v["A"], v["B"], v["out"], sk_sync=True, **a
                )
                assert (
                    forced.sk_sync
                    and not forced.sk_first
                    and "_skf" not in forced.template
                )
                forced, *_ = plan_dense_projection_gemm(
                    v["A"], v["B"], v["out"], sk=False, **a
                )
                assert (
                    not forced.sk_exact
                    and not forced.sk_first
                    and forced.template == "dense_proj_gemm_nn_n256_m256_pro7"
                )
                with pytest.raises(ValueError):
                    plan_dense_projection_gemm(
                        v["A"], v["B"], v["out"], sk=False, sk_first=True, **a
                    )
        # LGF-R2: the 4096 squares take the half-tail column-grouped programs (group 16), the N 5120 rows group 12
        for layout, template in (
            ("kk", "dense_proj_gemm_kk_n256_ht_cg_pro15"),
            ("kn", "dense_proj_gemm_kn_n256_ht_cg_pro15"),
        ):
            v = _class_views(layout, False, False, 16384, 4096, 4096)
            plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
            assert (
                plan.template == template
                and plan.htail
                and plan.cgrp
                and plan.group_m == 16
            ), (arch, layout)
            assert (
                plan.num_full,
                plan.tail_tiles,
                plan.sk_units,
                plan.iters_per_unit,
            ) == (988, 36, 72, 64)
        for layout, N, K, template in (
            ("kk", 5120, 5120, "dense_proj_gemm_kk_n256_cg_pro15"),
            ("kn", 5120, 5120, "dense_proj_gemm_kn_n256_cg_pro15"),
            ("kk", 17408, 5120, "dense_proj_gemm_kk_n256_cg_pro15"),
        ):
            v = _class_views(layout, False, False, 16384, N, K)
            plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
            assert plan.template == template and plan.cgrp and plan.group_m == 12, (
                arch,
                layout,
                N,
                K,
            )
        # TK-R2: the one-K-step rows take the half-tail load-warp program the lane measured on both K layers
        # (``_ht_tma2_lw``); the half-tail PLAN applies where the half items fit the 76 pairs (K 4096: 1024 tiles,
        # tail 36 -> 72 half items) and the ``_ht`` program runs the whole-tile plan where they do not (K 5120: 1280
        # tiles, tail 64 > 38 - the plan the lane measured); the K 128 rows the pd 2 load-warp program
        for K_layer, plan_fields, half in (
            (4096, (988, 36, 72, 1), True),
            (5120, (1280, 0, 0, 1), False),
        ):
            template = "dense_proj_gemm_kn_n256_ht_tma2_lw_pro15"
            for N_layer in (32, 48):
                v = _class_views("kn", False, False, 16384, K_layer, N_layer)
                plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
                assert plan.template == template and plan.lw and plan.pd == 0, (
                    arch,
                    K_layer,
                    N_layer,
                    plan.template,
                )
                assert plan.htail and plan.htail_plan == half
                assert (
                    plan.num_full,
                    plan.tail_tiles,
                    plan.sk_units,
                    plan.iters_per_unit,
                ) == plan_fields
                # the load-warp form yields to a caller-forced single-CTA form or a pro without bit 4
                forced, *_ = plan_dense_projection_gemm(
                    v["A"], v["B"], v["out"], cta1=True, **a
                )
                assert not forced.lw and forced.template.endswith("_c1")
                forced, *_ = plan_dense_projection_gemm(
                    v["A"], v["B"], v["out"], pro=3, **a
                )
                assert not forced.lw and forced.template.endswith("_pro3")
                with pytest.raises(ValueError):
                    plan_dense_projection_gemm(
                        v["A"], v["B"], v["out"], pro=3, lw=True, **a
                    )
        v = _class_views("kn", False, False, 16384, 5120, 128)
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
        assert (
            plan.template == "dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15"
            and plan.pd == 2
            and plan.lw
        )
        assert plan.group_m == (32 if arch == "sm_103a" else 16)
    # the GB300-only round-2 picks: the 4096 -> 64 load-warp program, the l2_256b promotion of the N 512 input gradients
    a = dict(kw, arch="sm_103a")
    v = _class_views("kn", False, False, 16384, 4096, 64)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
    assert (
        plan.template == "dense_proj_gemm_kn_n256_tma2_pd2_lw_pro15"
        and plan.group_m == 32
    )
    # (the two rows' eight-warp CTA form is forced off here - its own test is
    # test_eight_warp_form_rules_plan_their_programs; the ten-warp program below is the row's program where it yields)
    v = _class_views("kn", False, False, 16384, 4096, 512)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], form="", **a)
    assert (
        plan.template == "dense_proj_gemm_kn_n256_tma1_pd2_lw_mg_pro15"
        and plan.promo == "l2_256b"
    )  # + the whole-tile tile resolve (test_mg_rules_plan_the_whole_tile_programs)
    v = _class_views("kn", False, False, 16384, 5120, 512)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], form="", **a)
    assert (
        plan.template == "dense_proj_gemm_kn_n256_tma1_pd2_mg_sw25_pro15"
        and plan.promo == "l2_256b"
        and plan.group_m == 12
    )  # + the paced output stores (test_sw_rules_plan_the_paced_store_programs)
    # ... and the GB200 rows that keep round 1
    a = dict(kw, arch="sm_100a")
    v = _class_views("kn", False, False, 16384, 4096, 64)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
    assert plan.template == "dense_proj_gemm_kn_n256_tma2_pd3_c1" and not plan.lw
    v = _class_views("kn", False, False, 16384, 5120, 512)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
    assert (
        plan.template == "dense_proj_gemm_kn_n256_tma1_pd2_pro15"
        and plan.promo == "none"
        and plan.group_m == 8
    )
    # the column-grouped raster takes any group count; the pair raster still needs an even one
    v = _class_views("kk", False, False, 16384, 5120, 5120)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], group_m=3, **a)
    assert plan.group_m == 3 and plan.cgrp
    with pytest.raises(ValueError):
        plan_dense_projection_gemm(v["A"], v["B"], v["out"], group_m=3, cgrp=False, **a)
    # WGR2 E1: the 4096 -> 1024 bf16 weight gradient runs direct on sm_100a now (the sm_103a program)
    v = _class_views("nn", False, False, 1024, 4096, 16384)
    plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
    assert (
        plan.template == "dense_proj_gemm_nn_n224_hne_cg_pro15"
        and plan.block_n == 224
        and plan.cgrp
    )
    assert not wgrad_swapped(1024, 4096, arch="sm_100a", out_f32=False)


@pytest.mark.parametrize("T", [16172, 16231, 16384])
def test_mg_rules_plan_the_whole_tile_programs(T):
    """The whole-tile tile resolve (``mg``, instance-key field 35, ``_mg`` directly after ``_lw``): the three sm_103a
    input-gradient rules that carry it (4096 -> 512 with the load-warp form, 5120 -> 512, the kv_a 6144 -> 576) plan
    the ``_mg`` twin of the row's program and nothing else changes (every other plan field equals the plan with ``mg``
    forced off); the rows without the resolve (4096 -> 1024; 5120 -> 1024 carries the paced output stores alone), the
    fp32 twins and sm_100a keep their programs.
    A rule's ``mg`` yields where the plan carries stream-K units or a caller forces a form the program excludes; a
    caller-forced ``mg`` is refused there (``instance_key`` for the forms, the planner for the stream-K units)."""
    kw = dict(sm_count=152, l2_bytes=L2_BYTES, _fallback=False)
    a = dict(kw, arch="sm_103a")
    rows = (
        (
            4096,
            512,
            "dense_proj_gemm_kn_n256_tma1_pd2_lw_mg_pro15",
            16,
            "l2_256b",
            True,
        ),
        (
            5120,
            512,
            "dense_proj_gemm_kn_n256_tma1_pd2_mg_sw25_pro15",
            12,
            "l2_256b",
            False,
        ),
        (6144, 576, "dense_proj_gemm_kn_n256_tma1_pd2_mg_pro15", 12, "none", False),
    )
    for K_layer, N_layer, template, group_m, promo, lw in rows:
        key = ("sm_103a", False, True, False, False, False, K_layer, N_layer, None)
        rule = ROW_RULES[key]
        assert rule["mg"] is True and rule["sk"] is False, key
        assert rule["form"] in cake_backend.W8_FORMS, key
        skip = ("mg", "sk", "form", "group_m")
        assert {k: v for k, v in rule.items() if k not in skip} == {
            k: v
            for k, v in cake_backend._measured_row_rule(
                "sm_100a" if (K_layer, N_layer) == (6144, 576) else "sm_103a",
                *key[1:6],
                K_layer,
                N_layer,
                T,
            ).items()
            if k not in skip
        }, (
            key
        )  # the kv_a rule restates the inherited sm_100a knobs; the others extend their own rule
        # (these three rows carry the eight-warp CTA form too: it is forced off here, the whole-tile tile resolve is
        # the row's program where the form yields; test_eight_warp_form_rules_plan_their_programs tests the form)
        v = _class_views("kn", False, False, T, K_layer, N_layer)
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], form="", **a)
        assert plan.template == template, (key, T, plan.template)
        assert plan.mg and plan.lw == lw and plan.pd == 2
        assert (plan.epi, plan.slots, plan.group_m, plan.promo, plan.pro) == (
            "tma",
            1,
            group_m,
            promo,
            15,
        ), key
        assert plan.sk_units == 0 and plan.tail_tiles == 0
        assert plan.num_full == plan.pair_tiles and plan.ws_f32_elems == 0
        assert plan.grid == (2 * plan.pair_tiles, 1, 1) and not plan.cta1
        # the twin without the resolve: the same plan, only the program differs
        plain, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], mg=False, **a)
        assert not plain.mg and plain.template == template.replace("_mg", "")
        assert {**_plan_signature(plan), "template": None, "mg": None} == {
            **_plan_signature(plain),
            "template": None,
            "mg": None,
        }, key
        # the symbol decoder reads the knob back (the registry fallback's inverse) and names the program again
        knobs = cake_backend._template_knobs(template)
        assert knobs is not None and knobs["mg"] and knobs["lw"] == lw, template
        key_knobs = {k: v for k, v in knobs.items() if v is not None}
        key_knobs["sk_exact"] = bool(knobs["sk_exact"])
        assert (
            cake_backend.instance_symbol(
                cake_backend.instance_key(a_mn=False, b_mn=True, **key_knobs)
            )
            == template
        )
        # a caller forcing a form the program excludes: the rule's mg yields and the forced form plans
        forced, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cta1=True, **a
        )
        assert not forced.mg and forced.template.endswith("_c1"), forced.template
        forced, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cgrp=True, **a
        )
        assert not forced.mg and forced.cgrp and "_mg" not in forced.template
        assert "_cg" in forced.template
        with pytest.raises(ValueError, match=r"mg \(whole-tile"):
            plan_dense_projection_gemm(
                v["A"], v["B"], v["out"], cgrp=True, mg=True, **a
            )
    # the K 512 rows: ``sk=True`` admits no split (8 K steps = SK_MIN_ITERS: 36 units over the 36 tail tiles of the
    # 4096 -> 512 row), so the plan stays whole tiles and the resolve stays; kv_a (9 K steps, 1536 tiles on 76 pairs):
    # 18 units over the 16 tail tiles, so the rule's mg yields to the plain program, and forcing mg on that plan is
    # refused
    v = _class_views("kn", False, False, T, 4096, 512)
    sk, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], sk=True, form="", **a)
    assert sk.sk_units == 0 and sk.mg
    assert sk.template == "dense_proj_gemm_kn_n256_tma1_pd2_lw_mg_pro15"
    v = _class_views("kn", False, False, T, 6144, 576)
    sk, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], sk=True, **a)
    assert (sk.sk_units, sk.tail_tiles, sk.iters_per_unit) == (18, 16, 8)
    assert not sk.mg and sk.template == "dense_proj_gemm_kn_n256_tma1_pd2_pro15"
    with pytest.raises(ValueError, match="whole-tile plan"):
        plan_dense_projection_gemm(v["A"], v["B"], v["out"], sk=True, mg=True, **a)
    # the rows without the resolve keep their program (5120 -> 1024 paces its stores), the fp32 twins and sm_100a
    # never carry the resolve
    for K_layer, N_layer, template in (
        (4096, 1024, "dense_proj_gemm_kn_n256_tma1_pd2_pro15"),
        (5120, 1024, "dense_proj_gemm_kn_n256_tma1_pd2_sw25_pro15"),
    ):
        v = _class_views("kn", False, False, T, K_layer, N_layer)
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **a)
        assert not plan.mg, (K_layer, N_layer)
        assert plan.template == template, (K_layer, N_layer)
        assert (
            "mg"
            not in ROW_RULES[
                ("sm_103a", False, True, False, False, False, K_layer, N_layer, None)
            ]
        )
    for K_layer, N_layer, _t, _g, _p, _lw in rows:
        f = _class_views("kn", True, False, T, K_layer, N_layer)
        plan, *_ = plan_dense_projection_gemm(f["A"], f["B"], f["out"], **a)
        assert not plan.mg and "_mg" not in plan.template, (K_layer, N_layer, "f32")
        v = _class_views("kn", False, False, T, K_layer, N_layer)
        plan, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], **dict(kw, arch="sm_100a")
        )
        assert not plan.mg and "_mg" not in plan.template, (K_layer, N_layer, "sm_100a")
    for key in ROW_RULES:
        assert key[0] == "sm_103a" or "mg" not in ROW_RULES[key], key
    # the key: field 35, the symbol token after ``_lw``; the forms the program excludes are refused with it
    kn = dict(a_mn=False, b_mn=True, epi="tma", slots=1, pd=2, pro=15)
    key = cake_backend.instance_key(**kn, mg=True)
    assert len(key) == 37 and key[34] is True and key[33] is False
    assert (
        cake_backend.instance_symbol(key) == "dense_proj_gemm_kn_n256_tma1_pd2_mg_pro15"
    )
    assert (
        cake_backend.instance_symbol(cake_backend.instance_key(**kn, lw=True, mg=True))
        == "dense_proj_gemm_kn_n256_tma1_pd2_lw_mg_pro15"
    )
    assert cake_backend.instance_key(**kn)[34] is False
    plain = dict(a_mn=False, b_mn=True, pro=15)
    for bad in (
        dict(batch_group=4),
        dict(cgrp=True),
        dict(sk_exact=True),
        dict(sk_sync=True),
        dict(sk_sync=True, sk_slab=1),
    ):
        with pytest.raises(ValueError, match=r"mg \(whole-tile"):
            cake_backend.instance_key(**plain, mg=True, **bad)
    for bad in (dict(htail=True), dict(sk_exact=True, sk_first=True), dict(stt=True)):
        with pytest.raises(ValueError):
            cake_backend.instance_key(**plain, mg=True, **bad)
    # the single-CTA form admits the knob in the key (the planner keeps the measured pair programs' rules off it)
    assert (
        cake_backend.instance_symbol(
            cake_backend.instance_key(a_mn=False, b_mn=True, cta1=True, mg=True)
        )
        == "dense_proj_gemm_kn_n256_c1_mg"
    )
    assert "mg" not in cake_backend._CTA1_CLEARED


@pytest.mark.parametrize("T", [16172, 16231, 16384])
def test_sw_rules_plan_the_paced_store_programs(T):
    """The paced output stores (``sw``, instance-key field 36, ``_sw<n>`` directly after ``_mg`` and before ``_pro``):
    the two sm_103a input-gradient rules that carry it (5120 -> 512 with the whole-tile tile resolve, 5120 -> 1024
    without) plan the ``_sw25`` twin of the row's program and nothing else changes (every other plan field equals the
    plan with ``sw`` forced off); the other rows, the fp32 twins and sm_100a keep their programs.  A rule's ``sw``
    yields where the planned program is not the bf16 TMA-store pd-2 drain or a caller forces the single-CTA form; a
    caller-forced ``sw`` is refused there (``instance_key``, mirroring the kernel's refusal)."""
    kw = dict(sm_count=152, l2_bytes=L2_BYTES, _fallback=False)
    a = dict(kw, arch="sm_103a")
    rows = (
        (5120, 512, "dense_proj_gemm_kn_n256_tma1_pd2_mg_sw25_pro15", True),
        (5120, 1024, "dense_proj_gemm_kn_n256_tma1_pd2_sw25_pro15", False),
    )
    for K_layer, N_layer, template, mg in rows:
        key = ("sm_103a", False, True, False, False, False, K_layer, N_layer, None)
        assert ROW_RULES[key]["sw"] == 25, key
        # (the 5120 -> 512 row carries the eight-warp CTA form too: forced off here, the paced program is the row's
        # program where the form yields; test_eight_warp_form_rules_plan_their_programs tests the form)
        v = _class_views("kn", False, False, T, K_layer, N_layer)
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], form="", **a)
        assert plan.template == template, (key, T, plan.template)
        assert plan.sw == 25 and plan.mg == mg and not plan.out_f32
        assert (plan.epi, plan.slots, plan.pd, plan.pro) == ("tma", 1, 2, 15), key
        assert plan.sk_units == 0 and not plan.cta1
        # the twin without the pacing: the same plan, only the program differs
        plain, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], sw=0, **a)
        assert plain.sw == 0 and plain.template == template.replace("_sw25", "")
        assert {**_plan_signature(plan), "template": None, "sw": None} == {
            **_plan_signature(plain),
            "template": None,
            "sw": None,
        }, key
        # the symbol decoder reads the step count back and names the program again
        knobs = cake_backend._template_knobs(template)
        assert knobs is not None and knobs["sw"] == 25 and knobs["mg"] == mg, template
        key_knobs = {k: v for k, v in knobs.items() if v is not None}
        key_knobs["sk_exact"] = bool(knobs["sk_exact"])
        assert (
            cake_backend.instance_symbol(
                cake_backend.instance_key(a_mn=False, b_mn=True, **key_knobs)
            )
            == template
        )
        # a caller forcing a program outside the drain or the single-CTA form: the rule's sw yields and the forced
        # form plans; forcing sw with it is refused
        for forced_knobs in (dict(pd=0), dict(pd=3)):
            forced, *_ = plan_dense_projection_gemm(
                v["A"], v["B"], v["out"], **forced_knobs, **a
            )
            assert forced.sw == 0 and forced.pd == forced_knobs["pd"], forced.template
            assert "_sw" not in forced.template, forced.template
            with pytest.raises(ValueError, match=r"sw \(paced output stores\)"):
                plan_dense_projection_gemm(
                    v["A"], v["B"], v["out"], sw=25, **forced_knobs, **a
                )
        forced, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cta1=True, **a
        )
        assert forced.sw == 0 and forced.template.endswith("_c1"), forced.template
        # the fp32 twin and sm_100a never carry the pacing
        f = _class_views("kn", True, False, T, K_layer, N_layer)
        plan, *_ = plan_dense_projection_gemm(f["A"], f["B"], f["out"], **a)
        assert plan.sw == 0 and "_sw" not in plan.template, (K_layer, N_layer, "f32")
        plan, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], **dict(kw, arch="sm_100a")
        )
        assert plan.sw == 0 and "_sw" not in plan.template, (K_layer, N_layer)
    # the other sm_103a input-gradient rows keep their programs
    for K_layer, N_layer, template in (
        (4096, 512, "dense_proj_gemm_kn_n256_tma1_pd2_lw_mg_pro15"),
        (4096, 1024, "dense_proj_gemm_kn_n256_tma1_pd2_pro15"),
        (6144, 576, "dense_proj_gemm_kn_n256_tma1_pd2_mg_pro15"),
    ):
        v = _class_views("kn", False, False, T, K_layer, N_layer)
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], form="", **a)
        assert plan.sw == 0 and plan.template == template, (K_layer, N_layer)
    sw_rows = {("sm_103a", 5120, 512), ("sm_103a", 5120, 1024)}
    for key, rule in ROW_RULES.items():
        assert "sw" not in rule or (key[0], key[6], key[7]) in sw_rows, key
    # the key: field 36, the symbol token after ``_mg``; the knob is a step count on the bf16 TMA-store pd-2 drain
    kn = dict(a_mn=False, b_mn=True, epi="tma", slots=1, pd=2, pro=15)
    key = cake_backend.instance_key(**kn, mg=True, sw=25)
    assert len(key) == 37 and key[35] == 25 and key[34] is True
    assert (
        cake_backend.instance_symbol(key)
        == "dense_proj_gemm_kn_n256_tma1_pd2_mg_sw25_pro15"
    )
    assert (
        cake_backend.instance_symbol(cake_backend.instance_key(**kn, sw=25))
        == "dense_proj_gemm_kn_n256_tma1_pd2_sw25_pro15"
    )
    assert cake_backend.instance_key(**kn)[35] == 0
    assert cake_backend.instance_key(**kn, sw=0) == cake_backend.instance_key(**kn)
    assert cake_backend.instance_key(**kn, sw=12)[35] == 12
    for bad in (
        dict(**kn, sw=-1),
        dict(a_mn=False, b_mn=True, pro=15, sw=25),
        dict(a_mn=False, b_mn=True, epi="tma", slots=1, pd=0, pro=15, sw=25),
        dict(a_mn=False, b_mn=True, epi="tma", slots=1, pd=3, pro=15, sw=25),
        dict(
            a_mn=False,
            b_mn=True,
            out_f32=True,
            epi="tma",
            slots=2,
            pd=2,
            pro=15,
            sw=25,
        ),
    ):
        with pytest.raises(ValueError, match=r"sw \(paced output stores\)"):
            cake_backend.instance_key(**bad)
    assert "sw" not in cake_backend._CTA1_CLEARED


# the ROW_RULES keys that carry the eight-warp CTA form (the three sm_103a input-gradient rows)
EIGHT_WARP_ROWS = sorted(
    (key for key, rule in ROW_RULES.items() if rule.get("form")), key=repr
)


@pytest.mark.parametrize("T", [16172, 16231, 16384])
def test_eight_warp_form_rules_plan_their_programs(T, monkeypatch):
    """The eight-warp CTA form (``form``, instance-key field 37 = LAST, ``_w8r`` / ``_w8rs`` directly after
    ``_pro<n>``): the three sm_103a input-gradient rules that carry it (4096 -> 512 and the kv_a 6144 -> 576 with
    ``w8rs``, 5120 -> 512 with ``w8r``) plan the eight-warp program of the kn bf16 TMA-store class with the rule's
    launch parameters and none of the ten-warp pipeline knobs; the ``form=""`` twin is the row's ten-warp program with
    the same plan arithmetic; the launch adds the raster's four fast divisors.  A rule's form yields where the plan
    carries stream-K units or a caller forces a form the program excludes; a caller-forced form is refused there
    (``instance_key`` for the forms, the planner for the stream-K units)."""
    # the CPU dummies ``_k1_values`` creates here stay out of the module cache the device tests share
    monkeypatch.setattr(cake_backend, "_DUMMIES", {})
    kw = dict(sm_count=152, l2_bytes=L2_BYTES, arch="sm_103a")
    mirror = dict(kw, _fallback=False)
    registered = _registered("sm_103a")
    rows = {
        (4096, 512): ("dense_proj_gemm_kn_n256_tma1_w8rs", "w8rs", 16, "l2_256b"),
        (5120, 512): ("dense_proj_gemm_kn_n256_tma1_w8r", "w8r", 12, "l2_256b"),
        (6144, 576): ("dense_proj_gemm_kn_n256_tma1_w8rs", "w8rs", 12, "none"),
    }
    assert sorted(rows) == sorted((key[6], key[7]) for key in EIGHT_WARP_ROWS)
    for (N_layer, K_layer), (template, form, group_m, promo) in rows.items():
        key = ("sm_103a", False, True, False, False, False, N_layer, K_layer, None)
        rule = ROW_RULES[key]
        assert rule["form"] == form and rule["sk"] is False, key
        v = _class_views("kn", False, False, T, N_layer, K_layer)
        plan, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **mirror)
        assert plan.template == template, (key, T, plan.template)
        assert plan.form == form and plan.tma_out and plan.epi == "tma"
        assert (plan.slots, plan.stages, plan.block_n, plan.cta_rows) == (
            1,
            6,
            256,
            128,
        )
        assert (plan.group_m, plan.promo) == (group_m, promo), key
        assert plan.sk_units == 0 and plan.tail_tiles == 0 and plan.ws_f32_elems == 0
        assert plan.m_tiles % 2 == 0 and plan.num_full == plan.pair_tiles
        assert plan.grid == (2 * plan.pair_tiles, 1, 1) and not plan.cta1
        assert not plan.mg and not plan.lw and plan.sw == 0, key
        assert plan.pd == 0 and plan.pro == 0 and plan.pf == 0 and plan.sh == 0, key
        assert plan.hints == ("none", "none") and plan.store_hint == "none", key
        # the registered program plans directly (no fallback)
        served, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
        assert served.template == template and template in registered, (key, T)
        assert not (
            served.swap_fallback
            or served.rule_fallback
            or served.knob_fallback
            or served.wave_fallback
        ), key
        # the ten-warp twin: the same plan arithmetic, the row's program with its pipeline knobs
        plain, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], form="", **mirror
        )
        assert plain.form == "" and "_w8" not in plain.template, plain.template
        assert plain.template.endswith("_pro15") and plain.pd == 2, plain.template
        sig, plain_sig = _plan_signature(plan), _plan_signature(plain)
        differs = {name for name in sig if sig[name] != plain_sig[name]}
        expected = {"template", "form", "pd", "pro"} | {
            name for name in ("lw", "mg", "sw") if rule.get(name)
        }
        assert differs == expected, (key, T, sorted(differs))
        # the launch values: the four raster fast divisors of the kernel's division-free resolve, named by the
        # eight-warp programs' argument plan (``bind_launch`` orders them; a missing name would raise ``KeyError``)
        dummy = torch.zeros(1)
        values = cake_backend._k1_values(
            plan,
            torch.device("cpu", 0),
            A=dummy,
            B=dummy,
            out3=dummy,
            o_alias=dummy,
            ws=dummy,
            counters=dummy,
            ldo=plan.N,
            out_l=plan.M * plan.N,
        )
        assert values["fd_tiles_l"] == plan.m_tiles * plan.n_tiles
        assert values["fd_tiles_g"] == plan.group_m * plan.n_tiles
        assert values["fd_group"] == plan.group_m
        assert values["fd_last"] == (plan.m_tiles % plan.group_m or plan.group_m)
        assert 1 <= values["fd_last"] <= plan.group_m
        record = cake_jit.MODULES[cake_jit.KERNELS["sm_103a"][template]]
        arg_plan = [list(item) for item in cake_jit.arg_plan_of(record)]
        assert [item for item in arg_plan if item[1].startswith("fd_")] == [
            ["parameter", "fd_tiles_l"],
            ["parameter", "fd_tiles_g"],
            ["parameter", "fd_group"],
            ["parameter", "fd_last"],
        ], template
        ordered = cake_backend._order_arguments(
            record["library"], arg_plan, values, plan.grid
        )
        assert len(ordered) == len(arg_plan)
    # yields: a plan with stream-K units (kv_a with sk=True: 18 units over 16 tail tiles), the single-CTA form, the
    # register epilogue or a caller-forced pipeline knob plan the row's ten-warp program; a caller-forced form is
    # refused on every such plan
    v = _class_views("kn", False, False, T, 6144, 576)
    sk, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], sk=True, **mirror)
    assert (sk.sk_units, sk.tail_tiles) == (18, 16) and sk.form == ""
    assert sk.template == "dense_proj_gemm_kn_n256_tma1_pd2_pro15"
    with pytest.raises(ValueError, match="whole-tile"):
        plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], sk=True, form="w8rs", **mirror
        )
    for N_layer, K_layer in rows:
        v = _class_views("kn", False, False, T, N_layer, K_layer)
        forced, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], cta1=True, **mirror
        )
        assert forced.form == "" and forced.template.endswith("_c1"), forced.template
        forced, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], epi="reg", **mirror
        )
        assert forced.form == "" and forced.epi == "reg", forced.template
        assert "_w8" not in forced.template, forced.template
        forced, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], pro=7, **mirror
        )
        assert forced.form == "" and forced.pro == 7, forced.template
        assert forced.template.endswith("_pro7"), forced.template
        for bad in (dict(epi="reg"), dict(cta1=True), dict(pro=7), dict(pd=2)):
            with pytest.raises(ValueError, match="form"):
                plan_dense_projection_gemm(
                    v["A"], v["B"], v["out"], form="w8r", **bad, **mirror
                )
        # the fp32 twin and sm_100a carry no form
        f = _class_views("kn", True, False, T, N_layer, K_layer)
        plan, *_ = plan_dense_projection_gemm(f["A"], f["B"], f["out"], **mirror)
        assert plan.form == "" and "_w8" not in plan.template, (N_layer, K_layer)
        plan, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], **dict(mirror, arch="sm_100a")
        )
        assert plan.form == "" and "_w8" not in plan.template, (N_layer, K_layer)
    # the key: field 37, the symbol token after ``_pro<n>``; every field the form cannot carry is refused with it
    kn = dict(a_mn=False, b_mn=True, epi="tma", slots=1)
    key = cake_backend.instance_key(**kn, form="w8r")
    assert len(key) == 37 and key[36] == "w8r" and key[35] == 0 and key[34] is False
    assert cake_backend.instance_symbol(key) == "dense_proj_gemm_kn_n256_tma1_w8r"
    assert (
        cake_backend.instance_symbol(cake_backend.instance_key(**kn, form="w8rs"))
        == "dense_proj_gemm_kn_n256_tma1_w8rs"
    )
    assert cake_backend.instance_key(**kn, form="") == cake_backend.instance_key(**kn)
    assert cake_backend.instance_key(**kn)[36] == ""
    for bad in (
        dict(a_mn=True, b_mn=True, form="w8r"),
        dict(**kn, cta1=True, form="w8r"),
        dict(**kn, out_f32=True, form="w8r"),
        dict(a_mn=False, b_mn=True, block_n=128, epi="tma", slots=1, form="w8r"),
        dict(**kn, batch_group=4, form="w8r"),
        dict(**kn, pd=2, form="w8rs"),
        dict(**kn, pro=15, form="w8rs"),
        dict(**kn, lw=True, pro=15, form="w8rs"),
        dict(**kn, mg=True, form="w8r"),
        dict(a_mn=False, b_mn=True, epi="reg", form="w8r"),
        dict(**kn, stages=5, form="w8r"),
        dict(**kn, hints=("evict_first", "none"), form="w8r"),
        dict(**kn, store_hint="evict_last", form="w8r"),
        dict(**kn, form="w8"),
    ):
        with pytest.raises(ValueError, match=r"form"):
            cake_backend.instance_key(**bad)
    # the symbol decoder reads the form back and names the program again; a token the form refuses (``_pro15`` with
    # it) decodes to a key ``instance_key`` refuses
    knobs = cake_backend._template_knobs("dense_proj_gemm_kn_n256_tma1_w8r")
    assert knobs is not None and knobs["form"] == "w8r" and knobs["pro"] == 0
    assert (knobs["epi"], knobs["slots"], knobs["pd"], knobs["mg"]) == (
        "tma",
        1,
        0,
        False,
    )
    key_knobs = {k: v for k, v in knobs.items() if v is not None}
    key_knobs["sk_exact"] = bool(knobs["sk_exact"])
    assert (
        cake_backend.instance_symbol(
            cake_backend.instance_key(a_mn=False, b_mn=True, **key_knobs)
        )
        == "dense_proj_gemm_kn_n256_tma1_w8r"
    )
    knobs = cake_backend._template_knobs("dense_proj_gemm_kn_n256_tma1_w8r_pro15")
    assert knobs is not None and knobs["form"] == "w8r" and knobs["pro"] == 15
    key_knobs = {k: v for k, v in knobs.items() if v is not None}
    key_knobs["sk_exact"] = bool(knobs["sk_exact"])
    with pytest.raises(ValueError, match=r"form"):
        cake_backend.instance_key(a_mn=False, b_mn=True, **key_knobs)
    # the form is a class for the registry fallback: a ten-warp row never falls onto an eight-warp program, nor the
    # reverse (the eight-warp twin of the ``w8r`` program is the ``w8rs`` program alone)
    assert all(
        knobs["form"] == ""
        for knobs in cake_backend._registry_variants(
            "dense_proj_gemm_kn_n256_tma1_pd2_pro15", "sm_103a", False
        )
    )
    assert [
        knobs["form"]
        for knobs in cake_backend._registry_variants(
            "dense_proj_gemm_kn_n256_tma1_w8r", "sm_103a", False
        )
    ] == ["w8rs"]
    assert "form" not in cake_backend._CTA1_CLEARED
    for key, rule in ROW_RULES.items():
        assert "form" not in rule or (
            key[0] == "sm_103a" and rule["form"] in cake_backend.W8_FORMS
        ), key


def _n224_rule_rows():
    """Every (key, T) of a ``ROW_RULES`` entry that selects the exact-wave tile (BLOCK_N 224)."""
    return [
        (key, T)
        for key, rule in sorted(ROW_RULES.items(), key=repr)
        if rule.get("block_n") == cake_backend.EXACT_WAVE_BLOCK_N and not key[5]
        for T in (16231, 16384)
    ]


def test_half_tail_fit_gates_the_plan_not_the_program(monkeypatch):
    """The half-tail fit (``0 < 2 * tail <= pairs``) selects the plan, never the program: a rule or a caller that
    names ``htail`` keeps the ``_ht`` program (``plan.htail``), which runs the row's whole-tile plan where its half
    items do not fit the device's pairs (``plan.htail_plan`` False: the K 5120 thin input gradients, 1280 tiles on
    76 pairs, tail 64 - the plan the lane measured) and the half-tail plan where they do (K 4096: tail 36, 72 half
    items; K 5120 on 84 pairs: tail 20, 40 half items).  A row without a rule plans without the knob."""
    kw = dict(sm_count=152, l2_bytes=L2_BYTES, arch="sm_100a", _fallback=False)
    knobs = dict(epi="tma", slots=2, pro=15, htail=True, lw=True)
    for K_layer, fields, half in (
        (4096, (988, 36, 72, 1), True),
        (5120, (1280, 0, 0, 1), False),
    ):
        v = _class_views("kn", False, False, 16384, K_layer, 32)
        ruled, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
        assert ruled.template == "dense_proj_gemm_kn_n256_ht_tma2_lw_pro15"
        assert ruled.htail and ruled.htail_plan == half
        assert (
            ruled.num_full,
            ruled.tail_tiles,
            ruled.sk_units,
            ruled.iters_per_unit,
        ) == fields
        assert ruled.sk_iters == (72 if half else 0) and ruled.ws_f32_elems == 0
        with monkeypatch.context() as m:
            m.setattr(cake_backend, "ROW_RULES", {})
            plain, *_ = plan_dense_projection_gemm(v["A"], v["B"], v["out"], **kw)
            assert (
                not plain.htail and not plain.htail_plan and "_ht" not in plain.template
            )
            forced, *_ = plan_dense_projection_gemm(
                v["A"], v["B"], v["out"], **knobs, **kw
            )
            assert forced.template == ruled.template
            assert forced.htail and forced.htail_plan == half
            assert (
                forced.num_full,
                forced.tail_tiles,
                forced.sk_units,
                forced.iters_per_unit,
            ) == fields
    # the fit follows the device: on 84 pairs the K 5120 tail of 20 tiles fits as 40 half items
    v = _class_views("kn", False, False, 16384, 5120, 32)
    wide, *_ = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], **dict(kw, sm_count=168)
    )
    assert wide.htail and wide.htail_plan and "_ht_" in wide.template
    assert (wide.num_full, wide.tail_tiles, wide.sk_units, wide.iters_per_unit) == (
        1260,
        20,
        40,
        1,
    )
    assert wide.sk_iters == 40 and wide.ws_f32_elems == 0


def test_exact_wave_gate_follows_the_device_pair_count():
    """The exact-wave gate: a BLOCK_N 224 rule fills the 76 pairs of the measuring devices (152 SMs) in whole waves and
    plans as written there; on 74 pairs (148 SMs) its tiles need an extra, nearly empty wave, so the rule's BLOCK_N
    yields to the default tile (``wave_fallback``: the plan the rule's other knobs make at BLOCK_N 256); on 80 pairs
    (160 SMs) both tiles need the same waves and nothing changes; a caller-forced BLOCK_N is never touched."""
    assert exact_waves(152, 76) and exact_waves(76, 76)
    assert (
        not exact_waves(152, 74) and not exact_waves(76, 74) and not exact_waves(76, 80)
    )
    rows = _n224_rule_rows()
    assert rows and {key[0] for key, _ in rows} == {"sm_100a", "sm_103a"}
    fired = set()
    for key, T in rows:
        arch = key[0]
        v = _key_views(key, T)
        base = dict(
            l2_bytes=L2_BYTES, transposed_out=key[4], arch=arch, _fallback=False
        )
        plan, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], sm_count=152, **base
        )
        assert (
            plan.block_n == 224 and not plan.wave_fallback and "_n224_" in plan.template
        ), (key, T)
        tiles_224 = plan.pair_tiles
        tiles_256 = plan.L * (plan.m_tiles // 2) * -(-plan.N // 256)
        for sm_count in (148, 160):
            pairs = sm_count // 2
            plan, *_ = plan_dense_projection_gemm(
                v["A"], v["B"], v["out"], sm_count=sm_count, **base
            )
            expect = not exact_waves(tiles_224, pairs) and -(-tiles_224 // pairs) > -(
                -tiles_256 // pairs
            )
            assert plan.wave_fallback == expect, (key, T, sm_count)
            if sm_count == 160:
                assert not plan.wave_fallback, (
                    key,
                    T,
                )  # the same waves on 80 pairs: the rule's tile stays
        plan, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], sm_count=148, **base
        )
        if plan.wave_fallback:
            fired.add(key)
            assert plan.block_n == 256 and "_n256_" in plan.template, (key, T)
            forced, *_ = plan_dense_projection_gemm(
                v["A"], v["B"], v["out"], sm_count=148, block_n=256, **base
            )
            assert dataclasses.replace(plan, wave_fallback=False) == forced, (key, T)
            assert plan.pair_tiles == tiles_256 and -(-tiles_224 // 74) > -(
                -tiles_256 // 74
            ), (key, T)
        else:
            assert plan.block_n == 224 and (
                exact_waves(tiles_224, 74) or -(-tiles_224 // 74) == -(-tiles_256 // 74)
            ), (key, T)

        kept, *_ = plan_dense_projection_gemm(
            v["A"], v["B"], v["out"], sm_count=148, block_n=224, **base
        )
        assert kept.block_n == 224 and not kept.wave_fallback, (key, T)
    # the rows the gate protects on a 148-SM part: the 4096 x 4096 (152 tiles -> 128) and 4096 -> 1024 (76 -> 64)
    # weight gradients of both architectures (bf16 and the sm_103a fp32 form)
    assert fired == {
        key for key, _ in rows if (key[6], key[8]) in ((4096, 4096), (4096, 1024))
    }, sorted(fired, key=repr)
    for key in fired:
        plan, *_ = plan_dense_projection_gemm(
            *(_key_views(key, 16384)[k] for k in ("A", "B", "out")),
            sm_count=148,
            l2_bytes=L2_BYTES,
            transposed_out=key[4],
            arch=key[0],
            _fallback=False,
        )
        assert (
            plan.template
            == {
                (4096, False): "dense_proj_gemm_nn_n256_m256_pro7",
                (1024, False): "dense_proj_gemm_nn_n256_hne_cg_pro15",
                (1024, True): "dense_proj_gemm_nn_n256_f32_v8_ef_pro15",
            }[(key[8], key[3])]
        ), (key, plan.template)


def test_batched_rows_satisfy_the_kernel_output_layout_predicate():
    """The strides the host hands the kernel for the batched head-interleaved MLA rows (``[T, H, D]`` written as
    ``[H, T, D]``, ``D`` a slice of a wider slot) satisfy the generated bindings' output-layout predicate: with ``rows`` /
    ``cols`` the stored matrix's extents (``N`` / ``M`` under the transposed store), ``ldo >= cols``, ``out_l >= cols``
    and either batch-major (``out_l >= rows * ldo``) or interleaved within a row (``ldo >= L * out_l``); a batch of
    one passes the matrix span."""
    checked = 0
    for row in MLA_ROWS:
        for op in ("fwd", "dgrad", "wgrad"):
            for T in (1001, 16231):
                v = _views("mla", row, op, "bf16", T)
                plan, _, _, out3 = plan_dense_projection_gemm(
                    v["A"],
                    v["B"],
                    v["out"],
                    sm_count=152,
                    l2_bytes=L2_BYTES,
                    arch="sm_100a",
                )
                ldo, out_l = cake_backend._out_strides(out3)
                rows, cols = (
                    (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
                )
                assert ldo >= cols and out_l >= cols, (row, op, T, ldo, out_l)
                assert out_l >= rows * ldo or ldo >= plan.L * out_l, (
                    row,
                    op,
                    T,
                    ldo,
                    out_l,
                    rows,
                    plan.L,
                )
                assert MLA_ROWS[row]["H"] == plan.L
                checked += 1
    assert checked == 12
    # the interleaved store of the qabs input gradient (D = 192 of a 256-wide slot): the second form, with equality
    v = _views("mla", "qabs", "dgrad", "bf16", 33)
    plan, _, _, out3 = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], sm_count=152, l2_bytes=L2_BYTES
    )
    ldo, out_l = cake_backend._out_strides(out3)
    assert (
        (ldo, out_l, plan.L) == (64 * 256, 256, 64)
        and out_l < plan.M * ldo
        and ldo == plan.L * out_l
    )
    # a batch of one: the matrix span satisfies the batch-major form
    v = _views("proj", "q_a", "fwd", "bf16", 1001)
    plan, _, _, out3 = plan_dense_projection_gemm(
        v["A"], v["B"], v["out"], sm_count=152, l2_bytes=L2_BYTES
    )
    ldo, out_l = cake_backend._out_strides(out3)
    assert plan.L == 1 and ldo >= plan.N and out_l == plan.M * ldo


def test_router_convenience_views_match_the_planner():
    for op, T in (("fwd", 257), ("dgrad", 257), ("wgrad", 1001)):
        v = _router_inputs(op, T, seed=SEED + 5, device="cpu")
        plan, A_eff, B_eff = plan_router_fp32_gemm(v["A"], v["B"], v["out"])
        a_view, split_src = cake_backend._k2_views(v["A"], v["B"], plan)
        want_split = B_eff if plan.b_mn else B_eff.t()
        want_a = A_eff.t().unsqueeze(0) if plan.a_mn else A_eff.unsqueeze(0)
        assert cake_backend._same_view(a_view, want_a), op
        assert cake_backend._same_view(split_src, want_split), op
        assert tuple(a_view.shape[1:]) == (
            (plan.K, plan.M) if plan.a_mn else (plan.M, plan.K)
        )
        assert (plan.swapped, plan.splits) == (
            (op == "wgrad"),
            ROUTER_SPLITS[plan.layout],
        )
        # the fail-closed check of the convenience prepare: the planner's operands pass and yield the
        # per-call views; a rebuild that is not the planner's view (another buffer) refuses to cache
        checked = cake_backend._k2_checked_views(v["A"], v["B"], A_eff, B_eff, plan)
        assert all(
            cake_backend._same_view(got, want)
            for got, want in zip(checked, (a_view, split_src), strict=True)
        ), op
        for wrong_a, wrong_b in ((A_eff.clone(), B_eff), (A_eff, B_eff.clone())):
            with pytest.raises(RuntimeError, match="differ from the planner's"):
                cake_backend._k2_checked_views(v["A"], v["B"], wrong_a, wrong_b, plan)
    workspace = torch.empty(4, 6, 8)
    assert cake_backend._k2_workspace_strides(workspace) == (8, 48)
    out = torch.empty(6, 16)[:, :8]
    assert cake_backend._k2_out32(out).data_ptr() == out.data_ptr()
    assert (
        cake_backend._k2_out32(out).dim() == 1
        and cake_backend._k2_out32(torch.empty(6, 8)).dim() == 2
    )


def test_device_facts_are_keyed_by_the_device_index():
    assert cake_backend._cuda_index(torch.device("cuda", 1)) == 1
    assert cake_backend._cuda_index("cuda:2") == 2 and cake_backend._cuda_index(3) == 3
    with pytest.raises(ValueError, match="CUDA"):
        cake_backend._cuda_index(torch.device("cpu"))
    assert arch_for(torch.device("cpu")) is None
    facts = cake_backend.DeviceFacts(
        index=0, arch="sm_100a", sm_count=148, l2_bytes=L2_BYTES
    )
    assert (facts.arch, facts.sm_count, facts.l2_bytes) == ("sm_100a", 148, L2_BYTES)
    assert isinstance(cake_backend._DEVICE_FACTS, dict)


# ---------------------------------------------------------------------------
# GPU correctness (skips without a registered instance)
# ---------------------------------------------------------------------------


def _make_inputs(family, row, op, out_dtype, T, seed, device="cuda"):
    """Deterministic operands for one row (same distributions as the Cake eval harness)."""
    g = torch.Generator(device=device).manual_seed(seed)
    v = _views(family, row, op, out_dtype, T, device=device)

    def fill(t, scale=1.0):
        t.copy_(
            (torch.randn(t.shape, device=device, generator=g) * scale).to(
                torch.bfloat16
            )
        )

    if family == "proj":
        K, N = PROJECTION_ROWS[row]
        fill(v["X"])
        fill(v["W"], K**-0.5)
        fill(v["G"])
    else:
        spec = MLA_ROWS[row]
        fill(v["act"])
        fill(v["weight"], spec["D_in"] ** -0.5)
        fill(v["d_out"])
    if "buf" in v:
        v["buf"].fill_(float("nan"))  # the rope part of the slot must keep its sentinel
        v["untouched"] = v["buf"][..., MLA_ROWS[row]["D_in"] :]
    return v


def _check(actual, expected_f64, *, untouched=None):
    diff = (actual.double() - expected_f64).abs()
    assert bool(torch.isfinite(actual).all())
    if actual.dtype == torch.bfloat16:
        violations = int((diff > BF16_ATOL + BF16_RTOL * expected_f64.abs()).sum())
        assert violations == 0, (
            f"{violations} bf16 elements outside atol=rtol=1e-2 (max abs {float(diff.max()):.4g})"
        )
    else:
        tiny = torch.finfo(torch.float64).tiny
        rel_fro = float(diff.norm() / max(float(expected_f64.norm()), tiny))
        max_abs_ratio = float(diff.max()) / max(float(expected_f64.abs().max()), tiny)
        assert rel_fro <= F32_REL_FRO_MAX, rel_fro
        assert max_abs_ratio <= F32_MAX_ABS_OVER_REF_ABSMAX, max_abs_ratio
    if untouched is not None:
        assert bool(torch.isnan(untouched.float()).all())


def _plan_template(v, **knobs):
    plan, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=_sm_count(),
        l2_bytes=_l2_bytes(),
        transposed_out=v.get("transposed", False),
        **knobs,
    )
    return plan.template


@pytest.mark.parametrize("row, op, out_dtype, T", GPU_ROWS)
def test_projection_rows_match_fp64_reference(row, op, out_dtype, T):
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    v = _make_inputs("proj", row, op, out_dtype, T, seed=SEED)
    _require_program(_plan_template(v))
    if op == "wgrad":
        prepared = prepare_projection_wgrad(v["G"], v["X"], v["out"])
        assert prepared.plan.transposed_out == v["transposed"]
    else:
        prepared = prepare_dense_projection_gemm(v["A"], v["B"], v["out"])
    out = prepared.launch()
    torch.cuda.synchronize()
    assert out is v["out"]
    expected = torch.matmul(v["A"].double(), v["B"].double())
    _check(v["out"] if not v.get("transposed") else v["out"].t(), expected)


@pytest.mark.parametrize("T", [1001, 4100])
@pytest.mark.parametrize("key", EIGHT_WARP_ROWS, ids=lambda key: f"{key[6]}x{key[7]}")
def test_eight_warp_form_rows_match_fp64_reference(key, T):
    """The three sm_103a input-gradient rows run their eight-warp program at a short T against the FP64 reference of
    the exact BF16 operands (the package tolerance); the row's ten-warp program is launched on the same operands
    beside it and the two outputs are compared (reported, not asserted: bitwise agreement and the largest
    difference)."""
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    arch, a_mn, b_mn, out_f32, out_t, batched, N, K, M = key
    assert (a_mn, b_mn, out_f32, out_t, batched, M) == (
        False,
        True,
        False,
        False,
        False,
        None,
    )
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(SEED)
    A = torch.randn(T, K, device=device, generator=g).to(torch.bfloat16)
    B = (torch.randn(K, N, device=device, generator=g) * K**-0.5).to(torch.bfloat16)
    out = torch.empty(T, N, dtype=torch.bfloat16, device=device)
    plan, *_ = plan_dense_projection_gemm(
        A, B, out, sm_count=_sm_count(), l2_bytes=_l2_bytes(), arch=arch_for(device)
    )
    if not plan.form:
        pytest.skip(f"the eight-warp form is not a rule of {arch_for(device)}")
    _require_program(plan.template)
    prepared = prepare_dense_projection_gemm(A, B, out)
    assert prepared.template == plan.template and "_w8" in prepared.template
    assert prepared.plan.form == ROW_RULES[key]["form"]
    assert prepared.plan.sk_units == 0 and prepared.plan.m_tiles % 2 == 0
    result = prepared.launch()
    torch.cuda.synchronize()
    assert result is out
    expected = torch.matmul(A.double(), B.double())
    _check(out, expected)
    plain_out = torch.empty_like(out)
    plain = prepare_dense_projection_gemm(A, B, plain_out, form="")
    assert plain.plan.form == "" and "_w8" not in plain.template, plain.template
    plain.launch()
    torch.cuda.synchronize()
    _check(plain_out, expected)
    diff = (out.float() - plain_out.float()).abs()
    spacing = torch.exp2(
        torch.floor(torch.log2(expected.abs().float().clamp_min(2.0**-126))) - 7
    )
    print(
        f"eight-warp form vs the ten-warp program: {N} -> {K} T {T} {prepared.template} vs {plain.template}: "
        f"bitwise={torch.equal(out, plain_out)} mismatches={int((out != plain_out).sum())} "
        f"max_abs={float(diff.max()):.4g} max_bf16_ulp={float((diff / spacing).max()):.3g}"
    )


@pytest.mark.parametrize("row", list(MLA_ROWS))
@pytest.mark.parametrize("op", ["fwd", "dgrad", "wgrad"])
def test_mla_batched_rows_match_fp64_reference(row, op):
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    v = _make_inputs("mla", row, op, "bf16", 257, seed=SEED + 1)
    _require_program(_plan_template(v))
    prepared = prepare_dense_projection_gemm(v["A"], v["B"], v["out"])
    out = prepared.launch()
    torch.cuda.synchronize()
    assert out is v["out"] and prepared.plan.L == 64
    expected = torch.matmul(v["A"].double(), v["B"].double())
    _check(v["out"], expected, untouched=v.get("untouched"))


def test_strided_views_and_storage_offsets_match_fp64_reference():
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(SEED + 2)
    T, K, N = 301, 704, 264
    # A: a column slice inside a wider buffer starting at a 16-byte aligned offset; B: a row slice of a taller
    # weight; out: a slice inside a larger buffer with a padded row stride
    a_buf = (torch.randn(T, K + 64, device=device, generator=g)).to(torch.bfloat16)
    A = a_buf[:, 8 : 8 + K]
    w_buf = (torch.randn(N + 16, K, device=device, generator=g) * K**-0.5).to(
        torch.bfloat16
    )
    W = w_buf[16:]
    out_buf = torch.full(
        (T + 3, N + 24), float("nan"), device=device, dtype=torch.bfloat16
    )
    out = out_buf[3:, 8 : 8 + N]
    _require_program(_plan_template(dict(A=A, B=W.t(), out=out)))
    prepared = prepare_dense_projection_gemm(A, W.t(), out)
    prepared.launch()
    torch.cuda.synchronize()
    _check(out, torch.matmul(A.double(), W.double().t()))
    assert bool(torch.isnan(out_buf[:3].float()).all()) and bool(
        torch.isnan(out_buf[3:, :8].float()).all()
    )
    assert bool(torch.isnan(out_buf[3:, 8 + N :].float()).all())
    # the swapped small-N weight gradient X.T @ G (both operands MN-major) with the transposed fp32
    # store into a [N, K] output that sits inside a wider buffer (padded row stride)
    Nf = 96
    Gs = torch.randn(T, Nf, device=device, generator=g).to(torch.bfloat16)
    out_t_buf = torch.full(
        (Nf, K + 8), float("nan"), device=device, dtype=torch.float32
    )
    out_t = out_t_buf[:, :K]
    # (the 128-row fp32 transposed instance on sm_107a, the 64-row one on sm_100a: round 9 routes the sm_100a
    # small-N weight gradients to the 64-row family, so only that family's instance is exported there)
    v_t = dict(A=A.t(), B=Gs, out=out_t, transposed=True)
    knobs_t = _registered_knobs(v_t, {}, dict(cta_rows=64))
    prepared = prepare_projection_wgrad(Gs, A, out_t, **knobs_t)
    assert prepared.plan.transposed_out and prepared.template == _plan_template(
        v_t, **knobs_t
    )
    assert _nopro(prepared.template).startswith("dense_proj_gemm_nn_n128") and _nopro(
        prepared.template
    ).endswith("_f32_t")
    assert (
        prepared.plan.l2_bytes == _l2_bytes()
        and prepared.plan.sm_pairs == _sm_count() // 2
    )
    prepared.launch()
    torch.cuda.synchronize()
    _check(out_t, torch.matmul(Gs.double().t(), A.double()))
    assert bool(torch.isnan(out_t_buf[:, K:]).all())


def test_prepared_gemm_launches_without_allocation_and_is_deterministic():
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    # a swapped weight gradient on a half-width synthetic row (N = 32, K = 3072, T = 1001): stream-K "auto"
    # two-part split of a single partial wave with the deterministic in-kernel fixup.  The round-9 rules route
    # the GLM small-N weight gradients to whole tiles on sm_100a (64-row family: 48 tiles, which 74 pairs cannot
    # split), so the split is pinned on this row with whichever of the 64-row / 128-row instances this device's
    # export registered: the 64-row instance plans 24 pair tiles of 128 rows (sm_100a), the 128-row instance 12
    # pair tiles of 256 rows (sm_107a); the expectation follows the registered instance's tile height.
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(SEED + 3)
    T, K, N = 1001, 3072, 32
    X = torch.randn(T, K, device=device, generator=g).to(torch.bfloat16)
    G = torch.randn(T, N, device=device, generator=g).to(torch.bfloat16)
    out = torch.empty(N, K, device=device, dtype=torch.bfloat16)
    v = dict(A=X.t(), B=G, out=out, transposed=True)
    knobs = _registered_knobs(v, dict(cta_rows=64), dict(cta_rows=128))
    prepared = prepare_projection_wgrad(G, X, out, **knobs)
    pair_tiles = prepared.plan.M // (
        2 * prepared.plan.cta_rows
    )  # 24 (64-row instance) or 12 (128-row instance)
    assert prepared.plan.transposed_out and pair_tiles in (12, 24)
    assert prepared.plan.pair_tiles == pair_tiles
    if (
        prepared.plan.sm_pairs >= 2 * pair_tiles
    ):  # one partial wave whose halves fit the pairs: every tile in two parts
        assert (prepared.plan.tail_tiles, prepared.plan.sk_units) == (
            pair_tiles,
            2 * pair_tiles,
        )
    first = prepared.launch().clone()
    torch.cuda.synchronize()
    _check(out.t(), torch.matmul(X.t().double(), G.double()))
    before = torch.cuda.memory_stats()
    prepared.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert torch.equal(first, out)


def test_convenience_path_reuses_its_planned_launch_without_allocation():
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(SEED + 6)
    T, K, N = (
        257,
        6144,
        576,
    )  # the kv_a forward row (register epilogue, K-major operands)
    X = torch.randn(T, K, device=device, generator=g).to(torch.bfloat16)
    W = (torch.randn(N, K, device=device, generator=g) * K**-0.5).to(torch.bfloat16)
    out = torch.empty(T, N, device=device, dtype=torch.bfloat16)
    _require_program(_plan_template(dict(A=X, B=W.t(), out=out)))
    facts = cake_backend.device_facts(device)
    assert facts is cake_backend.device_facts(torch.device("cuda", facts.index))
    assert (facts.sm_count, facts.l2_bytes) == (_sm_count(), _l2_bytes())
    cake_backend.clear_convenience_caches()
    first = cake_backend.projection_forward(X, W, out).clone()
    torch.cuda.synchronize()
    _check(out, torch.matmul(X.double(), W.double().t()))
    assert len(cake_backend._K1_CACHE) == 1
    # new tensors of the same geometry hit the cached plan: no planning, no allocation (``out`` is given)
    X2 = torch.randn(T, K, device=device, generator=g).to(torch.bfloat16)
    out2 = torch.empty_like(out)
    before = torch.cuda.memory_stats()
    cake_backend.projection_forward(X2, W, out2)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert len(cake_backend._K1_CACHE) == 1
    _check(out2, torch.matmul(X2.double(), W.double().t()))
    assert torch.equal(cake_backend.projection_forward(X, W, out), first)
    assert len(cake_backend._ARENAS) == 1
    # another stream never shares an entry (nor its workspace arena)
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        cake_backend.projection_forward(X, W, out2)
    torch.cuda.synchronize()
    assert len(cake_backend._K1_CACHE) == 2 and torch.equal(out2, first)
    first_entry, second_entry = cake_backend._K1_CACHE.values()
    assert (
        first_entry.arena is not second_entry.arena and len(cake_backend._ARENAS) == 2
    )
    assert {e.arena.key for e in (first_entry, second_entry)} == set(
        cake_backend._ARENAS
    )
    # the prepared path of the same binding gives the same bits
    prepared = prepare_dense_projection_gemm(X, W.t(), out2)
    assert (
        prepared.template == next(iter(cake_backend._K1_CACHE.values())).plan.template
    )
    assert torch.equal(prepared.launch(), first)


def test_convenience_path_replays_a_stream_k_row_bit_exact():
    """The cached entry carves the stream-K workspace and the slice counters out of the arena of its device
    and stream: the second call of the same geometry reuses them without a memset (the kernel resets every
    counter it uses), stays allocation-free and reproduces the first result bit for bit."""
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(SEED + 7)
    T, K, N = (
        1001,
        3072,
        32,
    )  # the swapped small-N weight gradient of the prepared-launch test
    X = torch.randn(T, K, device=device, generator=g).to(torch.bfloat16)
    G = torch.randn(T, N, device=device, generator=g).to(torch.bfloat16)
    v = dict(
        A=X.t(),
        B=G,
        out=torch.empty(N, K, device=device, dtype=torch.bfloat16),
        transposed=True,
    )
    knobs = _registered_knobs(v, dict(cta_rows=64), dict(cta_rows=128))
    cake_backend.clear_convenience_caches()
    out1 = cake_backend.projection_wgrad(G, X, **knobs)  # ``out`` allocated here
    torch.cuda.synchronize()
    _check(out1.t(), torch.matmul(X.t().double(), G.double()))
    assert len(cake_backend._K1_CACHE) == 1
    entry = next(iter(cake_backend._K1_CACHE.values()))
    # the weight-gradient swap is the caller's (``wgrad_views`` -> ``transposed_out``); the batched small-M
    # swap of the planner (``entry.swapped``) does not apply to a 2-D row
    assert entry.plan.transposed_out and not entry.swapped
    out2 = torch.empty_like(out1)
    before = torch.cuda.memory_stats()
    cake_backend.projection_wgrad(G, X, out2, **knobs)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert len(cake_backend._K1_CACHE) == 1 and torch.equal(out1, out2)
    counters = entry.arena.buffers["k1_counters"]
    assert counters.numel() >= entry.plan.counters_alloc_u32
    assert entry.arena.buffers["k1_ws"].numel() >= max(entry.plan.ws_f32_elems, 1)
    if entry.plan.sk_units:
        # the whole shared counter buffer is back at zero when the launch completes (the zero contract every
        # entry of the arena relies on), not only the slice counters this launch used
        assert int(torch.count_nonzero(counters.view(torch.int32))) == 0
    # a different knob set is a different entry (``_registered_knobs`` picked the registered family)
    assert cake_backend.convenience_key(
        0, 0, (True, ("cta_rows", 64), ("sk", "auto")), G, X, out1
    ) != cake_backend.convenience_key(
        0, 0, (True, ("cta_rows", 128), ("sk", "auto")), G, X, out1
    )


def test_convenience_path_memory_is_bounded_across_token_counts():
    """A ragged token count (a new T every step) must not grow the memory the convenience path holds: the
    router entries of 23 distinct T share one arena per (device, stream), so the memory held after the loop is
    about the single-T high-water mark (each entry used to own ~200 MiB of workspace: 4.16 GiB after 23 T),
    and a cached replay still matches the prepared path bit for bit."""
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    for a_mn, b_mn, out_t in (
        (False, False, False),
        (False, True, False),
        (True, True, True),
    ):
        _require_program(
            cake_backend.router_symbol(a_mn, b_mn, out_t)
        )  # kk / kn / nn_t
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(SEED + 12)
    W = torch.randn(256, 6144, device=device, generator=g)

    def held_after(T_values):
        cake_backend.clear_convenience_caches()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        for T in T_values:
            X = torch.randn(T, 6144, device=device, generator=g)
            G = torch.randn(T, 256, device=device, generator=g)
            cake_backend.router_forward(X, W)
            cake_backend.router_dgrad(G, W)
            cake_backend.router_wgrad(G, X)
        del X, G
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() - base

    single = held_after([16022])
    assert len(cake_backend._K2_CACHE) == 3 and len(cake_backend._ARENAS) == 1
    (arena,) = cake_backend._ARENAS.values()
    assert arena.refs == 3 and single >= arena.nbytes() > 0
    loop = held_after(range(16000, 16023))
    assert len(cake_backend._K2_CACHE) == min(3 * 23, CONVENIENCE_CACHE_SIZE)
    assert loop <= 1.5 * single, (loop, single)
    assert loop < 2**30, loop  # far below the 4.16 GiB the per-entry workspaces held
    (arena,) = cake_backend._ARENAS.values()
    assert arena.refs == len(cake_backend._K2_CACHE)
    # the cached replay of the last T is allocation-free (``out`` given) and matches the prepared path
    T = 16022
    X = torch.randn(T, 6144, device=device, generator=g)
    G = torch.randn(T, 256, device=device, generator=g)
    outs = [torch.empty(T, 256, device=device) for _ in range(3)]
    cake_backend.router_forward(X, W, outs[0])
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    cake_backend.router_forward(X, W, outs[1])
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    prepared = prepare_router_fp32_gemm(X, W.t(), outs[2], splits=4)
    prepared.launch()
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    dw = cake_backend.router_wgrad(G, X)
    prepared_w = prepare_router_fp32_gemm(G.t(), X, torch.empty_like(dw), splits=6)
    assert torch.equal(dw, prepared_w.launch())
    cake_backend.clear_convenience_caches()
    assert cake_backend._ARENAS == {}


# longer than a cold-cache JIT build; the hang this test guards against never returns
@pytest.mark.timeout(300)
def test_pd2_stream_k_on_an_odd_k_completes():
    """The dgrad of T = 1024, N = 1032, K = 1000 is one partial wave of 16 tiles of 17 K steps.  Two units per tile
    would leave the last one without a K step, and the pd 2 program waits for that unit's accumulator forever; the
    planner plans the 31 units that hold one."""
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(SEED + 11)
    T, N, K = 1024, 1032, 1000
    G = torch.randn(T, N, device=device, generator=g).to(torch.bfloat16)
    W = (torch.randn(N, K, device=device, generator=g) * N**-0.5).to(torch.bfloat16)
    v = dict(
        A=G,
        B=W,
        out=torch.full((T, K), float("nan"), device=device, dtype=torch.bfloat16),
    )
    knobs = _registered_knobs(
        v,
        dict(pd=2, slots=2, epi="tma"),
        dict(pd=2, slots=2, epi="tma", pro=15),
        dict(pd=2, slots=1, epi="tma", pro=15),
    )
    prepared = prepare_dense_projection_gemm(v["A"], v["B"], v["out"], **knobs)
    plan = prepared.plan
    assert plan.pd == 2 and plan.k_blocks == 17 and plan.tail_tiles * 2 <= plan.sm_pairs
    assert (
        (plan.sk_units - 1) * plan.iters_per_unit
        < plan.sk_iters
        < 2 * plan.tail_tiles * plan.iters_per_unit
    )
    prepared.launch()
    torch.cuda.synchronize()
    _check(v["out"], G.double() @ W.double())


def _router_inputs(op, T, seed, K=6144, N=256, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(T, K, device=device, generator=g)
    W = torch.randn(N, K, device=device, generator=g) * 0.02
    G = torch.randn(T, N, device=device, generator=g) * 1e-3
    if op == "fwd":
        return dict(A=X, B=W.t(), out=torch.empty(T, N, device=device), X=X, W=W, G=G)
    if op == "dgrad":
        return dict(A=G, B=W, out=torch.empty(T, K, device=device), X=X, W=W, G=G)
    return dict(A=G.t(), B=X, out=torch.empty(N, K, device=device), X=X, W=W, G=G)


@pytest.mark.parametrize(
    "op, T", [("fwd", 257), ("dgrad", 257), ("wgrad", 1001), ("fwd", 129)]
)
def test_router_rows_match_fp64_reference_and_rerun_bit_exact(op, T):
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    v = _router_inputs(op, T, seed=SEED + 4)
    plan, *_ = plan_router_fp32_gemm(v["A"], v["B"], v["out"])
    _require_program(plan.template)
    prepared = prepare_router_fp32_gemm(v["A"], v["B"], v["out"])
    assert prepared.splits == ROUTER_SPLITS[plan.layout]
    first = prepared.launch().clone()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    prepared.launch()  # bit-exact rerun, no allocation (the split, the launch and the reduction included)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert torch.equal(first, v["out"])
    expected = torch.matmul(v["A"].double(), v["B"].double())
    diff = (v["out"].double() - expected).abs()
    rel_fro = float(diff.norm() / expected.norm())
    assert bool(torch.isfinite(v["out"]).all()) and rel_fro <= ROUTER_REL_FRO_MAX, (
        rel_fro
    )
    if (
        op == "fwd"
    ):  # top-8 routing agreement with the FP64 reference on the forward rows
        ref_top = torch.topk(expected, 8, dim=1).indices.sort(dim=1).values
        top = torch.topk(v["out"].double(), 8, dim=1).indices.sort(dim=1).values
        assert int((top != ref_top).any(dim=1).sum()) == 0
    # the eager entry point of the row gives the same bits
    eager_fn = dict(
        fwd=lambda: cake_backend.router_forward(v["X"], v["W"]),
        dgrad=lambda: cake_backend.router_dgrad(v["G"], v["W"]),
        wgrad=lambda: cake_backend.router_wgrad(v["G"], v["X"]),
    )[op]
    cake_backend.clear_convenience_caches()
    eager = eager_fn()
    torch.cuda.synchronize()
    assert torch.equal(eager, v["out"])
    assert len(cake_backend._K2_CACHE) == 1
    # the second eager call hits the cached plan: only ``out`` is allocated, the same bits come back
    before = torch.cuda.memory_stats()
    again = eager_fn()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 1
    assert len(cake_backend._K2_CACHE) == 1 and torch.equal(again, v["out"])


def test_fused_split_matches_the_torch_chain_bit_exactly():
    """The fused split program writes the bits of the torch chain for W [256, 6144], a ragged element count
    and contiguous sources at 4-, 8- and 16-byte storage offsets; the K2 paths bind it and allocate no fp32
    scratch.  Runs only where this checkout registers the program for the device."""
    if not _device_supported():
        pytest.skip(
            "requires a device with a registered generated instance "
            f"(compute capability {_registered_capabilities()})"
        )
    device = torch.device("cuda")
    arch = cake_backend.arch_for(device)
    module = cake_backend.fused_split_module(arch)
    if module is None:
        pytest.skip(
            f"fused split program {cake_backend.SPLIT_TEMPLATE!r} not registered for {arch}"
        )
    g = torch.Generator(device=device).manual_seed(SEED + 21)
    W = torch.randn(256, 6144, device=device, generator=g) * torch.logspace(
        -3, 3, 6144, device=device
    )
    ragged = torch.randn(
        7, 1001, device=device, generator=g
    )  # 7007: no multiple of 8 or of a block
    base = torch.randn(3 * 1001 + 8, device=device, generator=g)
    sources = [W, ragged] + [base[k : k + 3 * 1001].view(3, 1001) for k in (1, 2, 4)]
    index = cake_backend._device_index(device)
    for src in sources:
        n = src.numel()
        fused = torch.empty((3, *src.shape), device=device, dtype=torch.bfloat16)
        chain = torch.empty_like(fused)
        split_fp32_to_bf16x3_(src, chain, torch.empty_like(src))
        launch = bind_launch(
            module,
            cake_backend._split_values(src, fused, n),
            cake_backend._split_grid(n),
            arch,
        )
        with cake_backend._ffi_stream_context(index):
            launch()
        torch.cuda.synchronize()
        assert torch.equal(fused.view(torch.int16), chain.view(torch.int16)), tuple(
            src.shape
        )
    # the K2 paths take the program: no fp32 scratch is allocated (prepared) or carved (convenience) and the
    # two paths agree bit for bit
    _require_program("router_fp32_gemm_kk")
    X = torch.randn(257, 6144, device=device, generator=g)
    prepared = prepare_router_fp32_gemm(
        X, W.t(), torch.empty(257, 256, device=device), splits=4
    )
    assert prepared.fused_split and "resid" not in prepared.tensors
    first = prepared.launch().clone()
    cake_backend.clear_convenience_caches()
    eager = cake_backend.router_forward(X, W)
    cached = next(iter(cake_backend._K2_CACHE.values()))
    assert cached.split is not None and "k2_resid" not in cached.arena.buffers
    assert "resid" not in cake_backend._entry_views(cached)
    torch.cuda.synchronize()
    assert torch.equal(first, eager)
    cake_backend.clear_convenience_caches()
