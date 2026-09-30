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

import re
from types import SimpleNamespace

import pytest
import torch

from flashinfer.experimental.dense_projection_gemm import cake_backend, cake_jit
from flashinfer.experimental.dense_projection_gemm.cake_backend import (
    BLOCK_K,
    CTA_GROUP,
    ROUTER_SPLITS,
    SK_DUMMY_BASE,
    SK_MIN_ITERS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    bind_launch,
    default_block_n,
    default_group_m,
    default_hints,
    default_stages,
    device_l2_bytes,
    epi_mode,
    epi_slots,
    generated_program_available,
    instance_key,
    instance_symbol,
    operand_view,
    plan_dense_projection_gemm,
    plan_router_fp32_gemm,
    ROW_RULES,
    row_rule,
    swap_small_m,
    prepare_dense_projection_gemm,
    prepare_projection_wgrad,
    prepare_router_fp32_gemm,
    router_layout_class,
    split_fp32_to_bf16x3_,
    stream_k_plan,
    wave_working_set,
    wgrad_views,
)

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
# The instance every GLM-5.2 row resolves to on those devices, and the set the export registers.
# GENERATED from the Cake kernel host (adapter _plan_k1 with the mirrored sm_100a rule table) by the Cake
# workspace tool stage/r2/gen_fi_test_tables.py; the exported set is cake_jit.KERNELS of the delivered tree. Do not hand-edit.
# Table values are a template, or {(T, sm_count): template} when the plan depends on T / the device.
# --- BEGIN GENERATED TABLES ---
L2_BYTES = 132120576
EXPECTED_TEMPLATES = {
    ("o_proj", "fwd", "bf16"): {
        (1001, 148): "dense_proj_gemm_kk_n256_m256_g8",
        (1001, 212): "dense_proj_gemm_kk_n256",
        (2049, 148): "dense_proj_gemm_kk_n256_m256_g8",
        (2049, 212): "dense_proj_gemm_kk_n256",
    },
    ("o_proj", "dgrad", "bf16"): "dense_proj_gemm_kn_n256",
    ("o_proj", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n256_f32_v8",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_f32_v8",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("o_proj", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_g8_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_g8",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("o_proj", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("q_a", "fwd", "bf16"): {
        (1001, 148): "dense_proj_gemm_kk_n256_g8",
        (1001, 212): "dense_proj_gemm_kk_n256",
        (2049, 148): "dense_proj_gemm_kk_n256_g8",
        (2049, 212): "dense_proj_gemm_kk_n256",
    },
    ("q_a", "dgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8",
        (1001, 212): "dense_proj_gemm_kn_n256",
        (2049, 148): "dense_proj_gemm_kn_n256_g8",
        (2049, 212): "dense_proj_gemm_kn_n256",
    },
    ("q_a", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("q_a", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("q_a", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_g32_f32_v8",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_g32_f32_v8",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("q_b", "fwd", "bf16"): {
        (1001, 148): "dense_proj_gemm_kk_n256_g32",
        (1001, 212): "dense_proj_gemm_kk_n256",
        (2049, 148): "dense_proj_gemm_kk_n256_g32",
        (2049, 212): "dense_proj_gemm_kk_n256",
    },
    ("q_b", "dgrad", "bf16"): "dense_proj_gemm_kn_n256",
    ("q_b", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("q_b", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("q_b", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_g8_f32_v8",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_g8_f32_v8",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("kv_a", "fwd", "bf16"): {
        (1001, 148): "dense_proj_gemm_kk_n128",
        (1001, 212): "dense_proj_gemm_kk_n256",
        (2049, 148): "dense_proj_gemm_kk_n128",
        (2049, 212): "dense_proj_gemm_kk_n256",
    },
    ("kv_a", "dgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8_tma1",
        (1001, 212): "dense_proj_gemm_kn_n256_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_g8_tma1",
        (2049, 212): "dense_proj_gemm_kn_n256_tma1",
    },
    ("kv_a", "dgrad", "f32"): "dense_proj_gemm_kn_n256_f32_tma1",
    ("kv_a", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("kv_a", "wgrad", "f32"): "dense_proj_gemm_nn_n256_f32_tma1",
    ("shared_gate_up", "fwd", "bf16"): {
        (1001, 148): "dense_proj_gemm_kk_n256_g8",
        (1001, 212): "dense_proj_gemm_kk_n256",
        (2049, 148): "dense_proj_gemm_kk_n256_g8",
        (2049, 212): "dense_proj_gemm_kk_n256",
    },
    ("shared_gate_up", "dgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8",
        (1001, 212): "dense_proj_gemm_kn_n256",
        (2049, 148): "dense_proj_gemm_kn_n256_g8",
        (2049, 212): "dense_proj_gemm_kn_n256",
    },
    ("shared_gate_up", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("shared_gate_up", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("shared_gate_up", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_g32_f32_v8",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_g32_f32_v8",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("shared_down", "fwd", "bf16"): "dense_proj_gemm_kk_n256",
    ("shared_down", "dgrad", "bf16"): "dense_proj_gemm_kn_n256",
    ("shared_down", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("shared_down", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_g4_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_g4",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("shared_down", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_g8_f32_v8",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_g8_f32_v8",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("dense_gate_up", "fwd", "bf16"): "dense_proj_gemm_kk_n256",
    ("dense_gate_up", "dgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_kn_n256_m256_g8",
        (1001, 212): "dense_proj_gemm_kn_n256",
        (2049, 148): "dense_proj_gemm_kn_n256_m256_g8",
        (2049, 212): "dense_proj_gemm_kn_n256",
    },
    ("dense_gate_up", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n256_f32_v8",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_f32_v8",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("dense_gate_up", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_g8_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_g8",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("dense_gate_up", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("dense_down", "fwd", "bf16"): {
        (1001, 148): "dense_proj_gemm_kk_n256_m256",
        (1001, 212): "dense_proj_gemm_kk_n256",
        (2049, 148): "dense_proj_gemm_kk_n256_m256",
        (2049, 212): "dense_proj_gemm_kk_n256",
    },
    ("dense_down", "dgrad", "bf16"): "dense_proj_gemm_kn_n256",
    ("dense_down", "dgrad", "f32"): "dense_proj_gemm_kn_n256_f32_tma1",
    ("dense_down", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_g8_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_g8",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("dense_down", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("indexer_q", "fwd", "bf16"): "dense_proj_gemm_kk_n256",
    ("indexer_q", "dgrad", "bf16"): "dense_proj_gemm_kn_n256",
    ("indexer_q", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_g8_f32_v8",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("indexer_q", "wgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256",
        (2049, 212): "dense_proj_gemm_nn_n256",
    },
    ("indexer_q", "wgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (1001, 212): "dense_proj_gemm_nn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_f32_tma1",
        (2049, 212): "dense_proj_gemm_nn_n256_f32_tma1",
    },
    ("indexer_k", "fwd", "bf16"): "dense_proj_gemm_kk_n128",
    ("indexer_k", "dgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_kn_n256_m256_tma1",
        (1001, 212): "dense_proj_gemm_kn_n256_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_m256_tma1",
        (2049, 212): "dense_proj_gemm_kn_n256_tma1",
    },
    ("indexer_k", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n128_f32_tma1",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n128_f32_tma1",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("indexer_k", "wgrad", "bf16"): "dense_proj_gemm_nn_n128_t",
    ("indexer_k", "wgrad", "f32"): "dense_proj_gemm_nn_n128_f32_t",
    ("indexer_hw", "fwd", "bf16"): "dense_proj_gemm_kk_n128",
    ("indexer_hw", "dgrad", "bf16"): {
        (1001, 148): "dense_proj_gemm_kn_n256_m256_tma1",
        (1001, 212): "dense_proj_gemm_kn_n256_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_m256_tma1",
        (2049, 212): "dense_proj_gemm_kn_n256_tma1",
    },
    ("indexer_hw", "dgrad", "f32"): {
        (1001, 148): "dense_proj_gemm_kn_n128_f32_tma1",
        (1001, 212): "dense_proj_gemm_kn_n256_f32_tma1",
        (2049, 148): "dense_proj_gemm_kn_n128_f32_tma1",
        (2049, 212): "dense_proj_gemm_kn_n256_f32_tma1",
    },
    ("indexer_hw", "wgrad", "bf16"): "dense_proj_gemm_nn_n128_t",
    ("indexer_hw", "wgrad", "f32"): "dense_proj_gemm_nn_n128_f32_t",
}
MLA_TEMPLATES = {
    ("qabs", "fwd"): {
        (33, 148): "dense_proj_gemm_nk_n128_t",
        (33, 212): "dense_proj_gemm_nk_n128_t",
        (1001, 148): "dense_proj_gemm_kn_n256_tma1",
        (1001, 212): "dense_proj_gemm_kn_n256_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_tma1",
        (2049, 212): "dense_proj_gemm_kn_n256_tma1",
    },
    ("qabs", "dgrad"): {
        (33, 148): "dense_proj_gemm_kk_n128_t",
        (33, 212): "dense_proj_gemm_kk_n128_t",
        (1001, 148): "dense_proj_gemm_kk_n256_tma1",
        (1001, 212): "dense_proj_gemm_kk_n256_tma1",
        (2049, 148): "dense_proj_gemm_kk_n256_tma1",
        (2049, 212): "dense_proj_gemm_kk_n256_tma1",
    },
    ("qabs", "wgrad"): {
        (33, 148): "dense_proj_gemm_nn_n256_m256_t",
        (33, 212): "dense_proj_gemm_nn_n256_t",
        (1001, 148): "dense_proj_gemm_nn_n256_m256_t",
        (1001, 212): "dense_proj_gemm_nn_n256_t",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_t",
        (2049, 212): "dense_proj_gemm_nn_n256_t",
    },
    ("vproj", "fwd"): {
        (33, 148): "dense_proj_gemm_kk_n128_t",
        (33, 212): "dense_proj_gemm_kk_n128_t",
        (1001, 148): "dense_proj_gemm_kk_n256_tma1",
        (1001, 212): "dense_proj_gemm_kk_n256_tma1",
        (2049, 148): "dense_proj_gemm_kk_n256_tma1",
        (2049, 212): "dense_proj_gemm_kk_n256_tma1",
    },
    ("vproj", "dgrad"): {
        (33, 148): "dense_proj_gemm_nk_n128_t",
        (33, 212): "dense_proj_gemm_nk_n128_t",
        (1001, 148): "dense_proj_gemm_kn_n256_tma1",
        (1001, 212): "dense_proj_gemm_kn_n256_tma1",
        (2049, 148): "dense_proj_gemm_kn_n256_tma1",
        (2049, 212): "dense_proj_gemm_kn_n256_tma1",
    },
    ("vproj", "wgrad"): {
        (33, 148): "dense_proj_gemm_nn_n256_m256_t",
        (33, 212): "dense_proj_gemm_nn_n256_t",
        (1001, 148): "dense_proj_gemm_nn_n256_m256_t",
        (1001, 212): "dense_proj_gemm_nn_n256_t",
        (2049, 148): "dense_proj_gemm_nn_n256_m256_t",
        (2049, 212): "dense_proj_gemm_nn_n256_t",
    },
}
EXPORTED_TEMPLATES = frozenset(
    {
        "dense_proj_gemm_kk_n128",
        "dense_proj_gemm_kk_n128_hen",
        "dense_proj_gemm_kk_n256",
        "dense_proj_gemm_kk_n256_g32",
        "dense_proj_gemm_kk_n256_g8",
        "dense_proj_gemm_kk_n256_m256",
        "dense_proj_gemm_kk_n256_m256_g8",
        "dense_proj_gemm_kk_n256_tma1",
        "dense_proj_gemm_kn_n128_f32_tma1",
        "dense_proj_gemm_kn_n256",
        "dense_proj_gemm_kn_n256_f32_tma1",
        "dense_proj_gemm_kn_n256_f32_v8",
        "dense_proj_gemm_kn_n256_g8",
        "dense_proj_gemm_kn_n256_g8_f32_v8",
        "dense_proj_gemm_kn_n256_g8_tma1",
        "dense_proj_gemm_kn_n256_m256_g8",
        "dense_proj_gemm_kn_n256_m256_tma1",
        "dense_proj_gemm_kn_n256_tma1",
        "dense_proj_gemm_nn_n128_f32_t",
        "dense_proj_gemm_nn_n128_hen_f32_t",
        "dense_proj_gemm_nn_n128_hen_t",
        "dense_proj_gemm_nn_n128_t",
        "dense_proj_gemm_nn_n256",
        "dense_proj_gemm_nn_n256_f32_tma1",
        "dense_proj_gemm_nn_n256_g32_f32_v8",
        "dense_proj_gemm_nn_n256_g4",
        "dense_proj_gemm_nn_n256_g8_f32_v8",
        "dense_proj_gemm_nn_n256_m256",
        "dense_proj_gemm_nn_n256_m256_f32_tma1",
        "dense_proj_gemm_nn_n256_m256_g8",
        "dense_proj_gemm_nn_n256_m256_g8_tma1",
        "dense_proj_gemm_nn_n256_m256_t",
        "dense_proj_gemm_nn_n256_t",
        "dense_proj_gemm_nn_n256_tma1",
    }
)
# --- END GENERATED TABLES ---


def _expected(table, key, T, sm_count):
    value = table[key]
    return value if isinstance(value, str) else value[(T, sm_count)]


ARCH_OF_SM = {148: "sm_100a", 212: "sm_107a"}


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


def _assert_mirrors_cake(plan, cake_template, where):
    """The plan is the Cake launcher's plan when that instance is a generated program; otherwise (the batched
    small-M swap on a tiny T outside the export contract, or a measured rule whose template exists only for
    the contract's T -- both of which the Cake host JIT-compiles) the FlashInfer planner must have fallen
    back (swap and / or rule dropped) onto a generated program."""
    if cake_template in EXPORTED_TEMPLATES:
        assert plan.template == cake_template, (*where, plan.template)
        assert not (plan.swap_fallback or plan.rule_fallback or plan.knob_fallback), where
    else:
        assert plan.swap_fallback or plan.rule_fallback or plan.knob_fallback, (
            *where,
            cake_template,
            plan.template,
        )
        assert plan.template in EXPORTED_TEMPLATES, (*where, cake_template, plan.template)


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(template: str):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda"), template):
        pytest.skip(f"generated instance {template!r} not registered for this device")


def _sm_count() -> int:
    return int(torch.cuda.get_device_properties(0).multi_processor_count)


def _l2_bytes() -> int:
    return device_l2_bytes(torch.device("cuda", 0))


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


def test_registry_records_are_well_formed():
    for arch, table in cake_jit.KERNELS.items():
        assert arch in cake_jit.ARCH_NVCC_FLAGS
        for template, name in table.items():
            record = cake_jit.MODULES[name]
            assert record["arch"] == arch and record["template"] == template
            assert cake_jit.select_module(arch, template) == name
    for name, record in cake_jit.MODULES.items():
        assert record["arch"] in cake_jit.ARCH_NVCC_FLAGS
        assert cake_jit.KERNELS[record["arch"]][record["template"]] == name
        assert len(record["sources"]) == 2
        assert all(
            s.startswith("cake_dense_projection_gemm/") for s in record["sources"]
        )
        assert len(record["closure_sha256"]) == 64
        assert (
            int(record["tma_workspace_bytes"]) > 0
        )  # pointer TMA ABI: caller-owned descriptors
        kinds = {kind for kind, _ in record["arg_plan"]}
        assert kinds <= {"buffer", "tma_buffer", "workspace", "parameter", "grid"}
        assert ["workspace", "tma_descriptor_workspace"] in [
            list(item) for item in record["arg_plan"]
        ]
        launch = record["launch"]
        assert list(launch["cluster"]) == [CTA_GROUP, 1, 1]
        assert len(launch["block"]) == 3 and all(int(b) >= 1 for b in launch["block"])
    with pytest.raises(NotImplementedError, match="not registered"):
        cake_jit.select_module("sm_100a", "dense_proj_gemm_not_a_template")


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
    with pytest.raises(ValueError, match="transposed"):
        epi_mode(False, True, epi="tma")
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
        (
            dict(a_mn=False, b_mn=False, promo="l2_128b"),
            "dense_proj_gemm_kk_n256_l2_128b",
        ),
        (dict(a_mn=False, b_mn=False, pf=4), "dense_proj_gemm_kk_n256_pf4"),
        # raster group: 16 CTA row tiles per group is the default; other even counts carry ``_g<n>``
        (dict(a_mn=False, b_mn=False, group_m=16), "dense_proj_gemm_kk_n256"),
        (dict(a_mn=False, b_mn=False, group_m=8), "dense_proj_gemm_kk_n256_g8"),
        (
            dict(
                a_mn=True,
                b_mn=True,
                out_f32=True,
                slots=1,
                hints=("evict_first", "none"),
                group_m=4,
            ),
            "dense_proj_gemm_nn_n256_hen_g4_f32_tma1",
        ),
        (
            dict(
                a_mn=False,
                b_mn=False,
                pf=2,
                promo="l2_256b",
                hints=("evict_first", "none"),
                epi="tma",
                slots=1,
            ),
            "dense_proj_gemm_kk_n256_pf2_l2_256b_hen_tma1",
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
    ],
)
def test_instance_symbols(kwargs, symbol):
    assert instance_symbol(instance_key(**kwargs)) == symbol


def test_default_stages_by_tile_shape():
    # 32 KiB stages at BLOCK_N = 256, 24 KiB stages at BLOCK_N = 128 (deeper pipeline for the
    # streaming-bound small-N rows), 48 KiB stages for tall 256-row tiles; one fewer per staging slot
    assert [default_stages(s, 128, 256) for s in (0, 1, 2)] == [7, 6, 5]
    assert [default_stages(s, 128, 128) for s in (0, 1, 2)] == [9, 8, 6]
    assert [default_stages(s, 256, 256) for s in (0, 1, 2)] == [4, 4, 3]
    assert [default_stages(s, 256, 128) for s in (0, 1, 2)] == [4, 4, 3]
    assert default_stages(0) == 7 and default_stages(1) == 6
    # the default stage count carries no symbol suffix; anything else does
    assert instance_key(a_mn=False, b_mn=False, block_n=128)[5] == 9
    assert instance_key(a_mn=False, b_mn=False, block_n=128, epi="tma", slots=1)[5] == 8
    assert instance_key(a_mn=False, b_mn=False, cta_rows=256)[5] == 4


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
        instance_key(a_mn=False, b_mn=False, cta_rows=64)
    with pytest.raises(ValueError, match="SMEM"):
        instance_key(a_mn=False, b_mn=False, stages=8)
    with pytest.raises(ValueError, match="SMEM"):
        instance_key(
            a_mn=False, b_mn=False, block_n=128, stages=10
        )  # 9 is the deepest 24 KiB pipeline
    with pytest.raises(ValueError, match="diagnostic"):
        instance_key(a_mn=False, b_mn=False, diag=("no_mma",))
    with pytest.raises(ValueError, match="promo"):
        instance_key(a_mn=False, b_mn=False, promo="l2_512b")
    with pytest.raises(ValueError, match="hints"):
        instance_key(a_mn=False, b_mn=False, hints=("evict_first", "keep"))
    with pytest.raises(ValueError, match="prefetch"):
        instance_key(a_mn=False, b_mn=False, pf=17)
    with pytest.raises(ValueError, match="group_m"):
        instance_key(
            a_mn=False, b_mn=False, group_m=7
        )  # CTA pairs are adjacent row tiles: even groups only
    with pytest.raises(ValueError, match="group_m"):
        instance_key(a_mn=False, b_mn=False, group_m=0)
    key = instance_key(a_mn=False, b_mn=False)
    assert len(key) == 16 and key[11:] == (0, "none", ("none", "none"), 16, False)


@pytest.mark.parametrize("pairs", [74, 106])  # 148 SMs (B200) / 212 SMs (R200)
def test_stream_k_plan_rules(pairs):
    # --- sk="auto" (the launcher default): a K-aligned two-way split of a single partial wave ---
    # every tile has exactly two parts (unit u = tile u // 2, K half u % 2), so tiles that share an A / B
    # panel stay K-synchronised and the fixup reads one slab
    assert stream_k_plan(24, 96, pairs, sk="auto") == (0, 24, 48, 48)
    assert stream_k_plan(pairs // 2, 96, pairs, sk="auto") == (
        0,
        pairs // 2,
        2 * (pairs // 2),
        48,
    )
    assert stream_k_plan(10, 17, pairs, sk="auto") == (
        0,
        10,
        20,
        9,
    )  # odd K steps: ceil(k / 2)
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


def test_planner_defaults_to_auto_stream_k_and_guards_the_slice_counters():
    # the 24-tile swapped weight gradient (indexer_hw, T = 1001): one partial wave, K = 16 steps -> split
    v = _views("proj", "indexer_hw", "wgrad", "bf16", 1001)
    plan, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=148,
        l2_bytes=L2_BYTES,
        transposed_out=v["transposed"],
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
        transposed_out=v["transposed"],
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
        and off.template == plan.template
    )
    # stride-0 rows keep these planner-only views tiny: 300 tiles of 256 x 256 on a 1000-SM device (500 pairs)
    # with the linearised policy put 300 tail tiles in flight -> 4800 slice counters exceed SK_DUMMY_BASE
    A = torch.empty(1024, dtype=torch.bfloat16).as_strided((300 * 256, 1024), (0, 1))
    W = torch.empty(1024, dtype=torch.bfloat16).as_strided((256, 1024), (0, 1))
    out = torch.empty(256, dtype=torch.bfloat16).as_strided((300 * 256, 256), (0, 1))
    assert SK_DUMMY_BASE == 4096
    with pytest.raises(ValueError, match="slice-counter budget"):
        plan_dense_projection_gemm(
            A, W.t(), out, sm_count=1000, l2_bytes=L2_BYTES, sk=True
        )
    plan, *_ = plan_dense_projection_gemm(
        A, W.t(), out, sm_count=1000, l2_bytes=L2_BYTES
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
                )
                assert plan.sm_pairs == sm_count // 2
                assert plan.grid == ((plan.num_full + plan.sk_units) * CTA_GROUP, 1, 1)
                assert plan.m_tiles % CTA_GROUP == 0
                assert plan.k_blocks == -(-plan.K // BLOCK_K)
                assert (
                    plan.pair_tiles
                    == plan.L * (plan.m_tiles // CTA_GROUP) * plan.n_tiles
                )
                assert a_desc.stride(2) == 1 and b_desc.stride(2) == 1
                # the knob defaults of the launcher (stage depth by tile shape, raster group, working-set-gated
                # hints) unless the row's measured rule (ROW_RULES, round 2) pins a knob
                assert plan.cta_rows == rule.get("cta_rows", 128)
                assert plan.pf == rule.get("pf", 0) and plan.promo == rule.get("promo", "none")
                assert plan.group_m == rule.get(
                    "group_m",
                    default_group_m(
                        plan.a_mn,
                        plan.b_mn,
                        plan.m_tiles,
                        plan.pair_tiles,
                        plan.sm_pairs,
                    ),
                )
                assert (re.search(r"_g\d+", plan.template) is None) == (
                    plan.group_m == 16
                )  # raster-group suffix only away from the default ("_gemm" is not one)
                assert plan.stages == rule.get(
                    "stages", default_stages(plan.slots, plan.cta_rows, plan.block_n)
                )
                if "stages" not in rule and plan.cta_rows == 128:
                    assert (
                        plan.stages
                        == {(256, 0): 7, (256, 1): 6, (128, 0): 9, (128, 1): 8}[
                            (plan.block_n, plan.slots)
                        ]
                    )
                assert plan.l2_bytes == L2_BYTES
                assert plan.wave_working_set_bytes == wave_working_set(
                    plan.m_tiles, plan.n_tiles, plan.K, plan.group_m, plan.sm_pairs
                )
                assert plan.hints == tuple(
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
                if "hints" not in rule:
                    if plan.wave_working_set_bytes <= L2_BYTES:
                        assert plan.hints == ("none", "none")
                    if not (plan.a_mn and plan.b_mn) and plan.n_tiles > 1:
                        assert plan.hints == ("none", "none")
                assert (plan.n_tiles == 1) == (plan.N <= 256)
                # stream-K "auto": a two-part K-aligned split of a single partial wave, else whole tiles
                if plan.sk_units:
                    assert (
                        plan.pair_tiles <= plan.sm_pairs
                        and 2 * plan.tail_tiles <= plan.sm_pairs
                    )
                    assert plan.k_blocks >= 2 * SK_MIN_ITERS
                    assert (plan.num_full, plan.tail_tiles, plan.sk_units) == (
                        0,
                        plan.pair_tiles,
                        2 * plan.pair_tiles,
                    )
                    assert plan.iters_per_unit == -(-plan.k_blocks // 2)
                    assert (
                        plan.ws_f32_elems
                        == (plan.sk_units + plan.tail_tiles)
                        * 2
                        * plan.cta_rows
                        * plan.block_n
                    )
                else:
                    assert (plan.num_full, plan.tail_tiles, plan.iters_per_unit) == (
                        plan.pair_tiles,
                        0,
                        plan.k_blocks,
                    )
                    assert (
                        plan.pair_tiles > plan.sm_pairs
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
                elif N < 256:  # swapped: X.T @ G with the transposed store
                    assert (plan.M, plan.N, plan.K) == (K, N, T) and plan.transposed_out
                else:
                    assert (
                        (plan.M, plan.N, plan.K) == (N, K, T)
                        and plan.a_mn
                        and plan.b_mn
                    )
                assert plan.block_n == rule.get(
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
                plan, _expected(MLA_TEMPLATES, (row, op), T, sm_count), (row, op, T, sm_count)
            )
            # the swapped plans (weight gradients; the tiny-T forward / input gradient) store transposed
            assert plan.transposed_out == plan.template.endswith("_t")
            rule = _rule_of(plan, sm_count)
            assert plan.hints == tuple(
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
            assert plan.group_m == rule.get(
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
    # K = D_out = 512 <= 1024: TMA-store epilogue
    _assert_mirrors_cake(
        plan, _expected(MLA_TEMPLATES, ("qabs", "dgrad"), 33, 148), ("qabs", "dgrad", 33, 148)
    )
    assert plan.template.startswith(
        "dense_proj_gemm_kk_n256"
    ) and plan.template.endswith("_tma1")


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
    assert plan.transposed_out and plan.epi == "reg" and plan.template.endswith("_t")


def test_wgrad_swap_rule():
    G = torch.empty(100, 32, dtype=torch.bfloat16)
    X = torch.empty(100, 64, dtype=torch.bfloat16)
    A, B, transposed = wgrad_views(G, X)
    assert transposed and A.shape == (64, 100) and B is G
    G = torch.empty(100, 256, dtype=torch.bfloat16)
    A, B, transposed = wgrad_views(G, X)
    assert not transposed and A.shape == (256, 100) and B is X


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
        Q.permute(1, 2, 0), dL.permute(1, 0, 2), out, sm_count=148, l2_bytes=L2_BYTES, _fallback=False
    )
    assert (plan.L, plan.M, plan.N, plan.K) == (H, N, M, T)
    assert plan.a_mn and plan.b_mn and plan.transposed_out and plan.block_n == 256
    assert plan.template.startswith(
        "dense_proj_gemm_nn_n256"
    ) and plan.template.endswith("_t")
    assert tuple(out3.shape) == (H, M, N)
    # the rule table: exact (N, K, M) first, then the ragged-M (N, K, None) key, then the ragged-K (N, None, M) key
    ident = ("sm_100a", True, True, False, True, True)
    saved = dict(ROW_RULES)
    try:
        ROW_RULES.clear()
        ROW_RULES[ident + (M, None, N)] = {"cta_rows": 256}
        assert row_rule("sm_100a", True, True, False, True, True, M, T, N) == {
            "cta_rows": 256
        }
        assert row_rule("sm_107a", True, True, False, True, True, M, T, N) == {}
        ROW_RULES[ident + (M, T, None)] = {"group_m": 8}
        assert row_rule("sm_100a", True, True, False, True, True, M, T, N) == {
            "group_m": 8
        }
        ROW_RULES[ident + (M, T, N)] = {"group_m": 4}
        assert row_rule("sm_100a", True, True, False, True, True, M, T, N) == {
            "group_m": 4
        }
        # a rule fills only the knobs the caller left unset
        ROW_RULES.clear()
        ROW_RULES[ident + (M, None, N)] = {"cta_rows": 256, "group_m": 8}
        ruled, *_ = plan_dense_projection_gemm(
            Q.permute(1, 2, 0),
            dL.permute(1, 0, 2),
            out,
            sm_count=148,
            l2_bytes=L2_BYTES, _fallback=False,
            arch="sm_100a",
        )
        assert (
            ruled.cta_rows == 256
            and ruled.group_m == 8
            and "_m256" in ruled.template
            and "_g8" in ruled.template
        )
        pinned, *_ = plan_dense_projection_gemm(
            Q.permute(1, 2, 0),
            dL.permute(1, 0, 2),
            out,
            sm_count=148,
            l2_bytes=L2_BYTES, _fallback=False,
            arch="sm_100a",
            cta_rows=128,
        )
        assert pinned.cta_rows == 128 and pinned.group_m == 8
    finally:
        ROW_RULES.clear()
        ROW_RULES.update(saved)


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
        11,
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
        "arch": "sm_100a",
        "template": "dense_proj_gemm_kk_n256",
        "kernel": "kernel_fake",
        "cache_name": "fake",
        "sources": [],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["parameter", "M"],
            ["workspace", "tma_descriptor_workspace"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 512,
        "closure_sha256": "0" * 64,
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
        cake_backend, "load_cake_dense_projection_gemm_module", lambda name: fake
    )
    A = torch.zeros(1, 8, 64, dtype=torch.bfloat16)
    ws = torch.zeros(512, dtype=torch.uint8)
    launch = bind_launch(
        "cake_dense_projection_gemm_fake",
        dict(A=A, M=8, tma_descriptor_workspace=ws),
        (4, 1, 1),
    )
    assert launch.arguments == (A, 8, ws, 4, 1, 1)
    launch()
    assert calls == [(A, 8, ws, 4, 1, 1)]
    with pytest.raises(KeyError, match="'M'"):
        bind_launch(
            "cake_dense_projection_gemm_fake",
            dict(A=A, tma_descriptor_workspace=ws),
            (4, 1, 1),
        )
    with pytest.raises(ValueError, match="cluster"):
        bind_launch(
            "cake_dense_projection_gemm_fake",
            dict(A=A, M=8, tma_descriptor_workspace=ws),
            (3, 1, 1),
        )
    assert (
        cake_jit.select_module("sm_100a", "dense_proj_gemm_kk_n256")
        == "cake_dense_projection_gemm_fake"
    )


# ---------------------------------------------------------------------------
# GPU correctness (skips without a registered instance)
# ---------------------------------------------------------------------------


def _make_inputs(family, row, op, out_dtype, T, seed, device="cuda"):
    """Deterministic operands for one row (same distributions as the CAKE-758 eval harness)."""
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


def _plan_template(v):
    plan, *_ = plan_dense_projection_gemm(
        v["A"],
        v["B"],
        v["out"],
        sm_count=_sm_count(),
        l2_bytes=_l2_bytes(),
        transposed_out=v.get("transposed", False),
    )
    return plan.template


@pytest.mark.parametrize("row, op, out_dtype, T", GPU_ROWS)
def test_projection_rows_match_fp64_reference(row, op, out_dtype, T):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
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


@pytest.mark.parametrize("row", list(MLA_ROWS))
@pytest.mark.parametrize("op", ["fwd", "dgrad", "wgrad"])
def test_mla_batched_rows_match_fp64_reference(row, op):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
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
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
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
    _require_program(_plan_template(dict(A=A.t(), B=Gs, out=out_t, transposed=True)))
    prepared = prepare_projection_wgrad(Gs, A, out_t)
    assert prepared.plan.transposed_out and prepared.template == _plan_template(
        dict(A=A.t(), B=Gs, out=out_t, transposed=True)
    )
    assert prepared.template.startswith(
        "dense_proj_gemm_nn_n128"
    ) and prepared.template.endswith("_f32_t")
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
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    # the 24-tile swapped weight gradient: stream-K "auto" two-part split (deterministic in-kernel fixup)
    v = _make_inputs("proj", "indexer_hw", "wgrad", "bf16", 1001, seed=SEED + 3)
    _require_program(_plan_template(v))
    prepared = prepare_projection_wgrad(v["G"], v["X"], v["out"])
    assert prepared.plan.transposed_out and prepared.plan.pair_tiles == 24
    if prepared.plan.sm_pairs >= 48:
        assert (prepared.plan.tail_tiles, prepared.plan.sk_units) == (24, 48)
    first = prepared.launch().clone()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    prepared.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert torch.equal(first, v["out"])


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
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
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
    eager = dict(
        fwd=lambda: cake_backend.router_forward(v["X"], v["W"]),
        dgrad=lambda: cake_backend.router_dgrad(v["G"], v["W"]),
        wgrad=lambda: cake_backend.router_wgrad(v["G"], v["X"]),
    )[op]()
    torch.cuda.synchronize()
    assert torch.equal(eager, v["out"])
