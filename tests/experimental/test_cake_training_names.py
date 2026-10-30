# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU import/identity checks for deterministic training backend names."""

import importlib

import pytest

MODULE_ALIASES = [
    (
        "flashinfer.experimental.cake_kda_train.cake_kcp",
        "flashinfer.experimental.cake_kda_det_train.cake_kcp",
    ),
    (
        "flashinfer.experimental.cake_kda_train.cake_backend",
        "flashinfer.experimental.cake_kda_det_train.cake_backend",
    ),
    (
        "flashinfer.experimental.cake_kda_train.cake_jit",
        "flashinfer.experimental.cake_kda_det_train.cake_jit",
    ),
    (
        "flashinfer.experimental.cake_kda_train.cake_build",
        "flashinfer.experimental.cake_kda_det_train.cake_build",
    ),
    (
        "flashinfer.experimental.cake_kda_train.cake_kcp_plan",
        "flashinfer.experimental.cake_kda_det_train.cake_kcp_plan",
    ),
    (
        "flashinfer.experimental.cake_kda_train.cake_kcp_jit",
        "flashinfer.experimental.cake_kda_det_train.cake_kcp_jit",
    ),
    (
        "flashinfer.experimental.cake_lm_head_loss.cake_backend",
        "flashinfer.experimental.cake_lm_head_det_loss.cake_backend",
    ),
    (
        "flashinfer.experimental.cake_lm_head_loss.cake_jit",
        "flashinfer.experimental.cake_lm_head_det_loss.cake_jit",
    ),
    (
        "flashinfer.experimental.cake_lm_head_loss.cake_support",
        "flashinfer.experimental.cake_lm_head_det_loss.cake_support",
    ),
    (
        "flashinfer.experimental.cake_lm_head_loss.cake_build",
        "flashinfer.experimental.cake_lm_head_det_loss.cake_build",
    ),
    (
        "flashinfer.experimental.cake_lm_head_loss.cake_backend_opt_in",
        "flashinfer.experimental.cake_lm_head_det_loss.cake_backend_opt_in",
    ),
    (
        "flashinfer.experimental.cake_moe_grouped_gemm.cake_backend",
        "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_backend",
    ),
    (
        "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit",
        "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_jit",
    ),
    (
        "flashinfer.experimental.cake_moe_grouped_gemm.cake_build",
        "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_build",
    ),
    (
        "flashinfer.experimental.cake_moe_grouped_gemm.cake_autograd",
        "flashinfer.experimental.cake_moe_grouped_det_gemm.cake_autograd",
    ),
    (
        "flashinfer.experimental.dense_projection_gemm.cake_backend",
        "flashinfer.experimental.dense_projection_det_gemm.cake_backend",
    ),
    (
        "flashinfer.experimental.dense_projection_gemm.cake_jit",
        "flashinfer.experimental.dense_projection_det_gemm.cake_jit",
    ),
    (
        "flashinfer.experimental.dense_projection_gemm.cake_build",
        "flashinfer.experimental.dense_projection_det_gemm.cake_build",
    ),
    (
        "flashinfer.cake_mhc_train_backend.cake_jit",
        "flashinfer.cake_mhc_det_train_backend.cake_jit",
    ),
    (
        "flashinfer.cake_mhc_train_backend.cake_build",
        "flashinfer.cake_mhc_det_train_backend.cake_build",
    ),
    (
        "flashinfer.cake_mhc_train_head_backend.cake_jit",
        "flashinfer.cake_mhc_det_train_head_backend.cake_jit",
    ),
    (
        "flashinfer.cake_mhc_train_head_backend.cake_build",
        "flashinfer.cake_mhc_det_train_head_backend.cake_build",
    ),
    (
        "flashinfer.cake_rmsnorm_train_backend.cake_jit",
        "flashinfer.cake_rmsnorm_det_train_backend.cake_jit",
    ),
    (
        "flashinfer.cake_rmsnorm_train_backend.cake_build",
        "flashinfer.cake_rmsnorm_det_train_backend.cake_build",
    ),
    (
        "flashinfer.cake_rmsnorm_silu_gate_train_backend.cake_jit",
        "flashinfer.cake_rmsnorm_silu_gate_det_train_backend.cake_jit",
    ),
    (
        "flashinfer.cake_rmsnorm_silu_gate_train_backend.cake_build",
        "flashinfer.cake_rmsnorm_silu_gate_det_train_backend.cake_build",
    ),
    (
        "flashinfer.cake_causal_conv1d_train_backend.kcp_jit",
        "flashinfer.cake_causal_conv1d_det_train_backend.kcp_jit",
    ),
    (
        "flashinfer.cake_causal_conv1d_train_backend.cake_jit",
        "flashinfer.cake_causal_conv1d_det_train_backend.cake_jit",
    ),
    (
        "flashinfer.cake_causal_conv1d_train_backend.kcp",
        "flashinfer.cake_causal_conv1d_det_train_backend.kcp",
    ),
    (
        "flashinfer.cake_causal_conv1d_train_backend.cake_build",
        "flashinfer.cake_causal_conv1d_det_train_backend.cake_build",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_tables",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_tables",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_kcp",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_kcp",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_fake_backend",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_fake_backend",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_backend",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_backend",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_jit",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_jit",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_options",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_options",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_bytes",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_bytes",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_build",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_build",
    ),
    (
        "flashinfer.experimental.cake_gdn_train.cake_kcp_plan",
        "flashinfer.experimental.cake_gdn_det_train_kcp.cake_kcp_plan",
    ),
    ("flashinfer.cake_mhc_train", "flashinfer.cake_mhc_det_train"),
    ("flashinfer.cake_rmsnorm_train", "flashinfer.cake_rmsnorm_det_train"),
    (
        "flashinfer.cake_rmsnorm_silu_gate_train",
        "flashinfer.cake_rmsnorm_silu_gate_det_train",
    ),
    ("flashinfer.cake_causal_conv1d_train", "flashinfer.cake_causal_conv1d_det_train"),
]

PACKAGE_ALIASES = [
    (
        "flashinfer.experimental.cake_kda_train",
        "flashinfer.experimental.cake_kda_det_train",
    ),
    (
        "flashinfer.experimental.cake_lm_head_loss",
        "flashinfer.experimental.cake_lm_head_det_loss",
    ),
    (
        "flashinfer.experimental.cake_moe_grouped_gemm",
        "flashinfer.experimental.cake_moe_grouped_det_gemm",
    ),
    (
        "flashinfer.experimental.dense_projection_gemm",
        "flashinfer.experimental.dense_projection_det_gemm",
    ),
    ("flashinfer.cake_mhc_train_backend", "flashinfer.cake_mhc_det_train_backend"),
    (
        "flashinfer.cake_mhc_train_head_backend",
        "flashinfer.cake_mhc_det_train_head_backend",
    ),
    (
        "flashinfer.cake_rmsnorm_train_backend",
        "flashinfer.cake_rmsnorm_det_train_backend",
    ),
    (
        "flashinfer.cake_rmsnorm_silu_gate_train_backend",
        "flashinfer.cake_rmsnorm_silu_gate_det_train_backend",
    ),
    (
        "flashinfer.cake_causal_conv1d_train_backend",
        "flashinfer.cake_causal_conv1d_det_train_backend",
    ),
    (
        "flashinfer.experimental.cake_gdn_train",
        "flashinfer.experimental.cake_gdn_det_train_kcp",
    ),
]


@pytest.mark.parametrize("old,new", MODULE_ALIASES)
def test_compatibility_modules_share_identity(old, new):
    legacy = importlib.import_module(old)
    canonical = importlib.import_module(new)
    assert legacy is canonical


@pytest.mark.parametrize("old,new", PACKAGE_ALIASES)
def test_compatibility_exports_share_classes_and_functions(old, new):
    legacy = importlib.import_module(old)
    canonical = importlib.import_module(new)
    for name in getattr(canonical, "__all__", ()):
        assert getattr(legacy, name) is getattr(canonical, name), name


@pytest.mark.parametrize(
    "facade,implementation,entry",
    [
        ("cake_kda_det_train", "kda_train", "chunk_kda"),
        ("cake_kda_det_train_kcp", "kda_train", "chunk_kda_kcp"),
        ("cake_gdn_det_train_kcp", "gdn_train", "chunk_gdn_kcp"),
        (
            "cake_causal_conv1d_det_train_kcp",
            "cake_causal_conv1d_det_train",
            "cake_causal_conv1d_kcp",
        ),
    ],
)
def test_canonical_route_preserves_callable(facade, implementation, entry):
    named = importlib.import_module(f"flashinfer.{facade}")
    original = importlib.import_module(f"flashinfer.{implementation}")
    assert getattr(named, entry) is getattr(original, entry)
    assert entry in named.__all__ and entry in dir(named)
    with pytest.raises(AttributeError):
        _ = named.no_such_entry


def test_kcp_packages_share_existing_implementation():
    kda = importlib.import_module("flashinfer.experimental.cake_kda_det_train_kcp")
    shared = importlib.import_module(
        "flashinfer.experimental.cake_kda_det_train.cake_kcp"
    )
    assert kda.cake_kcp is shared
    conv = importlib.import_module(
        "flashinfer.cake_causal_conv1d_det_train_kcp_backend"
    )
    shared = importlib.import_module(
        "flashinfer.cake_causal_conv1d_det_train_backend.kcp"
    )
    assert conv.kcp is shared
