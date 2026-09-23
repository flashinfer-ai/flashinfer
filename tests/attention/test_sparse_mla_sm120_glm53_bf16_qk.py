# Copyright (c) 2026 Francesco Parisio
# SPDX-License-Identifier: Apache-2.0
"""Strict GLM53 BF16-QK routing, workspace and packed-cache numerics."""

import importlib.util
from pathlib import Path

import pytest
import torch

from flashinfer.mla import SparseMLASm120Wrapper
from flashinfer.mla._sparse_mla_sm120 import _execution as execution
from flashinfer.mla._sparse_mla_sm120 import _policy as policy


@pytest.fixture
def sm12x():
    """Skip native checks unless a supported SM12x CUDA device is available."""
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("GLM53 BF16-QK requires SM12x")


def _plan(**overrides):
    """Build a strict GLM53 policy request with optional geometry overrides."""
    values = dict(
        num_tokens=1,
        num_heads=16,
        topk=2112,
        model_type=3,
        page_block_size=64,
        has_extra=False,
        prefill_impl_pref=0,
        device=torch.device("cuda"),
        compute_precision="bf16_qk",
    )
    values.update(overrides)
    return policy.plan(**values)


@pytest.mark.parametrize("tokens", [1, 7, 65])
@pytest.mark.parametrize("width", [2112, 2176])
def test_fixed_sg_policy_does_not_use_fp8_calibration(monkeypatch, tokens, width):
    """Select SG without loading a module or consulting FP8 timing profiles."""

    def unexpected(*args, **kwargs):
        """Fail if strict planning reaches a forbidden calibration or JIT path."""
        raise AssertionError(
            "strict SG must not consult FP8 calibration or load a module"
        )

    monkeypatch.setattr(execution, "get_sparse_mla_sm120_module", unexpected)
    monkeypatch.setattr(policy._cpb, "get_ordinary_profile", unexpected)
    monkeypatch.setattr(policy, "_resolve_cpb", unexpected)
    monkeypatch.setattr(policy, "canonical_profile_layout", unexpected)
    assert policy.profile_selection(None, torch.device("cuda"), "bf16_qk") is None
    planned = _plan(num_tokens=tokens, topk=width)
    assert planned.variant is policy.KernelVariant.PREFILL_SG
    assert planned.cpb == -1


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_tokens": 0},
        {"num_heads": 8},
        {"num_heads": 32},
        {"topk": 2048},
        {"topk": 2051},
        {"topk": 2240},
        {"page_block_size": 32},
        {"has_extra": True},
        {"extra_topk": 64},
        {"extra_fp4": True},
        *({"model_type": model} for model in (0, 1, 2, 4, 5)),
        *({"prefill_impl_pref": preference} for preference in (1, 2, 3)),
    ],
)
def test_policy_rejects_unsupported_bf16_qk(overrides):
    """Reject unsupported models, shapes, extra caches and forced prefill choices."""
    with pytest.raises(ValueError, match="bf16_qk"):
        _plan(**overrides)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"kv_scale_format": "ue8m0_g32"},
        {"kv_scale_format": "arbitrary_fp32", "d_v": 1024},
        {"kv_scale_format": "arbitrary_fp32", "kv_cache_format": "nvfp4"},
    ],
)
def test_constructor_rejects_other_storage(kwargs):
    """Reject cache formats and value dimensions incompatible with GLM53 BF16 QK."""
    with pytest.raises(ValueError, match="bf16_qk requires GLM53"):
        SparseMLASm120Wrapper(compute_precision="bf16_qk", **kwargs)


def _metadata(tokens=1, width=2112, **overrides):
    """Build compact-cache SG metadata for native resolver and capability checks."""
    values = dict(
        model=3,
        tokens=tokens,
        heads=16,
        topk=width,
        extra_topk=0,
        page_size=64,
        extra_page_size=0,
        page_stride_bytes=64 * 528,
        extra_page_stride_bytes=0,
        row_stride_bytes=528,
        indices_stride=width,
        extra_indices_stride=0,
        lse_stride=16,
        has_lengths=True,
        has_extra_lengths=False,
        has_sink=False,
        extra_fp4=False,
        variant=1,
    )
    values.update(overrides)
    return execution.AttentionMetadata(**values)


def _resolve(metadata, precision="bf16_qk", max_shared=None):
    """Resolve with real GPU capabilities and an optional shared-memory limit."""
    props = torch.cuda.get_device_properties("cuda")
    return execution.resolve_attention(
        **metadata._asdict(),
        precision=precision,
        cpb=1,
        sm_count=props.multi_processor_count,
        max_shared_bytes=props.shared_memory_per_block_optin
        if max_shared is None
        else max_shared,
    )


@pytest.mark.parametrize("tokens", [1, 7, 65])
@pytest.mark.parametrize("width", [2112, 2176])
def test_descriptor_is_strict_and_accounts_for_bf16_shared_memory(sm12x, tokens, width):
    """Require hybrid SG, zero split scratch and adequate BF16 shared memory."""
    metadata = _metadata(tokens, width)
    strict = _resolve(metadata)
    ordinary_sg = _resolve(metadata, precision="default")
    facts = strict.inspect()
    assert facts["numeric_route"] == "hybrid"
    assert facts["implementation"] == "sg" and facts["variant"] == 1
    assert facts["merge"] == "direct" and facts["active_splits"] == 1
    assert facts["partial_bytes"] == facts["lse_bytes"] == 0
    assert all(requirement[2] == 0 for requirement in strict.workspace()[:2])
    assert ordinary_sg.inspect()["numeric_route"] == "fp8"
    assert facts["shared_bytes"] > ordinary_sg.inspect()["shared_bytes"]
    with pytest.raises(RuntimeError, match="shared memory exceeds"):
        _resolve(metadata, max_shared=int(facts["shared_bytes"]) - 1)
    props = torch.cuda.get_device_properties("cuda")
    legal = execution.metadata_candidates(
        metadata,
        "bf16_qk",
        props.multi_processor_count,
        props.shared_memory_per_block_optin,
    )
    assert set(legal) == {1}
    module = execution.get_sparse_mla_sm120_module()
    assert list(module.candidates(3, 16, width, 64, False, 0, "bf16_qk")) == [1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"heads": 8},
        {"topk": 2048, "indices_stride": 2048},
        {"topk": 2240, "indices_stride": 2240},
        *({"variant": variant} for variant in (0, 2, 3, 4)),
    ],
)
def test_resolver_does_not_fall_back_to_fp8(sm12x, overrides):
    """Reject unsupported native requests instead of silently selecting FP8 QK."""
    with pytest.raises(RuntimeError, match="bf16_qk requires"):
        _resolve(_metadata(**overrides))


def test_empty_strict_query_is_rejected(sm12x):
    """Reject zero-token strict calls before the wrapper's empty-query fast path."""
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="arbitrary_fp32", compute_precision="bf16_qk"
    )
    q = torch.empty(0, 16, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 64, 528, device="cuda", dtype=torch.uint8)
    indices = torch.empty(0, 2112, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="bf16_qk requires T>0"):
        wrapper.run(q, cache, indices, torch.empty_like(q), 256**-0.5)


@pytest.mark.parametrize("width", [2112, 2176])
def test_bf16_qk_packed_oracle_tails_masks_and_graphs(sm12x, width):
    """Pass all 13 packed-cache cases per width, restoring TF32 state afterward."""
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/repro_glm53_sm120_precision.py"
    )
    spec = importlib.util.spec_from_file_location("glm53_bf16_qk_repro", path)
    repro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        records = repro.run_suite(width, precision="bf16_qk")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert len(records) == 13
    assert all(record["passed"] for record in records), records
