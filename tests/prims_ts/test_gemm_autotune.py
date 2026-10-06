# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Support checks and autotune cache behavior for the dense PrimsTS GEMM."""

import pytest
import torch

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.prims_ts.gemm.config import PrimsTsGemmConfig
from flashinfer.prims_ts.gemm.runner import PrimsTsGemmRunner
from flashinfer.prims_ts.gemm.support import nvfp4_128x4_numel, validate_dense_gemm
from flashinfer.prims_ts.gemm.tactics import (
    MAX_PROFILED_TACTICS,
    config_from_tactic,
    default_tactic,
    fallback_cluster,
    fallback_config,
    legal_tactics,
    tactic_is_legal,
)
from flashinfer.utils import get_compute_capability


def _require_gemm_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("requires SM100, SM103, or SM107")


def _drop_dense_gemm_cache():
    tuner = AutoTuner.get()
    for key in list(tuner.profiling_cache):
        if str(getattr(key, "custom_op", "")).startswith("prims_ts_fp"):
            del tuner.profiling_cache[key]


def _two_tactics(self, inputs, profile):
    del inputs, profile
    identity = self.identity
    default = default_tactic(
        identity.arch,
        identity.operand_format,
        identity.output_format,
        identity.epilogue,
    )
    clc_off = default[:7] + (False,) + default[8:]
    return [default, clc_off]


def test_n_need_not_be_a_multiple_of_512():
    scale = torch.empty((nvfp4_128x4_numel(256, 256),), dtype=torch.uint8)
    validate_dense_gemm(
        arch=100,
        operand_format="nvfp4_e2m1",
        epilogue="linear",
        n=256,
        k=256,
        out_dtype=torch.bfloat16,
        weight_scale=scale,
        a_global_scale=1.0,
        weight_global_scale=1.0,
    )
    validate_dense_gemm(
        arch=100,
        operand_format="fp8_e4m3",
        epilogue="linear",
        n=128,
        k=128,
        out_dtype=torch.bfloat16,
        weight_scale=torch.empty((128,), dtype=torch.float32),
    )


def test_k_alignment_and_arch_are_refused_before_compile():
    with pytest.raises(ValueError, match="256"):
        validate_dense_gemm(
            arch=100,
            operand_format="nvfp4_e2m1",
            epilogue="linear",
            n=256,
            k=128,
            out_dtype=torch.bfloat16,
            a_global_scale=1.0,
            weight_global_scale=1.0,
        )
    with pytest.raises(ValueError, match="128"):
        validate_dense_gemm(
            arch=100,
            operand_format="fp8_e4m3",
            epilogue="linear",
            n=128,
            k=64,
            out_dtype=torch.bfloat16,
            weight_scale=torch.empty((128,), dtype=torch.float32),
        )
    with pytest.raises(RuntimeError, match="SM100"):
        validate_dense_gemm(
            arch=90,
            operand_format="fp8_e4m3",
            epilogue="linear",
            n=128,
            k=128,
            out_dtype=torch.bfloat16,
            weight_scale=torch.empty((128,), dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="even N"):
        validate_dense_gemm(
            arch=100,
            operand_format="fp8_e4m3",
            epilogue="swiglu",
            n=129,
            k=128,
            out_dtype=torch.bfloat16,
            weight_scale=torch.empty((129,), dtype=torch.float32),
        )


def test_fallback_cluster_and_default_config():
    assert fallback_cluster((4, 2, 1)) == (2, 1, 1)
    assert fallback_cluster((2, 2, 1)) == (2, 1, 1)
    assert fallback_cluster((2, 1, 1)) is None
    fp8 = fallback_config(
        arch=100,
        operand_format="fp8_e4m3",
        output_format="bf16",
        epilogue="linear",
        has_bias=False,
        head_dim=None,
        is_neox=None,
        has_qkv_scale=False,
    )
    assert fp8 == PrimsTsGemmConfig(
        100, "fp8_e4m3", "bf16", "linear", False, None, None
    )
    assert fp8.ab_stages is None
    fp4 = fallback_config(
        arch=100,
        operand_format="nvfp4_e2m1",
        output_format="bf16",
        epilogue="linear",
        has_bias=False,
        head_dim=None,
        is_neox=None,
        has_qkv_scale=False,
    )
    assert fp4 == PrimsTsGemmConfig(
        100, "nvfp4_e2m1", "bf16", "linear", False, None, None, tile_k=256
    )


def test_legal_tactics_cover_each_axis_and_drop_illegal_ones():
    sm100 = legal_tactics(100, "nvfp4_e2m1", "bf16", "linear")
    assert sm100[0] == default_tactic(100, "nvfp4_e2m1", "bf16", "linear")
    assert len(sm100) <= MAX_PROFILED_TACTICS
    assert all(tactic[6] == 64 for tactic in sm100)
    assert {tactic[7] for tactic in sm100} == {True, False}
    assert {tactic[8] for tactic in sm100} == {4, 8}
    assert {tactic[9] for tactic in sm100} == {True, False}
    assert all(not (tactic[4] and tactic[9]) for tactic in sm100)
    assert all(tactic[0] in (2, 4) and tactic[1] in (1, 2, 4) for tactic in sm100)
    assert all(tactic[3] != 768 for tactic in sm100)

    sm103 = legal_tactics(103, "nvfp4_e2m1", "bf16", "linear")
    assert any(tactic[6] == 96 and tactic[3] == 256 for tactic in sm103)
    assert all(tactic[3] == 256 for tactic in sm103 if tactic[6] == 96)
    assert len(sm103) <= MAX_PROFILED_TACTICS

    swiglu = legal_tactics(103, "nvfp4_e2m1", "bf16", "swiglu")
    assert all(not tactic[4] and tactic[8] == 8 for tactic in swiglu)
    assert {tactic[9] for tactic in swiglu} == {True, False}
    assert any((tactic[0], tactic[1]) == (4, 4) for tactic in swiglu)

    qkv = legal_tactics(100, "nvfp4_e2m1", "bf16", "qkv_qknorm_rope")
    assert {tactic[9] for tactic in qkv} == {True, False}
    assert all(tactic[5] == 4 and tactic[8] == 8 and not tactic[4] for tactic in qkv)

    fp8 = legal_tactics(100, "fp8_e4m3", "bf16", "linear")
    assert all(tactic[3] in (128, 256) and tactic[6] == 64 for tactic in fp8)
    assert fp8[0][3] == 128
    assert fp8[0][5] == 6

    for tactics in (sm100, sm103, swiglu, fp8):
        for tactic in tactics:
            cluster = (tactic[0], tactic[1], 1)
            fallback = fallback_cluster(cluster)
            if fallback is None:
                assert cluster == (2, 1, 1)
            else:
                assert fallback == (2, 1, 1)
                assert all(
                    preferred % smaller == 0
                    for smaller, preferred in zip(fallback, cluster, strict=True)
                )

    illegal_cluster = (1, 1, 256, 256, False, 5, 64, True, 4, False)
    illegal_overlap = (2, 2, 256, 256, True, 5, 64, True, 8, False)
    illegal_tma = (2, 2, 256, 256, True, 5, 64, True, 8, True)
    assert not tactic_is_legal(100, "nvfp4_e2m1", "bf16", "linear", illegal_cluster)
    assert not tactic_is_legal(103, "nvfp4_e2m1", "bf16", "swiglu", illegal_overlap)
    assert not tactic_is_legal(100, "nvfp4_e2m1", "bf16", "linear", illegal_tma)
    assert not tactic_is_legal(
        100,
        "nvfp4_e2m1",
        "nvfp4_e2m1",
        "swiglu",
        (2, 2, 256, 256, False, 5, 64, True, 8, True),
    )
    assert tactic_is_legal(
        100,
        "nvfp4_e2m1",
        "bf16",
        "swiglu",
        (4, 4, 256, 256, False, 5, 64, True, 8, True),
    )
    assert illegal_cluster not in sm100
    assert all(not tactic[4] for tactic in swiglu)

    narrowed = config_from_tactic(
        arch=100,
        operand_format="fp8_e4m3",
        output_format="bf16",
        epilogue="linear",
        has_bias=False,
        head_dim=None,
        is_neox=None,
        has_qkv_scale=False,
        tactic=(2, 2, 256, 128, False, 5, 64, False, 8, True),
    )
    assert narrowed.ab_stages == 5
    assert narrowed.epilogue_warps == 8
    assert narrowed.scheduler == "static"
    assert narrowed.use_tma_store is True


def test_explicit_config_does_not_call_the_tuner(monkeypatch):
    _require_gemm_gpu()
    from flashinfer.gemm import fp8_linear

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("explicit config must not call the autotuner")

    monkeypatch.setattr(AutoTuner.get(), "choose_one", fail_if_called)
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    m, n, k = 32, 256, 128
    a = torch.randn((m, k), device=device).to(torch.float8_e4m3fn)
    weight = torch.randn((n, k), device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones((m,), device=device)
    weight_scale = torch.ones((n,), device=device)
    config = PrimsTsGemmConfig(
        major * 10 + minor, "fp8_e4m3", "bf16", "linear", False, None, None
    )
    fp8_linear(a, weight, a_scale, weight_scale, config=config)


def test_autotune_second_call_hits_cache_and_epilogues_stay_separate(monkeypatch):
    _require_gemm_gpu()
    from flashinfer.gemm import fp8_linear, fp8_linear_swiglu

    calls = {"n": 0}

    def counting(self, inputs, profile):
        calls["n"] += 1
        return _two_tactics(self, inputs, profile)

    monkeypatch.setattr(PrimsTsGemmRunner, "get_valid_tactics", counting)
    _drop_dense_gemm_cache()
    device = torch.device("cuda")
    m, n, k = 32, 256, 128
    a = torch.randn((m, k), device=device).to(torch.float8_e4m3fn)
    weight = torch.randn((n, k), device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones((m,), device=device)
    weight_scale = torch.ones((n,), device=device)
    with autotune(True, tuning_buckets=(32,)):
        fp8_linear(a, weight, a_scale, weight_scale)
        assert calls["n"] == 1
        fp8_linear(a, weight, a_scale, weight_scale)
        assert calls["n"] == 1
        ops = {key.custom_op for key in AutoTuner.get().profiling_cache}
        assert "prims_ts_fp8_linear" in ops
        assert "prims_ts_fp8_swiglu" not in ops
        fp8_linear_swiglu(a, weight, a_scale, weight_scale)
        ops = {key.custom_op for key in AutoTuner.get().profiling_cache}
        assert "prims_ts_fp8_swiglu" in ops
        assert calls["n"] == 2
