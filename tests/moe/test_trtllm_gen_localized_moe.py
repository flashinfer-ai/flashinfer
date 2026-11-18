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

uGPU localization of the trtllm-gen NVFP4 MoE: both expert GEMMs split across a
GPU's two locality domains, each writing a slice of the shared GEMM outputs.
On Rubin the domain streams are green contexts; on other SM10x GPUs two plain
streams with half the SMs exercise the same split and slice writes.
"""

import functools

import pytest
import torch

from flashinfer import RoutingMethodType, fp4_quantize
from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    split_fp4_moe_weights_by_locality_domain,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.tllm_enums import ActivationType, SfLayout
from flashinfer.utils import get_compute_capability

_FP4_GLOBAL = 448.0 * 6.0
_EPILOGUE_TILE_M = 128


def _skip_unless_sm10x():
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm-gen NVFP4 MoE needs an SM10x GPU")


@functools.cache
def _domain_streams(device_index: int):
    """Two streams with their per-domain SM count: green contexts when the GPU has
    locality domains (Rubin), otherwise plain streams sized to half the SMs."""
    device = torch.device("cuda", device_index)
    try:
        from flashinfer.green_ctx import (
            get_locality_domain_count,
            split_device_green_ctx_by_locality_domain,
        )

        if get_locality_domain_count(device) >= 2:
            streams, resources = split_device_green_ctx_by_locality_domain(device)
            return streams[:2], min(int(r.sm.smCount) for r in resources[:2])
    except (ImportError, RuntimeError):
        pass
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return [torch.cuda.Stream(device), torch.cuda.Stream(device)], sm_count // 2


def _quantize(x: torch.Tensor):
    """NVFP4 with linear block scales: ([rows, K/2] uint8, [rows, K/16] float8)."""
    k = x.shape[-1]
    packed, scales = fp4_quantize(
        x.reshape(-1, k),
        torch.tensor([_FP4_GLOBAL], device=x.device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    lead = x.shape[:-1]
    return (
        packed.reshape(*lead, k // 2),
        scales.view(torch.float8_e4m3fn).reshape(*lead, k // 16),
    )


def _shuffle(w1, sf1, w2, sf2, cache):
    """The trtllm-gen weight layout: gated row interleave + shuffle + scale interleave."""
    num_experts = w1.shape[0]
    out = {key: [] for key in ("w1", "sf1", "w2", "sf2")}
    for e in range(num_experts):
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w1[e].view(torch.uint8), _EPILOGUE_TILE_M, is_gated_act_gemm=True
        )
        out["w1"].append(w1[e].view(torch.uint8)[perm.to(w1.device)].contiguous())
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            sf1[e].view(torch.uint8),
            _EPILOGUE_TILE_M,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        out["sf1"].append(
            block_scale_interleave(
                sf1[e].view(torch.uint8)[perm.to(w1.device)].contiguous()
            )
        )
        perm = get_w2_permute_indices_with_cache(
            cache, w2[e].view(torch.uint8), _EPILOGUE_TILE_M, is_gated_act_gemm=True
        )
        out["w2"].append(w2[e].view(torch.uint8)[perm.to(w2.device)].contiguous())
        perm = get_w2_permute_indices_with_cache(
            cache,
            sf2[e].view(torch.uint8),
            _EPILOGUE_TILE_M,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        out["sf2"].append(
            block_scale_interleave(
                sf2[e].view(torch.uint8)[perm.to(w2.device)].contiguous()
            )
        )
    return (
        torch.stack(out["w1"]),
        torch.stack(out["sf1"]).view(torch.float8_e4m3fn).reshape(sf1.shape),
        torch.stack(out["w2"]),
        torch.stack(out["sf2"]).view(torch.float8_e4m3fn).reshape(sf2.shape),
    )


def _make_problem(num_tokens, hidden_size, intermediate_size, num_experts):
    device = torch.device("cuda")
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    x, x_sf = _quantize(hidden * 0.1)
    w1_q, w1_sf = _quantize(w1)
    w2_q, w2_sf = _quantize(w2)
    scale = 1.0 / _FP4_GLOBAL / _FP4_GLOBAL
    return {
        "routing_logits": torch.rand(num_tokens, num_experts, device=device).to(
            torch.bfloat16
        ),
        "hidden_states": x,
        "hidden_states_scale": x_sf,
        "raw_weights": (w1_q, w1_sf, w2_q, w2_sf),
        "output1_scale_scalar": torch.full((num_experts,), scale, device=device),
        "output1_scale_gate_scalar": torch.full((num_experts,), scale, device=device),
        "output2_scale_scalar": torch.full((num_experts,), scale, device=device),
    }


def _localized_shards(raw_weights, cache):
    shards = split_fp4_moe_weights_by_locality_domain(*raw_weights)
    localized = []
    for shard in shards:
        w1, sf1, w2, sf2 = _shuffle(
            shard["gemm1_weights"],
            shard["gemm1_weights_scale"],
            shard["gemm2_weights"],
            shard["gemm2_weights_scale"],
            cache,
        )
        localized.append(
            {
                "gemm1_weights": w1,
                "gemm1_weights_scale": sf1,
                "gemm2_weights": w2,
                "gemm2_weights_scale": sf2,
            }
        )
    return localized


def _run(problem, weights, num_experts, top_k, intermediate_size, **localized):
    w1, sf1, w2, sf2 = weights
    return trtllm_fp4_block_scale_moe(
        problem["routing_logits"],
        None,
        problem["hidden_states"],
        problem["hidden_states_scale"],
        w1,
        sf1,
        None,
        None,
        None,
        None,
        w2,
        sf2,
        None,
        problem["output1_scale_scalar"],
        problem["output1_scale_gate_scalar"],
        problem["output2_scale_scalar"],
        num_experts,
        top_k,
        None,
        None,
        intermediate_size,
        0,
        num_experts,
        None,
        RoutingMethodType.Renormalize.value,
        True,
        None,
        ActivationType.Swiglu.value,
        hidden_states_scale_layout=SfLayout.layout_linear,
        **localized,
    )[0].float()


def _relative_error(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def test_split_fp4_moe_weights_by_locality_domain_rows():
    # Each domain gets its rows of both FC1 halves and its FC2 hidden rows.
    e, inter, hidden = 2, 8, 6
    w1 = torch.arange(e * 2 * inter * 3, dtype=torch.int32).reshape(e, 2 * inter, 3)
    w2 = torch.arange(e * hidden * 4, dtype=torch.int32).reshape(e, hidden, 4)
    shards = split_fp4_moe_weights_by_locality_domain(w1, w1, w2, w2)
    assert len(shards) == 2
    for d, shard in enumerate(shards):
        rows = list(range(d * 4, d * 4 + 4))
        expected_fc1 = torch.cat((w1[:, rows], w1[:, [inter + r for r in rows]]), 1)
        torch.testing.assert_close(shard["gemm1_weights"], expected_fc1)
        torch.testing.assert_close(shard["gemm1_weights_scale"], expected_fc1)
        torch.testing.assert_close(shard["gemm2_weights"], w2[:, d * 3 : (d + 1) * 3])
        assert shard["gemm2_weights"].is_contiguous()


@pytest.mark.parametrize("num_tokens", [1, 64, 1024])
@pytest.mark.parametrize("hidden_size,intermediate_size", [(1024, 512), (2048, 768)])
def test_localized_matches_full_gpu(num_tokens, hidden_size, intermediate_size):
    _skip_unless_sm10x()
    num_experts, top_k = 8, 2
    problem = _make_problem(num_tokens, hidden_size, intermediate_size, num_experts)
    cache = {}
    full = _shuffle(*problem["raw_weights"], cache)
    streams, sm_count = _domain_streams(torch.cuda.current_device())

    expected = _run(problem, full, num_experts, top_k, intermediate_size)
    actual = _run(
        problem,
        full,
        num_experts,
        top_k,
        intermediate_size,
        localized_weights=_localized_shards(problem["raw_weights"], cache),
        localized_streams=streams,
        localized_sm_count=sm_count,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    # Same math on each half of every output; only the tactic (and so the FP32
    # accumulation order) may differ from the full-GPU launch.
    assert _relative_error(actual, expected) < 2e-2


def test_localized_routed_and_cuda_graph_gpu():
    _skip_unless_sm10x()
    num_tokens, hidden_size, intermediate_size = 32, 1024, 512
    num_experts, top_k = 8, 2
    problem = _make_problem(num_tokens, hidden_size, intermediate_size, num_experts)
    cache = {}
    full = _shuffle(*problem["raw_weights"], cache)
    shards = _localized_shards(problem["raw_weights"], cache)
    streams, sm_count = _domain_streams(torch.cuda.current_device())
    topk_weights, topk_ids = torch.topk(
        torch.softmax(problem["routing_logits"].float(), -1), top_k
    )
    routing = (topk_ids.to(torch.int32), topk_weights.to(torch.bfloat16))
    output = torch.empty(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)

    def call(**localized):
        w1, sf1, w2, sf2 = full
        return trtllm_fp4_block_scale_routed_moe(
            routing,
            None,
            problem["hidden_states"],
            problem["hidden_states_scale"],
            w1,
            sf1,
            None,
            None,
            None,
            None,
            w2,
            sf2,
            None,
            problem["output1_scale_scalar"],
            problem["output1_scale_gate_scalar"],
            problem["output2_scale_scalar"],
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            0,
            num_experts,
            None,
            RoutingMethodType.TopK.value,
            True,
            None,
            ActivationType.Swiglu.value,
            output=output,
            hidden_states_scale_layout=SfLayout.layout_linear,
            **localized,
        )[0]

    expected = call().float().clone()
    localized = dict(
        localized_weights=shards, localized_streams=streams, localized_sm_count=sm_count
    )
    eager = call(**localized).float().clone()
    assert _relative_error(eager, expected) < 2e-2

    # Fork and join must be captured: replay recomputes the same output.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(**localized)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), eager)


def test_localized_autotune_gpu():
    _skip_unless_sm10x()
    num_tokens, hidden_size, intermediate_size = 256, 1024, 512
    num_experts, top_k = 8, 2
    problem = _make_problem(num_tokens, hidden_size, intermediate_size, num_experts)
    cache = {}
    full = _shuffle(*problem["raw_weights"], cache)
    streams, sm_count = _domain_streams(torch.cuda.current_device())
    localized = dict(
        localized_weights=_localized_shards(problem["raw_weights"], cache),
        localized_streams=streams,
        localized_sm_count=sm_count,
    )
    expected = _run(problem, full, num_experts, top_k, intermediate_size)
    with autotune(True):
        _run(problem, full, num_experts, top_k, intermediate_size, **localized)
    actual = _run(problem, full, num_experts, top_k, intermediate_size, **localized)
    assert _relative_error(actual, expected) < 2e-2


def test_localized_rejects_bad_arguments_gpu():
    _skip_unless_sm10x()
    num_experts, top_k, hidden_size, intermediate_size = 8, 2, 1024, 512
    problem = _make_problem(4, hidden_size, intermediate_size, num_experts)
    cache = {}
    full = _shuffle(*problem["raw_weights"], cache)
    shards = _localized_shards(problem["raw_weights"], cache)
    streams, sm_count = _domain_streams(torch.cuda.current_device())
    with pytest.raises(ValueError, match="exactly two shards"):
        _run(
            problem,
            full,
            num_experts,
            top_k,
            intermediate_size,
            localized_weights=shards[:1],
            localized_streams=streams[:1],
            localized_sm_count=sm_count,
        )
    with pytest.raises(ValueError, match="localized_sm_count"):
        _run(
            problem,
            full,
            num_experts,
            top_k,
            intermediate_size,
            localized_weights=shards,
            localized_streams=streams,
        )
    with pytest.raises(ValueError, match="distinct"):
        _run(
            problem,
            full,
            num_experts,
            top_k,
            intermediate_size,
            localized_weights=shards,
            localized_streams=[streams[0], streams[0]],
            localized_sm_count=sm_count,
        )
    # A shard cut the wrong way (full-width FC2 rows) fails the native shape check.
    bad = [dict(shard) for shard in shards]
    bad[1]["gemm2_weights"] = full[2]
    with pytest.raises(Exception, match="gemm2_weights shard 1"):
        _run(
            problem,
            full,
            num_experts,
            top_k,
            intermediate_size,
            localized_weights=bad,
            localized_streams=streams,
            localized_sm_count=sm_count,
        )
