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

"""Tests for the "specialized" SM120 backend of ``cutlass_fused_moe`` with
``use_deepseek_fp8_block_scale=True`` (agent-generated fp8 [128,128]
block-scale MoE kernels at the contract E=256, top_k=8, hidden=2048,
intermediate=512, silu-gated, bf16 activations).

On SM120 the stock path refuses this contract outright, so the dispatch
tests verify fail-closed semantics: auto routes to the specialized kernels
exactly on the shipped token-count allowlist and keeps the stock refusal
everywhere else; an explicitly selected backend is never silently swapped;
the kill switch restores the stock refusal. Numeric correctness is checked
against an fp32 block-dequant reference MoE, plus a CUDA-graph
capture/replay smoke.
"""

import pytest
import torch


def _is_validated_sm120_device():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    # The backend's dispatch surface requires the validated device profile
    # (CC 12.0 and the exact SM count its cooperative grids are sized for),
    # so other SM120 parts skip rather than exercise the stock refusal.
    return (props.major, props.minor) == (12, 0) and props.multi_processor_count == 188


def _has_min_cuda_toolkit():
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        v = get_cuda_version()
        # The kernels use cvt.rn.bf16x2.e4m3x2, which ptxas accepts only
        # from PTX ISA 9.2 (CUDA 13.2); CUDA 13.0/13.1 fail at JIT.
        return (v.major, v.minor) >= (13, 2)
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _is_validated_sm120_device(),
        reason="the specialized fp8 block-scale MoE backend serves only the "
        "validated SM120 device profile (CC 12.0, 188 SMs)",
    ),
    pytest.mark.skipif(
        not _has_min_cuda_toolkit(),
        reason="the specialized backend JIT-builds an sm_120a module "
        "(CUDA 13.2+ toolkit required for PTX ISA 9.2)",
    ),
]

DEVICE = "cuda"
E, TOPK, H, I = 256, 8, 2048, 512
REL_TOL = 2e-2


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cpu").manual_seed(20260102)
    w13 = torch.randn(E, 2 * I, H, generator=gen).to(torch.float8_e4m3fn)
    w2 = torch.randn(E, H, I, generator=gen).to(torch.float8_e4m3fn)
    w13_scale = torch.rand(E, (2 * I) // 128, H // 128, generator=gen)
    w2_scale = torch.rand(E, H // 128, I // 128, generator=gen)
    w13_scale = (w13_scale * 0.02 + 0.01).float()
    w2_scale = (w2_scale * 0.02 + 0.01).float()
    return (
        w13.to(DEVICE),
        w2.to(DEVICE),
        w13_scale.contiguous().to(DEVICE),
        w2_scale.contiguous().to(DEVICE),
    )


def _routing(t, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(t, E, generator=gen)
    ids = torch.topk(scores, TOPK, dim=-1).indices.to(torch.int32)
    tw = torch.softmax(torch.rand(t, TOPK, generator=gen), dim=-1)
    x = torch.randn(t, H, generator=gen).to(torch.bfloat16)
    return (
        x.to(DEVICE),
        ids.contiguous().to(DEVICE),
        tw.float().contiguous().to(DEVICE),
    )


def _dequant_block(w_fp8, scale):
    rows, cols = w_fp8.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(128, dim=0)[:rows]
    s = s.repeat_interleave(128, dim=1)[:, :cols]
    return w_fp8.to(torch.float32) * s


def _reference(x, ids, tw, weights):
    w13, w2, w13_scale, w2_scale = weights
    t = x.shape[0]
    x32 = x.to(torch.float32)
    out = torch.zeros(t, H, dtype=torch.float32, device=x.device)
    ids64 = ids.to(torch.int64)
    for e in torch.unique(ids64).tolist():
        tok, slot = (ids64 == e).nonzero(as_tuple=True)
        w13_e = _dequant_block(w13[e], w13_scale[e])
        w2_e = _dequant_block(w2[e], w2_scale[e])
        gate_up = x32[tok] @ w13_e.T
        gate, up = gate_up[:, :I], gate_up[:, I:]
        inter = (gate * torch.sigmoid(gate)) * up
        out.index_add_(0, tok, (inter @ w2_e.T) * tw[tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


def _call(x, ids, tw, weights, output=None, **overrides):
    from flashinfer.fused_moe import cutlass_fused_moe

    w13, w2, w13_scale, w2_scale = weights
    kwargs = dict(
        input=x,
        token_selected_experts=ids,
        token_final_scales=tw,
        fc1_expert_weights=w13,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[w13_scale, w2_scale],
        use_deepseek_fp8_block_scale=True,
    )
    if output is not None:
        kwargs["output"] = output
    kwargs.update(overrides)
    out = cutlass_fused_moe(**kwargs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _assert_close_to_reference(out, ref):
    scale = ref.float().abs().max().clamp(min=1e-3).item()
    rel_l2 = (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    ).item()
    max_abs = (out.float() - ref.float()).abs().max().item()
    assert rel_l2 < REL_TOL, f"rel_l2 {rel_l2}"
    assert max_abs < 0.05 * scale, f"max_abs {max_abs}"


@pytest.mark.parametrize("t", [4, 16])
def test_auto_routes_on_allowlist_and_matches_reference(weights, t):
    """On SM120 only the specialized backend can serve this contract (the
    stock path raises), so a successful auto call proves the dispatch; the
    output must match the fp32 block-dequant reference."""
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        specialized_moe_fp8_blockscale_stats,
    )

    x, ids, tw = _routing(t, seed=100 + t)
    ref = _reference(x, ids, tw, weights)
    before = specialized_moe_fp8_blockscale_stats()["specialized_dispatches"]
    out = _call(x, ids, tw, weights)
    torch.cuda.synchronize()
    after = specialized_moe_fp8_blockscale_stats()["specialized_dispatches"]
    assert after == before + 1
    _assert_close_to_reference(out, ref)


def test_auto_off_allowlist_keeps_stock_refusal(weights):
    """A token count off the shipped allowlist falls through to the stock
    path, which still refuses the contract on SM120 (fail-closed)."""
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        specialized_moe_fp8_blockscale_stats,
    )

    x, ids, tw = _routing(7, seed=1)
    with pytest.raises(NotImplementedError):
        _call(x, ids, tw, weights)
    reasons = specialized_moe_fp8_blockscale_stats()["fallback_reasons"]
    assert reasons.get("num_tokens_off_allowlist", 0) >= 1


def test_explicit_cutlass_never_swapped(weights):
    """An explicitly selected stock backend must never receive the
    specialized kernels: it keeps today's refusal even on-allowlist."""
    x, ids, tw = _routing(16, seed=2)
    with pytest.raises(NotImplementedError):
        _call(x, ids, tw, weights, backend="cutlass")


def test_explicit_specialized_off_allowlist_raises(weights):
    x, ids, tw = _routing(7, seed=3)
    with pytest.raises(NotImplementedError, match="specialized"):
        _call(x, ids, tw, weights, backend="specialized")


def test_bogus_backend_raises(weights):
    x, ids, tw = _routing(4, seed=4)
    with pytest.raises(ValueError):
        _call(x, ids, tw, weights, backend="bogus")


def test_kill_switch_restores_refusal(weights, monkeypatch):
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    x, ids, tw = _routing(16, seed=5)
    with pytest.raises(NotImplementedError):
        _call(x, ids, tw, weights)
    with pytest.raises(NotImplementedError, match="kill_switch"):
        _call(x, ids, tw, weights, backend="specialized")


def test_cuda_graph_replay_matches_eager(weights):
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        warmup,
    )

    w13, w2, w13_scale, w2_scale = weights
    warmup(w13, w2, [w13_scale, w2_scale], t_values=(16,))
    t = 16
    x, ids, tw = _routing(t, seed=6)
    out = torch.zeros(t, H, dtype=torch.bfloat16, device=DEVICE)
    _call(x, ids, tw, weights, output=out)
    torch.cuda.synchronize()
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(x, ids, tw, weights, output=out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        _call(x, ids, tw, weights, output=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    # a few bf16 ulp at the output magnitude: accumulation order may
    # differ between eager and replay (bf16-atomics class)
    tol = 2.0**-5 * eager.float().abs().max().clamp(min=1e-3).item()
    max_abs = (out.float() - eager.float()).abs().max().item()
    assert max_abs <= tol, f"graphed vs eager max_abs {max_abs} > {tol}"
