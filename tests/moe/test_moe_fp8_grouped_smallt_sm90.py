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

import pytest
import torch

from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.smallt_sm90 import (
    _eligibility,
    _moe_smallt_sm90_stats,
    _reset_moe_smallt_sm90_stats,
)

DISABLE_ENV = "FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90"
E, K, H, I = 16, 9, 512, 256  # small test geometry (contract class, not size)
# W8A8 fused-MoE kernels quantize activations internally, so elementwise
# tolerance vs an fp32-dequant reference is noise-dominated; the class gate
# is token-sum relative error + max per-token L2 relative error (the same
# two criteria the measured verification uses).
TOKEN_SUM_REL, TOKEN_L2_REL = 0.15, 0.10


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    _reset_moe_smallt_sm90_stats()
    yield
    _reset_moe_smallt_sm90_stats()


def _sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")


def _make(T, device, seed=0, e=E, k=K, h=H, i=I):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def f8(shape):
        return (
            torch.randn(shape, generator=g, device=device, dtype=torch.float32)
        ).to(torch.float8_e4m3fn)

    pow2 = torch.tensor([0.0625, 0.125, 0.25], device=device)
    s1 = pow2[
        torch.randint(0, 3, (e, (2 * i + 127) // 128, (h + 127) // 128),
                      generator=g, device=device)
    ].float().contiguous()
    s2 = pow2[
        torch.randint(0, 3, (e, (h + 127) // 128, (i + 127) // 128),
                      generator=g, device=device)
    ].float().contiguous()
    ids = torch.randint(0, e, (T, k), generator=g, device=device,
                        dtype=torch.int32)
    tw = torch.rand((T, k), generator=g, device=device, dtype=torch.float32)
    tw = tw / tw.sum(-1, keepdim=True)
    x = torch.randn((T, h), generator=g, device=device,
                    dtype=torch.float32).to(torch.bfloat16)
    return dict(x=x, w13=f8((e, 2 * i, h)), s13=s1, w2=f8((e, h, i)), s2=s2,
                ids=ids, tw=tw)


def _dequant(w, s, block=128):
    N, Kd = w.shape
    sf = s.repeat_interleave(block, dim=0)[:N].repeat_interleave(block, dim=1)[:, :Kd]
    return w.float() * sf


def _reference(b):
    x = b["x"].float()
    ids = b["ids"].long()
    tw = b["tw"].float()
    T = x.shape[0]
    i_sz = b["w2"].shape[2]
    out = torch.zeros(T, x.shape[1], dtype=torch.float32, device=x.device)
    for e_ in torch.unique(ids).tolist():
        rows, kth = torch.where(ids == e_)
        w13e = _dequant(b["w13"][e_], b["s13"][e_])
        w2e = _dequant(b["w2"][e_], b["s2"][e_])
        xe = x[rows]
        up = xe @ w13e[:i_sz].t()
        gate = xe @ w13e[i_sz:].t()
        hmid = gate * torch.sigmoid(gate) * up
        out.index_add_(0, rows, (hmid @ w2e.t()) * tw[rows, kth, None])
    return out


def _gate(y, ref):
    yf = y.float()
    ts = float((yf.sum(0) - ref.sum(0)).abs().max() / (ref.sum(0).abs().max() + 1e-9))
    l2 = float(((yf - ref).norm(dim=1) / (ref.norm(dim=1) + 1e-9)).max())
    return ts, l2


def _call(b, **kw):
    return cutlass_fused_moe(
        input=b["x"],
        token_selected_experts=b["ids"],
        token_final_scales=b["tw"],
        fc1_expert_weights=b["w13"],
        fc2_expert_weights=b["w2"],
        output_dtype=torch.bfloat16,
        quant_scales=[b["s13"], b["s2"]],
        use_deepseek_fp8_block_scale=True,
        **kw,
    )


# --------------------------------------------------------------------------
# Allowlist guard (GPU-free)
# --------------------------------------------------------------------------


def _elig_args(device="cpu", T=4, **over):
    b = {}
    b["x"] = torch.zeros((T, H), dtype=torch.bfloat16)
    b["w13"] = torch.zeros((E, 2 * I, H), dtype=torch.float8_e4m3fn)
    b["w2"] = torch.zeros((E, H, I), dtype=torch.float8_e4m3fn)
    b["s13"] = torch.zeros((E, (2 * I + 127) // 128, (H + 127) // 128))
    b["s2"] = torch.zeros((E, (H + 127) // 128, (I + 127) // 128))
    b["ids"] = torch.zeros((T, K), dtype=torch.int32)
    b["tw"] = torch.zeros((T, K), dtype=torch.float32)
    args = dict(
        input=b["x"],
        token_selected_experts=b["ids"],
        token_final_scales=b["tw"],
        fc1_expert_weights=b["w13"],
        fc2_expert_weights=b["w2"],
        output_dtype=torch.bfloat16,
        quant_scales=[b["s13"], b["s2"]],
        output=None,
        flags={},
    )
    args.update(over)
    return args


def test_allowlist_rejects_large_batches_and_flags():
    # CPU tensors: everything up to the device check must already reject
    a = _elig_args(T=64)
    a["input"] = torch.zeros((64, H), dtype=torch.bfloat16)
    a["token_selected_experts"] = torch.zeros((64, K), dtype=torch.int32)
    a["token_final_scales"] = torch.zeros((64, K), dtype=torch.float32)
    assert "num_tokens=64" in _eligibility(**a)
    assert "min_latency_mode" == _eligibility(
        **_elig_args(flags={"min_latency_mode": True})
    )
    bad = _elig_args()
    bad["output_dtype"] = torch.float16
    assert "output_dtype" in _eligibility(**bad)
    bad = _elig_args()
    bad["quant_scales"] = [bad["quant_scales"][0]]
    assert "quant_scales" in _eligibility(**bad)
    # CPU device is the final rejection for an otherwise-conforming call
    assert _eligibility(**_elig_args()) == "device cpu"


def test_stats_report_build_state_and_counters():
    stats = _moe_smallt_sm90_stats()
    for key in ("calls_seen", "runs_dispatched", "module_loaded", "module_error"):
        assert key in stats


# --------------------------------------------------------------------------
# Device tests (SM90): reference equivalence, kill-switch, CUDA graph
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 4, 16])
def test_specialized_matches_reference_and_stock(T, monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make(T, device)
    ref = _reference(b)

    out = _call(b)
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    ts, l2 = _gate(out, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)

    # determinism: the kernel is atomics-free and bit-stable
    out2 = _call(b)
    assert torch.equal(out, out2)

    # kill-switch: the same call is served by the stock cutlass path
    monkeypatch.setenv(DISABLE_ENV, "1")
    out_stock = _call(b)
    stats = _moe_smallt_sm90_stats()
    assert stats["calls_disabled"] == 1, stats
    assert stats["runs_dispatched"] == 2  # unchanged by the disabled call
    ts, l2 = _gate(out_stock, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


def test_above_envelope_takes_the_stock_path():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make(64, device)
    ref = _reference(b)
    out = _call(b)
    stats = _moe_smallt_sm90_stats()
    # the core.py pre-filter (T <= 16) keeps the hook out of the call
    assert stats["runs_dispatched"] == 0, stats
    ts, l2 = _gate(out, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


def test_graph_capture_replays_bitwise_equal_to_eager():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make(4, device)
    out = torch.empty((4, H), dtype=torch.bfloat16, device=device)
    eager = _call(b).clone()
    assert _moe_smallt_sm90_stats()["runs_dispatched"] == 1

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(b, output=out)  # warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        _call(b, output=out)
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched_capturing"] == 1, stats

    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)

    # new activations through the same buffers replay correctly
    b["x"].copy_(_make(4, device, seed=7)["x"])
    graph.replay()
    torch.cuda.synchronize()
    eager2 = _call(b)
    assert torch.equal(out, eager2)
