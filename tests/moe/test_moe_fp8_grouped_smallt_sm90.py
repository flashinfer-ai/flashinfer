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
    CUTLASS_FUSED_MOE_BACKENDS,
    _dispatch_table,
    _eligibility,
    _moe_smallt_sm90_stats,
    _reset_moe_smallt_sm90_stats,
)

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISABLE_ENV_LEGACY = "FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90"
E, K, H, I = 16, 9, 512, 256  # small test geometry (contract class, not size)
# W8A8 fused-MoE kernels quantize activations internally, so elementwise
# tolerance vs an fp32-dequant reference is noise-dominated; the class gate
# is token-sum relative error + max per-token L2 relative error (the same
# two criteria the measured verification uses).
TOKEN_SUM_REL, TOKEN_L2_REL = 0.15, 0.10


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV_LEGACY, raising=False)
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
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32)).to(
            torch.float8_e4m3fn
        )

    pow2 = torch.tensor([0.0625, 0.125, 0.25], device=device)
    s1 = (
        pow2[
            torch.randint(
                0,
                3,
                (e, (2 * i + 127) // 128, (h + 127) // 128),
                generator=g,
                device=device,
            )
        ]
        .float()
        .contiguous()
    )
    s2 = (
        pow2[
            torch.randint(
                0,
                3,
                (e, (h + 127) // 128, (i + 127) // 128),
                generator=g,
                device=device,
            )
        ]
        .float()
        .contiguous()
    )
    # Distinct expert ids per token (as real top-k routing produces): the
    # stock route's semantics for duplicate ids differ from an additive sum,
    # so duplicates would compare arms on an off-contract input.
    ids = torch.stack(
        [torch.randperm(e, generator=g, device=device)[:k] for _ in range(T)]
    ).to(torch.int32)
    tw = torch.rand((T, k), generator=g, device=device, dtype=torch.float32)
    tw = tw / tw.sum(-1, keepdim=True)
    x = torch.randn((T, h), generator=g, device=device, dtype=torch.float32).to(
        torch.bfloat16
    )
    return dict(
        x=x, w13=f8((e, 2 * i, h)), s13=s1, w2=f8((e, h, i)), s2=s2, ids=ids, tw=tw
    )


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
    r = cutlass_fused_moe(
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
    # The stock cutlass route returns [output] (list); the specialized route
    # returns the output tensor. Unwrap so arms compare uniformly.
    if isinstance(r, (tuple, list)):
        r = r[0]
    return r


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
    # the 17..255 gap (containing the measured FI-held T=64 class) rejects
    a = _elig_args(T=64)
    a["input"] = torch.zeros((64, H), dtype=torch.bfloat16)
    a["token_selected_experts"] = torch.zeros((64, K), dtype=torch.int32)
    a["token_final_scales"] = torch.zeros((64, K), dtype=torch.float32)
    assert "num_tokens=64" in _eligibility(**a)
    # the expanded band [256, 4096] admits, interior points by interpolation
    for t_in in (256, 512, 2048, 4096):
        b = _elig_args(T=t_in)
        b["input"] = torch.zeros((t_in, H), dtype=torch.bfloat16)
        b["token_selected_experts"] = torch.zeros((t_in, K), dtype=torch.int32)
        b["token_final_scales"] = torch.zeros((t_in, K), dtype=torch.float32)
        assert _eligibility(**b) == "device cpu"  # passes every band guard
    c = _elig_args(T=5000)
    c["input"] = torch.zeros((5000, H), dtype=torch.bfloat16)
    c["token_selected_experts"] = torch.zeros((5000, K), dtype=torch.int32)
    c["token_final_scales"] = torch.zeros((5000, K), dtype=torch.float32)
    assert "num_tokens=5000" in _eligibility(**c)
    assert (
        _eligibility(**_elig_args(flags={"min_latency_mode": True}))
        == "min_latency_mode"
    )
    bad = _elig_args()
    bad["output_dtype"] = torch.float16
    assert "output_dtype" in _eligibility(**bad)
    bad = _elig_args()
    bad["quant_scales"] = [bad["quant_scales"][0]]
    assert "quant_scales" in _eligibility(**bad)
    # CPU device is the final rejection for an otherwise-conforming call
    assert _eligibility(**_elig_args()) == "device cpu"


def test_allowlist_rejects_off_contract_kernel_geometry():
    # T == 1 runs the 512-half-swizzled fc1 staging and the fixed
    # 256-column / 9-slot fc2 pair; the grouped band runs the 288-thread
    # prefix scan, 256-column fp16 fc2 tiles, the 2x128-block fp8 fc2 and
    # the 9-slot reductions. Off-contract geometry must be rejected by the
    # Python guard (GPU-free: all of these reject before the device check);
    # the native validation in the .cu mirrors every one of them.
    def one(T, e=E, k=K, h=H, i=I):
        return _eligibility(
            input=torch.zeros((T, h), dtype=torch.bfloat16),
            token_selected_experts=torch.zeros((T, k), dtype=torch.int32),
            token_final_scales=torch.zeros((T, k), dtype=torch.float32),
            fc1_expert_weights=torch.zeros((e, 2 * i, h), dtype=torch.float8_e4m3fn),
            fc2_expert_weights=torch.zeros((e, h, i), dtype=torch.float8_e4m3fn),
            output_dtype=torch.bfloat16,
            quant_scales=[
                torch.zeros((e, (2 * i + 127) // 128, (h + 127) // 128)),
                torch.zeros((e, (h + 127) // 128, (i + 127) // 128)),
            ],
            output=None,
            flags={},
        )

    assert "512-aligned" in one(1, h=384)
    assert "I=512 != 256" in one(1, i=512)
    assert "top_k=8" in one(1, k=8)
    assert one(1) == "device cpu"  # the contract geometry passes every guard
    assert "top_k=8" in one(256, k=8)
    assert "num_experts=512" in one(256, e=512)
    assert "256-aligned" in one(256, h=384)
    assert "I=512 != 256" in one(256, i=512)
    assert one(256) == "device cpu"
    # T in [2, 16] keeps the generic small-path envelope
    assert one(4, k=8, e=512, h=384) == "device cpu"


def test_stats_report_build_state_and_counters():
    stats = _moe_smallt_sm90_stats()
    for key in (
        "calls_seen",
        "runs_dispatched",
        "module_loaded",
        "module_error",
        "backends",
        "backend_dispatches",
        "fallback_reasons",
        "dispatch_table_loaded",
    ):
        assert key in stats
    assert stats["backends"] == CUTLASS_FUSED_MOE_BACKENDS
    assert set(stats["backend_dispatches"]) == {"specialized", "stock"}


def test_backends_tuple_and_dispatch_table():
    assert CUTLASS_FUSED_MOE_BACKENDS == ("auto", "specialized", "cutlass", "cake")
    table = _dispatch_table()
    assert table is not None
    assert table["kill_switch"] == DISABLE_ENV
    op = table["ops"]["cutlass_fused_moe_dsfp8"]
    assert op["token_bands"] == [[1, 16], [256, 4096]]
    assert op["compute_capability"] == [9, 0]


def test_unknown_backend_raises_gpu_free():
    x = torch.zeros((1, H), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="never silently swapped"):
        cutlass_fused_moe(
            input=x,
            token_selected_experts=torch.zeros((1, K), dtype=torch.int32),
            token_final_scales=torch.zeros((1, K)),
            fc1_expert_weights=torch.zeros((E, 2 * I, H), dtype=torch.float8_e4m3fn),
            fc2_expert_weights=torch.zeros((E, H, I), dtype=torch.float8_e4m3fn),
            output_dtype=torch.bfloat16,
            quant_scales=[],
            use_deepseek_fp8_block_scale=True,
            backend="fa2",
        )


# --------------------------------------------------------------------------
# Device tests (SM90): reference equivalence, kill-switch, CUDA graph
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 4, 16, 256, 1024, 2048])
def test_specialized_matches_reference_and_stock(T, monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make(T, device)
    ref = _reference(b)

    out = _call(b)  # backend defaults to "auto" -> allowlisted -> dispatched
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    ts, l2 = _gate(out, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)

    # determinism: the kernel is atomics-free and bit-stable
    out2 = _call(b)
    assert torch.equal(out, out2)

    # NEVER-SWAPPED: explicit backend="cutlass" must not reach the kernel
    out_pin = _call(b, backend="cutlass")
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 2, stats  # unchanged by the pinned call
    ts, l2 = _gate(out_pin, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)

    # explicit backend="specialized" dispatches
    out_spec = _call(b, backend="specialized")
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 3, stats
    assert torch.equal(out_spec, out)

    # kill switches (new + deprecated alias): the stock cutlass path serves
    for env, nexpected in ((DISABLE_ENV, 1), (DISABLE_ENV_LEGACY, 2)):
        monkeypatch.setenv(env, "1")
        out_stock = _call(b)
        monkeypatch.delenv(env)
        stats = _moe_smallt_sm90_stats()
        assert stats["calls_disabled"] == nexpected, stats
        assert stats["runs_dispatched"] == 3  # unchanged by disabled calls
        ts, l2 = _gate(out_stock, ref)
        assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


def test_above_envelope_takes_the_stock_path():
    # T=64 sits in the excluded 17..255 gap: the measured class where the
    # stock cutlass kernel held at contract
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


@pytest.mark.parametrize("T", [1, 4])
def test_fc1_half_chunk_headroom_with_activation_outliers(T):
    # Regression (CodeRabbit): fc1 accumulates each 16-element chunk in
    # __half against RAW e4m3 mantissas, which the [128,128] block
    # quantizer drives to ~448 at every block's peak. Without the 1/16
    # activation prescale, an outlier channel of ~25 against such a peak
    # overflows the half chunk partial to inf; magnitudes like these occur
    # in real post-norm hidden states (outlier channels).
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make(T, device)
    b["s13"].fill_(0.0625)
    b["s2"].fill_(0.0625)
    b["w13"][:, :I, :16] = 448.0  # quantization peaks in the up branch
    b["x"][:, :16] = 25.0  # outlier activation channels
    ref = _reference(b)
    out = _call(b, backend="specialized")
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    assert torch.isfinite(out.float()).all()
    ts, l2 = _gate(out, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


def test_nvfp4_4over6_pinned_recipe_declines_specialized():
    # An eligible dsfp8 call with an explicitly pinned nvfp4_4over6 recipe
    # must raise the stock route's ValueError, not be silently served by
    # the specialized kernel (which would ignore the recipe). An explicit
    # None (recipe off) stays eligible.
    _sm90_or_skip()
    from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

    device = torch.device("cuda:0")
    b = _make(4, device)
    with pytest.raises(ValueError, match="nvfp4_4over6 requires"):
        _call(b, nvfp4_4over6=NVFP44Over6Config())
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    assert any("nvfp4_4over6" in r for r in stats["fallback_reasons"]), stats
    out = _call(b, nvfp4_4over6=None)
    assert _moe_smallt_sm90_stats()["runs_dispatched"] == 1
    ts, l2 = _gate(out, _reference(b))
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


def test_workspace_is_per_stream_and_replaced_buffers_are_retained():
    # Regression (CodeRabbit): the native run launches on the caller's
    # current stream, so two streams must not share one cached workspace;
    # and a workspace replaced by a larger one must be retained, since a
    # CUDA graph captured earlier may still replay into its address.
    _sm90_or_skip()
    import flashinfer.fused_moe.smallt_sm90 as smallt

    device = torch.device("cuda:0")
    # start from a clean cache without freeing buffers a captured graph
    # from an earlier test may still replay into
    smallt._retired_workspaces.extend(smallt._workspaces.values())
    smallt._workspaces.clear()
    _reset_moe_smallt_sm90_stats()

    b = _make(4, device)
    ref = _reference(b)
    out_default = _call(b)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out_side = _call(b)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 2, stats
    assert stats["workspace_allocs"] == 2, stats  # one per (device, stream)
    assert len(smallt._workspaces) == 2
    for y in (out_default, out_side):
        ts, l2 = _gate(y, ref)
        assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)

    # growing the default stream's workspace (band-2 capacity) retires the
    # old buffer instead of dropping the only reference to it
    retired_before = len(smallt._retired_workspaces)
    big = _make(256, device)
    out_big = _call(big)
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 3, stats
    assert stats["workspace_allocs"] == 3, stats
    assert len(smallt._retired_workspaces) == retired_before + 1
    ts, l2 = _gate(out_big, _reference(big))
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="needs an sm90 device",
)
def test_return_type_parity_across_backends(monkeypatch):
    """The backend taken must not be observable in the return type:
    cutlass_fused_moe returns [output] on the stock route, on the
    specialized route, under auto, and with the kill switch forcing
    stock."""
    device = torch.device("cuda")
    b = _make(4, device)

    def raw(**kw):
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

    results = {
        "auto": raw(),
        "specialized": raw(backend="specialized"),
        "cutlass": raw(backend="cutlass"),
    }
    monkeypatch.setenv(DISABLE_ENV, "1")
    results["auto+kill"] = raw()
    monkeypatch.delenv(DISABLE_ENV)
    # The stock route's own length varies by path (it may carry aux tensors
    # after the output), so the stable cross-backend contract is: a list
    # whose FIRST element is the output tensor.
    for name, r in results.items():
        assert isinstance(r, list) and len(r) >= 1, (name, type(r))
        assert isinstance(r[0], torch.Tensor) and r[0].dtype == torch.bfloat16, name
        assert r[0].shape == b["x"].shape, name
