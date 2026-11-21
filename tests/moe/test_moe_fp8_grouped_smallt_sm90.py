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

from flashinfer.fused_moe import (
    BackendOptions,
    CutlassFp8BlockConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SpecializedFp8BlockSmallTConfig,
    SpecializedFp8BlockSmallTRunner,
    SwiGLU,
    cutlass_fused_moe,
)
from flashinfer.fused_moe.smallt_sm90 import (
    _eligibility,
    _moe_smallt_sm90_dispatch,
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
    monkeypatch.delenv("FLASHINFER_MOE_SMALLT_SHARE_GRAPH_WORKSPACE", raising=False)
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


def _stock_call(b, **kw):
    """The stock flat entry (negative-control arm): always the CUTLASS route."""
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
    if isinstance(r, (tuple, list)):
        r = r[0]
    return r


def _dispatch(b, output=None):
    """Call the specialized kernel's own module directly (what the unified
    SpecializedFp8BlockSmallTRunner does); None means the call was declined."""
    return _moe_smallt_sm90_dispatch(
        b["x"],
        b["ids"],
        b["tw"],
        b["w13"],
        b["w2"],
        torch.bfloat16,
        [b["s13"], b["s2"]],
        output,
        flags={},
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
    # ... bounded by the small-path launches' dynamic shared memory, which
    # is data-dependent and launched without the >48KiB opt-in
    assert "fc2 smem" in one(4, i=4096)  # top_k * I halves
    assert "fc1 smem" in one(4, h=25600)  # H halves + the fc1 scale grid
    assert "fc1 staging smem" in one(1, h=25088, i=256)  # T=1: H halves


def test_allowlist_rejects_misaligned_views():
    # Contiguity does not imply address alignment: a contiguous view with a
    # storage offset can be only element-aligned, while the kernels read
    # activations and weights through 16-byte (uint4) vectors (GPU-free:
    # the rejection precedes the device check; mirrored natively).
    T = 4
    a = _elig_args()
    base = torch.zeros((T * H + 8,), dtype=torch.bfloat16)
    a["input"] = base[1 : 1 + T * H].reshape(T, H)  # 2-byte-aligned view
    assert a["input"].is_contiguous()
    assert "input not 16-byte aligned" in _eligibility(**a)
    a = _elig_args()
    wbase = torch.zeros((E * 2 * I * H + 16,), dtype=torch.float8_e4m3fn)
    a["fc1_expert_weights"] = wbase[1 : 1 + E * 2 * I * H].reshape(E, 2 * I, H)
    assert "fc1 not 16-byte aligned" in _eligibility(**a)
    a = _elig_args()
    obase = torch.zeros((T * H + 8,), dtype=torch.bfloat16)
    a["output"] = obase[1 : 1 + T * H].reshape(T, H)
    assert "output not 16-byte aligned" in _eligibility(**a)
    # a 16-byte storage offset stays aligned and passes through to the
    # device check
    a = _elig_args()
    base = torch.zeros((T * H + 8,), dtype=torch.bfloat16)
    a["input"] = base[8 : 8 + T * H].reshape(T, H)
    assert _eligibility(**a) == "device cpu"


def test_stats_report_build_state_and_counters():
    stats = _moe_smallt_sm90_stats()
    for key in (
        "calls_seen",
        "runs_dispatched",
        "module_loaded",
        "module_error",
        "backend_dispatches",
        "fallback_reasons",
    ):
        assert key in stats
    assert set(stats["backend_dispatches"]) == {"specialized", "declined"}


def test_flat_entry_is_untouched_negative_control():
    """The specialized kernel is NOT a cutlass_fused_moe backend: the flat
    vendor entry keeps its stock ("cutlass", "cake") surface and default,
    and rejects "specialized"/"auto" — the kernel's only public home is the
    unified MoELayer surface (GPU-free: the rejection precedes device
    work)."""
    import inspect

    assert (
        inspect.signature(cutlass_fused_moe).parameters["backend"].default == "cutlass"
    )
    for name in ("specialized", "auto"):
        with pytest.raises(ValueError, match="unsupported fused MoE backend"):
            cutlass_fused_moe(
                input=torch.zeros((1, H), dtype=torch.bfloat16),
                token_selected_experts=torch.zeros((1, K), dtype=torch.int32),
                token_final_scales=torch.zeros((1, K)),
                fc1_expert_weights=torch.zeros(
                    (E, 2 * I, H), dtype=torch.float8_e4m3fn
                ),
                fc2_expert_weights=torch.zeros((E, H, I), dtype=torch.float8_e4m3fn),
                output_dtype=torch.bfloat16,
                quant_scales=[],
                use_deepseek_fp8_block_scale=True,
                backend=name,
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

    out = _dispatch(b)  # the kernel's own dispatch, in-band -> served
    assert out is not None
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    ts, l2 = _gate(out, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)

    # determinism: the kernel is atomics-free and bit-stable
    out2 = _dispatch(b)
    assert torch.equal(out, out2)

    # NEGATIVE CONTROL: the flat entry is the stock kernel, always — the
    # specialized kernel is not reachable through it.
    out_stock = _stock_call(b)
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 2, stats  # unchanged by the flat call
    ts, l2 = _gate(out_stock, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)

    # kill switches (new + deprecated alias): the dispatch declines
    for env, nexpected in ((DISABLE_ENV, 1), (DISABLE_ENV_LEGACY, 2)):
        monkeypatch.setenv(env, "1")
        assert _dispatch(b) is None
        monkeypatch.delenv(env)
        stats = _moe_smallt_sm90_stats()
        assert stats["calls_disabled"] == nexpected, stats
        assert stats["runs_dispatched"] == 2  # unchanged by disabled calls
        assert stats["backend_dispatches"]["declined"] >= nexpected


def test_above_envelope_is_declined():
    # T=64 sits in the excluded 17..255 gap: the measured class where the
    # stock cutlass kernel held at contract. The dispatch declines; the
    # unified layer serves such calls with another configured backend.
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make(64, device)
    assert _dispatch(b) is None
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    assert any("outside the measured won bands" in r for r in stats["fallback_reasons"])


def test_graph_capture_replays_bitwise_equal_to_eager():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make(4, device)
    out = torch.empty((4, H), dtype=torch.bfloat16, device=device)
    eager = _dispatch(b).clone()
    assert _moe_smallt_sm90_stats()["runs_dispatched"] == 1

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert _dispatch(b, output=out) is not None  # warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        assert _dispatch(b, output=out) is not None
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
    eager2 = _dispatch(b)
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
    out = _dispatch(b)
    assert out is not None
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    assert torch.isfinite(out.float()).all()
    ts, l2 = _gate(out, ref)
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


@pytest.mark.parametrize("T", [1, 4, 16, 256, 2048])
@pytest.mark.parametrize(
    "w2_value,expected",
    [(2.0**-12, 1024.0), (2.0**-18, 16.0), (0.125, 524288.0)],
)
def test_block_peak_uniform_weights_match_analytic(T, w2_value, expected):
    # Regression (Codex, H100-reproduced): uniform BF16 weights through the
    # PUBLIC preparer put every e4m3 mantissa at the block peak (448) with an
    # ordinary fp32 block scale, and three fp16-range defects then produced
    # Inf/0 where the analytic (and stock-CUTLASS) result is exact and
    # finite: (1) the small-T fc2 inner product accumulated raw-mantissa
    # products in half and overflowed before the fp32 block scale was
    # applied; (2) the grouped dequant narrowed the fp32 scale to half
    # before the multiply, flushing small scales (e.g. 2**-18/448) to zero;
    # (3) the grouped per-slot partial buffer was half, overflowing at
    # 65504 while the BF16 output range is far larger.
    # With x = 1, W13 = 0.25 (H=512): up = gate = 128, h = silu(128)*128 =
    # 16384, out = 16384 * w2 * I. All three expectations are exact BF16.
    _sm90_or_skip()
    from flashinfer.fused_moe.prepare import prepare_cutlass_fp8_block_weights

    device = torch.device("cuda:0")
    w13 = torch.full((E, 2 * I, H), 0.25, dtype=torch.bfloat16, device=device)
    w2 = torch.full((E, H, I), w2_value, dtype=torch.bfloat16, device=device)
    view = prepare_cutlass_fp8_block_weights(
        w13, w2, num_local_experts=E, hidden_size=H, intermediate_size=I
    )
    g = torch.Generator(device=device)
    g.manual_seed(0)
    ids = torch.stack(
        [torch.randperm(E, generator=g, device=device)[:K] for _ in range(T)]
    ).to(torch.int32)
    b = dict(
        x=torch.ones((T, H), dtype=torch.bfloat16, device=device),
        w13=view["fc1_expert_weights"],
        s13=view["fc1_block_scale"],
        w2=view["fc2_expert_weights"],
        s2=view["fc2_block_scale"],
        ids=ids,
        tw=torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device),
    )
    out = _dispatch(b)
    assert out is not None
    assert torch.isfinite(out.float()).all(), (
        float(out.float().min()),
        float(out.float().max()),
    )
    exp = torch.full((T, H), expected, dtype=torch.bfloat16, device=device)
    assert torch.equal(out, exp), (
        float(out.float().min()),
        float(out.float().max()),
        expected,
    )


@pytest.mark.parametrize("T", [1, 4])
def test_fc1_overflow_before_block_scale(T):
    _sm90_or_skip()
    # Six 512-column steps plus a tail in fc1_t1_u4. Put the up peak
    # in the unrolled loop and the gate peak in the tail, then swap.
    b = _make(T, torch.device("cuda"), h=3584, e=K)
    b["x"].zero_()
    b["x"][:, :16] = 300
    b["x"][:, -16:] = 300
    b["s13"].fill_(2.0**-20)
    b["s2"].fill_(1.0 / I)
    b["w2"].fill_(1)
    for up, gate in (
        (slice(0, 16), slice(-16, None)),
        (slice(-16, None), slice(0, 16)),
    ):
        b["w13"].zero_()
        b["w13"][:, :I, up] = 448
        b["w13"][:, I:, gate] = 448
        out = _dispatch(b)
        assert out is not None
        ref = _reference(b)
        assert (ref > 1).all() and torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), ref, atol=0, rtol=0.01)


@pytest.mark.parametrize("first_scale", [2.0**-19, 2.0**110])
def test_native_fp8_fc2_independent_block_scales(first_scale):
    _sm90_or_skip()
    T, e, h = 2048, K, 256
    device = torch.device("cuda")
    w13 = torch.full((e, 2 * I, h), 0.5, device=device, dtype=torch.bfloat16)
    # First 128 up rows cancel on x=1, without any zero-scale FC1 block.
    w13[:, :128, h // 2 :] = -0.5
    second_scale = 2.0**-20
    w2 = torch.empty((e, h, I), device=device, dtype=torch.bfloat16)
    w2[:, :, :128] = 448 * first_scale
    w2[:, :, 128:] = 448 * second_scale
    view = SpecializedFp8BlockSmallTConfig.prepare_weights(
        w13, w2, num_local_experts=e, hidden_size=h, intermediate_size=I
    )
    scales = view["fc2_block_scale"]
    assert (scales[..., 0] == first_scale).all()
    assert (scales[..., 1] == second_scale).all()
    b = dict(
        x=torch.ones((T, h), device=device, dtype=torch.bfloat16),
        w13=view["fc1_expert_weights"],
        s13=view["fc1_block_scale"],
        w2=view["fc2_expert_weights"],
        s2=scales,
        ids=torch.arange(K, device=device, dtype=torch.int32).repeat(T, 1),
        tw=torch.full((T, K), 1 / K, device=device, dtype=torch.float32),
    )
    out = _dispatch(b)
    assert out is not None
    # Identical experts: first dot is zero, second has 128 terms with
    # up=gate=128 (SiLU(128) rounds to 128) and physical weight w2.
    expected = 128 * (128 * 128) * (448 * second_scale)
    assert torch.equal(out, torch.full_like(out, expected))


def test_workspace_planner_bounds_and_layout():
    _sm90_or_skip()
    from flashinfer.jit.moe_fp8_grouped_smallt_sm90 import (
        gen_moe_fp8_grouped_smallt_sm90_module,
    )

    plan = (
        gen_moe_fp8_grouped_smallt_sm90_module().build_and_load().plan_workspace_bytes
    )
    assert plan(4, 9, 256, 512, 16) == 18432
    t, k, i, h, e = 257, 9, 256, 512, 16
    p = t * k
    segments = (
        4 * p * i,
        2 * p * i,
        2 * p * h,
        4 * (4 * e + 1 + p),
        8 * (e + (p + 63) // 64),
    )
    assert plan(t, k, i, h, e) == sum(16 * ((n + 15) // 16) for n in segments)
    for dims in (
        (4096, 9, 256, 1 << 51, 16),
        (4, 9, 1 << 60, 512, 16),
        (16, 1 << 30, 256, 512, 16),
        ((1 << 31) - 256, 1, 1 << 29, (1 << 31) - 128, 16),
    ):
        with pytest.raises(RuntimeError):
            plan(*dims)


def _trace_capture_scratch(monkeypatch):
    """Observe addresses, never retain the scratch tensor or its storage."""
    import flashinfer.fused_moe.smallt_sm90 as smallt

    native = smallt._get_module()
    addresses = []

    class Traced:
        plan_workspace_bytes = staticmethod(native.plan_workspace_bytes)

        @staticmethod
        def run(*args):
            if torch.cuda.is_current_stream_capturing():
                addresses.append(args[-1].data_ptr())
            return native.run(*args)

    monkeypatch.setattr(smallt, "_module", Traced())
    return addresses


@pytest.mark.parametrize("legacy_share", [False, True])
def test_independent_graph_captures_get_distinct_scratch(monkeypatch, legacy_share):
    # Regression (Codex, H100-reproduced): torch.cuda.graph reuses one
    # default capture stream for every capture in the process, so keying
    # graph scratch by (device, capture stream) handed two INDEPENDENT
    # graphs the same buffer.
    _sm90_or_skip()
    if legacy_share:
        monkeypatch.setenv("FLASHINFER_MOE_SMALLT_SHARE_GRAPH_WORKSPACE", "1")
    addresses = _trace_capture_scratch(monkeypatch)

    device = torch.device("cuda:0")
    T = 16
    b1, b2 = _make(T, device, seed=11), _make(T, device, seed=22)
    ref1 = _dispatch(b1).clone()
    ref2 = _dispatch(b2).clone()
    out1 = torch.empty((T, H), dtype=torch.bfloat16, device=device)
    out2 = torch.empty_like(out1)

    def capture(b, out):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            assert _dispatch(b, output=out) is not None  # primes a workspace
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):  # the shared default capture stream
            assert _dispatch(b, output=out) is not None
        return g

    g1 = capture(b1, out1)
    g2 = capture(b2, out2)
    assert len(addresses) == 2
    assert addresses[0] != addresses[1]

    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(100):
        out1.zero_()
        out2.zero_()
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            g1.replay()
        with torch.cuda.stream(s2):
            g2.replay()
        torch.cuda.synchronize()
        assert torch.equal(out1, ref1)
        assert torch.equal(out2, ref2)


def test_two_moe_layers_concurrent_graph_replay():
    # The public form of the same regression: two independent MoELayer
    # instances, distinct weights/activations, no shared graph pool, each
    # captured via torch.cuda.graph (shared default capture stream) and
    # replayed concurrently, must match their eager outputs bitwise.
    _sm90_or_skip()
    layer1, act1, weights1, _b1 = _unified_layer(16, None, seed=31)
    layer2, act2, weights2, _b2 = _unified_layer(16, None, seed=32)
    ref1 = layer1(act1, weights1).clone()
    ref2 = layer2(act2, weights2).clone()
    # torch.cuda.graph's priming pattern: eager warmup on a side stream
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        layer1(act1, weights1)
        layer2(act2, weights2)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        out1 = layer1(act1, weights1)
    with torch.cuda.graph(g2):
        out2 = layer2(act2, weights2)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(100):
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            g1.replay()
        with torch.cuda.stream(s2):
            g2.replay()
        torch.cuda.synchronize()
        assert torch.equal(out1, ref1)
        assert torch.equal(out2, ref2)


@pytest.mark.parametrize("T", [1, 4, 16, 256])
@pytest.mark.parametrize("w13_value", [2.0**-14, 2.0**-15, 2.0**-16])
def test_block_peak_small_products_survive(T, w13_value):
    # Regression (Codex, round 2): a FIXED 2^-14 weight prescale made the
    # half FC2 chain overflow-safe but flushed small products to zero
    # before the fp32 unscale: with W13=2^-15 and W2=64 through the public
    # preparer, T<=16 returned 0 where the analytic result is ~2.0156 (the
    # pre-fix kernel returned 2.0, within 0.8%), and W13=2^-14 was off by
    # 12.3%. The chain now runs the original half arithmetic (bit-for-bit
    # the pre-existing lower-range behavior) and falls back to fp32
    # products only for chunks that actually overflow, so BOTH ends of the
    # admitted range survive.
    _sm90_or_skip()
    from flashinfer.fused_moe.prepare import prepare_cutlass_fp8_block_weights

    device = torch.device("cuda:0")
    w13 = torch.full((E, 2 * I, H), w13_value, dtype=torch.bfloat16, device=device)
    w2 = torch.full((E, H, I), 64.0, dtype=torch.bfloat16, device=device)
    view = prepare_cutlass_fp8_block_weights(
        w13, w2, num_local_experts=E, hidden_size=H, intermediate_size=I
    )
    g = torch.Generator(device=device)
    g.manual_seed(0)
    ids = torch.stack(
        [torch.randperm(E, generator=g, device=device)[:K] for _ in range(T)]
    ).to(torch.int32)
    b = dict(
        x=torch.ones((T, H), dtype=torch.bfloat16, device=device),
        w13=view["fc1_expert_weights"],
        s13=view["fc1_block_scale"],
        w2=view["fc2_expert_weights"],
        s2=view["fc2_block_scale"],
        ids=ids,
        tw=torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device),
    )
    ref = _reference(b)
    out = _dispatch(b)
    assert out is not None
    rel = float(((out.float() - ref).abs() / ref.abs().clamp_min(1e-30)).max())
    # 2% envelope: the fp16 intermediate is half-subnormal at these
    # magnitudes, so ~0.8% is inherent (and was the pre-existing kernel's
    # behavior); zero outputs or the 12.3% prescale warp must fail.
    assert rel <= 0.02, (rel, float(out.float().min()), float(ref.min()))


@pytest.mark.parametrize("legacy_share", [False, True])
def test_fork_join_branches_within_one_capture_get_distinct_scratch(
    monkeypatch, legacy_share
):
    # Regression (Codex, round 2): keying graph scratch by capture id ALONE
    # aliased parallel fork/join branches recorded on different streams
    # inside one capture — those branches run concurrently at replay (only
    # same-stream work within a capture is ordered), so two MoELayer calls
    # in forked branches corrupted each other 100/100. Scratch is owned per
    # (capture, capture-time stream) now.
    _sm90_or_skip()
    if legacy_share:
        monkeypatch.setenv("FLASHINFER_MOE_SMALLT_SHARE_GRAPH_WORKSPACE", "1")
    addresses = _trace_capture_scratch(monkeypatch)

    layer1, act1, weights1, _b1 = _unified_layer(16, None, seed=11)
    layer2, act2, weights2, _b2 = _unified_layer(16, None, seed=83)
    ref1 = layer1(act1, weights1).clone()
    ref2 = layer2(act2, weights2).clone()
    # prime one eager workspace per branch stream (the warmup pattern)
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    sa.wait_stream(torch.cuda.current_stream())
    sb.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sa):
        layer1(act1, weights1)
    with torch.cuda.stream(sb):
        layer2(act2, weights2)
    torch.cuda.current_stream().wait_stream(sa)
    torch.cuda.current_stream().wait_stream(sb)
    torch.cuda.synchronize()

    addresses.clear()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=sa):
        fork = torch.cuda.Event()
        fork.record()
        out1 = layer1(act1, weights1)  # branch 0 on the capture stream
        with torch.cuda.stream(sb):
            sb.wait_event(fork)
            out2 = layer2(act2, weights2)  # branch 1, forked stream
        torch.cuda.current_stream().wait_stream(sb)  # join
    assert len(addresses) == 2
    assert addresses[0] != addresses[1]

    for _ in range(100):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out1, ref1)
        assert torch.equal(out2, ref2)


def test_pack_inputs_binds_no_cutlass_workspace():
    # The specialized runner never consumes the CUTLASS workspace, so
    # neither packing path (pack_inputs or the tuning-input hook) may load
    # the stock CUTLASS module or retain its workspace allocation.
    _sm90_or_skip()
    layer, act, weights, _ = _unified_layer(4, None)
    (runner,) = layer.runners
    inputs = runner.pack_inputs(act, weights)
    runner._prepare_tuning_inputs(list(inputs))
    assert runner._workspace is None
    assert not runner._workspace_cache


def test_nvfp4_4over6_pinned_recipe_rejects_at_config_time():
    # A pinned nvfp4_4over6 recipe can never reach the specialized dsfp8
    # backend: the unified QuantConfig rejects it on a non-NVFP4 activation
    # outright, so no silently-dropped-recipe path exists (GPU-free).
    from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

    with pytest.raises(ValueError, match="nvfp4_4over6"):
        QuantConfig(
            weight=QuantFormat.DeepSeekFp8,
            activation=QuantFormat.DeepSeekFp8,
            nvfp4_4over6=NVFP44Over6Config(),
        )


def test_eager_workspace_stream_ordering_and_resize():
    _sm90_or_skip()

    device = torch.device("cuda:0")
    b = _make(4, device)
    big = _make(256, device)
    ref = _reference(b)
    out_default = _dispatch(b)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out_side = _dispatch(b)
    # No wait for the side-stream kernel before requesting grouped scratch.
    out_big = _dispatch(big)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 3, stats
    assert stats["workspace_allocs"] == 3, stats
    for y in (out_default, out_side):
        ts, l2 = _gate(y, ref)
        assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)

    ts, l2 = _gate(out_big, _reference(big))
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL, (ts, l2)


@pytest.mark.parametrize("timed", [False, True], ids=["public-graph", "candidate"])
def test_transient_graph_scratch_plateaus(timed):
    import gc

    from flashinfer.testing.utils import bench_gpu_time

    _sm90_or_skip()
    layer, act, weights, _ = _unified_layer(256, tune_max=256)
    (runner,) = layer.runners
    eager = layer(act, weights).clone()

    def candidate():
        inputs = runner.pack_inputs(act, weights)
        return runner.forward(inputs, tactic=-1, **runner.launch_kwargs_for(inputs))

    def cycle():
        before = _moe_smallt_sm90_stats()["runs_dispatched_capturing"]
        if timed:
            bench_gpu_time(
                candidate, dry_run_iters=5, repeat_iters=30, use_cuda_graph=True
            )
        else:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = layer(act, weights)
            out.zero_()
            g.replay()
            assert torch.equal(out, eager)
            g.reset()
        assert _moe_smallt_sm90_stats()["runs_dispatched_capturing"] > before

    samples = []
    for _ in range(4):
        cycle()
        torch.cuda.synchronize()
        gc.collect()
        samples.append(torch.cuda.memory_allocated(eager.device))
    # Allow incidental small allocations, not another ~90 MiB band buffer.
    assert max(samples[1:]) <= samples[0] + (1 << 20), samples


def test_sequential_grouped_calls_reuse_capture_scratch(monkeypatch):
    _sm90_or_skip()
    b = _make(256, torch.device("cuda"))
    eager = _dispatch(b)
    out = torch.empty_like(eager)
    addresses = _trace_capture_scratch(monkeypatch)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8):
            assert _dispatch(b, output=out) is out
    assert len(addresses) == 8 and len(set(addresses)) == 1
    g.replay()
    assert torch.equal(out, eager)


# --------------------------------------------------------------------------
# Unified MoELayer surface: the kernel's public home (default-visible)
# --------------------------------------------------------------------------


def _unified_layer(T, backends=None, tune_max=64, seed=0, *, e=E, k=K, h=H, i=I):
    """Build a dsfp8 MoELayer over the contract geometry plus matching packs."""
    if backends is None:
        backends = (SpecializedFp8BlockSmallTConfig(),)
    device = torch.device("cuda", torch.cuda.current_device())
    b = _make(T, device, seed=seed, e=e, k=k, h=h, i=i)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=e, top_k=k),
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8,
            activation=QuantFormat.DeepSeekFp8,
        ),
        experts=ExpertConfig(intermediate_size=i),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=backends),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
    )
    act = MoEActivationPack(b["x"], None, b["ids"], b["tw"])
    weights = MoEWeightPack()
    # The stock dsfp8 view serves both backends (weight_view_aliases).
    weights.prepare_for(
        "cutlass_fp8_block",
        {
            "fc1_expert_weights": b["w13"],
            "fc2_expert_weights": b["w2"],
            "fc1_block_scale": b["s13"],
            "fc2_block_scale": b["s2"],
        },
    )
    return MoELayer(config), act, weights, b


def test_unified_config_gating_and_registration():
    """GPU-free: SM90-only gating, runner registration, DEFAULT visibility."""
    from flashinfer.fused_moe.api import _DEFAULT_BACKEND, ALL_BACKEND_CONFIGS
    from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

    assert SpecializedFp8BlockSmallTConfig.supported(90)
    for arch in (80, 89, 100, 103, 107, 110, 120, 121):
        assert not SpecializedFp8BlockSmallTConfig.supported(arch)
    # First-class registration on the unified surface...
    assert (
        _BACKEND_RUNNERS[SpecializedFp8BlockSmallTConfig]
        is SpecializedFp8BlockSmallTRunner
    )
    assert SpecializedFp8BlockSmallTConfig in ALL_BACKEND_CONFIGS
    # ... and DEFAULT-VISIBLE: a matching layer reaches the kernel through
    # backend=auto with no opt-in, on SM90 only (supported() gates the arch,
    # the runner gates the contract and token bands).
    assert any(isinstance(c, SpecializedFp8BlockSmallTConfig) for c in _DEFAULT_BACKEND)
    assert any(
        isinstance(c, SpecializedFp8BlockSmallTConfig)
        for c in _DEFAULT_BACKEND.valid_for(90)
    )
    assert not any(
        isinstance(c, SpecializedFp8BlockSmallTConfig)
        for c in _DEFAULT_BACKEND.valid_for(120)
    )
    # The incumbent stock dsfp8 backend is ALSO a default candidate: it
    # serves every SM90 dsfp8 call the specialized runner declines (the
    # 17..255 gap, off-contract geometry, the kill switch), so default-auto
    # dsfp8 layers on SM90 always have a named backend.
    assert any(isinstance(c, CutlassFp8BlockConfig) for c in _DEFAULT_BACKEND)
    assert any(
        isinstance(c, CutlassFp8BlockConfig) for c in _DEFAULT_BACKEND.valid_for(90)
    )
    # Same prepared-weight contract as the stock dsfp8 backend.
    assert SpecializedFp8BlockSmallTRunner.weight_view_aliases == ("cutlass_fp8_block",)
    assert (
        SpecializedFp8BlockSmallTConfig.prepare_weights
        is CutlassFp8BlockConfig.prepare_weights
    )


def test_unified_smem_rejection_precedes_specialized_forward(monkeypatch):
    _sm90_or_skip()
    layer, act, weights, _ = _unified_layer(
        4,
        (SpecializedFp8BlockSmallTConfig(), CutlassFp8BlockConfig()),
        e=2,
        k=2,
        h=4096,
        i=14336,
    )
    assert {r.backend_key for r in layer.runners} == {
        "specialized_fp8_block_smallt",
        "cutlass_fp8_block",
    }

    def unexpected(*args, **kwargs):
        pytest.fail("specialized forward entered for 57344-byte FC2 staging")

    monkeypatch.setattr(SpecializedFp8BlockSmallTRunner, "forward", unexpected)
    out = layer(act, weights)
    assert out.shape == (4, 4096)
    assert layer.winner_backend == "cutlass_fp8_block"


@pytest.mark.parametrize("T", [1, 4, 16])
def test_unified_layer_dispatches_specialized_kernel(T):
    """MoELayer with the opt-in backend serves an in-band call with the
    specialized kernel (the determinism premise: eligible shape => the
    specialized kernel, not the stock route)."""
    _sm90_or_skip()
    layer, act, weights, b = _unified_layer(T, None)
    assert [r.backend_key for r in layer.runners] == ["specialized_fp8_block_smallt"]
    before = _moe_smallt_sm90_stats()
    out = layer(act, weights)
    after = _moe_smallt_sm90_stats()
    assert layer.winner_backend == "specialized_fp8_block_smallt"
    assert after["runs_dispatched"] > before["runs_dispatched"]
    assert after["backend_dispatches"]["specialized"] > 0
    ts, l2 = _gate(out, _reference(b))
    assert ts < TOKEN_SUM_REL and l2 < TOKEN_L2_REL, (ts, l2)


def test_unified_autotune_reaches_both_backends():
    """Cross-backend shootout: the specialized runner is a first-class timed
    candidate next to the stock dsfp8 runner, not an allowlist hook inside
    it."""
    _sm90_or_skip()
    layer, act, weights, b = _unified_layer(
        4,
        (SpecializedFp8BlockSmallTConfig(), CutlassFp8BlockConfig()),
    )
    keys = [r.backend_key for r in layer.runners]
    assert keys == ["specialized_fp8_block_smallt", "cutlass_fp8_block"]
    # The tuning surface enumerates BOTH backends as candidates.
    candidates = layer._tuning_candidates(act, weights)
    assert {c["backend"] for c in candidates} == set(keys)
    # The cross-backend shootout actually launches the specialized kernel
    # while timing its candidate.
    before = _moe_smallt_sm90_stats()["runs_dispatched"]
    out = layer(act, weights)
    assert _moe_smallt_sm90_stats()["runs_dispatched"] > before
    assert layer.winner_backend in keys
    ts, l2 = _gate(out, _reference(b))
    assert ts < TOKEN_SUM_REL and l2 < TOKEN_L2_REL, (ts, l2)


def test_unified_kill_switch_declines_and_stock_serves(monkeypatch):
    """With the kill switch set, the specialized runner declines every call
    (it never serves through another kernel); a layer that also configures
    the stock dsfp8 backend keeps working through it."""
    _sm90_or_skip()
    monkeypatch.setenv(DISABLE_ENV, "1")
    layer, act, weights, b = _unified_layer(
        4, (SpecializedFp8BlockSmallTConfig(), CutlassFp8BlockConfig())
    )
    out = layer(act, weights)
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert layer.winner_backend == "cutlass_fp8_block"
    ts, l2 = _gate(out, _reference(b))
    assert ts < TOKEN_SUM_REL and l2 < TOKEN_L2_REL, (ts, l2)
    # A specialized-ONLY layer has no eligible backend while disabled: it
    # raises instead of silently serving the call with a different kernel.
    layer_only, act2, weights2, _ = _unified_layer(4, None)
    with pytest.raises(NotImplementedError, match="routing_input_mode"):
        layer_only(act2, weights2)


def test_unified_out_of_band_tokens_route_to_stock():
    """Out-of-band token counts (the 17..255 gap) are declined at
    eligibility time: auto routes them to the stock dsfp8 backend, and a
    specialized-only layer refuses them instead of serving a different
    kernel (negative control for the removed in-runner fallback)."""
    _sm90_or_skip()
    layer, act, weights, b = _unified_layer(
        64, (SpecializedFp8BlockSmallTConfig(), CutlassFp8BlockConfig())
    )
    out = layer(act, weights)
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert layer.winner_backend == "cutlass_fp8_block"
    ts, l2 = _gate(out, _reference(b))
    assert ts < TOKEN_SUM_REL and l2 < TOKEN_L2_REL, (ts, l2)
    layer_only, act2, weights2, _ = _unified_layer(64, None)
    with pytest.raises(NotImplementedError, match="routing_input_mode"):
        layer_only(act2, weights2)


def test_unified_override_views_route_to_stock():
    """A stock dsfp8 view carrying per-expert activation overrides (a
    stock-kernel feature the specialized kernel does not implement) is
    declined at eligibility time: auto serves the call with the stock
    backend instead of failing inside the specialized runner's
    pack_inputs."""
    _sm90_or_skip()
    layer, act, weights, b = _unified_layer(
        4, (SpecializedFp8BlockSmallTConfig(), CutlassFp8BlockConfig())
    )
    device = b["x"].device
    view = dict(weights.get_view("cutlass_fp8_block"))
    view["gemm1_beta"] = torch.ones((E,), dtype=torch.float32, device=device)
    weights_over = MoEWeightPack()
    weights_over.prepare_for("cutlass_fp8_block", view)
    runners = {r.backend_key: r for r in layer.runners}
    assert not runners["specialized_fp8_block_smallt"].accepts(act, weights_over)
    assert runners["cutlass_fp8_block"].accepts(act, weights_over)
    out = layer(act, weights_over)
    stats = _moe_smallt_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert layer.winner_backend == "cutlass_fp8_block"
    assert torch.isfinite(out.float()).all()
    # without the override the same layer reaches the specialized kernel
    out2 = layer(act, weights)
    assert _moe_smallt_sm90_stats()["runs_dispatched"] > 0
    ts, l2 = _gate(out2, _reference(b))
    assert ts < TOKEN_SUM_REL and l2 < TOKEN_L2_REL, (ts, l2)


def test_disabled_constructor_defers_real_jit_until_eager(monkeypatch):
    _sm90_or_skip()
    import flashinfer.fused_moe.smallt_sm90 as smallt
    import flashinfer.jit.moe_fp8_grouped_smallt_sm90 as recipe

    generate = recipe.gen_moe_fp8_grouped_smallt_sm90_module
    calls = []

    def observed():
        calls.append(None)
        return generate()

    monkeypatch.setattr(recipe, "gen_moe_fp8_grouped_smallt_sm90_module", observed)
    monkeypatch.setattr(smallt, "_module", None)
    monkeypatch.setattr(smallt, "_module_error", None)
    monkeypatch.setenv(DISABLE_ENV, "1")
    layer, act, weights, _ = _unified_layer(4)
    assert not calls and smallt._module is None
    monkeypatch.delenv(DISABLE_ENV)
    out = layer(act, weights)
    torch.cuda.synchronize()
    assert len(calls) == 1 and smallt._module is not None
    assert torch.isfinite(out).all()


def test_cold_capture_on_input_device_with_other_device_current(monkeypatch):
    _sm90_or_skip()
    import flashinfer.fused_moe.smallt_sm90 as smallt

    if torch.cuda.device_count() < 2:
        pytest.skip("requires a second CUDA device")
    if torch.cuda.get_device_capability(1) != (9, 0):
        pytest.skip("requires SM90 on the second device")
    b = _make(4, torch.device("cuda:1"))
    torch.cuda.synchronize(1)
    monkeypatch.setattr(smallt, "_module", None)

    def forbidden():
        raise AssertionError("cold capture must not load or compile a module")

    monkeypatch.setattr(smallt, "_get_module", forbidden)
    previous = torch.cuda.current_device()
    try:
        with torch.cuda.device(1):
            stream = torch.cuda.Stream()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                b["x"].add_(0)
                torch.cuda.set_device(0)
                try:
                    assert _dispatch(b) is None
                    assert torch.cuda.current_device() == 0
                finally:
                    torch.cuda.set_device(1)
            g.replay()
            torch.cuda.synchronize(1)
    finally:
        torch.cuda.set_device(previous)


def test_deferred_jit_failure_excluded_before_selection(monkeypatch):
    _sm90_or_skip()
    import flashinfer.fused_moe.smallt_sm90 as smallt
    import flashinfer.jit.moe_fp8_grouped_smallt_sm90 as recipe

    calls = []

    def failed_build():
        calls.append(None)
        raise RuntimeError("injected compiler failure")

    monkeypatch.setattr(recipe, "gen_moe_fp8_grouped_smallt_sm90_module", failed_build)
    monkeypatch.setattr(smallt, "_module", None)
    monkeypatch.setattr(smallt, "_module_error", None)
    monkeypatch.setenv(DISABLE_ENV, "1")
    layer, act, weights, b = _unified_layer(
        4, (SpecializedFp8BlockSmallTConfig(), CutlassFp8BlockConfig())
    )
    assert [r.backend_key for r in layer.runners] == [
        "specialized_fp8_block_smallt",
        "cutlass_fp8_block",
    ]
    assert layer._eligible_runners(act, weights) == (1,)
    assert not calls
    monkeypatch.delenv(DISABLE_ENV)
    ref = _reference(b)
    for _ in range(2):
        out = layer(act, weights)
        assert layer.winner_backend == "cutlass_fp8_block"
        assert layer._eligible_runners(act, weights) == (1,)
        ts, l2 = _gate(out, ref)
        assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL
    assert len(calls) == 1
    assert "injected compiler failure" in smallt._module_error
    assert _moe_smallt_sm90_stats()["calls_seen"] == 0


def test_deferred_jit_checks_structure_first(monkeypatch):
    _sm90_or_skip()
    import flashinfer.fused_moe.smallt_sm90 as smallt

    monkeypatch.setattr(smallt, "_module", None)
    monkeypatch.setattr(smallt, "_module_error", None)
    monkeypatch.setattr(smallt, "_get_module", lambda: pytest.fail("unexpected JIT"))
    monkeypatch.setenv(DISABLE_ENV, "1")
    layer, act, weights, b = _unified_layer(4)
    gap, gap_act, gap_weights, _ = _unified_layer(64)
    monkeypatch.delenv(DISABLE_ENV)
    assert not gap.runners[0].accepts(gap_act, gap_weights)
    view = dict(weights.get_view("cutlass_fp8_block"), gemm1_beta=b["tw"][0])
    overrides = MoEWeightPack()
    overrides.prepare_for("cutlass_fp8_block", view)
    assert not layer.runners[0].accepts(act, overrides)
    buf = b["x"].new_empty(4 * H + 1)
    misaligned = MoEActivationPack(buf[1:].view(4, H), None, b["ids"], b["tw"])
    assert not layer.runners[0].accepts(misaligned, weights)
    assert smallt._module_error is None


def test_deferred_selection_capture_device_and_warm_switch(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    target = next(
        (
            i
            for i in range(torch.cuda.device_count())
            if torch.cuda.get_device_capability(i) == (9, 0)
        ),
        None,
    )
    if target is None:
        pytest.skip("requires an SM90 input device")
    import flashinfer.fused_moe.smallt_sm90 as smallt

    monkeypatch.setattr(smallt, "_module", None)
    monkeypatch.setattr(smallt, "_module_error", None)
    monkeypatch.setenv(DISABLE_ENV, "1")
    with torch.cuda.device(target):
        layer, act, weights, _ = _unified_layer(4)
    monkeypatch.delenv(DISABLE_ENV)
    probes = []

    def capturing():
        probes.append(torch.cuda.current_device())
        return probes[-1] == target

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    monkeypatch.setattr(smallt, "_get_module", lambda: pytest.fail("unexpected JIT"))
    other = 1 if target == 0 else 0
    with torch.cuda.device(other):
        assert not layer.runners[0].accepts(act, weights)
        assert probes == [target] and torch.cuda.current_device() == other
        assert smallt._module_error is None
        monkeypatch.setattr(smallt, "_module", object())
        assert layer._eligible_runners(act, weights) == (0,)
        monkeypatch.setenv(DISABLE_ENV, "1")
        assert not layer.runners[0].accepts(act, weights)
        monkeypatch.delenv(DISABLE_ENV)
        assert layer._eligible_runners(act, weights) == (0,)
        assert probes == [target]


def test_warm_invocation_failure_propagates(monkeypatch):
    _sm90_or_skip()
    import flashinfer.fused_moe.smallt_sm90 as smallt

    class BrokenLaunch:
        @staticmethod
        def plan_workspace_bytes(*args):
            return 16

        @staticmethod
        def run(*args):
            raise RuntimeError("injected invocation failure")

    monkeypatch.setattr(smallt, "_module", BrokenLaunch())
    monkeypatch.setattr(smallt, "_module_error", None)
    layer, act, weights, _ = _unified_layer(4)
    assert layer._eligible_runners(act, weights) == (0,)
    with pytest.raises(RuntimeError, match="injected invocation failure"):
        layer(act, weights)
    assert smallt._module_error is None


def test_deferred_alignment_remains_ineligible_after_warmup(monkeypatch):
    _sm90_or_skip()
    from unittest.mock import Mock

    import flashinfer.fused_moe.smallt_sm90 as smallt

    monkeypatch.setattr(smallt, "_module", None)
    monkeypatch.setattr(smallt, "_module_error", None)
    monkeypatch.setenv(DISABLE_ENV, "1")
    layer, act, weights, b = _unified_layer(
        4, (SpecializedFp8BlockSmallTConfig(), CutlassFp8BlockConfig())
    )
    runner = layer.runners[0]

    def offset(t, elements):
        buf = t.new_empty(t.numel() + elements)
        out = buf[elements:].view_as(t)
        assert out.is_contiguous()
        assert out.data_ptr() % 16 == (elements * t.element_size()) % 16
        return out

    misaligned = MoEActivationPack(offset(b["x"], 1), None, b["ids"], b["tw"])
    cases = [(misaligned, weights)]
    for name in ("fc1_expert_weights", "fc2_expert_weights"):
        view = dict(weights.get_view("cutlass_fp8_block"))
        view[name] = offset(view[name], 1)
        bad_weights = MoEWeightPack()
        bad_weights.prepare_for("cutlass_fp8_block", view)
        cases.append((act, bad_weights))
    monkeypatch.delenv(DISABLE_ENV)
    with monkeypatch.context() as ctx:
        ctx.setattr(
            runner, "pack_inputs", Mock(side_effect=AssertionError("unexpected pack"))
        )
        for bad_act, bad_weights in cases:
            assert layer._eligible_runners(bad_act, bad_weights) == (1,)
    assert smallt._module is None and runner._jit_deferred
    assert layer._eligible_runners(act, weights) == (0, 1)
    assert smallt._module is not None and not runner._jit_deferred
    before = _moe_smallt_sm90_stats()["runs_dispatched"]
    out = runner.forward(runner.pack_inputs(act, weights))
    torch.cuda.synchronize(act.hidden_states_q.device)
    assert _moe_smallt_sm90_stats()["runs_dispatched"] == before + 1
    ts, l2 = _gate(out, _reference(b))
    assert ts <= TOKEN_SUM_REL and l2 <= TOKEN_L2_REL
    aligned_offset = MoEActivationPack(offset(b["x"], 8), None, b["ids"], b["tw"])
    with monkeypatch.context() as ctx:
        ctx.setattr(
            runner, "pack_inputs", Mock(side_effect=AssertionError("unexpected pack"))
        )
        for bad_act, bad_weights in cases:
            assert layer._eligible_runners(bad_act, bad_weights) == (1,)
        assert layer._eligible_runners(act, weights) == (0, 1)
        assert layer._eligible_runners(aligned_offset, weights) == (0, 1)
