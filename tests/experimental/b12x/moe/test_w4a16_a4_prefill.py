"""Opt-in A4 prefill over W4A16 packed weights.

Gates the NVFP4-activation prefill pipeline (route pack 128 -> quantize ->
A4 FC1 -> A4 FC2 -> FP32-weighted top-k sum) against a float64 torch emulation
of the same quantization contract, for one activation plane (NVFP4) and two
planes (NVFP4 value + NVFP4 residual). Checks explicit per-call precision
selection, calibrated-scale requirements, and graph replay across live token
counts with the same compiled callables.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x

E, H, I, TOPK = 8, 512, 256, 2

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_table(device):
    mags = torch.tensor(_E2M1, device=device)
    return torch.cat((mags, -mags))


def _dequant_weight(codes_u8, scale_e4m3, gscale):
    table = _e2m1_table(codes_u8.device)
    lo = table[(codes_u8 & 0xF).long()]
    hi = table[(codes_u8 >> 4).long()]
    w = torch.stack((lo, hi), dim=-1).reshape(codes_u8.shape[0], -1).double()
    return w * scale_e4m3.double().repeat_interleave(16, dim=1) * float(gscale)


def _e2m1_round(v):
    grid = torch.tensor(_E2M1, device=v.device, dtype=v.dtype)
    mids = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=v.device, dtype=v.dtype
    )
    mag = v.abs()
    idx = torch.bucketize(mag, mids)
    tie = (idx < 7) & (mag == mids[idx.clamp(max=6)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.sign(v) * grid[idx]


def _nvfp4_qdq(x, gs):
    """quantize_block_fp4 semantics per 16 values along the last dim."""
    xb = x.float().reshape(-1, 16)
    gsb = gs.float().reshape(-1, 1)
    if gsb.shape[0] != 1:
        gsb = gsb.repeat_interleave(xb.shape[0] // gsb.shape[0], dim=0)
    amax = xb.abs().amax(dim=1, keepdim=True)
    sf = (amax * gsb / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    vs = sf / gsb
    q = torch.where(
        vs > 0,
        _e2m1_round(xb / torch.where(vs > 0, vs, torch.ones_like(vs))),
        torch.zeros_like(xb),
    )
    return (q * vs).reshape(x.shape)


def _qdq(x, gs, terms):
    q1 = _nvfp4_qdq(x, gs)
    if terms == 1:
        return q1
    return q1 + _nvfp4_qdq(x.float() - q1, gs)


def _make_case(seed, intermediate_size=I, hidden_size=H, num_experts=E, top_k=TOPK):
    import numpy as np

    from b12x.moe import fused_moe as moe
    from b12x.moe._shared.kernels.w4a16.host import unswizzle_expert_scales

    E, H, I = num_experts, hidden_size, intermediate_size
    dev = torch.device("cuda")
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=gen
    )
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=gen
    )

    def scales(rows, cols):
        base = rng.integers(0x28, 0x48, size=(E, rows, 1))
        raw = (base + rng.integers(0, 14, size=(E, rows, cols))).astype(np.uint8)
        return torch.from_numpy(raw).to(dev).view(torch.float8_e4m3fn)

    s13, s2 = scales(2 * I, H // 16), scales(H, I // 16)
    g13 = (torch.rand(E, device=dev, generator=gen) * 0.5 + 0.75) * 0.02
    g2 = (torch.rand(E, device=dev, generator=gen) * 0.5 + 0.75) * 0.02
    return dict(
        top_k=top_k,
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        g13=g13,
        g2=g2,
        s13_log=unswizzle_expert_scales(s13, rows=2 * I, cols=H),
        s2_log=unswizzle_expert_scales(s2, rows=H, cols=I),
        plan=moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(
                mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
        ),
    )


def _env(monkeypatch, enabled, terms):
    # Plan-time options: preparation may run lazily at the first bind, so they
    # stay set for the whole test.
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL", str(int(enabled)))
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_TERMS", str(terms))


def _prepare(case, a1g, a2g, *, csf=False):
    from b12x.moe import fused_moe as moe

    H = case["plan"].geometry.hidden_size
    I = case["plan"].geometry.intermediate_size

    weights = moe.PackedWeights(
        w13=case["w13"].clone(),
        w2=case["w2"].clone(),
        w13_block_scales=case["s13"].clone(),
        w2_block_scales=case["s2"].clone(),
        w13_global_scales=case["g13"],
        w2_global_scales=case["g2"],
        input_scale=a1g,
        intermediate_scale=a2g,
        immutable_input_scales=True,
    )
    if csf:
        from .test_nvfp4_csf import compress_fixture

        weights = moe.Nvfp4CsfWeights(
            packed=replace(
                weights,
                w13_block_scales=torch.empty_like(case["s13"]),
                w2_block_scales=torch.empty_like(case["s2"]),
            ),
            w13_scales=compress_fixture(case["s13"], 2 * I, H // 16),
            w2_scales=compress_fixture(case["s2"], H, I // 16),
        )
    return moe.prepare_weights(plan=case["plan"], weights=weights)


def _plan(experts, tokens, top_k=TOPK):
    from b12x.moe import fused_moe as moe

    return moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=top_k),
        invocation={"fast_math": True},
    )


def _bind(xp, x, ids, wts, out, scratch, **kwargs):
    from b12x.moe import fused_moe as moe

    return moe.bind(
        xp,
        a=x,
        topk_ids=ids,
        topk_weights=wts,
        output=out,
        scratch=scratch,
        input_scales_static=True,
        **kwargs,
    )


def _emulate(case, x, ids, wts, a1g, a2g, terms):
    H = x.shape[-1]
    TOPK = ids.shape[-1]
    I = case["plan"].geometry.intermediate_size
    tokens = x.shape[0]
    flat = ids.flatten().long()
    xq = _qdq(x.float(), a1g.amin().reshape(1), terms).double()
    y = torch.zeros(tokens * TOPK, H, device=x.device, dtype=torch.float64)
    for e in torch.unique(flat).tolist():
        r = (flat == e).nonzero().flatten()
        w13 = _dequant_weight(case["w13"][e], case["s13_log"][e], case["g13"][e])
        gate = (xq[r // TOPK] @ w13[I:].T).float().to(torch.bfloat16).float()
        up = (xq[r // TOPK] @ w13[:I].T).float().to(torch.bfloat16).float()
        limit = case["plan"].activation.swiglu_limit
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        silu = (gate * torch.sigmoid(gate)).to(torch.bfloat16).float()
        act = (silu * up.to(torch.bfloat16).float()).to(torch.bfloat16).float()
        act = _qdq(act, a2g[e].reshape(1), terms).double()
        w2 = _dequant_weight(case["w2"][e], case["s2_log"][e], case["g2"][e])
        y[r] = (act @ w2.T).float().to(torch.bfloat16).double()
    return (y.view(tokens, TOPK, H) * wts.double().view(tokens, TOPK, 1)).sum(dim=1)


def _inputs(tokens, seed, hidden_size=H, num_experts=E, top_k=TOPK):
    E, H, TOPK = num_experts, hidden_size, top_k
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(tokens, H, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
    ids = torch.argsort(torch.rand(tokens, E, device=dev, generator=gen), dim=1)[
        :, :TOPK
    ]
    ids = ids.to(torch.int32).contiguous()
    wts = torch.softmax(
        torch.randn(tokens, TOPK, device=dev, generator=gen), dim=-1
    ).float()
    return x, ids, wts.contiguous()


def _scales(case):
    """Calibration-like global scales: 448 * 6 / amax with headroom. Odd factors
    keep the synthetic data off exact E2M1 midpoints."""
    H = case["plan"].geometry.hidden_size
    E = case["plan"].geometry.num_experts
    TOPK = case["top_k"]
    I = case["plan"].geometry.intermediate_size
    dev = torch.device("cuda")
    x, ids, _ = _inputs(64, 999, H, E, TOPK)
    flat = ids.flatten().long()
    amax = 0.0
    for e in torch.unique(flat).tolist():
        r = (flat == e).nonzero().flatten()
        w13 = _dequant_weight(case["w13"][e], case["s13_log"][e], case["g13"][e])
        gate = x[r // TOPK].double() @ w13[I:].T
        up = x[r // TOPK].double() @ w13[:I].T
        amax = max(amax, float((gate * torch.sigmoid(gate) * up).abs().max()))
    a1g = torch.full((E,), 448.0 * 6.0 / 2.75 * 1.0123457, device=dev)
    a2g = torch.full((E,), 448.0 * 6.0 / (1.5 * amax) * 0.9876543, device=dev)
    return a1g, a2g


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("terms", [1, 2])
@pytest.mark.parametrize(
    "hidden_size,intermediate_size,num_experts,top_k,swiglu_limit",
    [
        (512, 256, 8, 2, None),
        (512, 256, 8, 2, 2.03),
        (512, 256, 8, 2, 10.0),
        (512, 320, 8, 2, None),
        (2560, 320, 16, 10, None),
    ],
)
def test_a4_prefill_matches_float64_emulation(
    terms, hidden_size, intermediate_size, num_experts, top_k, swiglu_limit, monkeypatch
):
    require_b12x()
    _env(monkeypatch, True, terms)
    case = _make_case(11, intermediate_size, hidden_size, num_experts, top_k)
    from b12x.moe import fused_moe as moe

    case["plan"] = moe.plan_weights(
        source=case["plan"].source,
        activation=replace(case["plan"].activation, swiglu_limit=swiglu_limit),
        geometry=case["plan"].geometry,
    )
    a1g, a2g = _scales(case)
    a1g = a1g * torch.linspace(0.9, 1.1, num_experts, device=a1g.device)
    a2g = a2g * torch.linspace(1.1, 0.9, num_experts, device=a2g.device)
    experts = _prepare(case, a1g, a2g)
    assert experts._impl.a4_prefill_scales
    tokens = 16 if top_k == 10 else 192
    xp = _plan(experts, 192, top_k)
    x, ids, wts = _inputs(tokens, 5, hidden_size, num_experts, top_k)
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=x.device) for s in xp.scratch_specs()
    )
    out = torch.empty_like(x)
    binding = _bind(xp, x, ids, wts, out, scratch, a4_prefill=True)
    launches = getattr(binding, "_impl", binding).a4_prefill_launches
    assert launches is not None and launches.terms == terms
    from b12x.moe import fused_moe as moe

    moe.run(binding=binding)
    torch.cuda.synchronize()
    ref = _emulate(case, x, ids, wts, a1g, a2g, terms)
    if swiglu_limit is not None:
        unclamped = dict(case)
        unclamped["plan"] = replace(
            case["plan"], activation=replace(case["plan"].activation, swiglu_limit=None)
        )
        assert not torch.equal(ref, _emulate(unclamped, x, ids, wts, a1g, a2g, terms))
    assert torch.isfinite(out.float()).all() and out.float().abs().max() > 0
    rel = ((out.double() - ref).norm() / ref.norm()).item()
    # BF16 output rounding plus the BF16 top-k sum bound the difference.
    assert rel < 4e-3, rel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a4_prefill_requires_explicit_choice_and_calibrated_scales(monkeypatch):
    require_b12x()
    _env(monkeypatch, True, 1)
    case = _make_case(12)
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g)
    xp = _plan(experts, 256)
    dev = torch.device("cuda")
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=dev) for s in xp.scratch_specs()
    )
    for tokens in (4, 64, 128, 256):
        x, ids, wts = _inputs(tokens, tokens)
        binding = _bind(xp, x, ids, wts, torch.empty_like(x), scratch)
        assert getattr(binding, "_impl", binding).a4_prefill_launches is None, tokens
    # Without calibrated scales (or with invalid ones) the weights stay W4A16 only.
    bad = a1g.clone()
    bad[3] = float("nan")
    ones = torch.ones_like(a1g)
    for scales in ((None, None), (bad, a2g), (ones, ones)):
        plain = _prepare(case, scales[0], scales[1])
        assert not plain._impl.a4_prefill_scales
        xp_plain = _plan(plain, 256)
        x, ids, wts = _inputs(256, 3)
        scratch_plain = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=dev)
            for s in xp_plain.scratch_specs()
        )
        binding = _bind(
            xp_plain, x, ids, wts, torch.empty_like(x), scratch_plain, a4_prefill=True
        )
        assert getattr(binding, "_impl", binding).a4_prefill_launches is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enabled", [False, True])
def test_a4_prefill_per_call_choice_is_independent_of_token_count(enabled, monkeypatch):
    """A caller that knows which rows are prefill picks the path per call."""
    require_b12x()
    _env(monkeypatch, enabled, 1)
    case = _make_case(13)
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g)
    xp = _plan(experts, 256)
    dev = torch.device("cuda")
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=dev) for s in xp.scratch_specs()
    )
    for tokens in (4, 256):
        for choice in (None, False, True):
            x, ids, wts = _inputs(tokens, tokens)
            binding = _bind(
                xp, x, ids, wts, torch.empty_like(x), scratch, a4_prefill=choice
            )
            launches = getattr(binding, "_impl", binding).a4_prefill_launches
            assert (launches is not None) == (enabled and choice is True)
    if not enabled:
        return
    # An explicitly selected small prefill computes the same A4 math.
    from b12x.moe import fused_moe as moe

    x, ids, wts = _inputs(48, 21)
    out = torch.empty_like(x)
    moe.run(binding=_bind(xp, x, ids, wts, out, scratch, a4_prefill=True))
    torch.cuda.synchronize()
    ref = _emulate(case, x, ids, wts, a1g, a2g, 1)
    rel = ((out.double() - ref).norm() / ref.norm()).item()
    assert rel < 4e-3, rel
    # Weights without calibrated scales stay W4A16 even when A4 is asked for.
    plain = _prepare(case, None, None)
    xp_plain = _plan(plain, 256)
    scratch_plain = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=dev)
        for s in xp_plain.scratch_specs()
    )
    x, ids, wts = _inputs(256, 4)
    binding = _bind(
        xp_plain, x, ids, wts, torch.empty_like(x), scratch_plain, a4_prefill=True
    )
    assert getattr(binding, "_impl", binding).a4_prefill_launches is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a4_prefill_option_leaves_fp4_activation_plans_alone(monkeypatch):
    """The option keeps calibrated scales for W4A16 weights only: an FP4-activation
    plan prepares the same per-expert input scales with or without it."""
    require_b12x()
    from b12x.moe import fused_moe as moe

    case = _make_case(13)
    a1g, a2g = _scales(case)
    a1g = a1g * torch.linspace(0.9, 1.1, E, device=a1g.device)
    case["plan"] = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
    )
    _env(monkeypatch, False, 1)
    without = _prepare(case, a1g, a2g)._impl
    _env(monkeypatch, True, 1)
    with_option = _prepare(case, a1g, a2g)._impl
    assert not with_option.a4_prefill_scales
    assert torch.equal(without.a1_gscale, with_option.a1_gscale)
    assert torch.equal(without.a2_gscale, with_option.a2_gscale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("csf", [False, True])
def test_a4_prefill_explicit_choice_ignores_auto_a16_cutoff(csf, monkeypatch):
    require_b12x()
    from b12x.moe import fused_moe as moe

    _env(monkeypatch, True, 1)
    case = _make_case(16)
    case["plan"] = moe.plan_weights(
        source=case["plan"].source,
        activation=replace(
            case["plan"].activation, a16_max_tokens=128, swiglu_limit=10.0
        ),
        geometry=case["plan"].geometry,
    )
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g, csf=csf)
    assert experts._impl.a4_prefill_scales
    xp = _plan(experts, 192)
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device="cuda") for s in xp.scratch_specs()
    )
    for tokens in (96, 128, 192):
        x, ids, wts = _inputs(tokens, 24)
        out = torch.empty_like(x)
        binding = _bind(xp, x, ids, wts, out, scratch, a4_prefill=True)
        launches = getattr(binding, "_impl", binding).a4_prefill_launches
        assert launches is not None
        moe.run(binding=binding)
        ref = _emulate(case, x, ids, wts, a1g, a2g, 1)
        rel = ((out.double() - ref).norm() / ref.norm()).item()
        assert rel < 4e-3, (csf, rel)
        for choice in (None, False):
            a16 = _bind(xp, x, ids, wts, out, scratch, a4_prefill=choice)
            assert getattr(a16, "_impl", a16).a4_prefill_launches is None


@pytest.mark.parametrize("calibrated", [False, True])
def test_csf_prefetch_query_matches_exact_variant_and_precision(
    calibrated, monkeypatch
):
    """Prefetch follows the selected reader, including uncalibrated A16 fallback."""
    from types import MappingProxyType

    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe import fused_moe as moe

    device = require_b12x()
    _env(monkeypatch, True, 1)
    case = _make_case(18)
    a1g, a2g = _scales(case) if calibrated else (None, None)
    owner = _prepare(case, a1g, a2g, csf=True)
    plan = moe.plan_execution(
        experts=owner,
        capacity=moe.ExecutionCapacity(
            max_tokens=3072, top_k=TOPK, warmup_token_counts=(4,)
        ),
        invocation={"fast_math": True},
    )
    with pytest.raises(RuntimeError, match="require a prepared MoE plan"):
        moe.uses_expanded_nvfp4_scales(plan, num_tokens=4)
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=device) for s in plan.scratch_specs()
    )
    x, ids, wts = _inputs(128, 52)
    output = torch.empty_like(x)
    _bind(plan, x, ids, wts, output, scratch)
    decoded = []
    decode = Nvfp4CsfDecoder.decode

    def record_decode(self, *args, **kwargs):
        decoded.append(self)
        return decode(self, *args, **kwargs)

    monkeypatch.setattr(Nvfp4CsfDecoder, "decode", record_decode)
    for tokens in (4, 6, 128):
        for choice in (None, True):
            expected = True
            with kernel_resolution_guard("prepared scale-consumer metadata"):
                before = torch.cuda.memory_allocated()
                assert (
                    moe.uses_expanded_nvfp4_scales(
                        plan, num_tokens=tokens, a4_prefill=choice
                    )
                    is expected
                )
                assert torch.cuda.memory_allocated() == before
            assert not decoded
            binding = _bind(
                plan,
                x[:tokens],
                ids[:tokens],
                wts[:tokens],
                output[:tokens],
                scratch,
                a4_prefill=choice,
            )
            assert (binding.a4_prefill_launches is not None) == (
                calibrated and choice is True
            ), (tokens, choice)
            if calibrated and choice is True and tokens == 4:
                assert binding.a4_prefill_launches is (
                    plan._prepared.state.variants[3072].w4a16_launches.a4_prefill
                )
            moe.run(binding=binding)
            torch.cuda.synchronize()
            assert bool(decoded) is expected
            decoded.clear()
    for tokens in (0, 3073):
        with pytest.raises(ValueError, match="capacity"):
            moe.uses_expanded_nvfp4_scales(plan, num_tokens=tokens)
    if calibrated:
        # Without a larger prepared variant, the exact four-row plan retains
        # A16 when its intermediate buffers cannot hold padded A4 routes.
        small = _plan(owner, 4)
        small_scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in small.scratch_specs()
        )
        binding = _bind(
            small,
            x[:4],
            ids[:4],
            wts[:4],
            output[:4],
            small_scratch,
            a4_prefill=True,
        )
        assert binding.a4_prefill_launches is None
        assert moe.uses_expanded_nvfp4_scales(small, num_tokens=4, a4_prefill=True)
        # A4 cannot use a plan whose intermediate buffers do not fit its planes.
        root = plan._prepared.state
        state = root.variants[3072]
        core = state.scratch._core_workspace_plan
        undersized = replace(
            core,
            tensor_specs=tuple(
                replace(spec, shape=(1,))
                if spec.name == "intermediate_cache13"
                else spec
                for spec in core.tensor_specs
            ),
        )
        variants = dict(root.variants)
        variants[3072] = replace(
            state, scratch=replace(state.scratch, _core_workspace_plan=undersized)
        )
        monkeypatch.setattr(
            plan._prepared, "state", replace(root, variants=MappingProxyType(variants))
        )
        assert moe.uses_expanded_nvfp4_scales(plan, num_tokens=128, a4_prefill=True)


def test_a4_prefetch_reader_requires_live_size_and_ready_scales(monkeypatch):
    """Prospective prefetch and binding agree without changing A4 precision."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe import fused_moe as moe

    device = require_b12x()
    _env(monkeypatch, True, 1)
    case = _make_case(19)
    owner = _prepare(case, *_scales(case), csf=True)
    plan = moe.plan_execution(
        experts=owner,
        capacity=moe.ExecutionCapacity(
            max_tokens=3072, top_k=TOPK, warmup_token_counts=(4,)
        ),
        invocation={"fast_math": True},
    )
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=device) for s in plan.scratch_specs()
    )
    x, ids, wts = _inputs(1537, 53)
    output = torch.empty_like(x)
    _bind(plan, x, ids, wts, output, scratch, a4_prefill=True)
    with kernel_resolution_guard("prepared A4 scale-readiness selection"):
        before = torch.cuda.memory_allocated()
        for tokens in (4, 128, 1536, 1537):
            for ready in (False, True):
                for choice in (None, False, True):
                    binding = _bind(
                        plan,
                        x[:tokens],
                        ids[:tokens],
                        wts[:tokens],
                        output[:tokens],
                        scratch,
                        a4_prefill=choice,
                        scales_expanded=ready,
                    )
                    uses_expanded = moe.uses_expanded_nvfp4_scales(
                        plan,
                        num_tokens=tokens,
                        a4_prefill=choice,
                        scales_expanded=ready,
                    )
                    launch = binding.a4_prefill_launches
                    if choice is True:
                        assert launch is not None
                        expected = True
                    else:
                        assert launch is None
                        expected = True
                    assert uses_expanded is expected, (tokens, ready, choice)
        assert torch.cuda.memory_allocated() == before


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_a4_prefill_launchers_are_owned_by_each_device(monkeypatch):
    from b12x.moe import fused_moe as moe

    _env(monkeypatch, True, 1)
    owners = []
    for ordinal in (0, 1):
        with torch.cuda.device(ordinal):
            require_b12x()
            case = _make_case(15)
            a1g, a2g = _scales(case)
            experts = _prepare(case, a1g, a2g)
            xp = _plan(experts, 192)
            x, ids, wts = _inputs(192, 23)
            out = torch.empty_like(x)
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=x.device)
                for s in xp.scratch_specs()
            )
            binding = _bind(xp, x, ids, wts, out, scratch, a4_prefill=True)
            launches = getattr(binding, "_impl", binding).a4_prefill_launches
            assert launches is not None
            owners.append((launches.quant, launches.fc1, launches.fc2))
            moe.run(binding=binding)
            ref = _emulate(case, x, ids, wts, a1g, a2g, 1)
            rel = ((out.double() - ref).norm() / ref.norm()).item()
            assert rel < 4e-3, (ordinal, rel)
    assert all(first is not second for first, second in zip(*owners, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("terms", [1, 2])
def test_a4_prefill_graph_replay_reuses_launches(terms, monkeypatch):
    require_b12x()
    from b12x.moe import fused_moe as moe
    from b12x._lib.runtime_control import kernel_resolution_guard

    _env(monkeypatch, True, terms)
    case = _make_case(13)
    a1g, a2g = _scales(case)
    experts = _prepare(case, a1g, a2g)
    xp = _plan(experts, 320)
    dev = torch.device("cuda")
    scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=dev) for s in xp.scratch_specs()
    )
    seen = set()
    for tokens in (96, 320, 200):
        x, ids, wts = _inputs(tokens, 100 + tokens)
        eager_out = torch.empty_like(x)
        binding = _bind(xp, x, ids, wts, eager_out, scratch, a4_prefill=True)
        launches = getattr(binding, "_impl", binding).a4_prefill_launches
        assert launches is not None
        seen.add((id(launches.quant), id(launches.fc1), id(launches.fc2)))
        moe.run(binding=binding)
        torch.cuda.synchronize()
        expected = eager_out.clone()
        eager_out.zero_()
        changed_x, changed_ids, changed_wts = _inputs(tokens, 500 + tokens)
        graph = torch.cuda.CUDAGraph()
        pointers = tuple(t.data_ptr() for t in (*scratch, eager_out))
        try:
            with kernel_resolution_guard("A4 prepared launches"):
                with torch.cuda.graph(graph):
                    moe.run(binding=binding)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(eager_out, expected), tokens
                x.copy_(changed_x)
                ids.copy_(changed_ids)
                wts.copy_(changed_wts)
                moe.run(binding=binding)
                changed_expected = eager_out.clone()
                assert not torch.equal(expected, changed_expected)
                for buffer in scratch:
                    buffer.fill_(77)
                eager_out.fill_(float("nan"))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                assert torch.cuda.max_memory_allocated() == allocated
                assert pointers == tuple(t.data_ptr() for t in (*scratch, eager_out))
                assert torch.equal(eager_out, changed_expected), tokens
        finally:
            graph.reset()
    # One compiled pipeline serves every live token count of the capacity.
    assert len(seen) == 1


@pytest.mark.parametrize(
    "intermediate_size,terms,capacity,warps,tokens,ready",
    [
        (256, 1, 320, 8, 131, False),
        (256, 1, 3072, 8, 131, False),
        (256, 2, 3072, 8, 131, False),
        (256, 1, 3072, 16, 131, False),
        (256, 1, 3072, 8, 131, True),
        (256, 1, 3072, 8, 1537, False),
        (256, 1, 3072, 8, 1537, True),
        (320, 1, 320, 8, 131, False),
        (320, 2, 3072, 16, 131, False),
        (320, 1, 3072, 8, 1537, False),
        (320, 2, 3072, 8, 1537, True),
    ],
)
def test_a4_csf_scale_readers_preserve_dense_graph_output(
    intermediate_size, terms, capacity, warps, tokens, ready, monkeypatch
):
    """Scale expansion preserves native arithmetic and caller stream ordering."""
    import numpy as np

    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe import fused_moe as moe
    from .test_w4a16_csf_tp6 import _swizzled

    device = require_b12x()
    _env(monkeypatch, True, terms)
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_WARPS", str(warps))
    I = intermediate_size
    case = _make_case(17, I)
    case["plan"] = moe.plan_weights(
        source=case["plan"].source,
        activation=replace(case["plan"].activation, swiglu_limit=10.0),
        geometry=case["plan"].geometry,
    )
    for name, rows, columns in (("s13", 2 * I, H // 16), ("s2", H, I // 16)):
        expert = np.arange(E)[:, None, None]
        row = np.arange(rows)[None, :, None]
        group = np.arange(columns)[None, None, :]
        logical = 0x28 + (row // 64 + expert * 3) % 16 + (group + row) % 14
        # Two groups per atom force global replacement-word reads; other groups
        # retain base/code words.
        logical[:, :, 1::2] = 0x68 + (row + expert) % 8
        logical[:, 5, ::8] = 0
        logical = logical.astype(np.uint8)
        case[name] = _swizzled(logical, device)
        case[name + "_log"] = (
            torch.from_numpy(logical).to(device).view(torch.float8_e4m3fn)
        )
    a1g = torch.linspace(800, 1100, E, device=device)
    a2g = torch.linspace(24, 56, E, device=device)
    owners = [_prepare(case, a1g, a2g, csf=csf) for csf in (False, True)]
    x, ids, wts = _inputs(tokens, 51)
    bindings, outputs, scratches = [], [], []
    for owner in owners:
        plan = _plan(owner, capacity)
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in plan.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = _bind(
            plan,
            x,
            ids,
            wts,
            output,
            scratch,
            a4_prefill=True,
            scales_expanded=ready,
        )
        bindings.append(binding)
        outputs.append(output)
        scratches.append(scratch)
    launches = [binding.a4_prefill_launches for binding in bindings]
    assert launches[0].fc1 is launches[1].fc1
    assert launches[0].fc2 is launches[1].fc2
    packed = owners[1]._impl.representation.value
    scale_scratch = (packed.w13_scale, packed.w2_scale)
    consumes_expanded = ready

    def reject_expansion(*args, **kwargs):
        raise AssertionError("Caller-expanded scales must skip per-call expansion")

    if ready:
        monkeypatch.setattr(Nvfp4CsfDecoder, "decode", reject_expansion)
    for buffer in scale_scratch:
        buffer.view(torch.uint8).fill_(0x7F)
    if consumes_expanded:
        assert moe.expand_scales(owners[1])
    for binding in bindings:
        moe.run(binding=binding)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    assert torch.isfinite(outputs[1]).all() and torch.count_nonzero(outputs[1])
    graph = torch.cuda.CUDAGraph()
    stable = (*scratches[1], *scale_scratch, outputs[1])
    pointers = tuple(t.data_ptr() for t in stable)
    try:
        with kernel_resolution_guard("A4 CSF prepared readers"):
            with torch.cuda.graph(graph):
                moe.run(binding=bindings[1])
            for seed in (61, 62):
                changed = _inputs(x.shape[0], seed)
                for target, source in zip((x, ids, wts), changed, strict=True):
                    target.copy_(source)
                moe.run(binding=bindings[0])
                for buffer in (*scratches[1], *scale_scratch):
                    buffer.view(torch.uint8).fill_(0x7F)
                if consumes_expanded:
                    assert moe.expand_scales(owners[1])
                outputs[1].fill_(float("nan"))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                assert torch.cuda.max_memory_allocated() == allocated
                assert pointers == tuple(t.data_ptr() for t in stable)
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                if not consumes_expanded:
                    assert all(
                        not torch.all(buffer.view(torch.uint8) == 0x7F)
                        for buffer in scale_scratch
                    )
    finally:
        graph.reset()
