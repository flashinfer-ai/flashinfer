# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""The specialized SM107 FP8 backend of qsa_sparse_paged_attention against a
chunked PyTorch reference, plus backend semantics (explicit-never-swapped,
kill switch, stats), determinism and CUDA-graph warmup."""

import itertools

import pytest
import torch

from flashinfer import qsa_sparse_paged_attention

HQ = 6  # TP4 shard of Qwen QSA's 24 query heads
KV = 1
D = 256
PAGE = 64  # specialized backend's page size
PWIDTH = 2052  # selection width + count column
WIDTH = PWIDTH - 1
SCALE = D**-0.5
CR = 4  # tokens per selection block
BLOCK_TOPK = 512  # selected blocks per query
REF_CHUNK = 256  # reference gather chunk (memory bound)

# Acceptance gate: signal-scaled element match ratio plus an absolute error
# cap, both consistent with the validation gate the kernel was accepted under.
MATCH_ATOL = 2e-3
MATCH_RTOL = 2e-2
MIN_MATCHED_RATIO = 0.99
MAX_ABS_ERR = 0.01

SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(not SM107, reason="requires SM107 (Rubin)")


@pytest.fixture(autouse=True)
def _no_experimental_auto_opt_in(monkeypatch):
    """The specialized backend is stable-classed: backend="auto" must reach
    it WITHOUT FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS (which gates only
    experimental-classed backends, e.g. cake). Keep the variable unset so
    every auto-routing test proves default reachability."""
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)


def ref_qsa_sparse_attention(q, gate, k_cache, v_cache, packed, cu_seqlens, page_base):
    """Chunked exact reference: per-token gather of the selected K/V slots,
    FP32 softmax at scale 256**-0.5, BF16 rounding, then the sigmoid gate."""
    dev = q.device
    T = q.shape[0]
    cu = cu_seqlens.long()
    lens = cu[1:] - cu[:-1]
    req = torch.repeat_interleave(torch.arange(lens.numel(), device=dev), lens)
    kf = k_cache.reshape(-1, KV, D).float()
    vf = v_cache.reshape(-1, KV, D).float()
    idx = packed[:, :WIDTH].long()
    valid = idx >= 0
    base = page_base.long()[req] * PAGE
    out = torch.empty((T, HQ, D), dtype=torch.bfloat16, device=dev)
    gs = HQ // KV
    for c0 in range(0, T, REF_CHUNK):
        c1 = min(c0 + REF_CHUNK, T)
        slots = base[c0:c1].unsqueeze(1) + idx[c0:c1].clamp_min(0)
        keys = kf[slots]
        vals = vf[slots]
        qc = q[c0:c1].float()
        vmask = valid[c0:c1]
        outs = []
        for h in range(KV):
            qg = qc[:, h * gs : (h + 1) * gs]
            sc = torch.einsum("thd,twd->thw", qg, keys[:, :, h]) * SCALE
            sc = sc.masked_fill(~vmask.unsqueeze(1), float("-inf"))
            pr = torch.softmax(sc, dim=-1).nan_to_num(0.0)
            outs.append(torch.einsum("thw,twd->thd", pr, vals[:, :, h]))
        attn = torch.cat(outs, 1).to(torch.bfloat16).float()
        gated = attn * torch.sigmoid(gate[c0:c1].float())
        has = vmask.any(1)
        gated = torch.where(has.view(-1, 1, 1), gated, torch.zeros_like(gated))
        out[c0:c1] = gated.to(torch.bfloat16)
    return out


def build_selection(pos, seed, dev):
    """Causally-valid synthetic selection: up to 512 random complete blocks of
    the (pos+1)//4 visible, expanded x4, plus the causal tail; ascending ids,
    -1 pad, count in the last column."""
    T = pos.numel()
    vis = ((pos + 1) // CR).long()
    vmax = max(int(vis.max().item()), 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand((T, vmax), generator=g).to(dev)
    cols = torch.arange(vmax, device=dev)
    scores = scores.masked_fill(cols.unsqueeze(0) >= vis.unsqueeze(1), -1.0)
    k = min(BLOCK_TOPK, vmax)
    tv, ti = torch.topk(scores, k, dim=1)
    blocks = ti.masked_fill(tv < -0.5, -1)
    off = torch.arange(CR, device=dev)
    toks = torch.where(
        blocks.unsqueeze(-1) >= 0,
        blocks.unsqueeze(-1) * CR + off,
        torch.full((1, 1, CR), -1, dtype=torch.long, device=dev),
    ).reshape(T, k * CR)
    tcol = torch.arange(CR - 1, device=dev)
    tail_start = (pos + 1) // CR * CR
    tail = tail_start.unsqueeze(1) + tcol
    tail = torch.where(
        tcol.unsqueeze(0) < (pos + 1 - tail_start).unsqueeze(1),
        tail,
        torch.full_like(tail, -1),
    )
    allt = torch.cat([toks, tail], 1)
    big = torch.where(allt >= 0, allt, torch.full_like(allt, 2**31 - 1))
    srt = torch.sort(big, dim=1)[0]
    counts = (allt >= 0).sum(1, dtype=torch.int32)
    packed = torch.full((T, PWIDTH), -1, dtype=torch.int32, device=dev)
    sel = torch.where(srt < 2**31 - 1, srt, torch.full_like(srt, -1)).int()
    packed[:, : srt.shape[1]] = sel
    packed[:, PWIDTH - 1] = counts
    return packed


def make_inputs(lengths, seed=0, device="cuda", q_scale=0.5, spare_pages=1):
    """Packed requests of the given lengths over contiguous paged FP8 K/V
    caches (``spare_pages`` spare pages per request; 0 sizes the cache
    exactly); synthetic seeded values. Returns the qsa_sparse_paged_attention
    argument dict plus the reference-side views (cu_seqlens / page_base,
    which the block table encodes)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cu = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int32)
    T = int(cu[-1])
    npages = [(length + PAGE - 1) // PAGE + spare_pages for length in lengths]
    page_base = torch.tensor(
        [0] + list(itertools.accumulate(npages))[:-1], dtype=torch.int32
    )
    num_pages = sum(npages)
    # Contiguous per-request block ranges, as the specialized backend requires.
    max_blocks = max(npages)
    block_table = torch.zeros((len(lengths), max_blocks), dtype=torch.int32)
    for r, (base, n) in enumerate(zip(page_base.tolist(), npages, strict=True)):
        block_table[r, :n] = torch.arange(base, base + n, dtype=torch.int32)
    token_to_req = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int64),
    ).int()
    q = (torch.randn(T, HQ, D, generator=gen) * q_scale).to(torch.bfloat16)
    gate = torch.randn(T, HQ, D, generator=gen).to(torch.bfloat16)
    k_cache = (torch.randn(num_pages, PAGE, KV, D, generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    v_cache = (torch.randn(num_pages, PAGE, KV, D, generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    pos = torch.cat([torch.arange(length) for length in lengths])
    packed = build_selection(pos.to(device), seed * 31 + T, torch.device(device))
    return {
        "q": q.to(device),
        "k_cache": k_cache.to(device),
        "v_cache": v_cache.to(device),
        "logical_indices": packed,
        "block_table": block_table.to(device),
        "token_to_req": token_to_req.to(device),
        "output_gate": gate.to(device),
        "cu_seqlens": cu.to(device),
        "page_base": page_base.to(device),
    }


def run_api(inputs, out=None, backend="auto"):
    return qsa_sparse_paged_attention(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["logical_indices"],
        inputs["block_table"],
        inputs["token_to_req"],
        True,  # use_prefill_config: signature compatibility, no effect
        out,
        1.0,  # k_scale (the specialized backend serves unit scales)
        1.0,  # v_scale
        output_gate=inputs["output_gate"],
        backend=backend,
    )


def run_ref(inputs):
    return ref_qsa_sparse_attention(
        inputs["q"],
        inputs["output_gate"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["logical_indices"],
        inputs["cu_seqlens"],
        inputs["page_base"],
    )


def assert_close_at_gate(
    out, ref, atol=MATCH_ATOL, rtol=MATCH_RTOL, abs_cap=MAX_ABS_ERR
):
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    diff = (out.float() - ref.float()).abs()
    max_abs_err = float(diff.max()) if diff.numel() else 0.0
    matched = diff <= atol + rtol * ref.float().abs()
    matched_ratio = float(matched.float().mean()) if diff.numel() else 1.0
    assert max_abs_err <= abs_cap, f"max_abs_err {max_abs_err}"
    assert matched_ratio >= MIN_MATCHED_RATIO, f"matched_ratio {matched_ratio}"


@pytest.mark.parametrize(
    "lengths",
    [
        (24,),  # tiny per-token route
        (96,),  # short-row route
        (256,),  # short-row route, upper bound
        (512,),  # neighbor-class mid shape (general split path)
        (768,),  # the shape whose dedicated route is excluded: general path
        (300, 212, 180),  # batch > 1 multi-request mix
        (2048, 1552),  # batch > 1 long rows (descriptor-based sweep)
    ],
)
def test_correctness(lengths):
    inputs = make_inputs(lengths, seed=sum(lengths))
    out = run_api(inputs)
    ref = run_ref(inputs)
    assert_close_at_gate(out, ref)


@pytest.mark.parametrize("lengths", [(192,), (130,), (250, 190)])
def test_exact_capacity_cache(lengths):
    """A cache with no spare pages: the short-row KV windows must keep their
    logical origin when the last window overhangs the slot count (origin
    clamping read the wrong slots), and every route must stay in bounds."""
    inputs = make_inputs(lengths, seed=sum(lengths) + 1, spare_pages=0)
    out = run_api(inputs)
    ref = run_ref(inputs)
    assert_close_at_gate(out, ref)


def make_uniform_dense_prefix_inputs(lengths):
    """Full causal-prefix selections over all-ones Q/K/V and a zero gate.

    The softmax is exactly uniform and V is constant, so every output element
    is ``bf16(1.0) * sigmoid(0) = 0.5`` regardless of row length. The uniform
    logits of ``256 * 256**-0.5 = 16`` put the FP8 sweep's fixed-reference
    probabilities near ``exp(16)``: far beyond an E4M3 PV operand's range
    (~448) while the FP32 denominator stays finite and self-consistent — the
    case the probability-range guard must route to the exact retry."""
    dev = "cuda"
    cu = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int32)
    T = int(cu[-1])
    npages = [(length + PAGE - 1) // PAGE for length in lengths]
    page_base = torch.tensor(
        [0] + list(itertools.accumulate(npages))[:-1], dtype=torch.int32
    )
    num_pages = sum(npages)
    max_blocks = max(npages)
    block_table = torch.zeros((len(lengths), max_blocks), dtype=torch.int32)
    for r, (base, n) in enumerate(zip(page_base.tolist(), npages, strict=True)):
        block_table[r, :n] = torch.arange(base, base + n, dtype=torch.int32)
    token_to_req = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int64),
    ).int()
    pos = torch.cat([torch.arange(length) for length in lengths]).to(dev)
    cols = torch.arange(WIDTH, device=dev, dtype=torch.int32)
    sel = torch.where(
        cols.unsqueeze(0) <= pos.unsqueeze(1),
        cols.unsqueeze(0).expand(T, WIDTH),
        torch.full((T, WIDTH), -1, dtype=torch.int32, device=dev),
    )
    packed = torch.cat([sel, (pos + 1).to(torch.int32).unsqueeze(1)], dim=1)
    ones = torch.ones(T, HQ, D, dtype=torch.bfloat16, device=dev)
    kv_ones = torch.ones(num_pages, PAGE, KV, D, device=dev)
    return {
        "q": ones,
        "k_cache": kv_ones.to(torch.float8_e4m3fn),
        "v_cache": kv_ones.to(torch.float8_e4m3fn),
        "logical_indices": packed,
        "block_table": block_table.to(dev),
        "token_to_req": token_to_req.to(dev),
        "output_gate": torch.zeros_like(ones),
        "cu_seqlens": cu.to(dev),
        "page_base": page_base.to(dev),
    }


@pytest.mark.parametrize(
    "lengths",
    [
        (2029, 2028),  # T=4057, B=2 dedicated descriptor route
        (2046, 2045, 2045, 2045),  # T=8181, B=4 dedicated descriptor route
    ],
)
def test_uniform_logits_stay_in_fp8_probability_range(lengths):
    """Uniformly large logits must not silently saturate the FP8 probability
    operands: the dedicated descriptor routes' range guard has to hand such
    tiles to the exact retry (the denominator check alone cannot see it)."""
    inputs = make_uniform_dense_prefix_inputs(lengths)
    out = run_api(inputs)
    ref = run_ref(inputs)
    assert_close_at_gate(out, ref)
    assert float((out.float() - 0.5).abs().max()) <= MAX_ABS_ERR


def test_auto_reaches_specialized_by_default(monkeypatch):
    """The specialized backend is a DEFAULT auto candidate: backend="auto"
    routes to it with FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS unset or 0
    (the opt-in gates only experimental-classed backends, such as cake)."""
    from flashinfer import sparse as sparse_mod

    inputs = make_inputs((64,), seed=21)
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "0")
    before = sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"]
    auto = run_api(inputs, backend="auto")
    assert (
        sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"]
        == before + 1
    )
    spec = run_api(inputs, backend="specialized")
    assert torch.equal(auto, spec)


def test_wider_query_magnitude_stays_at_gate():
    """No dispatched route may assume the queries' magnitude: x4-scaled
    queries (which would saturate a fixed x64 FP8 scale) must still track the
    reference on the shape whose fixed-scale route was excluded from dispatch.

    The gate is magnitude-scaled: every dispatched route quantizes queries
    with a dynamic (per-row/per-tile max-based) or relative-precision FP8
    scale, so the absolute score error — and with it the output's absolute
    error — grows linearly with the input magnitude even though nothing
    clips. A relative-dominant match plus an abs cap scaled by the x4
    magnitude ratio is therefore the calibrated form of the base gate here;
    a fixed quantization scale on the route would saturate (|q| > 3.75 at
    the excluded route's x64 constant) and fail even this scaled gate by
    orders of magnitude.
    """
    mag_ratio = 4.0  # q_scale 2.0 vs the serve-distribution 0.5
    inputs = make_inputs((768,), seed=17, q_scale=2.0)
    out = run_api(inputs)
    ref = run_ref(inputs)
    assert_close_at_gate(
        out,
        ref,
        atol=MATCH_ATOL * mag_ratio,
        rtol=0.05,
        abs_cap=MAX_ABS_ERR * mag_ratio,
    )


def test_preallocated_out_and_determinism():
    inputs = make_inputs((768,), seed=7)
    out = torch.full_like(inputs["q"], 123.0)
    ret = run_api(inputs, out=out)
    assert ret.data_ptr() == out.data_ptr()
    again = run_api(inputs)
    assert torch.equal(out, again)  # bitwise repeatable, split-K included
    multi = make_inputs((300, 212, 180), seed=8)
    assert torch.equal(run_api(multi), run_api(multi))


def test_warmup_then_graph_capture():
    """The documented capture discipline: one same-shape eager warmup call,
    then capture replays bitwise-identically to eager execution."""
    # The docstring must carry the warmup-before-capture claim.
    doc = " ".join((qsa_sparse_paged_attention.__doc__ or "").lower().split())
    assert "before capturing" in doc
    inputs = make_inputs((768,), seed=5)
    out = torch.empty_like(inputs["q"])
    eager = run_api(inputs).clone()  # warmup pre-sizes scratch
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_api(inputs, out=out)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_graph_replay_survives_larger_eager_call():
    """A graph captured at one shape must still replay correctly after a
    LARGER eager call grows the persistent scratch: every buffer a capture
    recorded is retained for the process lifetime (the graph baked in its
    addresses), so growth replaces cache entries without freeing them."""
    small = make_inputs((512,), seed=11)
    big = make_inputs((1024,), seed=12)
    out = torch.empty_like(small["q"])
    eager = run_api(small).clone()  # same-shape warmup pre-sizes scratch
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_api(small, out=out)
    run_api(big)  # grows the split-K scratch after the capture
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_backend_kwarg_explicit_matches_auto():
    inputs = make_inputs((300, 212), seed=5)
    auto = run_api(inputs, backend="auto")
    spec = run_api(inputs, backend="specialized")
    assert torch.equal(auto, spec)
    with pytest.raises(ValueError, match="backend"):
        run_api(inputs, backend="fa2")


def test_explicit_default_never_swapped():
    """backend='default' on SM107 must keep reporting the unsupported device,
    never silently run the specialized kernel."""
    from flashinfer import sparse as sparse_mod

    inputs = make_inputs((64,), seed=12)
    before = sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"]
    with pytest.raises(RuntimeError, match="SM80 through SM90"):
        run_api(inputs, backend="default")
    assert sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"] == before


def test_kill_switch_forces_refusal(monkeypatch):
    """FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1 must never dispatch the
    specialized kernel: 'auto' falls through to 'default' (which reports the
    unsupported device on SM107, as before this backend existed) and explicit
    'specialized' raises NotImplementedError; the reason is recorded."""
    from flashinfer import sparse as sparse_mod

    inputs = make_inputs((64,), seed=9)
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    before_reasons = len(sparse_mod.qsa_sparse_paged_attention_fallback_reasons)
    before_stats = sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"]
    with pytest.raises(RuntimeError, match="SM80 through SM90"):
        run_api(inputs, backend="auto")
    with pytest.raises(NotImplementedError, match="DISABLE"):
        run_api(inputs, backend="specialized")
    reasons = sparse_mod.qsa_sparse_paged_attention_fallback_reasons
    assert len(reasons) == before_reasons + 2
    assert all("DISABLE" in r for r in reasons[-2:])
    assert (
        sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"]
        == before_stats
    )


def test_dispatch_stats_recorded():
    from flashinfer import sparse as sparse_mod

    inputs = make_inputs((64,), seed=10)
    before = sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"]
    run_api(inputs)
    assert (
        sparse_mod.qsa_sparse_paged_attention_dispatch_stats["specialized"]
        == before + 1
    )


def test_out_of_scope_calls_raise_not_implemented():
    """Valid-for-the-op calls outside the specialized contract are refused by
    name under explicit backend='specialized' (never served by another
    backend silently)."""
    inputs = make_inputs((64,), seed=11)
    # BF16 cache: upstream-valid, outside the specialized scope.
    bf16 = dict(inputs)
    bf16["k_cache"] = torch.randn_like(inputs["k_cache"], dtype=torch.bfloat16)
    bf16["v_cache"] = torch.randn_like(inputs["v_cache"], dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="float8_e4m3fn"):
        run_api(bf16, backend="specialized")
    # Wrong head count for this backend (12 heads is upstream-valid).
    hq12 = dict(inputs)
    hq12["q"] = torch.cat([inputs["q"], inputs["q"]], dim=1).contiguous()
    hq12["output_gate"] = torch.cat(
        [inputs["output_gate"], inputs["output_gate"]], dim=1
    ).contiguous()
    with pytest.raises(NotImplementedError, match="queries"):
        run_api(hq12, backend="specialized")
    # Non-unit scales.
    with pytest.raises(NotImplementedError, match="unit k_scale"):
        qsa_sparse_paged_attention(
            inputs["q"],
            inputs["k_cache"],
            inputs["v_cache"],
            inputs["logical_indices"],
            inputs["block_table"],
            inputs["token_to_req"],
            True,
            None,
            2.0,
            1.0,
            output_gate=inputs["output_gate"],
            backend="specialized",
        )
    # Narrower selection width (upstream-valid, outside this contract).
    narrow = dict(inputs)
    narrow["logical_indices"] = inputs["logical_indices"][:, -8:].contiguous()
    with pytest.raises(NotImplementedError, match="selections"):
        run_api(narrow, backend="specialized")


def test_contract_violations_raise_value_error():
    inputs = make_inputs((64,), seed=13)
    bad_q = dict(inputs)
    bad_q["q"] = inputs["q"].float()
    bad_q["output_gate"] = inputs["output_gate"]
    with pytest.raises((ValueError, NotImplementedError), match="bfloat16|queries"):
        run_api(bad_q, backend="specialized")
    bad_req = dict(inputs)
    bad_req["token_to_req"] = inputs["token_to_req"][:-1]
    with pytest.raises(ValueError, match="token_to_req"):
        run_api(bad_req, backend="specialized")
    bad_gate = dict(inputs)
    strided = torch.zeros(
        inputs["q"].shape[0], HQ, D + 16, dtype=torch.bfloat16, device="cuda"
    )[:, :, :D]
    assert strided.shape == inputs["q"].shape and not strided.is_contiguous()
    bad_gate["output_gate"] = strided
    with pytest.raises(ValueError, match="output_gate"):
        run_api(bad_gate, backend="specialized")
