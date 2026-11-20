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
        (1024,),  # mid-range upper bound (general split path)
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
    packed = torch.cat([sel, (pos + 1).clamp_max(WIDTH).int().unsqueeze(1)], dim=1)
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


def make_dense_causal_cancellation_inputs(length):
    """Reviewer-reported cancellation construction on the mid-range path:
    every query repeats the BF16 pair ``(1.046875, -1)`` along the head dim,
    even cache token rows are all ones and odd rows all zeros, the gate is
    zero, and each row selects its full dense causal prefix. The exact scaled
    logit of a ones-row is ``128 * 0.046875 / 16 = 0.375``; a single
    dynamically scaled FP8 query loses that cancellation structure, so the
    query residual must be live on this route too."""
    dev = "cuda"
    T = length
    npages = (T + PAGE - 1) // PAGE
    block_table = torch.arange(npages, dtype=torch.int32, device=dev).unsqueeze(0)
    token_to_req = torch.zeros(T, dtype=torch.int32, device=dev)
    pos = torch.arange(T, device=dev)
    cols = torch.arange(WIDTH, device=dev, dtype=torch.int32)
    sel = torch.where(
        cols.unsqueeze(0) <= pos.unsqueeze(1),
        cols.unsqueeze(0).expand(T, WIDTH),
        torch.full((T, WIDTH), -1, dtype=torch.int32, device=dev),
    )
    packed = torch.cat([sel, (pos + 1).to(torch.int32).unsqueeze(1)], dim=1)
    q = torch.empty(T, HQ, D, dtype=torch.bfloat16, device=dev)
    q[..., 0::2] = 1.046875
    q[..., 1::2] = -1.0
    kv = torch.zeros(npages * PAGE, KV, D, device=dev)
    kv[0:T:2] = 1.0
    kv = kv.view(npages, PAGE, KV, D)
    return {
        "q": q,
        "k_cache": kv.to(torch.float8_e4m3fn),
        "v_cache": kv.to(torch.float8_e4m3fn),
        "logical_indices": packed,
        "block_table": block_table,
        "token_to_req": token_to_req,
        "output_gate": torch.zeros_like(q),
        "cu_seqlens": torch.tensor([0, T], dtype=torch.int32, device=dev),
        "page_base": torch.tensor([0], dtype=torch.int32, device=dev),
    }


@pytest.mark.parametrize("length", [512, 768, 1024])
def test_dense_causal_cancellation_mid_range(length):
    """Ordinary finite inputs, sorted complete causal selections, on the
    single-request 512..1024 split path: the output must stay inside the
    absolute error cap (requires the FP8 query residual on this route; a
    direct dynamically-scaled FP8 query exceeded the cap on SM107 silicon
    with ~all rows failing)."""
    inputs = make_dense_causal_cancellation_inputs(length)
    out = run_api(inputs)
    ref = run_ref(inputs)
    assert_close_at_gate(out, ref)
    # Even-count rows have an analytic expectation: m ones-rows at scaled
    # logit 0.375 against m zero-rows, V constant one, gate sigmoid(0):
    # bf16(sigmoid(0.375)) * 0.5 = 0.296875.
    even = (torch.arange(length, device=out.device) % 2) == 1
    if bool(even.any()):
        err = (out[even].float() - 0.296875).abs().max()
        assert float(err) <= MAX_ABS_ERR, f"even-prefix rows off by {float(err)}"


def run_route(inputs, route, monkeypatch):
    """Record a successful launch, not just a specialized API dispatch."""
    from flashinfer.triton.kernels import qsa_sparse_attention as mod

    name = {"w6": "_qsa_w6", "w7": "_qsa_w7_tma", "spv": "_qsa", "tiny": "_qsa_tiny"}[
        route
    ]
    kernel = getattr(mod, name)
    launches = []

    class Trace:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                result = kernel[grid](*args, **kw)
                launches.append(kw)
                return result

            return launch

    monkeypatch.setattr(mod, name, Trace())
    monkeypatch.setattr(mod, "_NO_WSPEC", {"w7": True} if route == "spv" else {})
    monkeypatch.setattr(mod, "_wspec_compile_failure", lambda exc: False)
    out = run_api(inputs, backend="specialized")
    assert len(launches) == 1
    if route == "spv":
        assert launches[0]["SPV"] and launches[0]["USEQL"]
    return out


@pytest.mark.parametrize(
    "route,lengths",
    [
        ("w6", (2029, 2028)),
        ("w6", (2046, 2045, 2045, 2045)),
        ("w7", (8191, 8191)),
        ("spv", (8191, 8191)),
    ],
)
def test_long_query_cancellation(route, lengths, monkeypatch, record_property):
    x = make_uniform_dense_prefix_inputs(lengths)
    q = x["q"]
    q[..., ::2] = 1.046875
    q[..., 1::2] = -1
    # Independent PV probe: four nonzero lanes, no Q cancellation.
    q[:, -1] = 0
    q[:, -1, :4] = 1.046875
    k = torch.zeros_like(x["k_cache"], dtype=torch.float32)
    k[:, ::2] = 1  # page-aligned even logical slots
    x["v_cache"] = (8 * k - 4).to(torch.float8_e4m3fn)
    p = x["logical_indices"]
    if route != "w6":
        # Both requests exercise full windows + masked tails, then sparse
        # [0,3] membership. Request 1 adds the SAME score to both key
        # classes, putting the cancelling heads above the zero-anchor band
        # and forcing the whole tile (including the PV probe) through redo.
        k[int(x["page_base"][1]) :, ..., ::2] += 1
        for start in (0, lengths[0]):
            tail = p[start + 256 : start + lengths[0]]
            tail.fill_(-1)
            tail[:, 0] = 0
            tail[:, 1] = 3
            tail[:, -1] = 2
    x["k_cache"] = k.to(torch.float8_e4m3fn)
    out = run_route(x, route, monkeypatch)

    # In either selection, ceil(count/2) even ids have value +4 and
    # floor(count/2) odd ids have value -4. The common logit cancels.
    count = p[:, -1:]
    delta = torch.full((HQ,), 0.375, device=q.device)
    delta[-1] = 4 * 1.046875 * SCALE
    even_mass = ((count + 1) // 2) * delta.exp()
    odd_mass = count // 2
    att = (4 * (even_mass - odd_mass) / (even_mass + odd_mass)).bfloat16()
    ref = (att.float() * 0.5).bfloat16().unsqueeze(-1).expand_as(out)
    err = (out.float() - ref.float()).abs()
    record_property("head_max_abs", err.amax(dim=(0, 2)).tolist())
    record_property("head_mean_abs", err.mean(dim=(0, 2)).tolist())
    assert_close_at_gate(out, ref)


@pytest.mark.parametrize(
    "route,lengths",
    [
        ("w6", (2046, 2045, 2045, 2045)),  # _gmul_bf
        ("w6", (2029, 2028)),  # _gmul
        ("tiny", (2,)),  # _gate_bf16
    ],
)
def test_positive_gate_absolute_accuracy(route, lengths, monkeypatch):
    x = make_uniform_dense_prefix_inputs(lengths)
    x["q"].zero_()  # uniform weights, no exponent redo
    dev = x["q"].device
    v = torch.tensor([256, -256, 4, -4, 1, -1, 2, -2], device=dev).repeat(D // 8)
    x["v_cache"] = v.expand_as(x["v_cache"]).to(torch.float8_e4m3fn).contiguous()
    # Include g=4 and positive gates near BF16 output rounding boundaries.
    # No large negative-gate lane may trigger the old whole-tile retry.
    g = ((torch.arange(HQ * D, device=dev).view(HQ, D) // 8 + 1) / 32).bfloat16()
    x["output_gate"].copy_(g)
    out = run_route(x, route, monkeypatch)
    ref = (v.float() * g.float().sigmoid()).bfloat16().expand_as(out)
    assert_close_at_gate(out, ref)


@pytest.mark.parametrize(
    "route,lengths",
    [("w6", (2046, 2045, 2045, 2045)), ("w7", (6001, 5999))],
)
def test_small_attention_gate_boundary(route, lengths, monkeypatch):
    x = make_uniform_dense_prefix_inputs(lengths)
    x["q"].zero_()
    x["v_cache"].fill_(2.0)  # Every nonempty prefix gives att_half = 1.
    gate = x["output_gate"]
    values = torch.tensor(
        [-16, -10.0625, -10, -9.9375, -9, -8, -6, -4, -1, -1 / 128, 0, 1 / 128, 4],
        dtype=torch.bfloat16,
        device=gate.device,
    )
    # Bands wider than a W6/W7 tile: -10 cannot trigger the tile redo on
    # behalf of every nearby negative gate, and nonnegative tiles stay packed.
    band = torch.arange(gate.shape[0], device=gate.device) // 128
    g = values[band % values.numel()][:, None, None]
    gate.copy_(g)
    out = run_route(x, route, monkeypatch)
    ref = (2.0 * g.float().sigmoid()).bfloat16().expand_as(out)
    assert_close_at_gate(out, ref)


def make_negative_gate_inputs(lengths):
    """Large in-contract attention values under a strongly negative gate:
    V is a constant 256 (E4M3-exact), K all ones (uniform softmax, so the
    attention output is exactly 256 for every selected row) and the gate is
    -10 everywhere. The true gated output is 256 * sigmoid(-10) ~ 0.0116; a
    half-precision tanh-form sigmoid collapses to 0 or ~2.4e-4 and misses the
    0.01 absolute cap on this small output."""
    inputs = make_inputs(lengths, seed=sum(lengths) + 5)
    inputs["k_cache"] = torch.ones_like(inputs["k_cache"].float()).to(
        torch.float8_e4m3fn
    )
    inputs["v_cache"] = torch.full_like(inputs["v_cache"].float(), 256.0).to(
        torch.float8_e4m3fn
    )
    inputs["output_gate"] = torch.full_like(inputs["output_gate"], -10.0)
    return inputs


@pytest.mark.parametrize(
    "lengths",
    [
        (2,),  # per-token exact route
        (96,),  # short-row route
        (768,),  # mid-range split path
        (300, 212, 180),  # general multi-request route
        (2029, 2028),  # dedicated descriptor route
    ],
)
def test_negative_gate_keeps_small_sigmoid_accuracy(lengths):
    """Every route's sigmoid gate must keep absolute accuracy for small
    values: with |att| = 256 (in the E4M3 value range) and gate -10, the
    output must stay inside the cap (requires an FP32-accurate sigmoid; the
    packed fp16/bf16 tanh form fails here by construction)."""
    inputs = make_negative_gate_inputs(lengths)
    out = run_api(inputs)
    ref = run_ref(inputs)
    assert_close_at_gate(out, ref)
    expected = 256.0 * torch.sigmoid(torch.tensor(-10.0)).item()
    has_sel = inputs["logical_indices"][:, -1] > 0
    err = (out[has_sel].float() - expected).abs().max()
    assert float(err) <= 1e-3, f"negative-gate output off by {float(err)}"


@pytest.mark.parametrize("readable", [False, True])
def test_foreign_allocator_during_failed_launch(monkeypatch, readable):
    """Model the process-global setter, including runtimes without a getter.

    An unrelated client reads the allocator *inside* the QSA launch, then
    installs its successor before QSA unwinds. Neither operation takes a QSA
    lock. Do not mutate the pytest process's real Triton allocator.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    from flashinfer.triton.kernels import qsa_sparse_attention as mod

    first = lambda size, alignment, stream: "foreign scratch"
    successor = lambda size, alignment, stream: "new scratch"
    state = SimpleNamespace(_allocator=first)
    monkeypatch.setattr(
        mod.triton,
        "set_allocator",
        lambda a: setattr(state, "_allocator", a),
        raising=False,
    )
    monkeypatch.setattr(
        mod, "_triton_allocation", state if readable else None, raising=False
    )
    monkeypatch.setattr(mod, "_LEGACY_ALLOC_INSTALLED", False, raising=False)
    entered, updated = Event(), Event()

    class FailLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                entered.set()
                assert updated.wait(10)
                raise RuntimeError("injected launch failure")

            return launch

    def foreign_client():
        assert entered.wait(10)
        try:
            observed = state._allocator
            mod.triton.set_allocator(successor)
            assert observed is first
            assert observed(128, 128, None) == "foreign scratch"
        finally:
            updated.set()

    monkeypatch.setattr(mod, "_qsa_tiny", FailLaunch())
    x = make_inputs((2,))
    with ThreadPoolExecutor(max_workers=1) as pool:
        other = pool.submit(foreign_client)
        with pytest.raises(RuntimeError, match="injected launch failure"):
            mod.run(
                x["q"],
                x["output_gate"],
                x["k_cache"],
                x["v_cache"],
                x["logical_indices"],
                x["cu_seqlens"],
                x["page_base"],
                torch.empty_like(x["q"]),
            )
        other.result()
    assert state._allocator is successor


@pytest.mark.parametrize("route,lengths", [("w6", (2029, 2028)), ("w7", (8191, 8191))])
def test_pointer_tiles_without_host_descriptors(monkeypatch, route, lengths):
    from flashinfer.triton.kernels import qsa_sparse_attention as mod

    monkeypatch.setattr(mod, "_HTMA", 0)
    x = make_uniform_dense_prefix_inputs(lengths)
    x["q"].zero_()  # exercise the hot sweep, not the large-logit retry
    out = run_route(x, route, monkeypatch)
    assert_close_at_gate(out, torch.full_like(out, 0.5))


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


@pytest.mark.parametrize("lengths", [(768,), (2029, 2028)])
def test_warmup_then_graph_capture(lengths):
    """The documented capture discipline: one same-shape eager warmup call,
    then capture replays bitwise-identically to eager execution."""
    # The docstring must carry the warmup-before-capture claim.
    doc = " ".join((qsa_sparse_paged_attention.__doc__ or "").lower().split())
    assert "before capturing" in doc
    inputs = make_inputs(lengths, seed=5)
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


@pytest.mark.parametrize("lengths", [(6001, 5999), (8001, 5999), (8192, 8191)])
def test_long_two_request_unequal_lengths_without_spare_pages(lengths, monkeypatch):
    inputs = make_inputs(lengths, seed=sum(lengths), spare_pages=0)
    out = run_route(inputs, "w7", monkeypatch)
    assert_close_at_gate(out, run_ref(inputs))
