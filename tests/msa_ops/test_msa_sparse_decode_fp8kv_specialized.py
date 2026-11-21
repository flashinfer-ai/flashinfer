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

Tests for the SM103 "specialized" block-sparse MSA decode backend of
msa_sparse_decode_attention (bf16 Q over paged fp8 E4M3 KV): correctness vs
an fp32 reference, backend semantics (auto / specialized / cake, kill
switch, off-allowlist raise), out= handling, and CUDA-graph replay == eager.
"""

import os

import pytest
import torch

import flashinfer.msa_sparse_decode_fp8kv_cuda as spec_mod
from flashinfer.msa_ops import msa_sparse_decode_attention

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized MSA fp8-KV decode kernel is SM103-only",
)

HQ, D, BLOCK, TOPK = 16, 128, 128, 16
SCALE = 0.08838834764831845  # 128 ** -0.5


def _make_inputs(B, SEQ, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    pages = (SEQ + BLOCK - 1) // BLOCK
    total = B * pages
    k = (torch.randn(total, 1, BLOCK, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn(total, 1, BLOCK, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    q = (torch.randn(B, HQ, D, generator=g, device=dev)).to(torch.bfloat16)
    topk = torch.full((1, B, TOPK), -1, dtype=torch.int32, device=dev)
    n_valid = min(TOPK, pages)
    if pages <= TOPK:
        topk[0, :, :n_valid] = torch.arange(pages, dtype=torch.int32, device=dev)
    else:
        picks = torch.argsort(torch.rand(B, pages - 1, generator=g, device=dev), dim=1)[
            :, : TOPK - 1
        ]
        last = torch.full((B, 1), pages - 1, dtype=torch.int64, device=dev)
        topk[0] = torch.cat([picks, last], 1).sort(1).values.to(torch.int32)
    gp = torch.Generator().manual_seed(seed + 7)
    pt = torch.randperm(total, generator=gp).to(torch.int32).to(dev).reshape(B, pages)
    seq_lens = torch.full((B,), SEQ, dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, topk=topk, pt=pt.contiguous(), seq_lens=seq_lens)


def _reference(inp):
    """fp32 block-sparse GQA decode: request b attends its selected blocks
    (ascending ids, -1 padded) gathered through the page table, masked to
    position < seq_lens[b]."""
    dev = inp["q"].device
    B = inp["q"].shape[0]
    kf = inp["k"].reshape(-1, BLOCK, D).to(torch.float32)
    vf = inp["v"].reshape(-1, BLOCK, D).to(torch.float32)
    blk = inp["topk"][0].to(torch.int64)
    blk_c = blk.clamp_min(0)
    page_ids = inp["pt"].to(torch.int64).gather(1, blk_c)
    K = kf[page_ids].reshape(B, TOPK * BLOCK, D)
    V = vf[page_ids].reshape(B, TOPK * BLOCK, D)
    pos = (blk_c.unsqueeze(-1) * BLOCK + torch.arange(BLOCK, device=dev)).reshape(
        B, TOPK * BLOCK
    )
    valid = (blk >= 0).unsqueeze(-1).expand(-1, -1, BLOCK).reshape(B, TOPK * BLOCK) & (
        pos < inp["seq_lens"].to(torch.int64).unsqueeze(1)
    )
    qf = inp["q"].to(torch.float32)
    logits = torch.einsum("bhd,bwd->bhw", qf, K) * SCALE
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("bhw,bwd->bhd", p, V).to(torch.bfloat16)


def _api(inp, out=None, backend="auto"):
    return msa_sparse_decode_attention(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["topk"],
        page_table=inp["pt"],
        seqused_k=inp["seq_lens"],
        seqlen_q=1,
        causal=True,
        softmax_scale=SCALE,
        out=out,
        backend=backend,
    )


def _assert_close(out, gold):
    diff = (out.float() - gold).abs()
    tol = 0.004 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0, float(diff.max())


@requires_sm103
@pytest.mark.parametrize(
    "B,SEQ",
    [
        (1, 384),  # fewer visible blocks than TOPK (-1 padding live)
        (8, 8192),  # full top-16 selection
        (64, 384),  # large batch, small kv
        (4, 300),  # partial final page (position mask live)
        (8, 512),  # 4-page regime (general route family)
    ],
)
def test_specialized_correct_and_routed(B, SEQ):
    inp = _make_inputs(B, SEQ, seed=9)
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    _assert_close(out, gold)
    # explicit specialized == auto bitwise; out= is honoured and returned
    buf = torch.empty_like(inp["q"])
    out2 = _api(inp, out=buf, backend="specialized")
    assert out2 is buf
    assert torch.equal(out, out2)


@requires_sm103
@pytest.mark.parametrize(
    "B,SEQ", [(2, 384), (8, 384), (64, 384), (2, 2048), (4, 2048), (8, 512), (304, 512)]
)
def test_finite_negative_logits(B, SEQ):
    inp = _make_inputs(B, SEQ)
    # Constant V: every nonempty convex combination is exactly one. Row 0
    # sees just one token (including wholly masked ranks in split pages);
    # the last row has no selections. No second backend is the oracle.
    for name in ("k", "v"):
        inp[name] = torch.ones_like(inp[name], dtype=torch.float16).to(
            torch.float8_e4m3fn
        )
    inp["seq_lens"][0] = 1
    inp["topk"][0, 0, 1:] = -1
    inp["topk"][0, -1] = -1
    expected = torch.ones_like(inp["q"])
    expected[-1].zero_()
    for exponent in (90, 100):
        inp["q"].fill_(-(2.0**exponent))
        for backend in ("specialized", "auto"):
            out = torch.full_like(expected, float("nan"))
            before = spec_mod.DISPATCH_STATS["specialized"]
            assert _api(inp, out=out, backend=backend) is out
            assert spec_mod.DISPATCH_STATS["specialized"] == before + 1
            assert torch.equal(out, expected), (exponent, backend)


@requires_sm103
def test_backend_semantics():
    inp = _make_inputs(4, 300, seed=1)
    gold = _reference(inp).float()
    # backend="cake" (the Cake-generated Blackwell programs, the former
    # arch-dispatch "native" label) never routes to the specialized kernel
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp, backend="cake")
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    _assert_close(out, gold)
    # explicit specialized raises off-allowlist (fp16 q) with reasons
    bad = dict(inp)
    bad["q"] = inp["q"].to(torch.float16)
    with pytest.raises(ValueError, match="fallback_reasons"):
        _api(bad, backend="specialized")
    with pytest.raises(ValueError, match="backend must be one of"):
        _api(inp, backend="bogus")


@requires_sm103
def test_aliased_out_auto_matches_stock():
    inp = _make_inputs(4, 512)
    gold = _reference(inp).float()

    def run(backend):
        args = dict(inp, q=inp["q"].clone())
        return _api(args, backend=backend, out=args["q"])

    before = dict(spec_mod.DISPATCH_STATS)
    try:
        stock = run("cake")
    except (ValueError, NotImplementedError) as exc:
        with pytest.raises(type(exc)) as auto_exc:
            run("auto")
        assert str(auto_exc.value) == str(exc)
    else:
        _assert_close(stock, gold)
        assert torch.equal(run("auto"), stock)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert spec_mod.DISPATCH_STATS["fallback"] == before["fallback"] + 1
    assert "out_overlaps_q" in spec_mod.LAST_FALLBACK_REASONS


@requires_sm103
def test_kill_switch_restores_stock():
    inp = _make_inputs(4, 300, seed=2)
    gold = _reference(inp).float()
    os.environ[spec_mod.KILL_SWITCH_ENV] = "1"
    try:
        before = dict(spec_mod.DISPATCH_STATS)
        out = _api(inp)
        assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
        assert spec_mod.LAST_FALLBACK_REASONS[0].startswith("kill_switch")
        _assert_close(out, gold)
        with pytest.raises(ValueError, match="fallback_reasons"):
            _api(inp, backend="specialized")
    finally:
        os.environ.pop(spec_mod.KILL_SWITCH_ENV, None)


@pytest.fixture(scope="module")
def fault_module(tmp_path_factory):
    # Private test build: change launch geometry, not the production error
    # check or Python exception path. No test switches in the shipped module.
    from flashinfer.jit import msa_sparse_decode_fp8kv as jit

    tmp = tmp_path_factory.mktemp("msa_launch_fault")
    sources = []
    for name in jit._SOURCES:
        src = jit.jit_env.FLASHINFER_CSRC_DIR / name
        if name.endswith("_f16core.cu"):
            text = src.read_text()
            for old, new in (
                ("65535 / OSPLIT", "1"),
                (
                    "dim3 grid(CLU, chunk * OSPLIT);",
                    "dim3 grid(B == 1 || base == 1 ? 0 : CLU, chunk * OSPLIT);",
                ),
            ):
                assert text.count(old) == 1
                text = text.replace(old, new)
            src = tmp / name
            src.write_text(text)
        elif name.endswith("_dispatch.cu"):
            # Only the attribute operation is replaced. Two consecutive
            # failures on the same kernel prove the first wasn't cached.
            text = src.read_text()
            old = "cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)"
            assert text.count(old) == 1
            text = "#include <cstdlib>\n" + text.replace(
                old,
                f'(std::getenv("MSA_TEST_FAIL_ATTR") ? cudaErrorInvalidValue : {old})',
            )
            src = tmp / name
            src.write_text(text)
        sources.append(src)
    return jit.gen_jit_spec(
        "msa_sparse_decode_fp8kv_sm103_launch_fault",
        sources,
        extra_cuda_cflags=jit.sm103a_nvcc_flags + ["--use_fast_math", "-lineinfo"],
    ).build_and_load()


@requires_sm103
@pytest.mark.parametrize("backend", ["specialized", "auto"])
@pytest.mark.parametrize("B", [1, 2])
def test_f16_immediate_launch_rejection(fault_module, monkeypatch, B, backend):
    monkeypatch.setattr(
        spec_mod, "get_msa_sparse_decode_fp8kv_module", lambda: fault_module
    )
    inp = _make_inputs(B, 512)
    out = torch.full_like(inp["q"], float("nan"))
    gold = _reference(inp).float()
    # The call itself must throw, without a stream sync or graph replay.
    with pytest.raises(RuntimeError, match="specialized msa decode launch failed"):
        _api(inp, out=out, backend=backend)
    assert torch.isnan(out[-1]).all()
    if B == 2:
        _assert_close(out[:1], gold[:1])  # chunk 0 was already submitted


@requires_sm103
def test_failed_attribute_setup_not_cached(fault_module, monkeypatch):
    monkeypatch.setattr(
        spec_mod, "get_msa_sparse_decode_fp8kv_module", lambda: fault_module
    )
    # A different family than the rejected F16 launches: cold first use.
    inp = _make_inputs(64, 384)
    gold = _reference(inp).float()
    out = torch.full_like(inp["q"], float("nan"))
    monkeypatch.setenv("MSA_TEST_FAIL_ATTR", "1")
    for backend in ("auto", "specialized"):
        with pytest.raises(
            RuntimeError, match="setting dynamic shared memory.*cudaErrorInvalidValue"
        ):
            _api(inp, out=out, backend=backend)
        assert torch.isnan(out).all()
    monkeypatch.delenv("MSA_TEST_FAIL_ATTR")
    _assert_close(_api(inp, out=out, backend="specialized"), gold)


@requires_sm103
def test_cuda_graph_replay_matches_eager():
    inp = _make_inputs(8, 8192, seed=4)
    buf = torch.empty_like(inp["q"])
    _api(inp, out=buf)  # eager warmup builds the module before capture
    eager = buf.clone()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _api(inp, out=buf)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        _api(inp, out=buf)
    for _ in range(3):
        buf.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(buf, eager)


@requires_sm103
def test_backend_names_reflect_implementations():
    """Round-3 naming: no arbitrary labels. "cake"/"cutedsl" name real
    implementations; "native" is gone; "auto" only selects."""
    from flashinfer.msa_ops.sparse_decode import MSA_SPARSE_DECODE_BACKENDS

    assert MSA_SPARSE_DECODE_BACKENDS == ("auto", "specialized", "cake", "cutedsl")
    inp = _make_inputs(B=2, SEQ=256, seed=31)
    with pytest.raises(ValueError, match="backend must be one of"):
        _api(inp, backend="native")
    # explicit cutedsl on SM103 raises (it serves SM120/SM121 here)
    with pytest.raises(ValueError, match="cutedsl"):
        _api(inp, backend="cutedsl")
    # explicit cake on SM103 equals what auto selects off-allowlist and
    # the stock Blackwell path
    out_cake = _api(inp, backend="cake")
    gold = _reference(inp)
    diff = (out_cake.float() - gold.float()).abs()
    assert float((diff <= 0.02 + 0.05 * gold.float().abs()).float().mean()) >= 0.99


@requires_sm103
@pytest.mark.parametrize(
    "B,SEQ",
    [
        (4, 300),  # direct-K short route, partial final page
        (304, 512),  # 4-page F16 general route, splits=1 single-cluster epilogue
    ],
)
def test_empty_selection_rows_produce_zero_output(B, SEQ):
    """A query row whose q2k_indices are all -1 selects nothing; the
    specialized route must return ZERO for that row (the packed backend's
    empty-effective-KV convention), never 0/0 = NaN."""
    inp = _make_inputs(B, SEQ, seed=47)
    inp["topk"][0, 2, :] = -1
    gold = _reference(inp)
    out = _api(inp, backend="specialized")
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    row = out.float()[2]
    assert float(row.abs().max()) == 0.0
    # the other rows stay reference-close
    keep = [i for i in range(out.shape[0]) if i != 2]
    diff = (out.float()[keep] - gold.float()[keep]).abs()
    assert (
        float((diff <= 0.02 + 0.05 * gold.float()[keep].abs()).float().mean()) >= 0.99
    )


# ---------------------------------------------------------------------------
# Numerical-gap regressions (constructions from the PR review, each measured
# on SM103 silicon against the fp32 reference / stock arm before the fix):
# 1. bf16 Q exponent spread must survive QK staging,
# 2. partial numerators must cross the cluster merge in fp32,
# 3. NaN bytes in the unwritten tail of a partial last KV page must not
#    reach the output through the PV mma,
# 4. on single-CTA F16, probabilities far below the max must survive P staging.
# The (B, SEQ) pairs below cover every kernel TU the dispatcher can select:
# pages==3 -> c12 (B<=2) / direct-K b8 (B<=16) / direct-K short (else);
# pages>=16 -> b8 (B<=2) / direct-K TMA (else); otherwise the native-F16
# core (single-CTA at large B, clustered merge at small B).
# ---------------------------------------------------------------------------


@requires_sm103
@pytest.mark.parametrize(
    "B,SEQ",
    [
        (2, 384),  # c12
        (8, 384),  # direct-K short b8
        (64, 384),  # direct-K short
        (2, 2048),  # b8, full top-16 selection
        (4, 2048),  # direct-K TMA
        (8, 512),  # native-F16 core
    ],
)
def test_query_exponent_spread_preserved(B, SEQ):
    """A bf16 query mixing a 2^40 channel with a 16.0 channel: the small
    channel must survive QK staging (fp16-staged Q flushed it to zero and
    returned 0 where the reference gives ~0.887)."""
    inp = _make_inputs(B, SEQ, seed=11)
    dev = inp["q"].device
    pages = (SEQ + BLOCK - 1) // BLOCK
    total = B * pages
    q = torch.zeros(B, HQ, D, device=dev)
    q[:, :, 0] = 2.0**40
    q[:, :, 1] = 16.0
    inp["q"] = q.to(torch.bfloat16)
    k = torch.zeros(total, 1, BLOCK, D, device=dev)
    k[:, :, : BLOCK // 2, 1] = 1.0
    k[:, :, BLOCK // 2 :, 1] = -1.0
    v = torch.empty(total, 1, BLOCK, D, device=dev)
    v[:, :, : BLOCK // 2, :] = 1.0
    v[:, :, BLOCK // 2 :, :] = -1.0
    inp["k"] = k.to(torch.float8_e4m3fn)
    inp["v"] = v.to(torch.float8_e4m3fn)
    gold = _reference(inp).float()
    assert float(gold.abs().max()) > 0.5  # non-degenerate construction
    out = _api(inp, backend="specialized").float()
    assert float((out - gold).abs().max()) <= 0.02


@requires_sm103
@pytest.mark.parametrize(
    "B,SEQ",
    [
        (8, 256),  # native-F16 core, 2-CTA cluster merge
        (2, 384),  # c12, 12-CTA quarter-page merge
        (8, 384),  # direct-K short b8 merge
        (64, 384),  # direct-K short merge
        (4, 2048),  # direct-K TMA, 16-CTA merge
    ],
)
def test_partial_numerator_cancellation_survives_merge(B, SEQ):
    """Q=K=0 (uniform attention); V page of block 0 holds +64 with a single
    +60 token, block 1 holds -64 everywhere, later blocks hold 0.  The exact
    answer is -4/SEQ on every output element; bf16-rounded partial
    numerators cancelled the residual entirely and returned 0."""
    inp = _make_inputs(B, SEQ, seed=13)
    dev = inp["q"].device
    pages = (SEQ + BLOCK - 1) // BLOCK
    total = B * pages
    inp["q"] = torch.zeros(B, HQ, D, device=dev).to(torch.bfloat16)
    vf = torch.zeros(total, 1, BLOCK, D, device=dev)
    p0 = inp["pt"][:, 0].long()
    p1 = inp["pt"][:, 1].long()
    vf[p0] = 64.0
    vf[p0, 0, 0, :] = 60.0
    vf[p1] = -64.0
    inp["v"] = vf.to(torch.float8_e4m3fn)
    inp["k"] = torch.zeros_like(vf).to(torch.float8_e4m3fn)
    gold = _reference(inp).float()
    expect = -4.0 / SEQ
    assert abs(float(gold.mean()) - expect) < 1e-4  # construction sanity
    out = _api(inp, backend="specialized").float()
    assert float((out - gold).abs().max()) <= 2.5e-4


@requires_sm103
@pytest.mark.parametrize(
    "B,SEQ",
    [
        (2, 300),  # c12
        (8, 300),  # direct-K short b8
        (64, 300),  # direct-K short
        (32, 2020),  # direct-K TMA (16 pages, partial last)
        (8, 484),  # native-F16 core (4 pages, partial last)
    ],
)
def test_nan_bytes_in_partial_last_page_tail(B, SEQ):
    """The unwritten tail of a partial last KV page is arbitrary allocator
    memory.  E4M3 NaN bytes (0x7F) there must not reach the output: the
    score mask zeroes P, but 0 * NaN is NaN in the PV mma unless the staged
    V tail itself is zeroed."""
    inp = _make_inputs(B, SEQ, seed=17)
    gold = _reference(inp).float()
    out_clean = _api(inp, backend="specialized").clone()
    rem = SEQ % BLOCK
    assert rem != 0
    pages = (SEQ + BLOCK - 1) // BLOCK
    last_phys = inp["pt"][:, pages - 1].long()
    for t in (inp["k"], inp["v"]):
        tb = t.view(torch.uint8)
        tb[last_phys, 0, rem:, :] = 0x7F
    out = _api(inp, backend="specialized")
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out, out_clean)
    _assert_close(out, gold)


@requires_sm103
def test_small_probabilities_survive_f16_staging():
    """Single-CTA F16 route (B=160, 15 full pages, selections [0..14, -1]):
    nonpeak probabilities of ~1e-8 keep ~2000 V contributions of 448 alive
    in the fp32 denominator; flushing them to zero in f16 P staging returned
    0 where the reference gives ~0.0095."""
    B, SEQ = 160, 15 * BLOCK
    inp = _make_inputs(B, SEQ, seed=19)
    dev = inp["q"].device
    pages = SEQ // BLOCK
    total = B * pages
    q = torch.zeros(B, HQ, D, device=dev)
    q[:, :, 0] = 16.0
    inp["q"] = q.to(torch.bfloat16)
    k = torch.zeros(total, 1, BLOCK, D, device=dev)
    v = torch.full((total, 1, BLOCK, D), 448.0, device=dev)
    first_phys = inp["pt"][:, 0].long()
    k[first_phys, 0, 0, 0] = 13.0
    v[first_phys, 0, 0, :] = 0.0
    inp["k"] = k.to(torch.float8_e4m3fn)
    inp["v"] = v.to(torch.float8_e4m3fn)
    gold = _reference(inp).float()
    assert 0.005 <= float(gold.mean()) <= 0.02  # construction sanity
    out = _api(inp, backend="specialized").float()
    assert float((out - gold).abs().max()) <= 2e-4
