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

---

Differential tests for the MSA CUDA-C++ -> CuTe-DSL kernel conversions.

Each converted kernel is checked against the original CUDA implementation
(which the 53-case ``test_msa_ops.py`` suite already validates) on randomized
inputs: bit-exact for the integer index kernels, tight tolerance for the
floating-point combine. The CUDA path is retained until its CuTe-DSL
replacement is green here and in ``test_msa_ops.py``, then deleted.
"""

import importlib

import pytest
import torch

from flashinfer.utils import is_sm12x_supported


def _skip_if_unsupported():
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("MSA ops require SM120 or SM121 and CUDA >= 12.8")


# the msa_sparse_attention *function* shadows the submodule on attribute import
def _sa():
    return importlib.import_module("flashinfer.msa_ops.sparse_attention")


# ===========================================================================
# Phase 1: sparse_combine  (CuTe-DSL vs CUDA)
# ===========================================================================

_COMBINE_DTYPES = [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float32, torch.bfloat16),
    (torch.float32, torch.float16),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.float16),
    (torch.float8_e4m3fn, torch.bfloat16),
    (torch.float8_e4m3fn, torch.float16),
]


def _rand_partials(topk, total_q, Hq, d, partial_dtype, dev, seed):
    torch.manual_seed(seed)
    base = torch.randn(topk, total_q, Hq, d, dtype=torch.float32, device=dev)
    if partial_dtype == torch.float8_e4m3fn:
        # keep magnitudes in the e4m3 sweet spot
        o_p = (base * 0.5).to(torch.float8_e4m3fn)
    else:
        o_p = base.to(partial_dtype)
    lse_p = torch.randn(topk, total_q, Hq, dtype=torch.float32, device=dev) * 4
    return o_p, lse_p


@pytest.mark.parametrize("partial_dtype,out_dtype", _COMBINE_DTYPES)
def test_combine_cudsl_vs_cuda_dtypes(partial_dtype, out_dtype):
    _skip_if_unsupported()
    sa = _sa()
    dev = "cuda"
    topk, total_q, Hq, G, d = 16, 200, 8, 4, 128
    Hkv = Hq // G
    o_p, lse_p = _rand_partials(topk, total_q, Hq, d, partial_dtype, dev, seed=11)
    counts = torch.randint(0, topk + 1, (total_q, Hkv), dtype=torch.int32, device=dev)

    cuda = sa._combine_partials_cuda(o_p, lse_p, counts, G, out_dtype)
    cudsl = sa._combine_partials_cudsl(o_p, lse_p, counts, G, out_dtype)
    torch.cuda.synchronize()

    # both quantize to the same low-precision out dtype from f32 weights; the
    # only divergence is exp2 (ex2.approx vs exp2f) -> a sub-ulp perturbation.
    diff = (cudsl.float() - cuda.float()).abs().max().item()
    assert diff < 5e-3, f"combine {partial_dtype}->{out_dtype} max|diff|={diff}"


@pytest.mark.parametrize("out_scale", [1.0, 0.375])
def test_combine_cudsl_vs_cuda_lse_and_scale(out_scale):
    _skip_if_unsupported()
    sa = _sa()
    dev = "cuda"
    topk, total_q, Hq, G, d = 16, 137, 8, 2, 128
    Hkv = Hq // G
    o_p, lse_p = _rand_partials(topk, total_q, Hq, d, torch.bfloat16, dev, seed=22)
    counts = torch.randint(0, topk + 1, (total_q, Hkv), dtype=torch.int32, device=dev)

    lse_cuda = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)
    lse_cudsl = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)
    o_cuda = sa._combine_partials_cuda(
        o_p, lse_p, counts, G, torch.bfloat16, lse_out=lse_cuda, out_scale=out_scale
    )
    o_cudsl = sa._combine_partials_cudsl(
        o_p, lse_p, counts, G, torch.bfloat16, lse_out=lse_cudsl, out_scale=out_scale
    )
    torch.cuda.synchronize()

    assert (o_cudsl.float() - o_cuda.float()).abs().max().item() < 5e-3
    # -inf rows (count==0) must match exactly; finite rows within log2/ln tol
    both_inf = torch.isinf(lse_cuda) & torch.isinf(lse_cudsl)
    finite = ~torch.isinf(lse_cuda)
    assert torch.equal(torch.isinf(lse_cuda), torch.isinf(lse_cudsl)), "inf mask"
    if finite.any():
        ld = (lse_cudsl[finite] - lse_cuda[finite]).abs().max().item()
        assert ld < 1e-3, f"lse max|diff|={ld}"
    assert both_inf.sum() + finite.sum() == lse_cuda.numel()


def test_combine_cudsl_vs_cuda_temperature_lse():
    _skip_if_unsupported()
    sa = _sa()
    dev = "cuda"
    topk, total_q, Hq, G, d = 8, 96, 4, 4, 128
    Hkv = Hq // G
    o_p, lse_p = _rand_partials(topk, total_q, Hq, d, torch.bfloat16, dev, seed=33)
    lse_t_p = torch.randn(topk, total_q, Hq, dtype=torch.float32, device=dev) * 3
    counts = torch.randint(0, topk + 1, (total_q, Hkv), dtype=torch.int32, device=dev)

    lse_cuda = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)
    lse_t_cuda = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)
    lse_cudsl = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)
    lse_t_cudsl = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)

    o_cuda = sa._combine_partials_cuda(
        o_p,
        lse_p,
        counts,
        G,
        torch.bfloat16,
        lse_out=lse_cuda,
        lse_t_partial=lse_t_p,
        lse_t_out=lse_t_cuda,
    )
    o_cudsl = sa._combine_partials_cudsl(
        o_p,
        lse_p,
        counts,
        G,
        torch.bfloat16,
        lse_out=lse_cudsl,
        lse_t_partial=lse_t_p,
        lse_t_out=lse_t_cudsl,
    )
    torch.cuda.synchronize()

    assert (o_cudsl.float() - o_cuda.float()).abs().max().item() < 5e-3
    for a, b, name in [
        (lse_cuda, lse_cudsl, "lse"),
        (lse_t_cuda, lse_t_cudsl, "lse_t"),
    ]:
        assert torch.equal(torch.isinf(a), torch.isinf(b)), f"{name} inf mask"
        fin = ~torch.isinf(a)
        if fin.any():
            d_ = (b[fin] - a[fin]).abs().max().item()
            assert d_ < 1e-3, f"{name} max|diff|={d_}"


def test_combine_cudsl_vs_cuda_fully_masked():
    """Rows where every selected slot has -inf LSE (count>0 but all masked):
    weights collapse to 0, output must be zero and LSE -inf in both paths."""
    _skip_if_unsupported()
    sa = _sa()
    dev = "cuda"
    topk, total_q, Hq, G, d = 16, 64, 4, 2, 128
    Hkv = Hq // G
    o_p, lse_p = _rand_partials(topk, total_q, Hq, d, torch.bfloat16, dev, seed=44)
    counts = torch.randint(1, topk + 1, (total_q, Hkv), dtype=torch.int32, device=dev)
    # force the first half of the rows fully -inf
    lse_p[:, : total_q // 2, :] = float("-inf")

    lse_cuda = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)
    lse_cudsl = torch.empty(total_q, Hq, dtype=torch.float32, device=dev)
    o_cuda = sa._combine_partials_cuda(
        o_p, lse_p, counts, G, torch.bfloat16, lse_out=lse_cuda
    )
    o_cudsl = sa._combine_partials_cudsl(
        o_p, lse_p, counts, G, torch.bfloat16, lse_out=lse_cudsl
    )
    torch.cuda.synchronize()

    assert torch.equal(o_cudsl.float(), o_cuda.float()), "masked output mismatch"
    assert torch.equal(torch.isinf(lse_cuda), torch.isinf(lse_cudsl))


# ===========================================================================
# Phase 2: build_k2q_csr / _schedule  (CuTe-DSL vs CUDA)
# ===========================================================================

BLK_KV = 128

_CSR_SHAPES = [
    (1, 2, 16, [37], [1024]),
    (3, 4, 16, [17, 64, 5], [512, 2048, 300]),
    (2, 1, 8, [100, 33], [4096, 700]),
    (2, 2, 32, [50, 21], [8192, 5000]),
    (4, 3, 4, [1, 2, 3, 4], [128, 256, 384, 129]),
    (1, 5, 16, [512], [600]),  # many heads, single batch
]


def _make_csr_inputs(B, H, topk, seqs_q, seqs_k, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    S_Q = int(cu_q[-1])
    q2k = torch.full((H, S_Q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        n = min(topk, nb)
        for h in range(H):
            for qi in range(lo, hi):
                sel = torch.randperm(nb)[:n].sort().values.to(torch.int32)
                q2k[h, qi, :n] = sel.to(dev)
    return q2k, cu_q, cu_k


@pytest.mark.parametrize("B,H,topk,seqs_q,seqs_k", _CSR_SHAPES)
def test_csr_basic_cudsl_vs_cuda(B, H, topk, seqs_q, seqs_k):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_build_k2q_csr

    q2k, cu_q, cu_k = _make_csr_inputs(B, H, topk, seqs_q, seqs_k, seed=7)

    rp_cuda, qi_cuda = msa_build_k2q_csr(q2k, cu_q, cu_k, _backend="cuda")
    rp_cudsl, qi_cudsl = msa_build_k2q_csr(q2k, cu_q, cu_k, _backend="cudsl")
    torch.cuda.synchronize()

    assert torch.equal(rp_cuda, rp_cudsl), "row_ptr mismatch"
    # q_indices is fully determined (ascending within each row), so bit-exact
    assert torch.equal(qi_cuda, qi_cudsl), "q_indices mismatch"


@pytest.mark.parametrize("B,H,topk,seqs_q,seqs_k", _CSR_SHAPES)
@pytest.mark.parametrize("target", [128, 64])
def test_csr_schedule_cudsl_vs_cuda(B, H, topk, seqs_q, seqs_k, target):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_build_k2q_csr_schedule

    q2k, cu_q, cu_k = _make_csr_inputs(B, H, topk, seqs_q, seqs_k, seed=9)

    a = msa_build_k2q_csr_schedule(
        q2k, cu_q, cu_k, target_q_per_cta=target, _backend="cuda"
    )
    b = msa_build_k2q_csr_schedule(
        q2k, cu_q, cu_k, target_q_per_cta=target, _backend="cudsl"
    )
    torch.cuda.synchronize()

    # fully-determined outputs: bit-exact
    assert torch.equal(a.row_ptr, b.row_ptr), "row_ptr"
    assert torch.equal(a.q_indices, b.q_indices), "q_indices"
    assert torch.equal(a.split_counts, b.split_counts), "split_counts"
    # qsplit_indices: the CUDA kernel leaves unused tail slots uninitialized
    # (the forward kernel only reads positions inside the CSR row ranges), so
    # compare only the valid entries (where q_indices != -1).
    valid = a.q_indices != -1
    assert torch.equal(a.qsplit_indices[valid], b.qsplit_indices[valid]), (
        "qsplit_indices (valid positions)"
    )

    wc_a = int(a.work_count.item())
    wc_b = int(b.work_count.item())
    assert wc_a == wc_b, f"work_count {wc_a} != {wc_b}"

    # scheduler_metadata: same set of work items, order is implementation-defined
    def _sorted_items(sched, n):
        items = sched[:n].cpu().tolist()
        return sorted(tuple(row) for row in items)

    assert _sorted_items(a.scheduler_metadata, wc_a) == _sorted_items(
        b.scheduler_metadata, wc_b
    ), "scheduler_metadata work-item set mismatch"


# ===========================================================================
# Phase 3: sparse_topk_select  (CuTe-DSL vs CUDA)
# ===========================================================================


@pytest.mark.parametrize(
    "H,P,S,nvp,fb,fe",
    [
        (2, 256, 64, 256, 0, 0),  # plain top-16, no clamp/force
        (2, 256, 64, 200, 0, 0),  # clamp only
        (2, 256, 64, 200, 3, 2),  # forced sink + window + clamp
        (1, 64, 40, 64, 4, 4),  # forced, full range
        (3, 128, 50, 128, 0, 0),  # more heads
        (2, 10, 32, 10, 0, 0),  # max_k_tiles < topk -> all valid, tail -1
        (2, 64, 16, 8, 0, 0),  # nvp < topk -> tail -1
    ],
)
def test_topk_cudsl_vs_cuda(H, P, S, nvp, fb, fe):
    """CUDA vs the CuTe-DSL radix port. Distinct random scores -> the selected
    set and its ascending-by-index order are fully determined, so the two are
    bit-exact."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    torch.manual_seed(13)
    dev = "cuda"
    max_score = torch.randn(H, P, S, dtype=torch.float32, device=dev)
    if nvp < P:
        max_score[:, nvp:, :] = float("-inf")

    def _run(backend):
        return msa_topk_select(
            max_score,
            16,
            num_valid_pages=nvp,
            force_begin_blocks=fb,
            force_end_blocks=fe,
            _backend=backend,
        )

    out_cuda = _run("cuda")
    out_radix = _run("cudsl_radix")
    torch.cuda.synchronize()

    ctx = f"H={H} P={P} S={S} nvp={nvp} fb={fb} fe={fe}"
    assert torch.equal(out_cuda, out_radix), (
        f"radix topk mismatch {ctx}\ncuda={out_cuda[0, 0]}\nradix={out_radix[0, 0]}"
    )


def _clustered_distinct(H, P, S, step_ulps, dev, seed):
    """Per-(head, token) column: a random permutation of ``P`` *distinct* floats
    packed into a narrow band starting at 1.0, spaced ``step_ulps`` ULPs apart.

    A narrow band makes many scores share the high radix digits of the
    order-preserving key, which overflows stage 1's threshold bin and forces the
    multi-stage refinement (stages 2/3). Distinct values keep the top-k boundary
    unambiguous, so the selected set (and its ascending-by-index order) is fully
    determined and must be bit-exact across backends.

    ``step_ulps == 1`` (consecutive representable floats) is maximally clustered:
    for ``P`` a power of two it collapses both the high (stage 1) and mid (stage
    2) digits to a single bin, so selection only resolves at the terminal stage 3.
    """
    ulp = 2.0**-23  # ULP of float32 in [1, 2)
    vals = 1.0 + torch.arange(P, dtype=torch.float64) * (step_ulps * ulp)
    vals = vals.to(torch.float32)
    assert vals.unique().numel() == P, "band too narrow: values collapsed"
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(H, P, S, dtype=torch.float32)
    for h in range(H):
        for s in range(S):
            out[h, :, s] = vals[torch.randperm(P, generator=g)]
    return out.to(dev)


def _topk_reference(max_score, topk, nvp, fb, fe):
    """Ground-truth MSA block selection for *distinct* scores (no tie ambiguity):
    forced sink/window blocks always selected, plus the top ``topk - n_forced``
    of the valid middle region by score, returned ascending-by-index with -1
    tail padding. Layout matches ``msa_topk_select``: (total_qo_len, H, topk)."""
    H, P, S = max_score.shape
    out = torch.full((S, H, topk), -1, dtype=torch.int32)
    forced = list(range(fb)) + list(range(nvp - fe, nvp))
    n_forced = len(forced)
    target = topk - n_forced
    mid = list(range(fb, nvp - fe))  # valid middle (invalid >= nvp excluded)
    for s in range(S):
        for h in range(H):
            chosen = []
            k = min(target, len(mid))
            if k > 0:
                mid_scores = max_score[h, mid, s]
                top = torch.topk(mid_scores, k).indices.tolist()
                chosen = [mid[i] for i in top]
            sel = sorted(forced + chosen)
            for i, v in enumerate(sel):
                out[s, h, i] = v
    return out.to(max_score.device)


@pytest.mark.parametrize(
    "H,P,S,step_ulps,nvp,fb,fe",
    [
        # step_ulps >= 4 keeps every value resolvable by the 30-bit radix
        # (it drops the bottom 2 bits), so the top-16 set is exact. The narrow
        # band still collapses the high digit -> stage 1 overflows -> the
        # selection resolves at stage 2 (the path that removes the 2048 cap).
        (2, 4096, 8, 4, 4096, 0, 0),  # 4 ULP, P=2^12
        (2, 4096, 8, 512, 4096, 0, 0),  # wider band
        (2, 3000, 8, 8, 3000, 0, 0),  # non-power-of-two count
        (2, 8192, 4, 4, 8192, 0, 0),  # P > 2x the old 2048 cap
        (2, 4096, 8, 8, 4000, 3, 2),  # + clamp + forced sink/window
        (1, 6000, 6, 64, 6000, 0, 0),  # single head, mid band
    ],
)
def test_topk_multistage_radix(H, P, S, step_ulps, nvp, fb, fe):
    """Exercises the multi-stage radix refinement (max_k_tiles > the old 2048
    staging cap, clustered-but-resolvable scores forcing stage 2). The top-16
    set is determined, so the radix port must match the torch reference.

    The CUDA kernel is cross-checked only for the no-forced regime: when forced
    (FLT_MAX) blocks coexist with a stage-0 overflow, CUDA's fp16 stage-0 emits
    them and its match-all fp32 stage-1 re-emits them, producing *duplicate*
    indices. The fp32-only port has no redundant match-all stage, so it is
    correct in that regime too (validated against the reference)."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    dev = "cuda"
    max_score = _clustered_distinct(H, P, S, step_ulps, dev, seed=101)
    if nvp < P:
        max_score[:, nvp:, :] = float("-inf")

    def _run(backend):
        return msa_topk_select(
            max_score,
            16,
            num_valid_pages=nvp,
            force_begin_blocks=fb,
            force_end_blocks=fe,
            _backend=backend,
        )

    ref = _topk_reference(max_score, 16, nvp, fb, fe)
    out_radix = _run("cudsl_radix")
    torch.cuda.synchronize()

    ctx = f"H={H} P={P} S={S} step={step_ulps} nvp={nvp} fb={fb} fe={fe}"
    assert torch.equal(ref, out_radix), (
        f"multi-stage radix vs reference mismatch {ctx}\n"
        f"ref  ={ref[0, 0]}\nradix={out_radix[0, 0]}"
    )

    if fb == 0 and fe == 0:
        out_cuda = _run("cuda")
        torch.cuda.synchronize()
        assert torch.equal(ref, out_cuda), (
            f"CUDA vs reference mismatch (no-forced regime) {ctx}\n"
            f"ref ={ref[0, 0]}\ncuda={out_cuda[0, 0]}"
        )


@pytest.mark.parametrize("nvp,fb,fe", [(4096, 0, 0), (4000, 3, 2)])
def test_topk_multistage_stage3_valid(nvp, fb, fe):
    """Drives the terminal stage 3: >2048 values spaced 1 ULP apart all share
    the top 30 radix bits, so stage 1 and stage 2 both collapse to one bin and
    selection only resolves at stage 3. Within a sub-4-ULP group the radix (like
    CUDA) cannot order by the bottom 2 bits, so the exact tie membership at the
    boundary is ambiguous; assert a *valid* top-k instead of exact equality:
    forced present, distinct + ascending, right count, and (tolerating the
    4-ULP radix resolution) no clearly-better middle block left unselected."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    H, P, S, topk = 2, 4096, 8, 16
    dev = "cuda"
    max_score = _clustered_distinct(H, P, S, step_ulps=1, dev=dev, seed=202)
    if nvp < P:
        max_score[:, nvp:, :] = float("-inf")

    out = msa_topk_select(
        max_score,
        topk,
        num_valid_pages=nvp,
        force_begin_blocks=fb,
        force_end_blocks=fe,
        _backend="cudsl_radix",
    ).reshape(S, H, topk)
    torch.cuda.synchronize()

    forced = set(range(fb)) | set(range(nvp - fe, nvp))
    mid = list(range(fb, nvp - fe))
    ulp = 2.0**-23
    for s in range(S):
        for h in range(H):
            sel = [int(x) for x in out[s, h] if x >= 0]
            assert len(sel) == len(set(sel)), f"dup at (s={s},h={h}): {sel}"
            assert sel == sorted(sel), f"not ascending at (s={s},h={h}): {sel}"
            assert len(sel) == min(topk, nvp), f"count at (s={s},h={h})"
            assert forced <= set(sel), f"forced missing at (s={s},h={h})"
            sel_mid = [i for i in sel if i in set(mid)]
            unsel_mid = [i for i in mid if i not in set(sel)]
            if sel_mid and unsel_mid:
                scores = max_score[h, :, s].cpu()
                # any "better" unselected block can only beat a selected one by
                # the unresolved bottom-2-bits (<= ~4 ULP near 1.0).
                gap = scores[unsel_mid].max() - scores[sel_mid].min()
                assert gap <= 4 * ulp + 1e-12, (
                    f"radix dropped a clearly-better block at (s={s},h={h}): "
                    f"gap={gap.item()}"
                )


def test_topk_tied_scores():
    """With deliberately tied scores the *exact* tie membership at the selection
    boundary is ambiguous (it depends on atomic emission order, and differs
    legitimately across backends). So instead of cross-backend equality we assert
    each backend produces a *valid* top-k per row: forced blocks present,
    ascending + distinct, the right count, and no unselected middle block scores
    strictly above a selected one."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    torch.manual_seed(7)
    dev = "cuda"
    H, P, S, nvp, fb, fe = 2, 128, 48, 120, 2, 2
    topk = 16
    # coarse quantization -> many exact ties among the per-block scores
    max_score = (torch.randn(H, P, S, device=dev) * 4).round() / 4
    max_score[:, nvp:, :] = float("-inf")
    forced = set(range(fb)) | set(range(nvp - fe, nvp))
    mid = list(range(fb, nvp - fe))

    def _assert_valid(out, backend):
        flat = out.reshape(H * S, topk).cpu()
        for r in range(flat.shape[0]):
            sel = [int(x) for x in flat[r] if x >= 0]
            assert len(sel) == len(set(sel)), f"{backend}: dup/order"
            assert sel == sorted(sel), f"{backend}: not ascending {sel}"
            assert len(sel) == min(topk, nvp), f"{backend}: count {len(sel)}"
            assert forced <= set(sel), f"{backend}: forced missing"

    for b in ("cuda", "cudsl_radix"):
        out = msa_topk_select(
            max_score,
            topk,
            num_valid_pages=nvp,
            force_begin_blocks=fb,
            force_end_blocks=fe,
            _backend=b,
        )
        torch.cuda.synchronize()
        _assert_valid(out, b)

    # valid-top-k by score (tie-break-independent): for the radix backend, the
    # min selected middle-score must be >= the max unselected middle-score.
    out = msa_topk_select(
        max_score,
        topk,
        num_valid_pages=nvp,
        force_begin_blocks=fb,
        force_end_blocks=fe,
        _backend="cudsl_radix",
    )
    torch.cuda.synchronize()
    out = out.reshape(S, H, topk).cpu()
    for s_idx in range(S):
        for h_idx in range(H):
            sel = {int(x) for x in out[s_idx, h_idx] if x >= 0}
            sel_mid = [i for i in sel if i in set(mid)]
            unsel_mid = [i for i in mid if i not in sel]
            if sel_mid and unsel_mid:
                scores = max_score[h_idx, :, s_idx].cpu()
                assert scores[sel_mid].min() >= scores[unsel_mid].max(), (
                    f"radix selected a worse middle block at (s={s_idx},h={h_idx})"
                )
