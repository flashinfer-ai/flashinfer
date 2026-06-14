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
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    torch.manual_seed(13)
    dev = "cuda"
    max_score = torch.randn(H, P, S, dtype=torch.float32, device=dev)
    if nvp < P:
        max_score[:, nvp:, :] = float("-inf")

    out_cuda = msa_topk_select(
        max_score,
        16,
        num_valid_pages=nvp,
        force_begin_blocks=fb,
        force_end_blocks=fe,
        _backend="cuda",
    )
    out_cudsl = msa_topk_select(
        max_score,
        16,
        num_valid_pages=nvp,
        force_begin_blocks=fb,
        force_end_blocks=fe,
        _backend="cudsl",
    )
    torch.cuda.synchronize()

    # ascending distinct indices with distinct random scores -> the selected set
    # and its order are fully determined, so bit-exact.
    assert torch.equal(out_cuda, out_cudsl), (
        f"topk mismatch H={H} P={P} S={S} nvp={nvp} fb={fb} fe={fe}\n"
        f"cuda={out_cuda[0, 0]}\ncudsl={out_cudsl[0, 0]}"
    )
