"""
Copyright (c) 2025 by FlashInfer team.

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

import math
import os
import statistics

import pytest
import torch

import flashinfer
from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm100a_supported

# ---------------------------------------------------------------------------
# Hardware gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="VSA Blackwell backend requires sm100a (Blackwell GPU)",
)

# BSA blk128 kernel block granularity: 128-token Q/KV blocks
R = C = 128
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_random_bsr(MB: int, NB: int, density: float, device: torch.device):
    """Return (indptr, indices) for a random BSR pattern; every row has >= 1 block."""
    rows = []
    for _ in range(MB):
        k = max(1, int(round(density * NB)))
        k = min(k, NB)
        col_indices = torch.randperm(NB, device="cpu")[:k].sort().values
        rows.append(col_indices)

    indptr = torch.zeros(MB + 1, dtype=torch.int32)
    indices_list = []
    for i, row in enumerate(rows):
        indptr[i + 1] = indptr[i] + len(row)
        indices_list.append(row)

    indices = torch.cat(indices_list).to(torch.int32)
    return indptr.to(device), indices.to(device)


def _bsr_to_dense_mask(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    MB: int,
    NB: int,
    R: int,
    C: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand BSR sparsity pattern into a token-level boolean mask [M, N]."""
    mask = torch.zeros(MB * R, NB * C, dtype=torch.bool, device=device)
    indptr_cpu = indptr.cpu()
    indices_cpu = indices.cpu()
    for i in range(MB):
        s, e = int(indptr_cpu[i]), int(indptr_cpu[i + 1])
        for j_blk in indices_cpu[s:e].tolist():
            mask[i * R : i * R + R, j_blk * C : j_blk * C + C] = True
    return mask


def _pytorch_ref(
    q: torch.Tensor,  # [M, H, D]
    k: torch.Tensor,  # [N, H, D]
    v: torch.Tensor,  # [N, H, D]
    indptr: torch.Tensor,
    indices: torch.Tensor,
    R: int,
    C: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """Dense PyTorch reference for block-sparse attention."""
    M, H, D = q.shape
    N = k.shape[0]
    MB, NB = M // R, N // C
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    mask = _bsr_to_dense_mask(indptr, indices, MB, NB, R, C, q.device)

    qf = q.float().permute(1, 0, 2)  # [H, M, D]
    kf = k.float().permute(1, 0, 2)  # [H, N, D]
    vf = v.float().permute(1, 0, 2)  # [H, N, D]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale  # [H, M, N]
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)  # [H, M, D]
    return out.permute(1, 0, 2).to(q.dtype)  # [M, H, D]


@pytest.fixture(scope="module")
def workspace():
    return torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _make_wrapper(workspace):
    return BlockSparseAttentionWrapper(workspace, backend="vsa_blackwell")


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,density,num_blocks,num_heads",
    [
        (torch.bfloat16, 0.25, 16, 8),
        (torch.bfloat16, 0.75, 4, 8),
        (torch.float16, 0.25, 16, 8),
        (torch.float16, 0.75, 4, 8),
    ],
)
def test_vsa_accuracy(dtype, density, num_blocks, num_heads, workspace):
    """VSA output must match PyTorch dense reference."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    M = N = num_blocks * R
    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)

    o_ref = _pytorch_ref(q, k, v, indptr, indices, R, C)

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("sm_scale", [0.5])
def test_vsa_sm_scale(sm_scale, workspace):
    """User-supplied sm_scale must propagate correctly."""
    device = torch.device("cuda")
    torch.manual_seed(4)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    o_ref = _pytorch_ref(q, k, v, indptr, indices, R, C, sm_scale=sm_scale)

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
        sm_scale=sm_scale,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


def test_vsa_vs_auto_80k(workspace):
    """Cross-validate VSA backend against auto backend at 80k context (1250 blocks × 64).

    PyTorch dense reference is skipped at this scale to avoid OOM.
    """
    device = torch.device("cuda")
    torch.manual_seed(8)
    num_heads = 8
    num_blocks = 625  # 625 * 128 = 80 000 tokens
    density = 0.01
    dtype = torch.bfloat16

    M = N = num_blocks * R
    assert M == 80_000

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)

    ref_w = BlockSparseAttentionWrapper(workspace, backend="auto")
    ref_w.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    o_ref = ref_w.run(q, k, v)

    vsa_w = BlockSparseAttentionWrapper(workspace, backend="vsa_blackwell")
    vsa_w.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    o_vsa = vsa_w.run(q, k, v)

    torch.testing.assert_close(o_ref.float(), o_vsa.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Per-head mask accuracy tests
# ---------------------------------------------------------------------------


def test_vsa_per_head_mask_correctness(workspace):
    """Per-head block_mask path must match PyTorch dense reference per head."""
    device = torch.device("cuda")
    torch.manual_seed(10)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    # Construct distinct per-head masks
    block_mask = torch.zeros(
        num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
    )
    for h in range(num_heads):
        # Each head attends to a different set of blocks
        chosen = torch.randperm(num_blocks)[: max(1, num_blocks // 2)]
        block_mask[h, :, chosen] = True

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        None,
        None,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
        block_mask=block_mask,
    )
    o_vsa = wrapper.run(q, k, v)

    # PyTorch reference: compute each head independently using its own mask
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    o_ref = torch.empty_like(o_vsa)
    for h in range(num_heads):
        qh = q[:, h, :].float()  # [M, D]
        kh = k[:, h, :].float()  # [N, D]
        vh = v[:, h, :].float()  # [N, D]
        # Expand block_mask[h] to token-level
        token_mask = torch.zeros(M, N, dtype=torch.bool, device=device)
        for qi in range(num_blocks):
            for ki in range(num_blocks):
                if block_mask[h, qi, ki]:
                    token_mask[qi * R : (qi + 1) * R, ki * C : (ki + 1) * C] = True
        scores = torch.matmul(qh, kh.t()) * sm_scale  # [M, N]
        scores = scores.masked_fill(~token_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o_ref[:, h, :] = torch.matmul(probs, vh).to(dtype)

    torch.testing.assert_close(o_ref, o_vsa, atol=1e-2, rtol=1e-2)


def test_vsa_per_head_mask_differs_across_heads(workspace):
    """Per-head masks produce different per-head outputs; head-averaged BSR cannot replicate."""
    device = torch.device("cuda")
    torch.manual_seed(11)
    num_heads, num_blocks = 4, 16
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    # Maximally different masks: head h attends only to block h (mod num_blocks)
    block_mask = torch.zeros(
        num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
    )
    for h in range(num_heads):
        block_mask[h, :, h % num_blocks] = True

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        None,
        None,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
        block_mask=block_mask,
    )
    o_per_head = wrapper.run(q, k, v)

    # Head-averaged BSR (union of all per-head blocks)
    union_mask_bool = block_mask.any(dim=0)  # [MB, NB]
    nz = union_mask_bool.nonzero(as_tuple=False)
    indptr = torch.zeros(num_blocks + 1, dtype=torch.int32, device=device)
    row_counts = union_mask_bool.sum(dim=1).to(torch.int32)
    indptr[1:] = row_counts.cumsum(0)
    indices = nz[:, 1].to(torch.int32)

    wrapper2 = _make_wrapper(workspace)
    wrapper2.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    o_head_avg = wrapper2.run(q, k, v)

    # Per-head output must differ from head-averaged (they use different masks)
    assert not torch.allclose(o_per_head.float(), o_head_avg.float(), atol=1e-3), (
        "Per-head and head-averaged outputs should differ when masks are distinct per head"
    )


# ---------------------------------------------------------------------------
# Fastvideo-style coarse mask utilities
# ---------------------------------------------------------------------------


def _compute_vsa_compress_and_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    R: int,
    C: int,
    topk: int,
    sm_scale: float = None,
):
    """Fastvideo-style compress + coarse-mask step.

    Fully mirrors video_sparse_attn (__init__.py):
      1. Mean-pool Q, K, V within each 64-token block.
      2. Run dense attention on pooled tokens → block_attn_score [H, MB, NB],
         output_compress_pooled [H, MB, D].
      3. Broadcast output_compress back to full seq_len → [M, H, D].
      4. Top-K KV-block selection per (head, Q-block) from block_attn_score
         → per-head block_mask [H, MB, NB].

    Returns
    -------
    output_compress : torch.Tensor  [M, H, D]
        Global-context output from compressed attention (same dtype as q).
    block_mask : torch.Tensor  [H, MB, NB]  bool
        Per-head block-level selection mask for the VSA sparse branch.
    density : float
        Average fraction of selected KV blocks across heads.
    """
    M, H, D = q.shape
    N = k.shape[0]
    MB = M // R
    NB = N // C
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Mean-pool within each block → [H, MB/NB, D]
    q_pooled = q.view(MB, R, H, D).float().mean(dim=1).permute(1, 0, 2)  # [H, MB, D]
    k_pooled = k.view(NB, C, H, D).float().mean(dim=1).permute(1, 0, 2)  # [H, NB, D]
    v_pooled = v.view(NB, C, H, D).float().mean(dim=1).permute(1, 0, 2)  # [H, NB, D]

    # Dense attention on pooled tokens
    scores = (
        torch.matmul(q_pooled, k_pooled.transpose(-1, -2)) * sm_scale
    )  # [H, MB, NB]
    block_attn_score = torch.softmax(scores, dim=-1)  # [H, MB, NB]
    out_pooled = torch.matmul(block_attn_score, v_pooled)  # [H, MB, D]

    # Broadcast compress output: [H, MB, D] → [M, H, D]
    # Each pooled block result is replicated for all R tokens in the block.
    output_compress = (
        out_pooled.permute(1, 0, 2)  # [MB, H, D]
        .unsqueeze(1)  # [MB, 1, H, D]
        .expand(-1, R, -1, -1)  # [MB, R, H, D]
        .reshape(M, H, D)  # [M, H, D]
        .to(q.dtype)
    )

    # Top-K per (head, Q-block) from softmax block scores → block_mask [H, MB, NB]
    k_actual = min(topk, NB)
    topk_idx = torch.topk(block_attn_score, k_actual, dim=-1).indices  # [H, MB, topk]
    block_mask = torch.zeros(H, MB, NB, dtype=torch.bool, device=q.device)
    block_mask.scatter_(-1, topk_idx, True)

    density = block_mask.float().mean().item()
    return output_compress, block_mask, density


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between two [M, H, D] tensors (per position per head)."""
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return cos.mean().item()


# ---------------------------------------------------------------------------
# Two-stage VSA vs dense: accuracy and performance sweep
# ---------------------------------------------------------------------------

# (seqlen, topk_fraction)
_SWEEP_CONFIGS = [
    (16384, 0.50),
    (16384, 0.25),
    (16384, 0.10),
    (32768, 0.25),
    (32768, 0.10),
    (80000, 0.25),
    (80000, 0.10),
]


@pytest.mark.skipif(
    not os.environ.get("FLASHINFER_TEST_PERF"),
    reason="performance sweep, set FLASHINFER_TEST_PERF=1 to run",
)
@pytest.mark.parametrize("seqlen,topk_frac", _SWEEP_CONFIGS)
def test_vsa_accuracy_vs_dense(seqlen, topk_frac, workspace):
    """Accuracy: full fastvideo VSA (compress + select) vs full dense attention.

    Stage 1: compress attention on mean-pooled tokens → output_compress [M, H, D]
             + per-head block_mask [H, MB, NB] from softmax block scores.
    Stage 2: BlockSparseAttentionWrapper vsa_blackwell → output_select [M, H, D].
    Final:   output_compress + output_select  (mirrors fastvideo final_output).
    Dense:   flashinfer.single_prefill_with_kv_cache(backend="auto").

    Metrics reported:
      MAE    – mean absolute error per element
      CosSim – mean cosine similarity per (token, head) vector
    VSA output must be finite; no strict accuracy bound since it is an approximation.
    """
    assert seqlen % R == 0, "seqlen must be a multiple of R=128"
    device = torch.device("cuda")
    torch.manual_seed(42)

    num_heads = 8
    num_blocks = seqlen // R
    topk = max(1, int(round(topk_frac * num_blocks)))
    dtype = torch.bfloat16

    q = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)

    # Stage 1: compress attention + per-head block mask
    output_compress, block_mask, actual_density = _compute_vsa_compress_and_mask(
        q,
        k,
        v,
        R,
        C,
        topk,
    )

    # Stage 2: block-sparse attention (select branch)
    vsa_w = BlockSparseAttentionWrapper(workspace, backend="vsa_blackwell")
    vsa_w.plan(
        None,
        None,
        seqlen,
        seqlen,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
        block_mask=block_mask,
    )
    output_select = vsa_w.run(q, k, v)

    # Combine: compress + select  (fastvideo: final_output = output_compress + output_select)
    o_vsa = output_compress + output_select

    # Dense reference
    o_dense = flashinfer.single_prefill_with_kv_cache(q, k, v, backend="auto")

    assert not torch.isnan(o_vsa).any(), "VSA output contains NaN"
    assert not torch.isinf(o_vsa).any(), "VSA output contains Inf"

    mae = (o_vsa.float() - o_dense.float()).abs().mean().item()
    cos_sim = _cosine_sim(o_vsa, o_dense)
    print(
        f"\n[accuracy] seqlen={seqlen:>6}  density={actual_density:.3f}"
        f"  MAE={mae:.5f}  CosSim={cos_sim:.4f}"
    )


@pytest.mark.skipif(
    not os.environ.get("FLASHINFER_TEST_PERF"),
    reason="performance benchmark, set FLASHINFER_TEST_PERF=1 to run",
)
def test_vsa_performance_vs_dense(workspace):
    """Performance table: full fastvideo VSA (compress + select) vs dense FlashInfer prefill.

    VSA timing is broken into two parts:
      compress_ms – Stage 1: compress attention (pool Q/K/V, dense attn on pooled tokens,
                             softmax top-K → block_mask).  This is the "free" byproduct:
                             output_compress is also produced here.
      attn_ms     – Stage 2: BlockSparseAttentionWrapper vsa_blackwell (output_select).
      total_ms    – compress_ms + attn_ms  (output_compress + output_select = final output)
    Dense:
      dense_ms    – flashinfer.single_prefill_with_kv_cache (auto backend)
    Speedup = dense_ms / total_ms.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)

    num_heads = 8
    dtype = torch.bfloat16

    header = (
        f"\n{'seqlen':>8}  {'density':>8}  "
        f"{'dense_ms':>10}  {'cmp_ms':>8}  {'attn_ms':>9}  {'total_ms':>10}  {'speedup':>8}"
    )
    sep = "-" * (len(header) - 1)
    print(header)
    print(sep)

    for seqlen, topk_frac in _SWEEP_CONFIGS:
        num_blocks = seqlen // R
        topk = max(1, int(round(topk_frac * num_blocks)))

        q = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
        k = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
        v = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)

        # Pre-run: get block_mask and plan wrapper
        output_compress, block_mask, actual_density = _compute_vsa_compress_and_mask(
            q,
            k,
            v,
            R,
            C,
            topk,
        )
        vsa_w = BlockSparseAttentionWrapper(workspace, backend="vsa_blackwell")
        vsa_w.plan(
            None,
            None,
            seqlen,
            seqlen,
            R,
            C,
            num_heads,
            num_heads,
            HEAD_DIM,
            q_data_type=dtype,
            block_mask=block_mask,
        )

        # Time Stage 1: compress attention (pool + dense attn on MB tokens + topk)
        compress_times = bench_gpu_time(
            lambda: _compute_vsa_compress_and_mask(q, k, v, R, C, topk),
        )

        # Time Stage 2: VSA sparse attention kernel
        attn_times = bench_gpu_time(lambda: vsa_w.run(q, k, v))

        # Time dense reference
        dense_times = bench_gpu_time(
            lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, backend="auto"),
        )

        compress_ms = statistics.median(compress_times)
        attn_ms = statistics.median(attn_times)
        total_ms = compress_ms + attn_ms
        dense_ms = statistics.median(dense_times)
        speedup = dense_ms / total_ms

        print(
            f"{seqlen:>8}  {actual_density:>8.3f}  "
            f"{dense_ms:>10.3f}  {compress_ms:>8.3f}  {attn_ms:>9.3f}  "
            f"{total_ms:>10.3f}  {speedup:>7.2f}x"
        )

    print(sep)


# ===========================================================================
# blk64 tests — BSA blk64 C++ kernel (kSparseBlockSize=64, kRows=64)
# ===========================================================================

R64 = C64 = 64  # blk64 block granularity
HEAD_DIM_BLK64 = 128  # blk64 kernel requires head_dim=128


def _make_wrapper_blk64(workspace):
    return BlockSparseAttentionWrapper(workspace, backend="vsa_blackwell_blk64")


@pytest.mark.parametrize(
    "density,num_blocks,num_heads",
    [
        (0.25, 16, 8),
        (0.75, 4, 8),
    ],
)
def test_vsa_blk64_accuracy(density, num_blocks, num_heads, workspace):
    """blk64 output must match PyTorch dense reference (bfloat16 only)."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    dtype = torch.bfloat16

    M = N = num_blocks * R64
    q = torch.randn(M, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)

    o_ref = _pytorch_ref(q, k, v, indptr, indices, R64, C64)

    wrapper = _make_wrapper_blk64(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R64,
        C64,
        num_heads,
        num_heads,
        HEAD_DIM_BLK64,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref.float(), o.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "seqlen,topk_frac",
    [
        (4096, 0.5),
    ],
)
def test_vsa_blk64_accuracy_vs_dense(seqlen, topk_frac, workspace):
    """blk64 output must be close to dense attention for the selected blocks."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    num_heads = 8
    dtype = torch.bfloat16

    assert seqlen % R64 == 0
    MB = NB = seqlen // R64
    M = N = seqlen

    q = torch.randn(M, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(MB, NB, topk_frac, device)
    o_ref = _pytorch_ref(q, k, v, indptr, indices, R64, C64)

    wrapper = _make_wrapper_blk64(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R64,
        C64,
        num_heads,
        num_heads,
        HEAD_DIM_BLK64,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    assert torch.isfinite(o).all(), "blk64 output contains non-finite values"
    torch.testing.assert_close(o_ref.float(), o.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# blk64 performance: seqlen × density sweep
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FLASHINFER_TEST_PERF"),
    reason="performance benchmark, set FLASHINFER_TEST_PERF=1 to run",
)
def test_vsa_blk64_perf_sweep(workspace):
    """blk64 kernel throughput across seqlen in [1024, 2048, 4096] and density in [0.25, 0.5, 0.75]."""
    device = torch.device("cuda")
    torch.manual_seed(0)

    num_heads = 8
    dtype = torch.bfloat16
    seqlens = [1024, 2048, 4096]
    densities = [0.25, 0.5, 0.75]

    header = f"\n{'seqlen':>8}  {'density':>8}  {'active_blks':>12}  {'median_ms':>10}  {'tflops':>8}"
    sep = "-" * (len(header) - 1)
    print(header)
    print(sep)

    for seqlen in seqlens:
        num_blocks = seqlen // R64
        q = torch.randn(seqlen, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
        k = torch.randn(seqlen, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
        v = torch.randn(seqlen, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)

        for density in densities:
            indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)
            active_blocks = len(indices)

            wrapper = _make_wrapper_blk64(workspace)
            wrapper.plan(
                indptr,
                indices,
                seqlen,
                seqlen,
                R64,
                C64,
                num_heads,
                num_heads,
                HEAD_DIM_BLK64,
                q_data_type=dtype,
            )
            wrapper.run(q, k, v)  # warm-up

            times = bench_gpu_time(
                lambda w=wrapper, _q=q, _k=k, _v=v: w.run(_q, _k, _v)
            )
            ms = statistics.median(times)

            # FLOPs = 2 * (QK matmul + PV matmul) per active block pair
            flops = (
                2
                * 2
                * num_blocks
                * active_blocks
                * R64
                * C64
                * num_heads
                * HEAD_DIM_BLK64
            )
            tflops = flops / (ms * 1e-3) / 1e12

            actual_density = active_blocks / (num_blocks * num_blocks)
            print(
                f"{seqlen:>8}  {actual_density:>8.3f}  {active_blocks:>12}  {ms:>10.3f}  {tflops:>8.2f}"
            )

        print(sep)


# ---------------------------------------------------------------------------
# blk64 validation / rejection tests


# ===========================================================================
# New testcases: gaps vs. Block-Sparse-Attention/test_flash_fwd.py
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. GQA / MQA — blk128
# ---------------------------------------------------------------------------


def _pytorch_ref_gqa(
    q: torch.Tensor,  # [M, Hq, D]
    k: torch.Tensor,  # [N, Hkv, D]
    v: torch.Tensor,  # [N, Hkv, D]
    indptr: torch.Tensor,
    indices: torch.Tensor,
    R: int,
    C: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """PyTorch reference for GQA block-sparse attention.
    QO head h attends to KV head h // (Hq // Hkv), using the shared BSR mask.
    """
    M, Hq, D = q.shape
    Hkv = k.shape[1]
    qhead_per_kvhead = Hq // Hkv
    MB, NB = M // R, k.shape[0] // C
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    mask = _bsr_to_dense_mask(indptr, indices, MB, NB, R, C, q.device)  # [M, N]
    output = torch.empty_like(q)
    for h in range(Hq):
        h_kv = h // qhead_per_kvhead
        qh = q[:, h, :].float()
        kh = k[:, h_kv, :].float()
        vh = v[:, h_kv, :].float()
        scores = torch.matmul(qh, kh.t()) * sm_scale  # [M, N]
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output[:, h, :] = torch.matmul(probs, vh).to(q.dtype)
    return output


@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads,dtype",
    [
        (8, 4, torch.bfloat16),  # GQA 2x
        (8, 2, torch.bfloat16),  # GQA 4x
        (8, 1, torch.bfloat16),  # MQA
        (8, 4, torch.float16),  # GQA 2x, fp16
    ],
)
def test_vsa_gqa_native_bsr(num_qo_heads, num_kv_heads, dtype, workspace):
    """Native GQA via BSR path: plan() with num_qo_heads != num_kv_heads.

    Passes real (num_qo_heads, num_kv_heads) to wrapper.plan() and verifies
    against a per-QO-head PyTorch reference that maps each QO head to its
    corresponding KV head (h_kv = h // qhead_per_kvhead).
    """
    device = torch.device("cuda")
    torch.manual_seed(42)
    num_blocks = 8
    M = N = num_blocks * R

    q = torch.randn(M, num_qo_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    o_ref = _pytorch_ref_gqa(q, k, v, indptr, indices, R, C)

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads",
    [
        (8, 4),  # GQA 2x
        (8, 2),  # GQA 4x
    ],
)
def test_vsa_gqa_native_block_mask(num_qo_heads, num_kv_heads, workspace):
    """Native GQA via per-head block_mask: accepts (num_qo_heads, MB, NB) shape.

    Builds a per-KV-head mask, broadcasts to QO heads, passes the
    (num_qo_heads, MB, NB) tensor to plan(), and verifies each QO head
    uses the correct KV-head's sparsity pattern.
    """
    device = torch.device("cuda")
    torch.manual_seed(43)
    num_blocks = 8
    M = N = num_blocks * R
    dtype = torch.bfloat16
    qhead_per_kvhead = num_qo_heads // num_kv_heads

    q = torch.randn(M, num_qo_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)

    # Per-KV-head mask [Hkv, MB, NB], broadcast to [Hq, MB, NB]
    block_mask_kv = torch.zeros(
        num_kv_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
    )
    for h in range(num_kv_heads):
        chosen = torch.randperm(num_blocks)[: max(1, num_blocks // 2)]
        block_mask_kv[h, :, chosen] = True
    block_mask_qo = block_mask_kv.repeat_interleave(
        qhead_per_kvhead, dim=0
    )  # [Hq, MB, NB]

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        None,
        None,
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        q_data_type=dtype,
        block_mask=block_mask_qo,
    )
    o = wrapper.run(q, k, v)

    # PyTorch reference: each QO head uses its KV head's block mask
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    o_ref = torch.empty_like(o)
    for h in range(num_qo_heads):
        h_kv = h // qhead_per_kvhead
        qh = q[:, h, :].float()
        kh = k[:, h_kv, :].float()
        vh = v[:, h_kv, :].float()
        token_mask = torch.zeros(M, N, dtype=torch.bool, device=device)
        for qi in range(num_blocks):
            for ki in range(num_blocks):
                if block_mask_kv[h_kv, qi, ki]:
                    token_mask[qi * R : (qi + 1) * R, ki * C : (ki + 1) * C] = True
        scores = torch.matmul(qh, kh.t()) * sm_scale
        scores = scores.masked_fill(~token_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o_ref[:, h, :] = torch.matmul(probs, vh).to(dtype)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 2. head_dim variants (64, 96) — blk128
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "head_dim,density,num_blocks",
    [
        (64, 0.5, 8),
        (64, 0.25, 16),
        (96, 0.5, 8),
        (96, 0.25, 16),
    ],
)
def test_vsa_head_dim_accuracy(head_dim, density, num_blocks, workspace):
    """VSA blk128 with head_dim=64/96 must match PyTorch dense reference."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    num_heads = 4
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, head_dim, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)
    o_ref = _pytorch_ref(q, k, v, indptr, indices, R, C)

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        head_dim,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 3. seqlen_q != seqlen_k (cross-attention shape) — blk128
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "MB,NB,num_heads,density",
    [
        (4, 8, 8, 0.5),
        (8, 4, 4, 0.5),
        (2, 16, 4, 0.25),
    ],
)
def test_vsa_asymmetric_seqlen(MB, NB, num_heads, density, workspace):
    """VSA blk128 with seqlen_q != seqlen_k must match PyTorch dense reference."""
    device = torch.device("cuda")
    torch.manual_seed(13)
    M, N = MB * R, NB * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(MB, NB, density, device)
    o_ref = _pytorch_ref(q, k, v, indptr, indices, R, C)

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 4. seqlen_q != seqlen_k — blk64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "MB64,NB64,density",
    [
        (4, 8, 0.5),
        (8, 4, 0.5),
    ],
)
def test_vsa_blk64_asymmetric_seqlen(MB64, NB64, density, workspace):
    """VSA blk64 with seqlen_q != seqlen_k must match PyTorch dense reference."""
    device = torch.device("cuda")
    torch.manual_seed(14)
    num_heads = 4
    M, N = MB64 * R64, NB64 * R64
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(MB64, NB64, density, device)
    o_ref = _pytorch_ref(q, k, v, indptr, indices, R64, C64)

    wrapper = _make_wrapper_blk64(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R64,
        C64,
        num_heads,
        num_heads,
        HEAD_DIM_BLK64,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref.float(), o.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 5. LSE output validation — blk128
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,num_blocks,num_heads",
    [
        (torch.bfloat16, 8, 4),
        (torch.float16, 8, 4),
    ],
)
def test_vsa_return_lse(dtype, num_blocks, num_heads, workspace):
    """return_lse=True must produce LSE values consistent with the attention output."""
    device = torch.device("cuda")
    torch.manual_seed(20)
    M = N = num_blocks * R

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    # Reference LSE from PyTorch
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mask = _bsr_to_dense_mask(indptr, indices, num_blocks, num_blocks, R, C, device)
    qf = q.float().permute(1, 0, 2)  # [H, M, D]
    kf = k.float().permute(1, 0, 2)  # [H, N, D]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale  # [H, M, N]
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1).permute(1, 0)  # [M, H]

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    _, lse = wrapper.run(q, k, v, return_lse=True)

    # Only compare finite positions (rows with no KV blocks produce -inf LSE)
    finite = lse_ref.isfinite()
    assert finite.any(), "no finite LSE positions"
    torch.testing.assert_close(
        lse[finite].float(), lse_ref[finite].float(), atol=1e-2, rtol=1e-2
    )


# ---------------------------------------------------------------------------
# 6. LSE output validation — blk64
# ---------------------------------------------------------------------------


def test_vsa_blk64_return_lse(workspace):
    """blk64 return_lse=True must produce finite LSE values matching PyTorch reference."""
    device = torch.device("cuda")
    torch.manual_seed(21)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R64
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM_BLK64, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_BLK64)
    mask = _bsr_to_dense_mask(indptr, indices, num_blocks, num_blocks, R64, C64, device)
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1).permute(1, 0)  # [M, H]

    wrapper = _make_wrapper_blk64(workspace)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R64,
        C64,
        num_heads,
        num_heads,
        HEAD_DIM_BLK64,
        q_data_type=dtype,
    )
    _, lse = wrapper.run(q, k, v, return_lse=True)

    finite = lse_ref.isfinite()
    assert finite.any()
    torch.testing.assert_close(
        lse[finite].float(), lse_ref[finite].float(), atol=1e-2, rtol=1e-2
    )


# ---------------------------------------------------------------------------
# 7. Per-q-block variable KV-block count via block_mask — blk128
# ---------------------------------------------------------------------------


def test_vsa_variable_blocks_per_q(workspace):
    """Per-head block_mask with variable KV count per Q-block must match per-head PyTorch ref.

    Each Q-block gets a different number of attended KV blocks (row-varying density),
    exercising the variable block count path (q2k_num differs across Q-blocks).
    """
    device = torch.device("cuda")
    torch.manual_seed(30)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    # Each Q-block attends to i+1 KV blocks (varying count per row, same across heads)
    block_mask = torch.zeros(
        num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
    )
    for i in range(num_blocks):
        cnt = i + 1  # Q-block 0 → 1 block, Q-block 7 → 8 blocks
        chosen = torch.randperm(num_blocks)[:cnt]
        block_mask[:, i, chosen] = True

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        None,
        None,
        M,
        N,
        R,
        C,
        num_heads,
        num_heads,
        HEAD_DIM,
        q_data_type=dtype,
        block_mask=block_mask,
    )
    o_vsa = wrapper.run(q, k, v)

    # PyTorch reference per head
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    o_ref = torch.empty_like(o_vsa)
    for h in range(num_heads):
        qh = q[:, h, :].float()
        kh = k[:, h, :].float()
        vh = v[:, h, :].float()
        token_mask = torch.zeros(M, N, dtype=torch.bool, device=device)
        for qi in range(num_blocks):
            for ki in range(num_blocks):
                if block_mask[h, qi, ki]:
                    token_mask[qi * R : (qi + 1) * R, ki * C : (ki + 1) * C] = True
        scores = torch.matmul(qh, kh.t()) * sm_scale
        scores = scores.masked_fill(~token_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o_ref[:, h, :] = torch.matmul(probs, vh).to(dtype)

    torch.testing.assert_close(o_ref, o_vsa, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
