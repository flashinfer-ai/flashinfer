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
import statistics

import numpy as np
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

# VSA Blackwell kernel has fixed constraints:
R = C = 64
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
    q: torch.Tensor,     # [M, H, D]
    k: torch.Tensor,     # [N, H, D]
    v: torch.Tensor,     # [N, H, D]
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

    qf = q.float().permute(1, 0, 2)           # [H, M, D]
    kf = k.float().permute(1, 0, 2)           # [H, N, D]
    vf = v.float().permute(1, 0, 2)           # [H, N, D]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale  # [H, M, N]
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)             # [H, M, D]
    return out.permute(1, 0, 2).to(q.dtype)  # [M, H, D]


def _make_wrapper(device):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return BlockSparseAttentionWrapper(ws, backend="vsa_blackwell")


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,density,num_blocks,num_heads", [
    (torch.bfloat16, 0.25,  4, 1),
    (torch.bfloat16, 0.25, 16, 8),
    (torch.bfloat16, 0.75,  4, 8),
    (torch.bfloat16, 0.75, 16, 1),
    (torch.float16,  0.25,  4, 8),
    (torch.float16,  0.25, 16, 1),
    (torch.float16,  0.75,  4, 1),
    (torch.float16,  0.75, 16, 8),
])
def test_vsa_accuracy(dtype, density, num_blocks, num_heads):
    """VSA output must match PyTorch dense reference."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    M = N = num_blocks * R
    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)

    o_ref = _pytorch_ref(q, k, v, indptr, indices, R, C)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


def test_vsa_preallocated_output():
    """run(out=...) must write into the provided tensor."""
    device = torch.device("cuda")
    torch.manual_seed(1)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )

    o = wrapper.run(q, k, v)
    o_buf = torch.empty_like(o)
    wrapper.run(q, k, v, out=o_buf)
    torch.testing.assert_close(o, o_buf)


def test_vsa_return_lse():
    """return_lse=True must yield correctly shaped, finite LSE."""
    device = torch.device("cuda")
    torch.manual_seed(2)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )

    o, lse = wrapper.run(q, k, v, return_lse=True)

    assert o.shape == (M, num_heads, HEAD_DIM)
    assert lse.shape == (M, num_heads)
    assert lse.dtype == torch.float32
    assert not torch.isnan(lse).any()
    assert not torch.isinf(lse).any()


def test_vsa_preallocated_lse():
    """run(lse=...) must write LSE into the provided tensor."""
    device = torch.device("cuda")
    torch.manual_seed(3)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )

    _, lse = wrapper.run(q, k, v, return_lse=True)
    lse_buf = torch.empty(M, num_heads, dtype=torch.float32, device=device)
    _, lse2 = wrapper.run(q, k, v, lse=lse_buf, return_lse=True)

    assert lse2 is lse_buf
    torch.testing.assert_close(lse, lse_buf)


@pytest.mark.parametrize("sm_scale", [0.5])
def test_vsa_sm_scale(sm_scale):
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

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
        sm_scale=sm_scale,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


def test_vsa_vs_flashinfer_default_backend():
    """Cross-validate VSA backend against FlashInfer's default backend."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    ref_w = BlockSparseAttentionWrapper(ws, backend="auto")
    ref_w.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o_ref = ref_w.run(q, k, v)

    vsa_w = BlockSparseAttentionWrapper(ws, backend="vsa_blackwell")
    vsa_w.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o_vsa = vsa_w.run(q, k, v)

    torch.testing.assert_close(o_ref.float(), o_vsa.float(), atol=1e-2, rtol=1e-2)


def test_vsa_vs_auto_80k():
    """Cross-validate VSA backend against auto backend at 80k context (1250 blocks × 64).

    PyTorch dense reference is skipped at this scale to avoid OOM.
    """
    device = torch.device("cuda")
    torch.manual_seed(8)
    num_heads = 8
    num_blocks = 1250  # 1250 * 64 = 80 000 tokens
    density = 0.01
    dtype = torch.bfloat16

    M = N = num_blocks * R
    assert M == 80_000

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)

    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

    ref_w = BlockSparseAttentionWrapper(ws, backend="auto")
    ref_w.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o_ref = ref_w.run(q, k, v)

    vsa_w = BlockSparseAttentionWrapper(ws, backend="vsa_blackwell")
    vsa_w.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o_vsa = vsa_w.run(q, k, v)

    torch.testing.assert_close(o_ref.float(), o_vsa.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# plan() constraint-violation tests
# ---------------------------------------------------------------------------


def _plan_indptr_indices(device, num_blocks=2):
    indptr = torch.arange(num_blocks + 1, dtype=torch.int32, device=device)
    indices = torch.arange(num_blocks, dtype=torch.int32, device=device)
    return indptr, indices


@pytest.mark.parametrize(
    "bad_R, bad_C",
    [(64, 32), (32, 64)],
)
def test_plan_rejects_bad_block_size(bad_R, bad_C):
    device = torch.device("cuda")
    indptr, indices = _plan_indptr_indices(device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="R == C == 64"):
        wrapper.plan(indptr, indices, bad_R * 2, bad_C * 2, bad_R, bad_C, 4, 4, HEAD_DIM)


def test_plan_rejects_bad_head_dim():
    device = torch.device("cuda")
    indptr, indices = _plan_indptr_indices(device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="head_dim == 128"):
        wrapper.plan(indptr, indices, 2 * R, 2 * C, R, C, 4, 4, 64)


def test_plan_rejects_gqa():
    device = torch.device("cuda")
    indptr, indices = _plan_indptr_indices(device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="GQA"):
        wrapper.plan(indptr, indices, 2 * R, 2 * C, R, C, 8, 4, HEAD_DIM)


def test_plan_rejects_mask():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    nnz = int(indices.shape[0])
    mask = torch.ones(nnz, R, C, dtype=torch.bool, device=device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="mask"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            mask=mask,
        )


def test_plan_rejects_packed_mask():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    packed_mask = torch.zeros(1, dtype=torch.uint8, device=device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="mask"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            packed_mask=packed_mask,
        )


def test_plan_rejects_causal():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="causal"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            causal=True,
        )


def test_plan_rejects_pos_encoding_mode():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="pos_encoding_mode"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            pos_encoding_mode="ROPE_LLAMA",
        )


def test_plan_rejects_logits_soft_cap():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="logits_soft_cap"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            logits_soft_cap=50.0,
        )


def test_plan_rejects_block_mask_wrong_shape():
    """block_mask with wrong shape must raise ValueError."""
    device = torch.device("cuda")
    num_blocks, num_heads = 4, 4
    wrapper = _make_wrapper(device)
    bad_mask = torch.ones(num_heads + 1, num_blocks, num_blocks, dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="block_mask must have shape"):
        wrapper.plan(
            None, None, num_blocks * R, num_blocks * C, R, C,
            num_heads, num_heads, HEAD_DIM,
            block_mask=bad_mask,
        )


def test_plan_rejects_block_mask_non_vsa():
    """block_mask must be rejected for non-vsa_blackwell backends."""
    device = torch.device("cuda")
    num_blocks, num_heads = 4, 4
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(ws, backend="auto")
    block_mask = torch.ones(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="vsa_blackwell"):
        wrapper.plan(
            None, None, num_blocks * R, num_blocks * C, R, C,
            num_heads, num_heads, HEAD_DIM,
            block_mask=block_mask,
        )


def test_plan_rejects_missing_indptr_indices():
    """vsa_blackwell without block_mask must require indptr/indices."""
    device = torch.device("cuda")
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="indptr"):
        wrapper.plan(
            None, None, 4 * R, 4 * C, R, C,
            4, 4, HEAD_DIM,
        )


# ---------------------------------------------------------------------------
# Per-head mask accuracy tests
# ---------------------------------------------------------------------------


def test_vsa_per_head_mask_correctness():
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
    block_mask = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)
    for h in range(num_heads):
        # Each head attends to a different set of blocks
        chosen = torch.randperm(num_blocks)[:max(1, num_blocks // 2)]
        block_mask[h, :, chosen] = True

    wrapper = _make_wrapper(device)
    wrapper.plan(
        None, None, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
        block_mask=block_mask,
    )
    o_vsa = wrapper.run(q, k, v)

    # PyTorch reference: compute each head independently using its own mask
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    o_ref = torch.empty_like(o_vsa)
    for h in range(num_heads):
        qh = q[:, h, :].float()    # [M, D]
        kh = k[:, h, :].float()    # [N, D]
        vh = v[:, h, :].float()    # [N, D]
        # Expand block_mask[h] to token-level
        token_mask = torch.zeros(M, N, dtype=torch.bool, device=device)
        for qi in range(num_blocks):
            for ki in range(num_blocks):
                if block_mask[h, qi, ki]:
                    token_mask[qi*R:(qi+1)*R, ki*C:(ki+1)*C] = True
        scores = torch.matmul(qh, kh.t()) * sm_scale      # [M, N]
        scores = scores.masked_fill(~token_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o_ref[:, h, :] = torch.matmul(probs, vh).to(dtype)

    torch.testing.assert_close(o_ref, o_vsa, atol=1e-2, rtol=1e-2)


def test_vsa_per_head_mask_differs_across_heads():
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
    block_mask = torch.zeros(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)
    for h in range(num_heads):
        block_mask[h, :, h % num_blocks] = True

    wrapper = _make_wrapper(device)
    wrapper.plan(
        None, None, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
        block_mask=block_mask,
    )
    o_per_head = wrapper.run(q, k, v)

    # Head-averaged BSR (union of all per-head blocks)
    union_mask_bool = block_mask.any(dim=0)   # [MB, NB]
    nz = union_mask_bool.nonzero(as_tuple=False)
    indptr = torch.zeros(num_blocks + 1, dtype=torch.int32, device=device)
    row_counts = union_mask_bool.sum(dim=1).to(torch.int32)
    indptr[1:] = row_counts.cumsum(0)
    indices = nz[:, 1].to(torch.int32)

    wrapper2 = _make_wrapper(device)
    wrapper2.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
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
    scores = torch.matmul(q_pooled, k_pooled.transpose(-1, -2)) * sm_scale  # [H, MB, NB]
    block_attn_score = torch.softmax(scores, dim=-1)                          # [H, MB, NB]
    out_pooled = torch.matmul(block_attn_score, v_pooled)                    # [H, MB, D]

    # Broadcast compress output: [H, MB, D] → [M, H, D]
    # Each pooled block result is replicated for all R tokens in the block.
    output_compress = (
        out_pooled.permute(1, 0, 2)          # [MB, H, D]
        .unsqueeze(1)                         # [MB, 1, H, D]
        .expand(-1, R, -1, -1)               # [MB, R, H, D]
        .reshape(M, H, D)                    # [M, H, D]
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
    (4096,  0.50),   # short-seq boundary: VSA break-even point
    (16384, 0.50),
    (16384, 0.25),
    (16384, 0.10),
    (32768, 0.25),
    (32768, 0.10),
    (80000, 0.25),
    (80000, 0.10),
]


@pytest.mark.parametrize("seqlen,topk_frac", _SWEEP_CONFIGS)
def test_vsa_accuracy_vs_dense(seqlen, topk_frac):
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
    assert seqlen % R == 0, "seqlen must be a multiple of R=64"
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
        q, k, v, R, C, topk,
    )

    # Stage 2: block-sparse attention (select branch)
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    vsa_w = BlockSparseAttentionWrapper(ws, backend="vsa_blackwell")
    vsa_w.plan(
        None, None, seqlen, seqlen, R, C,
        num_heads, num_heads, HEAD_DIM,
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


def test_vsa_performance_vs_dense():
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

    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

    for seqlen, topk_frac in _SWEEP_CONFIGS:
        num_blocks = seqlen // R
        topk = max(1, int(round(topk_frac * num_blocks)))

        q = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
        k = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
        v = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)

        # Pre-run: get block_mask and plan wrapper
        output_compress, block_mask, actual_density = _compute_vsa_compress_and_mask(
            q, k, v, R, C, topk,
        )
        vsa_w = BlockSparseAttentionWrapper(ws, backend="vsa_blackwell")
        vsa_w.plan(
            None, None, seqlen, seqlen, R, C,
            num_heads, num_heads, HEAD_DIM,
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

