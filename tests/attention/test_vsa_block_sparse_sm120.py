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

from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm12x_supported

# ---------------------------------------------------------------------------
# Hardware gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="vsa_sm120_blk64 backend requires SM120/SM121 GPU",
)

# sm120_blk64 kernel constants
R = C = 64
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Helpers  (mirrors test_vsa_block_sparse.py)
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
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Dense PyTorch reference for block-sparse attention."""
    M, _H, D = q.shape
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


def _pytorch_ref_gqa(
    q: torch.Tensor,  # [M, Hq, D]
    k: torch.Tensor,  # [N, Hkv, D]
    v: torch.Tensor,  # [N, Hkv, D]
    indptr: torch.Tensor,
    indices: torch.Tensor,
    R: int,
    C: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    M, Hq, D = q.shape
    Hkv = k.shape[1]
    qhead_per_kvhead = Hq // Hkv
    MB, NB = M // R, k.shape[0] // C
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    mask = _bsr_to_dense_mask(indptr, indices, MB, NB, R, C, q.device)
    output = torch.empty_like(q)
    for h in range(Hq):
        h_kv = h // qhead_per_kvhead
        qh = q[:, h, :].float()
        kh = k[:, h_kv, :].float()
        vh = v[:, h_kv, :].float()
        scores = torch.matmul(qh, kh.t()) * sm_scale
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output[:, h, :] = torch.matmul(probs, vh).to(q.dtype)
    return output


@pytest.fixture(scope="module")
def workspace():
    return torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _make_wrapper(workspace):
    return BlockSparseAttentionWrapper(workspace, backend="vsa_sm120_blk64")


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
def test_vsa_sm120_accuracy(dtype, density, num_blocks, num_heads, workspace):
    """sm120_blk64 output must match PyTorch dense block-sparse reference."""
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
        indptr, indices, M, N, R, C, num_heads, num_heads, HEAD_DIM, q_data_type=dtype
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("sm_scale", [0.5])
def test_vsa_sm120_sm_scale(sm_scale, workspace):
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


# ---------------------------------------------------------------------------
# Per-head mask accuracy tests
# ---------------------------------------------------------------------------


def test_vsa_sm120_per_head_mask_correctness(workspace):
    """Per-head block_mask path must match PyTorch dense reference per head."""
    device = torch.device("cuda")
    torch.manual_seed(10)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    block_mask = torch.zeros(
        num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
    )
    for h in range(num_heads):
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
# GQA / MQA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads,dtype",
    [
        (8, 4, torch.bfloat16),  # GQA 2x
        (8, 2, torch.bfloat16),  # GQA 4x
        (8, 1, torch.bfloat16),  # MQA
        (8, 4, torch.float16),  # GQA 2x, fp16
    ],
)
def test_vsa_sm120_gqa(num_qo_heads, num_kv_heads, dtype, workspace):
    """sm120_blk64 GQA must match per-head PyTorch reference."""
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


# ---------------------------------------------------------------------------
# Asymmetric seqlen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "MB,NB,num_heads,density",
    [
        (4, 8, 8, 0.5),
        (8, 4, 4, 0.5),
    ],
)
def test_vsa_sm120_asymmetric_seqlen(MB, NB, num_heads, density, workspace):
    """sm120_blk64 with seqlen_q != seqlen_k must match PyTorch reference."""
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
        indptr, indices, M, N, R, C, num_heads, num_heads, HEAD_DIM, q_data_type=dtype
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# LSE output validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,num_blocks,num_heads",
    [
        (torch.bfloat16, 8, 4),
        (torch.float16, 8, 4),
    ],
)
def test_vsa_sm120_return_lse(dtype, num_blocks, num_heads, workspace):
    """return_lse=True must produce LSE values consistent with PyTorch logsumexp."""
    device = torch.device("cuda")
    torch.manual_seed(20)
    M = N = num_blocks * R

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mask = _bsr_to_dense_mask(indptr, indices, num_blocks, num_blocks, R, C, device)
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1).permute(1, 0)  # [M, H]

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr, indices, M, N, R, C, num_heads, num_heads, HEAD_DIM, q_data_type=dtype
    )
    _, lse = wrapper.run(q, k, v, return_lse=True)

    finite = lse_ref.isfinite()
    assert finite.any()
    torch.testing.assert_close(
        lse[finite].float(), lse_ref[finite].float(), atol=1e-2, rtol=1e-2
    )


def test_vsa_sm120_empty_row(workspace):
    """Empty sparse rows (no KV blocks) must produce zero output and LSE=-inf."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    dtype = torch.bfloat16
    MB = NB = 4
    M = N = MB * R
    num_heads = 8

    # BSR path: first Q-block has no KV blocks
    indptr = torch.tensor([0, 0, 2, 3, 4], dtype=torch.int32, device=device)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    wrapper = _make_wrapper(workspace)
    wrapper.plan(indptr, indices, M, N, R, C, num_heads, num_heads, HEAD_DIM, q_data_type=dtype)
    out, lse = wrapper.run(q, k, v, return_lse=True)

    # Empty Q-block 0: output must be zero, LSE must be -inf
    assert torch.all(out[:R] == 0), "empty row output should be zero"
    assert torch.all(lse[:R].isinf() & (lse[:R] < 0)), "empty row LSE should be -inf"
    # Non-empty rows must produce finite LSE
    assert torch.all(lse[R:].isfinite()), "non-empty rows should have finite LSE"

    # block_mask path: second Q-block is all False
    block_mask = torch.ones(num_heads, MB, NB, dtype=torch.bool, device=device)
    block_mask[:, 1, :] = False

    wrapper2 = _make_wrapper(workspace)
    wrapper2.plan(None, None, M, N, R, C, num_heads, num_heads, HEAD_DIM, q_data_type=dtype, block_mask=block_mask)
    out2, lse2 = wrapper2.run(q, k, v, return_lse=True)

    empty_slice = slice(R, 2 * R)
    assert torch.all(out2[empty_slice] == 0), "empty row output should be zero (block_mask path)"
    assert torch.all(lse2[empty_slice].isinf() & (lse2[empty_slice] < 0)), "empty row LSE should be -inf (block_mask path)"


# ---------------------------------------------------------------------------
# Variable KV-block count via block_mask
# ---------------------------------------------------------------------------


def test_vsa_sm120_variable_blocks_per_q(workspace):
    """Per-head block_mask with variable KV count per Q-block must match PyTorch ref."""
    device = torch.device("cuda")
    torch.manual_seed(30)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    block_mask = torch.zeros(
        num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device
    )
    for i in range(num_blocks):
        cnt = i + 1
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
# Performance sweep (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FLASHINFER_TEST_PERF"),
    reason="performance benchmark, set FLASHINFER_TEST_PERF=1 to run",
)
def test_vsa_sm120_perf_sweep(workspace):
    """sm120_blk64 throughput across seqlen × density configurations."""
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
        num_blocks = seqlen // R
        q = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
        k = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)
        v = torch.randn(seqlen, num_heads, HEAD_DIM, dtype=dtype, device=device)

        for density in densities:
            indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)
            active_blocks = len(indices)

            wrapper = _make_wrapper(workspace)
            wrapper.plan(
                indptr,
                indices,
                seqlen,
                seqlen,
                R,
                C,
                num_heads,
                num_heads,
                HEAD_DIM,
                q_data_type=dtype,
            )
            wrapper.run(q, k, v)  # warm-up

            times = bench_gpu_time(
                lambda w=wrapper, _q=q, _k=k, _v=v: w.run(_q, _k, _v)
            )
            ms = statistics.median(times)

            flops = 2 * 2 * active_blocks * R * C * num_heads * HEAD_DIM
            tflops = flops / (ms * 1e-3) / 1e12
            actual_density = active_blocks / (num_blocks * num_blocks)
            print(
                f"{seqlen:>8}  {actual_density:>8.3f}  {active_blocks:>12}  {ms:>10.3f}  {tflops:>8.2f}"
            )

        print(sep)
