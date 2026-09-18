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
from flashinfer.utils import is_sm12x_supported, is_sm90a_supported

# ---------------------------------------------------------------------------
# Hardware gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (
        is_sm90a_supported(torch.device("cuda"))
        or is_sm12x_supported(torch.device("cuda"))
    ),
    reason="FP16/BF16 blk64 VSA requires an SM90 or SM12x GPU",
)

# Shared warp-MMA/TMA blk64 kernel constants
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


def _pytorch_ref_gqa_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    R: int,
    C: int,
) -> torch.Tensor:
    """Reference for a QO-head-level block mask with GQA K/V mapping."""
    M, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    qheads_per_kv_head = num_qo_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(q)

    for qo_head in range(num_qo_heads):
        kv_head = qo_head // qheads_per_kv_head
        token_mask = (
            block_mask[qo_head].repeat_interleave(R, dim=0).repeat_interleave(C, dim=1)
        )
        scores = (
            torch.matmul(q[:, qo_head, :].float(), k[:, kv_head, :].float().t())
            * sm_scale
        )
        probs = torch.softmax(scores.masked_fill(~token_mask, float("-inf")), dim=-1)
        output[:, qo_head, :] = torch.matmul(probs, v[:, kv_head, :].float()).to(
            q.dtype
        )

    return output


@pytest.fixture(scope="module")
def workspace():
    return torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _backend(device=None):
    major, _ = torch.cuda.get_device_capability(device)
    if major == 9:
        return "vsa_sm90_blk64"
    if major == 12:
        return "vsa_sm120_blk64"
    raise RuntimeError(f"unsupported compute capability on {device}")


def _make_wrapper(workspace):
    return BlockSparseAttentionWrapper(workspace, backend=_backend(workspace.device))


def _low_level_fwd(device=None):
    if _backend(device) == "vsa_sm90_blk64":
        from flashinfer.cute_dsl.sparse.bsa_attn_sm90 import (
            bsa_attn_sm90_blk64_fwd,
        )

        return bsa_attn_sm90_blk64_fwd
    from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
        bsa_attn_sm120_blk64_fwd,
    )

    return bsa_attn_sm120_blk64_fwd


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
def test_vsa_fp16_blk64_accuracy(dtype, density, num_blocks, num_heads, workspace):
    """FP16/BF16 blk64 output must match PyTorch dense block-sparse reference."""
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
def test_vsa_fp16_blk64_sm_scale(sm_scale, workspace):
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


def test_vsa_fp16_blk64_per_head_mask_correctness(workspace):
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
def test_vsa_fp16_blk64_gqa(num_qo_heads, num_kv_heads, dtype, workspace):
    """FP16/BF16 blk64 GQA must match per-head PyTorch reference."""
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


@pytest.mark.parametrize("mask_head_kind", ["qo", "kv"])
def test_vsa_fp16_blk64_gqa_block_mask(mask_head_kind, workspace):
    """GQA accepts QO-head masks; SM90 also broadcasts KV-head masks."""
    if mask_head_kind == "kv" and _backend() != "vsa_sm90_blk64":
        pytest.skip("KV-head block masks are specific to the SM90 backend")
    device = torch.device("cuda")
    torch.manual_seed(17)
    num_qo_heads, num_kv_heads, num_blocks = 8, 2, 4
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_qo_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)

    mask_heads = num_qo_heads if mask_head_kind == "qo" else num_kv_heads
    block_mask = torch.rand(mask_heads, num_blocks, num_blocks, device=device) > 0.5
    block_mask[:, :, 0] = True  # every row must contain at least one KV block
    if mask_head_kind == "kv":
        qo_block_mask = block_mask.repeat_interleave(
            num_qo_heads // num_kv_heads, dim=0
        )
    else:
        qo_block_mask = block_mask
    o_ref = _pytorch_ref_gqa_block_mask(q, k, v, qo_block_mask, R, C)

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
        block_mask=block_mask,
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
def test_vsa_fp16_blk64_asymmetric_seqlen(MB, NB, num_heads, density, workspace):
    """FP16/BF16 blk64 with seqlen_q != seqlen_k must match PyTorch reference."""
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
def test_vsa_fp16_blk64_return_lse(dtype, num_blocks, num_heads, workspace):
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


def test_vsa_fp16_blk64_empty_row(workspace):
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
    wrapper.plan(
        indptr, indices, M, N, R, C, num_heads, num_heads, HEAD_DIM, q_data_type=dtype
    )
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
    wrapper2.plan(
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
    out2, lse2 = wrapper2.run(q, k, v, return_lse=True)

    empty_slice = slice(R, 2 * R)
    assert torch.all(out2[empty_slice] == 0), (
        "empty row output should be zero (block_mask path)"
    )
    assert torch.all(lse2[empty_slice].isinf() & (lse2[empty_slice] < 0)), (
        "empty row LSE should be -inf (block_mask path)"
    )
    assert torch.all(lse2[:R].isfinite()), (
        "non-empty rows should have finite LSE (block_mask path)"
    )
    assert torch.all(lse2[2 * R :].isfinite()), (
        "non-empty rows should have finite LSE (block_mask path)"
    )


# ---------------------------------------------------------------------------
# Variable KV-block count via block_mask
# ---------------------------------------------------------------------------


def test_vsa_fp16_blk64_variable_blocks_per_q(workspace):
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


def test_vsa_fp16_blk64_preallocated_outputs(workspace):
    """Wrapper returns and fills caller-provided output and LSE tensors."""
    device = torch.device("cuda")
    torch.manual_seed(31)
    num_heads, num_blocks = 4, 4
    M = N = num_blocks * R
    dtype = torch.float16
    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    wrapper = _make_wrapper(workspace)
    wrapper.plan(
        indptr, indices, M, N, R, C, num_heads, num_heads, HEAD_DIM, q_data_type=dtype
    )
    out = torch.empty_like(q)
    lse = torch.empty(M, num_heads, dtype=torch.float32, device=device)
    returned_out, returned_lse = wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)

    assert returned_out is out
    assert returned_lse is lse
    assert torch.all(out.isfinite())
    assert torch.all(lse.isfinite())

    expected = wrapper.run(q, k, v)
    k_alias = k.clone()
    aliased_out = wrapper.run(q, k_alias, v, out=k_alias)
    torch.testing.assert_close(aliased_out, expected, atol=1e-2, rtol=1e-2)


def test_vsa_fp16_blk64_low_level_validation():
    """The public low-level entry rejects unsafe tensor shapes and outputs."""
    fwd = _low_level_fwd()

    device = torch.device("cuda")
    q = torch.randn(1, R, 4, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(1, C, 2, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    q2k = torch.zeros(1, 4, 1, 1, dtype=torch.int32, device=device)
    q2k_num = torch.ones(1, 4, 1, dtype=torch.int32, device=device)
    kwargs = {"q2k_block_nums": q2k_num, "block_sparse_num": 1}

    with pytest.raises(ValueError, match="q and k"):
        fwd(q, k[..., :64], v, q2k, **kwargs)
    with pytest.raises(ValueError, match="v must match"):
        fwd(q, k, v[:, :-1], q2k, **kwargs)
    with pytest.raises(ValueError, match="must be 4D"):
        fwd(q, k, v, q2k[0], **kwargs)

    out_cpu = torch.empty(q.shape, dtype=q.dtype, device="cpu")
    with pytest.raises(ValueError, match="out must have shape"):
        fwd(q, k, v, q2k, out=out_cpu, **kwargs)

    overlapping_out = torch.empty(
        1, R, 1, HEAD_DIM, dtype=q.dtype, device=device
    ).expand_as(q)
    with pytest.raises(ValueError, match="internal storage overlap"):
        fwd(q, k, v, q2k, out=overlapping_out, **kwargs)

    overlap_storage = torch.empty(65536, dtype=q.dtype, device=device)
    overlapping_strided_out = torch.as_strided(
        overlap_storage, q.shape, (65536, 256, 8192, 1)
    )
    with pytest.raises(ValueError, match="internal storage overlap"):
        fwd(q, k, v, q2k, out=overlapping_strided_out, **kwargs)

    out_storage = torch.empty(1, R, 4, HEAD_DIM + 1, dtype=q.dtype, device=device)
    out_view = out_storage[..., :HEAD_DIM]
    lse_storage = torch.empty(1, 4, R + 1, dtype=torch.float32, device=device)
    lse_view = lse_storage[..., :R]
    assert not out_view.is_contiguous() and out_view.stride(-1) == 1
    assert not lse_view.is_contiguous() and lse_view.stride(-1) == 1

    returned_out, returned_lse = fwd(
        q,
        k,
        v,
        q2k,
        out=out_view,
        lse=lse_view,
        return_lse=True,
        **kwargs,
    )
    assert returned_out is out_view
    assert returned_lse is lse_view
    assert torch.all(out_view.isfinite())
    assert torch.all(lse_view.isfinite())

    q_storage = torch.empty(q.numel() + 1, dtype=q.dtype, device=device)
    q_offset = q_storage[1:].view_as(q)
    q_offset.copy_(q)
    out_storage = torch.empty(q.numel() + 1, dtype=q.dtype, device=device)
    out_offset = out_storage[1:].view_as(q)
    returned_out, _ = fwd(q_offset, k, v, q2k, out=out_offset, **kwargs)
    assert returned_out is out_offset
    assert torch.all(out_offset.isfinite())

    q_mha = q[:, :, :2].contiguous()
    q2k_mha, q2k_num_mha = q2k[:, :2], q2k_num[:, :2]
    expected, _ = fwd(q_mha, k, v, q2k_mha, 1, q2k_block_nums=q2k_num_mha)
    for alias in ("k", "v"):
        k_arg, v_arg = k.clone(), v.clone()
        aliased_out = k_arg if alias == "k" else v_arg
        actual, _ = fwd(
            q_mha,
            k_arg,
            v_arg,
            q2k_mha,
            1,
            q2k_block_nums=q2k_num_mha,
            out=aliased_out,
        )
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    elements = q_mha.numel()
    shared = torch.empty(elements * 2, dtype=q.dtype, device=device)
    k_partial_alias = shared[elements:].view_as(k)
    k_partial_alias.copy_(k)
    out_partial_alias = torch.as_strided(
        shared,
        q_mha.shape,
        (R * 2 * (HEAD_DIM + 1), 2 * (HEAD_DIM + 1), HEAD_DIM + 1, 1),
    )
    expected, _ = fwd(
        q_mha,
        k_partial_alias.clone(),
        v,
        q2k_mha,
        1,
        q2k_block_nums=q2k_num_mha,
    )
    actual, _ = fwd(
        q_mha,
        k_partial_alias,
        v,
        q2k_mha,
        1,
        q2k_block_nums=q2k_num_mha,
        out=out_partial_alias,
    )
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_vsa_fp16_blk64_non_current_device():
    """The low-level entry uses the input device without changing the caller's device."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible SM90/SM12x GPUs")
    original_device = torch.cuda.current_device()
    other_indices = [
        index
        for index in range(torch.cuda.device_count())
        if index != original_device
        and torch.cuda.get_device_capability(index)[0] in (9, 12)
    ]
    if not other_indices:
        pytest.skip("requires another visible SM90/SM12x GPU")
    other_device = torch.device("cuda", other_indices[0])
    fwd = _low_level_fwd(other_device)
    q = torch.randn(1, R, 2, HEAD_DIM, dtype=torch.float16, device=other_device)
    k = torch.randn(1, C, 2, HEAD_DIM, dtype=torch.float16, device=other_device)
    v = torch.randn_like(k)
    q2k = torch.zeros(1, 2, 1, 1, dtype=torch.int32, device=other_device)
    q2k_num = torch.ones(1, 2, 1, dtype=torch.int32, device=other_device)

    out, _ = fwd(q, k, v, q2k, 1, q2k_block_nums=q2k_num)

    assert out.device == other_device
    assert torch.all(out.isfinite())
    assert torch.cuda.current_device() == original_device


def test_vsa_fp16_blk64_wrapper_validation(workspace):
    """SM90 plan-time metadata and run-time tensor contracts fail before launch."""
    if _backend() != "vsa_sm90_blk64":
        pytest.skip("SM90-specific wrapper contract")
    device = torch.device("cuda")
    num_qo_heads, num_kv_heads, num_blocks = 4, 2, 4
    M = N = num_blocks * R
    dtype = torch.float16
    q = torch.randn(M, num_qo_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_kv_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn_like(k)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

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
    # Preserve the historical convenience of accepting default torch int64
    # BSR tensors and normalizing them to the int32 kernel ABI at plan time.
    _make_wrapper(workspace).plan(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        q_data_type=dtype,
    )
    with pytest.raises(ValueError, match="q"):
        wrapper.run(q[:-1], k, v)

    non_bool_mask = torch.ones(
        num_kv_heads, num_blocks, num_blocks, dtype=torch.int32, device=device
    )
    with pytest.raises(TypeError, match="3D bool"):
        _make_wrapper(workspace).plan(
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
            block_mask=non_bool_mask,
        )

    for bad_offsets in ([0, 2, 1, 2, 3], [0, 2**32 + 1, 1, 2, 3]):
        bad_indptr = torch.tensor(bad_offsets, dtype=torch.int64, device=device)
        with pytest.raises(ValueError, match="row counts"):
            _make_wrapper(workspace).plan(
                bad_indptr,
                indices[:3],
                M,
                N,
                R,
                C,
                num_qo_heads,
                num_kv_heads,
                HEAD_DIM,
                q_data_type=dtype,
            )

    with pytest.raises(ValueError, match="matching Q/K/V dtypes"):
        _make_wrapper(workspace).plan(
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
            kv_data_type=torch.bfloat16,
        )


# ---------------------------------------------------------------------------
# Performance sweep (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FLASHINFER_TEST_PERF"),
    reason="performance benchmark, set FLASHINFER_TEST_PERF=1 to run",
)
def test_vsa_fp16_blk64_perf_sweep(workspace):
    """FP16/BF16 blk64 throughput across seqlen × density configurations."""
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
