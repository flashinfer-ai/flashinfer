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

import pytest
import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd
from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import (
    quantize_sage_kv_sm120,
    quantize_sage_q_sm120,
    quantize_sage_qkv_sm120,
)
from flashinfer.utils import is_sm12x_supported

# ---------------------------------------------------------------------------
# Hardware gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="sm120_blk64 Sage backend requires SM120/SM121 GPU with cc==(12, 0)",
)

BLOCK = 64
HEAD_DIM = 128
# Sage QK-INT8/PV-FP8 quantization is inherently lossy; use a looser
# tolerance than the bf16/fp16 vsa_sm120_blk64 tests (atol/rtol=1e-2).
ATOL, RTOL = 5e-2, 5e-2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_random_q2k(
    batch: int,
    heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    density: float,
    device: torch.device,
):
    """Return (q2k_block_index, q2k_block_nums) with a variable KV-block count
    per (batch, head, q_block) row; every row has >= 1 block."""
    capacity = num_kv_blocks
    index = torch.zeros(
        batch, heads, num_q_blocks, capacity, dtype=torch.int32, device=device
    )
    nums = torch.zeros(batch, heads, num_q_blocks, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(heads):
            for qi in range(num_q_blocks):
                k = max(1, int(round(density * num_kv_blocks)))
                k = min(k, num_kv_blocks)
                chosen = torch.randperm(num_kv_blocks)[:k].sort().values
                index[b, h, qi, :k] = chosen.to(torch.int32).to(device)
                nums[b, h, qi] = k
    return index, nums


def _dense_mask_from_q2k(
    q2k_index: torch.Tensor,
    q2k_nums: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand direct-selection q2k metadata into a per-(batch,head) token mask
    [B, H, seqlen_q, seqlen_k]."""
    batch, heads, num_q_blocks, _capacity = q2k_index.shape
    mask = torch.zeros(
        batch, heads, seqlen_q, seqlen_k, dtype=torch.bool, device=device
    )
    idx_cpu = q2k_index.cpu()
    nums_cpu = q2k_nums.cpu()
    for b in range(batch):
        for h in range(heads):
            for qi in range(num_q_blocks):
                q_lo = qi * BLOCK
                q_hi = min(q_lo + BLOCK, seqlen_q)
                cnt = int(nums_cpu[b, h, qi])
                for j in range(cnt):
                    kb = int(idx_cpu[b, h, qi, j])
                    k_lo = kb * BLOCK
                    k_hi = min(k_lo + BLOCK, seqlen_k)
                    mask[b, h, q_lo:q_hi, k_lo:k_hi] = True
    return mask


def _pytorch_ref(
    q: torch.Tensor,  # [B, H, Sq, D] bf16 (pre-quantization)
    k: torch.Tensor,  # [B, H, Sk, D] bf16
    v: torch.Tensor,  # [B, H, Sk, D] bf16
    mask: torch.Tensor,  # [B, H, Sq, Sk] bool
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Dense FP32 PyTorch reference computed on the pre-quantization tensors.

    This is the correctness bar for a lossy INT8/FP8 kernel: the quantized
    kernel output should be close to (not bit-exact with) the full-precision
    reference.
    """
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.einsum("bhsd,bhtd->bhst", qf, kf) * sm_scale
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhst,bhtd->bhsd", probs, vf).to(torch.bfloat16)


def _random_bf16_bhsd(batch, heads, seqlen, device, scale=0.1):
    return (
        torch.randn(batch, heads, seqlen, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * scale
    )


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,num_heads,num_blocks,density",
    [
        (1, 2, 2, 1.0),
        (1, 8, 8, 0.5),
        (2, 4, 4, 0.25),
        (2, 8, 16, 0.75),
    ],
)
def test_vsa_sm120_sage_accuracy(batch, num_heads, num_blocks, density):
    device = torch.device("cuda")
    torch.manual_seed(42)
    seqlen = num_blocks * BLOCK

    q = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    v = _random_bf16_bhsd(batch, num_heads, seqlen, device)

    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    q2k_index, q2k_nums = _build_random_q2k(
        batch, num_heads, num_blocks, num_blocks, density, device
    )
    mask = _dense_mask_from_q2k(q2k_index, q2k_nums, seqlen, seqlen, device)
    o_ref = _pytorch_ref(q, k, v, mask)

    o = bsa_attn_sm120_blk64_sage_fwd(
        q8,
        k8,
        v8,
        qs,
        ks,
        vs,
        q2k_index,
        block_sparse_num=num_blocks,
        q2k_block_nums=q2k_nums,
    )

    torch.testing.assert_close(o.float(), o_ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("sm_scale", [0.5, 1.0 / math.sqrt(HEAD_DIM)])
def test_vsa_sm120_sage_sm_scale(sm_scale):
    device = torch.device("cuda")
    torch.manual_seed(4)
    batch, num_heads, num_blocks = 1, 4, 4
    seqlen = num_blocks * BLOCK

    q = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    v = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    q2k_index, q2k_nums = _build_random_q2k(
        batch, num_heads, num_blocks, num_blocks, 0.5, device
    )
    mask = _dense_mask_from_q2k(q2k_index, q2k_nums, seqlen, seqlen, device)
    o_ref = _pytorch_ref(q, k, v, mask, sm_scale=sm_scale)

    o = bsa_attn_sm120_blk64_sage_fwd(
        q8,
        k8,
        v8,
        qs,
        ks,
        vs,
        q2k_index,
        block_sparse_num=num_blocks,
        q2k_block_nums=q2k_nums,
        softmax_scale=sm_scale,
    )

    torch.testing.assert_close(o.float(), o_ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (100, 128),  # non-64-aligned Q tail
        (128, 100),  # non-64-aligned K tail
        (100, 100),  # both tails
    ],
)
def test_vsa_sm120_sage_ragged_seqlen(seqlen_q, seqlen_k):
    """Non-64-aligned seqlens exercise the per-tile tail-masking path."""
    device = torch.device("cuda")
    torch.manual_seed(13)
    batch, num_heads = 1, 4
    num_q_blocks = (seqlen_q + BLOCK - 1) // BLOCK
    num_kv_blocks = (seqlen_k + BLOCK - 1) // BLOCK

    q = _random_bf16_bhsd(batch, num_heads, seqlen_q, device)
    k = _random_bf16_bhsd(batch, num_heads, seqlen_k, device)
    v = _random_bf16_bhsd(batch, num_heads, seqlen_k, device)
    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    # Every Q block attends to every KV block (dense, but seqlen is ragged).
    index = (
        torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
        .view(1, 1, 1, num_kv_blocks)
        .expand(batch, num_heads, num_q_blocks, num_kv_blocks)
        .contiguous()
    )
    mask = torch.zeros(
        batch, num_heads, seqlen_q, seqlen_k, dtype=torch.bool, device=device
    )
    mask[:] = True

    o_ref = _pytorch_ref(q, k, v, mask)
    o = bsa_attn_sm120_blk64_sage_fwd(
        q8,
        k8,
        v8,
        qs,
        ks,
        vs,
        index,
        block_sparse_num=num_kv_blocks,
    )

    torch.testing.assert_close(o.float(), o_ref.float(), atol=ATOL, rtol=RTOL)


def test_vsa_sm120_sage_empty_row():
    """A Q-block with zero selected KV blocks must produce exactly zero output."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    batch, num_heads, num_blocks = 1, 4, 4
    seqlen = num_blocks * BLOCK

    q = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    v = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    index = torch.zeros(
        batch, num_heads, num_blocks, num_blocks, dtype=torch.int32, device=device
    )
    nums = torch.full(
        (batch, num_heads, num_blocks), num_blocks, dtype=torch.int32, device=device
    )
    for kb in range(num_blocks):
        index[:, :, 1, kb] = kb
    nums[:, :, 0] = 0  # Q-block 0 selects nothing.

    o = bsa_attn_sm120_blk64_sage_fwd(
        q8,
        k8,
        v8,
        qs,
        ks,
        vs,
        index,
        block_sparse_num=num_blocks,
        q2k_block_nums=nums,
    )

    assert torch.all(o[:, :, :BLOCK, :] == 0), "empty Q-block output should be zero"
    assert torch.isfinite(o[:, :, BLOCK:, :]).all()


# ---------------------------------------------------------------------------
# API surface / argument plumbing
# ---------------------------------------------------------------------------


def test_vsa_sm120_sage_out_param():
    device = torch.device("cuda")
    torch.manual_seed(2)
    batch, num_heads, num_blocks = 1, 2, 2
    seqlen = num_blocks * BLOCK

    q = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    v = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    index = (
        torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        .expand(batch, num_heads, num_blocks, num_blocks)
        .contiguous()
    )
    preallocated = torch.empty(
        batch, num_heads, seqlen, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    out = bsa_attn_sm120_blk64_sage_fwd(
        q8,
        k8,
        v8,
        qs,
        ks,
        vs,
        index,
        block_sparse_num=num_blocks,
        out=preallocated,
    )
    assert out.data_ptr() == preallocated.data_ptr()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("block_sizes_mode", [1, 2, 3])
def test_vsa_sm120_sage_block_sizes(block_sizes_mode):
    """block_sizes (1D / 2D / 3D, matching _prepare_sm120_sparse_metadata's
    block_sizes_mode encoding) must mask out padding tokens in the last KV
    block."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    batch, num_heads = 1, 2
    num_blocks = 2
    seqlen = num_blocks * BLOCK
    valid_last_block = 40  # last KV block only has 40/64 valid tokens

    q = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    v = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    index = (
        torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        .expand(batch, num_heads, num_blocks, num_blocks)
        .contiguous()
    )

    if block_sizes_mode == 1:
        block_sizes = torch.tensor(
            [BLOCK, valid_last_block], dtype=torch.int32, device=device
        )
    elif block_sizes_mode == 2:
        block_sizes = (
            torch.tensor([[BLOCK, valid_last_block]], dtype=torch.int32, device=device)
            .expand(batch, num_blocks)
            .contiguous()
        )
    else:
        block_sizes = (
            torch.tensor([[BLOCK, valid_last_block]], dtype=torch.int32, device=device)
            .view(1, 1, num_blocks)
            .expand(batch, num_heads, num_blocks)
            .contiguous()
        )

    mask = torch.zeros(
        batch, num_heads, seqlen, seqlen, dtype=torch.bool, device=device
    )
    mask[:] = True
    mask[:, :, :, BLOCK + valid_last_block :] = False

    o_ref = _pytorch_ref(q, k, v, mask)
    o = bsa_attn_sm120_blk64_sage_fwd(
        q8,
        k8,
        v8,
        qs,
        ks,
        vs,
        index,
        block_sparse_num=num_blocks,
        block_sizes=block_sizes,
    )

    torch.testing.assert_close(o.float(), o_ref.float(), atol=ATOL, rtol=RTOL)


def test_vsa_sm120_sage_gqa_guard():
    """This function only supports MHA; num_kv_heads != num_heads must raise."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, num_heads, num_blocks = 1, 4, 2
    seqlen = num_blocks * BLOCK

    q = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = _random_bf16_bhsd(batch, 2, seqlen, device)
    v = _random_bf16_bhsd(batch, 2, seqlen, device)
    q8, qs = quantize_sage_q_sm120(q)
    k8, v8, ks, vs = quantize_sage_kv_sm120(k, v)

    index = (
        torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        .expand(batch, num_heads, num_blocks, num_blocks)
        .contiguous()
    )

    with pytest.raises(ValueError):
        bsa_attn_sm120_blk64_sage_fwd(
            q8,
            k8,
            v8,
            qs,
            ks,
            vs,
            index,
            block_sparse_num=num_blocks,
        )


def test_vsa_sm120_sage_wrong_arch_dtype_guards():
    """Dtype guards must reject non-int8 Q/K and non-fp8 V."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, num_heads, num_blocks = 1, 2, 2
    seqlen = num_blocks * BLOCK
    index = (
        torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        .expand(batch, num_heads, num_blocks, num_blocks)
        .contiguous()
    )

    q_bf16 = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = torch.zeros(batch, num_heads, seqlen, HEAD_DIM, dtype=torch.int8, device=device)
    v = torch.zeros(
        batch, num_heads, HEAD_DIM, seqlen, dtype=torch.float8_e4m3fn, device=device
    )
    qs = torch.ones(batch, num_heads, 4, dtype=torch.float32, device=device)
    ks = torch.ones(batch, num_heads, num_blocks, dtype=torch.float32, device=device)
    vs = torch.ones(batch, num_heads, HEAD_DIM, dtype=torch.float32, device=device)

    with pytest.raises(AssertionError):
        bsa_attn_sm120_blk64_sage_fwd(
            q_bf16,
            k,
            v,
            qs,
            ks,
            vs,
            index,
            block_sparse_num=num_blocks,
        )


# ---------------------------------------------------------------------------
# V permutation / PRMT consistency
#
# The V quantization kernel (_quantize_sage_kv_kernel in sage_quant_sm120.py)
# physically stores V under a 16-token permutation so the FP8 PV MMA can
# consume it without an in-kernel transpose. The forward kernel
# (flash_fwd_sm120_sage._make_acc_into_fp8_op) undoes this with a matching
# cute.arch.prmt byte-lane shuffle on the P fragment. If the two disagree,
# attention would still numerically "work" (finite, plausible-looking
# output) but silently retrieve the WRONG token's V row -- ordinary
# end-to-end accuracy tests with random data cannot distinguish this from a
# correct kernel, because averaging over many random tokens hides which
# specific token contributed what. This test is position-sensitive: query
# row t is engineered to attend almost exclusively to K/V token t, so a
# permutation mismatch would surface as row t recovering some *other*
# token's V value instead of its own.
# ---------------------------------------------------------------------------


def test_vsa_sm120_sage_v_permutation_consistency():
    """Query row t must recover V token t's value, exercising every token
    position (and hence every branch of the 16-token physical permutation
    formula) in a single kernel launch."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, num_heads = 1, 2
    S = BLOCK  # single K64 tile: sweep all 64 possible physical-permutation slots

    # Diagonal-dominant Q/K: row t has a large spike in its own feature
    # dimension t (plus small background noise), so Q[t]-K[j] is maximized
    # at j == t for every row simultaneously.
    bg, spike = 0.05, 10.0
    qf = torch.randn(batch, num_heads, S, HEAD_DIM, device=device) * bg
    kf = torch.randn(batch, num_heads, S, HEAD_DIM, device=device) * bg
    diag = torch.arange(S, device=device)
    qf[:, :, diag, diag] = spike
    kf[:, :, diag, diag] = spike
    q = qf.to(torch.bfloat16)
    k = kf.to(torch.bfloat16)

    # Every token gets a unique, monotonically increasing "fingerprint"
    # value, identical across all 128 channels.
    const = 0.01
    token_ids = torch.arange(S, device=device).float() + 1.0
    v = (
        (token_ids.view(1, 1, S, 1) * const)
        .expand(batch, num_heads, S, HEAD_DIM)
        .to(torch.bfloat16)
    )

    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    index = torch.zeros(batch, num_heads, 1, 1, dtype=torch.int32, device=device)
    # Large softmax_scale sharpens the (already diagonal-dominant) score
    # matrix into a near-one-hot distribution.
    out = bsa_attn_sm120_blk64_sage_fwd(
        q8, k8, v8, qs, ks, vs, index, block_sparse_num=1, softmax_scale=30.0
    )
    out_mean = out.float().mean(dim=-1)  # [batch, num_heads, S]

    # The correctness bar is the FP8-rounded V value (V itself is quantized
    # to float8_e4m3fn, ~6% per-octave precision), not the pre-quantization
    # float value -- comparing against the latter would conflate V's
    # inherent quantization error with an actual permutation bug.
    v_scale_bc = vs.view(batch, num_heads, 1, HEAD_DIM)
    v_fp8_sim = (v.float() / v_scale_bc).to(torch.float8_e4m3fn).float()
    expected = (v_fp8_sim * v_scale_bc).mean(dim=-1)  # [batch, num_heads, S]

    # Tight tolerance: correct behavior gives ~1e-3 residual (softmax not
    # perfectly one-hot); a real permutation bug would misroute row t to a
    # *different* token's fingerprint, which is spaced at least `const`
    # apart -- an order of magnitude larger than this bound.
    torch.testing.assert_close(out_mean, expected, atol=3e-3, rtol=0.0)


# ---------------------------------------------------------------------------
# Quantization unit tests
# ---------------------------------------------------------------------------


def test_quantize_sage_q_shapes_and_range():
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, heads, seqlen = 2, 3, 150  # not a multiple of 128 -> partial last group
    q = _random_bf16_bhsd(batch, heads, seqlen, device, scale=1.0)

    q8, q_scale = quantize_sage_q_sm120(q)
    assert q8.shape == q.shape
    assert q8.dtype == torch.int8
    num_groups = ((seqlen + 127) // 128) * 4
    assert q_scale.shape == (batch, heads, num_groups)
    assert q8.abs().max() <= 127

    # Dequantized value should be close to the original (per-32-token-group scale).
    group_of_row = (torch.arange(seqlen, device=device) // 32).clamp(max=num_groups - 1)
    scale_per_row = q_scale[:, :, group_of_row]  # [B, H, S]
    dequant = q8.float() * scale_per_row.unsqueeze(-1)
    torch.testing.assert_close(dequant, q.float(), atol=5e-2, rtol=5e-2)


def test_quantize_sage_kv_shapes_and_v_scale_cap():
    device = torch.device("cuda")
    torch.manual_seed(1)
    batch, heads, seqlen = 1, 2, 96  # not a multiple of 64
    k = _random_bf16_bhsd(batch, heads, seqlen, device, scale=1.0)
    v = _random_bf16_bhsd(batch, heads, seqlen, device, scale=1.0)

    k8, v8, k_scale, v_scale = quantize_sage_kv_sm120(k, v)
    padded = ((seqlen + BLOCK - 1) // BLOCK) * BLOCK
    assert k8.shape == k.shape
    assert k8.dtype == torch.int8
    assert v8.shape == (batch, heads, HEAD_DIM, padded)
    assert v8.dtype == torch.float8_e4m3fn
    assert k_scale.shape == (batch, heads, padded // BLOCK)
    assert v_scale.shape == (batch, heads, HEAD_DIM)
    # SAGE_V_SCALE_MAX cap: scale = max(amax, eps) / 2.25, so amax/scale should
    # not exceed 2.25 (up to fp32 rounding).
    assert (v.float().abs().amax(dim=2) / v_scale <= 2.25 + 1e-3).all()


@pytest.mark.parametrize(
    "bad_tensor_name,mutate",
    [
        ("q_scale", lambda t: t[:, :, :1].contiguous()),
        ("k_scale", lambda t: t[:, :, :1].contiguous()),
        ("v_scale", lambda t: t[:, :, :1].contiguous()),
        ("k_int8", lambda t: t[:, :1].contiguous()),  # wrong head count (not MHA)
    ],
)
def test_vsa_sm120_sage_rejects_mismatched_shapes(bad_tensor_name, mutate):
    """Wrong-shaped Q/K scale tensors or a non-MHA K must raise, not silently
    compute a wrong result (regression test for a bug found in review: an
    under-sized q_scale used to run to completion and return a finite but
    numerically wrong output)."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, num_heads, num_blocks = 1, 2, 2
    seqlen = num_blocks * BLOCK

    q = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    k = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    v = _random_bf16_bhsd(batch, num_heads, seqlen, device)
    q8, k8, v8, qs, ks, vs = quantize_sage_qkv_sm120(q, k, v)

    tensors = {
        "q_int8": q8,
        "k_int8": k8,
        "v_fp8": v8,
        "q_scale": qs,
        "k_scale": ks,
        "v_scale": vs,
    }
    tensors[bad_tensor_name] = mutate(tensors[bad_tensor_name])

    index = (
        torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        .expand(batch, num_heads, num_blocks, num_blocks)
        .contiguous()
    )

    with pytest.raises(ValueError):
        bsa_attn_sm120_blk64_sage_fwd(
            tensors["q_int8"],
            tensors["k_int8"],
            tensors["v_fp8"],
            tensors["q_scale"],
            tensors["k_scale"],
            tensors["v_scale"],
            index,
            block_sparse_num=num_blocks,
        )
