"""Merge kernels that give each head its own row of threads must handle more heads than
one block can hold.

MergeState, MergeStateInPlace, MergeStates (``num_index_sets < seq_len``) and
AttentionSum launch ``head_dim / vec_size`` threads per head. Past 1024 threads per
block (e.g. more than 64 heads of dim 128 in a 16-bit type) the launch failed with
"invalid configuration argument", which also broke the split-KV path of single prefill.
"""

import math

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import ref_single_prefill

# (head_dim, num_heads); the smaller count per head_dim is the largest that fit in one
# block before heads were tiled over gridDim.y.
HEAD_CONFIGS = [
    (64, 128),
    (64, 160),
    (128, 64),
    (128, 96),
    (128, 128),
    (256, 32),
    (256, 64),
    (512, 32),
    (512, 48),
]


def ref_merge_state(v_a, s_a, v_b, s_b):
    s_merged = torch.logaddexp2(s_a, s_b)
    w_a = torch.exp2(s_a - s_merged).unsqueeze(-1)
    w_b = torch.exp2(s_b - s_merged).unsqueeze(-1)
    return w_a * v_a.float() + w_b * v_b.float(), s_merged


def ref_merge_states(v, s):
    s_merged = torch.logsumexp(s * math.log(2.0), dim=1) / math.log(2.0)
    w = torch.exp2(s - s_merged.unsqueeze(1)).unsqueeze(-1)
    return (w * v.float()).sum(dim=1), s_merged


def _rand_state(*shape, dtype):
    v = torch.randn(*shape, dtype=dtype, device="cuda")
    s = torch.randn(*shape[:-1], dtype=torch.float32, device="cuda")
    return v, s


@pytest.mark.parametrize("head_dim,num_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_merge_state_many_heads(head_dim, num_heads, dtype):
    torch.manual_seed(0)
    seq_len = 17
    v_a, s_a = _rand_state(seq_len, num_heads, head_dim, dtype=dtype)
    v_b, s_b = _rand_state(seq_len, num_heads, head_dim, dtype=dtype)

    v, s = flashinfer.merge_state(v_a, s_a, v_b, s_b)

    v_ref, s_ref = ref_merge_state(v_a, s_a, v_b, s_b)
    torch.testing.assert_close(v.float(), v_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(s, s_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("head_dim,num_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_merge_state_in_place_many_heads(head_dim, num_heads, dtype):
    torch.manual_seed(0)
    seq_len = 17
    v, s = _rand_state(seq_len, num_heads, head_dim, dtype=dtype)
    v_other, s_other = _rand_state(seq_len, num_heads, head_dim, dtype=dtype)
    mask = torch.arange(seq_len, device="cuda") % 3 != 0
    v_orig, s_orig = v.clone(), s.clone()

    flashinfer.merge_state_in_place(v, s, v_other, s_other, mask=mask)

    v_ref, s_ref = ref_merge_state(v_orig, s_orig, v_other, s_other)
    torch.testing.assert_close(v[mask].float(), v_ref[mask], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(s[mask], s_ref[mask], rtol=1e-3, atol=1e-3)
    # Masked-out positions stay untouched.
    torch.testing.assert_close(v[~mask], v_orig[~mask], rtol=0, atol=0)
    torch.testing.assert_close(s[~mask], s_orig[~mask], rtol=0, atol=0)


# seq_len > num_index_sets takes the per-position MergeStatesKernel; the other case takes
# MergeStatesLargeNumIndexSetsKernel, which already tiled heads over the grid.
@pytest.mark.parametrize("seq_len,num_index_sets", [(17, 3), (3, 17)])
@pytest.mark.parametrize("head_dim,num_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_merge_states_many_heads(seq_len, num_index_sets, head_dim, num_heads, dtype):
    torch.manual_seed(0)
    v, s = _rand_state(seq_len, num_index_sets, num_heads, head_dim, dtype=dtype)

    v_merged, s_merged = flashinfer.merge_states(v, s)

    v_ref, s_ref = ref_merge_states(v, s)
    torch.testing.assert_close(v_merged.float(), v_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(s_merged, s_ref, rtol=1e-3, atol=1e-3)


# A short query over few KV heads with a large GQA group leaves SMs idle, so single
# prefill splits the KV range and merges the partial states with MergeStates over
# (qo_len, num_qo_heads); qo_len > num_chunks selects the per-position kernel.
@pytest.mark.parametrize("head_dim,num_qo_heads", [(128, 128), (256, 64)])
@pytest.mark.parametrize("qo_len", [16, 48])
def test_single_prefill_split_kv_many_heads(head_dim, num_qo_heads, qo_len):
    torch.manual_seed(0)
    num_kv_heads, kv_len = 8, 4096
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)

    o_ref, _ = ref_single_prefill(q, k, v, causal=False)
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
