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

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from flashinfer.utils import get_compute_capability, is_sm100a_supported
from flashinfer.gdn_prefill import chunk_gated_delta_rule


def _skip_if_not_sm90_or_sm100():
    device = torch.device("cuda")
    major, _ = get_compute_capability(device)
    if major not in (9, 10):
        pytest.skip("state_indices GDN prefill path requires SM90 or SM100/SM103")
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    if is_sm100a_supported(device) and cuda_major < 13:
        pytest.skip(f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}")


def _make_inputs(seq_lens, H, D, dtype, device, seed):
    torch.manual_seed(seed)
    total = sum(seq_lens)
    num_seqs = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    q = (
        F.normalize(
            torch.randn(total, H, D, dtype=torch.float32, device=device), dim=-1
        )
        .to(dtype)
        .contiguous()
    )
    k = (
        F.normalize(
            torch.randn(total, H, D, dtype=torch.float32, device=device), dim=-1
        )
        .to(dtype)
        .contiguous()
    )
    v = torch.randn(total, H, D, dtype=dtype, device=device).contiguous()
    # FlashInfer consumes linear-space alpha = exp(log_g)
    g_log = -F.softplus(
        torch.randn(total, H, dtype=torch.float32, device=device) * 0.5 - 2.0
    )
    g = torch.exp(g_log).contiguous()
    beta = torch.rand(total, H, dtype=torch.float32, device=device).contiguous()
    state_dtype = torch.float32 if get_compute_capability(device)[0] == 9 else dtype
    init_state = torch.randn(
        num_seqs, H, D, D, dtype=state_dtype, device=device
    ).contiguous()
    return q, k, v, g, beta, cu_seqlens, init_state


def _run(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    initial_state,
    output_state,
    state_indices,
    use_cp,
):
    total, H, D = q.shape
    out = torch.empty(total, H, D, dtype=q.dtype, device=q.device)
    output, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        None,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out,
        output_state=output_state,
        state_indices=state_indices,
        use_cp=use_cp,
    )
    return output, final


def _make_pool(init_state, perm, n_pool, pad, dtype, device):
    """Build a state pool holding init_state[i] at row perm[i].

    ``pad > 0`` gives the pool a padded (non-compact) *first-dimension* stride
    while keeping the inner ``[H, V, K]`` block fully contiguous -- exactly
    TRT-LLM's mamba cache layout, where conv+ssm are packed per slot so only the
    per-slot (dim-0) stride is padded. Built via ``as_strided`` over flat storage
    so the padding lands only between slots, not inside the state block.
    """
    _, H, D, _ = init_state.shape
    if pad == 0:
        pool = torch.zeros(n_pool, H, D, D, dtype=dtype, device=device)
    else:
        slot_stride = H * D * D + pad  # usable state block + inter-slot padding
        storage = torch.zeros(n_pool * slot_stride, dtype=dtype, device=device)
        pool = storage.as_strided((n_pool, H, D, D), (slot_stride, D * D, D, 1))
        assert not pool.is_contiguous()  # dim-0 stride padded
        assert pool.stride()[1:] == (D * D, D, 1)  # inner [H, V, K] contiguous
    for i, r in enumerate(perm):
        pool[r] = init_state[i]
    return pool


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "seq_lens",
    [[128], [256], [128, 192, 64], [64, 512]],
)
@pytest.mark.parametrize("H", [8, 16, 32])
@pytest.mark.parametrize("pad", [0, 96])  # 0 = compact pool, 96 = non-compact
@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_matches_packed(dtype, seq_lens, H, pad, use_cp):
    """A pool + state_indices in-place update must match the packed,
    sequence-ordered baseline bitwise (the kernel math is identical; only the
    addressed gmem row differs)."""
    _skip_if_not_sm90_or_sm100()
    device = torch.device("cuda")
    D = 128
    num_seqs = len(seq_lens)
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, dtype, device, seed=0
    )

    # (a) packed baseline, no state_indices
    state_dtype = init_state.dtype
    out_state_a = torch.empty(num_seqs, H, D, D, dtype=state_dtype, device=device)
    output_a, final_a = _run(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        init_state.clone(),
        out_state_a,
        None,
        use_cp,
    )
    torch.cuda.synchronize()

    # (b) indexed pool, in-place output_state == initial_state
    n_pool = num_seqs + 5
    perm = [(i * 3 + 2) % n_pool for i in range(num_seqs)]
    assert len(set(perm)) == num_seqs  # distinct slots
    pool = _make_pool(init_state, perm, n_pool, pad, state_dtype, device)
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    output_b, _ = _run(q, k, v, g, beta, cu_seqlens, pool, pool, idx, use_cp)
    torch.cuda.synchronize()

    assert not torch.isnan(output_b).any()
    assert torch.equal(output_a, output_b), "output differs from packed baseline"
    # final states landed in the indexed pool rows
    assert torch.equal(final_a, pool[perm]), "final state differs from baseline"

    # untouched pool rows stay zero (only requested rows written)
    untouched = [r for r in range(n_pool) if r not in perm]
    assert torch.equal(pool[untouched], torch.zeros_like(pool[untouched]))


def test_prefill_state_indices_requires_output_state_pool():
    """With state_indices set, output_state must be a caller-provided pool: an
    auto-allocated compact [num_seqs, ...] tensor would be indexed out of bounds
    by the pool slot ids, so output_state=None must be rejected."""
    _skip_if_not_sm90_or_sm100()
    device = torch.device("cuda")
    H, D = 16, 128
    seq_lens = [128, 64]
    num_seqs = len(seq_lens)
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=2
    )
    n_pool = num_seqs + 3
    perm = list(range(num_seqs))
    pool = _make_pool(init_state, perm, n_pool, 0, init_state.dtype, device)
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    out = torch.empty(sum(seq_lens), H, D, dtype=torch.bfloat16, device=device)
    # On supported SM90/SM100 paths this must be the output_state ValueError,
    # not NotImplementedError (which would mean the kernel was wrongly rejected).
    with pytest.raises(ValueError, match="explicit output_state pool"):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            None,
            initial_state=pool,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=out,
            output_state=None,  # must be rejected when state_indices is set
            state_indices=idx,
        )


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_with_state_checkpoints(use_cp):
    """Indexed final-state I/O must not change packed checkpoint ordering."""
    _skip_if_not_sm90_or_sm100()
    device = torch.device("cuda")
    if use_cp and get_compute_capability(device)[0] not in (9, 10):
        pytest.skip("CP state checkpointing requires SM90 or SM100")
    H, D = 16, 128
    seq_lens = [128, 512]
    num_seqs = len(seq_lens)
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=3
    )
    checkpoint_every = 64
    checkpoint_counts = [seq_len // checkpoint_every for seq_len in seq_lens]
    checkpoint_cu_starts = torch.tensor(
        [0, *torch.cumsum(torch.tensor(checkpoint_counts), 0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    checkpoint_shape = (sum(checkpoint_counts), H, D, D)

    packed_state = torch.empty_like(init_state)
    packed_checkpoints = torch.empty(
        checkpoint_shape, dtype=init_state.dtype, device=device
    )
    packed_output = torch.empty_like(q)
    output_a, final_a = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=init_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output=packed_output,
        output_state=packed_state,
        state_checkpoints=packed_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every,
        use_cp=use_cp,
    )

    n_pool = num_seqs + 4
    slots = [4, 1]
    pool = _make_pool(init_state, slots, n_pool, 96, init_state.dtype, device)
    state_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    indexed_checkpoints = torch.empty_like(packed_checkpoints)
    indexed_output = torch.empty_like(q)
    output_b, final_b = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output=indexed_output,
        output_state=pool,
        state_checkpoints=indexed_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every,
        state_indices=state_indices,
        use_cp=use_cp,
    )
    torch.cuda.synchronize()

    assert torch.equal(output_a, output_b)
    assert torch.equal(final_a, final_b[slots])
    assert torch.equal(packed_checkpoints, indexed_checkpoints)
    if use_cp:
        reference_checkpoints = torch.empty_like(packed_checkpoints)
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=init_state.clone(),
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            state_checkpoints=reference_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every,
            use_cp=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            packed_checkpoints,
            reference_checkpoints,
            atol=1e-2,
            rtol=5e-3,
        )


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_without_final_state(use_cp):
    """A state pool can supply initial state without requesting a final state."""
    _skip_if_not_sm90_or_sm100()
    device = torch.device("cuda")
    H, D = 16, 128
    seq_lens = [64, 512]
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=4
    )

    packed_output = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=init_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_cp=use_cp,
    )

    slots = [3, 0]
    pool = _make_pool(init_state, slots, 5, 96, init_state.dtype, device)
    state_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    indexed_output = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        use_cp=use_cp,
    )
    torch.cuda.synchronize()

    assert torch.equal(packed_output, indexed_output)


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_none_is_default(use_cp):
    """state_indices=None must reproduce the packed path exactly (default)."""
    _skip_if_not_sm90_or_sm100()
    device = torch.device("cuda")
    H, D = 16, 128
    seq_lens = [128, 192]
    num_seqs = len(seq_lens)
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=1
    )
    s1 = torch.empty(num_seqs, H, D, D, dtype=init_state.dtype, device=device)
    o1, f1 = _run(q, k, v, g, beta, cu_seqlens, init_state.clone(), s1, None, use_cp)
    s2 = torch.empty(num_seqs, H, D, D, dtype=init_state.dtype, device=device)
    o2, f2 = _run(q, k, v, g, beta, cu_seqlens, init_state.clone(), s2, None, use_cp)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)
    assert torch.equal(f1, f2)
