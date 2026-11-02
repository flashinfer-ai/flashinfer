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

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

from flashinfer.utils import get_compute_capability, is_sm100a_supported
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.jit.cake_gdn import CakeGDNUnsupportedError


INTEGER_DTYPES = (
    torch.int32,
    torch.int64,
)

# Cake CP is selected explicitly by backend="cake_gdn", use_cp=True.
# auto/flashinfer retain the baseline implementation for both CP and non-CP.
BACKEND_CP_CASES = (
    pytest.param("flashinfer", False, id="flashinfer-non-cp"),
    pytest.param("flashinfer", True, id="flashinfer-cp"),
    pytest.param("cake_gdn", False, id="cake_gdn-non-cp"),
    pytest.param("cake_gdn", True, id="cake_gdn-cp"),
)

# Non-CP Cake variants cover BF16 H=16; Cake CP shares the H=8 baseline cases.
BACKEND_CP_HEAD_CASES = (
    pytest.param("flashinfer", False, 8, id="flashinfer-non-cp"),
    pytest.param("flashinfer", True, 8, id="flashinfer-cp"),
    pytest.param("cake_gdn", False, 16, id="cake_gdn-non-cp"),
    pytest.param("cake_gdn", True, 8, id="cake_gdn-cp"),
)


def _skip_if_not_supported(backend, use_cp):
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    if backend == "cake_gdn" and (major, minor) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("cake_gdn prefill requires SM100, SM103 or SM107")
    if backend == "cake_gdn" and use_cp is True and (major, minor) == (10, 7):
        pytest.skip("Cake GDN CP prefill is not provided on SM107")
    if major not in (8, 9, 10, 12):
        pytest.skip(
            "state_indices GDN prefill path requires SM8x, SM90, SM100, or SM120"
        )
    cuda_version = tuple(
        int(part) for part in (torch.version.cuda or "0.0").split(".")[:2]
    )
    cuda_major = cuda_version[0]
    if backend == "cake_gdn" and use_cp is True and cuda_version < (12, 8):
        pytest.skip(f"Cake GDN CP requires CUDA 12.8+, got {torch.version.cuda}")
    if backend != "cake_gdn" and is_sm100a_supported(device) and cuda_major < 13:
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
    state_dtype = (
        torch.float32 if get_compute_capability(device)[0] in (9, 12) else dtype
    )
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
    backend,
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
        max_seqlen=total,
        backend=backend,
    )
    return output, final


def _make_pool(init_state, perm, n_pool, pad, dtype, device, inner_stride=1):
    """Build a state pool holding init_state[i] at row perm[i].

    ``pad > 0`` gives the pool a padded first-dimension stride, matching a state
    pool packed with other per-slot data. ``inner_stride > 1`` additionally
    exercises a fully strided ``[H, V, K]`` view.
    """
    _, H, D, _ = init_state.shape
    if pad == 0 and inner_stride == 1:
        pool = torch.zeros(n_pool, H, D, D, dtype=dtype, device=device)
    else:
        slot_stride = H * D * D * inner_stride + pad
        storage = torch.zeros(n_pool * slot_stride, dtype=dtype, device=device)
        pool = storage.as_strided(
            (n_pool, H, D, D),
            (
                slot_stride,
                D * D * inner_stride,
                D * inner_stride,
                inner_stride,
            ),
        )
        assert not pool.is_contiguous()
    for i, r in enumerate(perm):
        pool[r] = init_state[i]
    return pool


@pytest.mark.parametrize(
    "dtype,H,backend,use_cp",
    [
        (dtype, H, "flashinfer", use_cp)
        for dtype in (torch.bfloat16, torch.float16)
        for H in (8, 16, 32)
        for use_cp in (False, True)
    ]
    + [(torch.bfloat16, 16, "cake_gdn", False)]
    + [
        (dtype, H, "cake_gdn", True)
        for dtype in (torch.bfloat16, torch.float16)
        for H in (8, 16, 32)
    ],
)
@pytest.mark.parametrize(
    "seq_lens",
    [[128], [256], [128, 192, 64], [64, 512]],
)
@pytest.mark.parametrize("pad", [0, 96])  # 0 = compact pool, 96 = non-compact
def test_prefill_state_indices_matches_packed(dtype, seq_lens, H, pad, backend, use_cp):
    """A pool + state_indices in-place update must match the packed,
    sequence-ordered baseline bitwise (the kernel math is identical; only the
    addressed gmem row differs)."""
    _skip_if_not_supported(backend, use_cp)
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
        backend,
    )
    torch.cuda.synchronize()

    # (b) indexed pool, in-place output_state == initial_state
    n_pool = num_seqs + 5
    perm = [(i * 3 + 2) % n_pool for i in range(num_seqs)]
    assert len(set(perm)) == num_seqs  # distinct slots
    pool = _make_pool(init_state, perm, n_pool, pad, state_dtype, device)
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    output_b, _ = _run(q, k, v, g, beta, cu_seqlens, pool, pool, idx, use_cp, backend)
    torch.cuda.synchronize()

    assert not torch.isnan(output_b).any()
    assert torch.equal(output_a, output_b), "output differs from packed baseline"
    # final states landed in the indexed pool rows
    assert torch.equal(final_a, pool[perm]), "final state differs from baseline"

    # untouched pool rows stay zero (only requested rows written)
    untouched = [r for r in range(n_pool) if r not in perm]
    assert torch.equal(pool[untouched], torch.zeros_like(pool[untouched]))


def test_state_stride_divisibility():
    """The wrapper only assumes stride divisibility the pool satisfies."""
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
        _state_stride_divisibility,
    )

    H, D = 4, 128
    hvk = H * D * D
    compact = torch.empty(6, H, D, D)
    assert _state_stride_divisibility((compact, compact), D) == D
    # Several layers coalesced into each slot
    coalesced = torch.empty(6, 3, H, D, D)[:, 1]
    assert coalesced.stride(0) == 3 * hvk
    assert _state_stride_divisibility((coalesced, None), D) == D
    # Slot padded by 96 elements
    padded = torch.empty(6, hvk + 96)[:, :hvk].unflatten(1, (H, D, D))
    assert _state_stride_divisibility((padded, padded), D) == 32
    # The most constrained state wins
    assert _state_stride_divisibility((compact, padded), D) == 32
    # Non-unit K stride: no assumption
    strided_k = torch.empty(6, H, D, 2 * D)[..., ::2]
    assert _state_stride_divisibility((strided_k,), D) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill_state_indices_compact_and_padded_pools_in_one_process(dtype):
    """Alternating compact and padded pools must not reuse a mismatched kernel."""
    _skip_if_not_supported("flashinfer", False)
    device = torch.device("cuda")
    D, H = 128, 16
    seq_lens = [64, 32, 96, 16, 48, 80, 8, 128]
    num_seqs = len(seq_lens)
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, dtype, device, seed=7
    )
    packed_state = torch.empty_like(init_state)
    ref_output, ref_final = _run(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        init_state.clone(),
        packed_state,
        None,
        False,
        "flashinfer",
    )

    n_pool = num_seqs + 3
    perm = [(i * 5 + 1) % n_pool for i in range(num_seqs)]
    assert len(set(perm)) == num_seqs
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    for pad in (0, 96, 0, 96):
        pool = _make_pool(init_state, perm, n_pool, pad, init_state.dtype, device)
        output, _ = _run(
            q, k, v, g, beta, cu_seqlens, pool, pool, idx, False, "flashinfer"
        )
        torch.cuda.synchronize()
        assert torch.equal(output, ref_output), f"output differs (pad={pad})"
        assert torch.equal(pool[perm], ref_final), f"final state differs (pad={pad})"


@pytest.mark.parametrize("index_dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("backend,use_cp,H", BACKEND_CP_HEAD_CASES)
def test_prefill_integer_index_dtypes(index_dtype, backend, use_cp, H):
    """Sequence and state indices retain their integer dtype across dispatch."""
    _skip_if_not_supported(backend, use_cp)
    device = torch.device("cuda")
    D = 128
    seq_lens = [64]
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=5
    )
    cu_seqlens = cu_seqlens.to(index_dtype)

    packed_state = torch.empty_like(init_state)
    packed_output, packed_final = _run(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        init_state.clone(),
        packed_state,
        None,
        use_cp,
        backend,
    )

    slots = [2]
    pool = _make_pool(init_state, slots, 4, 96, init_state.dtype, device)
    state_indices = torch.tensor(slots, dtype=index_dtype, device=device)
    indexed_output, indexed_final = _run(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        pool,
        pool,
        state_indices,
        use_cp,
        backend,
    )
    torch.cuda.synchronize()

    assert torch.equal(packed_output, indexed_output)
    assert torch.equal(packed_final, indexed_final[slots])


@pytest.mark.parametrize("backend,use_cp,H", BACKEND_CP_HEAD_CASES)
def test_prefill_state_indices_preserves_inner_strides(backend, use_cp, H):
    """Indexed views preserve inner strides or reject unsupported layouts."""
    _skip_if_not_supported(backend, use_cp)
    device = torch.device("cuda")
    D = 128
    seq_lens = [64]
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=6
    )

    packed_state = torch.empty_like(init_state)
    packed_output, packed_final = _run(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        init_state.clone(),
        packed_state,
        None,
        use_cp,
        backend,
    )

    slots = [2]
    state_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    contiguous_pool = _make_pool(init_state, slots, 4, 96, init_state.dtype, device)
    contiguous_output, contiguous_final = _run(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        contiguous_pool,
        contiguous_pool,
        state_indices,
        use_cp,
        backend,
    )
    assert torch.equal(packed_output, contiguous_output)
    assert torch.equal(packed_final, contiguous_final[slots])

    pool = _make_pool(
        init_state,
        slots,
        4,
        96,
        init_state.dtype,
        device,
        inner_stride=2,
    )
    if backend == "cake_gdn" and use_cp is not True:
        with pytest.raises(CakeGDNUnsupportedError, match=r"contiguous \[H,V,K\] rows"):
            _run(
                q,
                k,
                v,
                g,
                beta,
                cu_seqlens,
                pool,
                pool,
                state_indices,
                use_cp,
                backend,
            )
        return
    indexed_output, indexed_final = _run(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        pool,
        pool,
        state_indices,
        use_cp,
        backend,
    )
    torch.cuda.synchronize()

    assert torch.equal(packed_output, indexed_output)
    assert torch.equal(packed_final, indexed_final[slots])


def _make_transposed_pool(init_state, perm, n_pool, dtype, device, fill):
    """Build a pool backed by an ``[N_pool, H, K, V]`` buffer and passed as its
    ``[N_pool, H, V, K]`` transpose, so K is not the unit-stride dimension."""
    _, H, D, _ = init_state.shape
    storage = torch.full((n_pool, H, D, D), fill, dtype=dtype, device=device)
    pool = storage.transpose(-1, -2)
    assert pool.stride(-1) == D and pool.stride(-2) == 1
    for i, r in enumerate(perm):
        pool[r] = init_state[i]
    return pool


@pytest.mark.parametrize(
    "seq_lens,H",
    [
        pytest.param([1024], 8, id="8-states"),
        pytest.param([256, 244, 500], 4, id="varlen-12-states"),
        pytest.param([1024, 1024], 8, id="16-states"),
        pytest.param([4096], 32, id="32-states"),
        pytest.param([512, 512, 512, 512], 32, id="128-states"),
    ],
)
@pytest.mark.parametrize("transposed", ["initial", "output", "in-place"])
def test_prefill_cp_transposed_state_pools(seq_lens, H, transposed):
    """The SM100/SM103 CP route honors [N, H, K, V]-backed (transposed) initial
    and output state pools: outputs and final states match contiguous pools,
    every requested row is written and no other row is touched. The cases cover
    the SIMT (<= 9 states) and the UTCMMA fixup state counts."""
    _skip_if_not_supported("flashinfer", True)
    device = torch.device("cuda")
    if not is_sm100a_supported(device):
        pytest.skip(
            "arbitrary pool layouts are only documented on SM100/SM103, and "
            "only the SM100 CP fixup is covered by this regression test"
        )
    D = 128
    num_seqs = len(seq_lens)
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=7
    )
    state_dtype = init_state.dtype
    n_pool = 2 * num_seqs + 1
    perm = [2 * (num_seqs - i) - 1 for i in range(num_seqs)]  # reversed odd rows
    untouched = [r for r in range(n_pool) if r not in perm]
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    nan = float("nan")

    ref_initial = _make_pool(init_state, perm, n_pool, 0, state_dtype, device)
    ref_final = torch.full_like(ref_initial, nan)
    ref_output, _ = _run(
        q, k, v, g, beta, cu_seqlens, ref_initial, ref_final, idx, True, "flashinfer"
    )

    if transposed == "in-place":
        initial = _make_transposed_pool(
            init_state, perm, n_pool, state_dtype, device, nan
        )
        final = initial
    elif transposed == "initial":
        initial = _make_transposed_pool(
            init_state, perm, n_pool, state_dtype, device, 0.0
        )
        final = torch.full_like(ref_initial, nan)
    else:
        initial = ref_initial.clone()
        final = _make_transposed_pool(
            torch.zeros_like(init_state), [], n_pool, state_dtype, device, nan
        )
    output, _ = _run(
        q, k, v, g, beta, cu_seqlens, initial, final, idx, True, "flashinfer"
    )
    torch.cuda.synchronize()

    assert torch.isfinite(final[perm]).all(), (
        "requested final-state rows not fully written"
    )
    assert torch.isnan(final[untouched]).all(), (
        "rows outside state_indices were written"
    )
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final[perm], ref_final[perm], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "backend,use_cp",
    [("auto", "auto"), ("cake_gdn", "auto"), ("cake_gdn", True)],
)
def test_prefill_state_indices_requires_output_state_pool(backend, use_cp):
    """With state_indices set, output_state must be a caller-provided pool: an
    auto-allocated compact [num_seqs, ...] tensor would be indexed out of bounds
    by the pool slot ids, so output_state=None must be rejected."""
    _skip_if_not_supported(backend, use_cp)
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
            backend=backend,
            use_cp=use_cp,
        )


@pytest.mark.parametrize("backend,use_cp", BACKEND_CP_CASES)
def test_prefill_state_indices_without_final_state(backend, use_cp):
    """Backends support read-only state pools or explicitly reject them."""
    _skip_if_not_supported(backend, use_cp)
    device = torch.device("cuda")
    H, D = 16, 128
    seq_lens = [64, 512]
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=4
    )

    slots = [3, 0]
    pool = _make_pool(init_state, slots, 5, 96, init_state.dtype, device)
    state_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    if backend == "cake_gdn" and use_cp is not True:
        with pytest.raises(
            CakeGDNUnsupportedError,
            match="indexed state requires initial and final state",
        ):
            chunk_gated_delta_rule(
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
                backend=backend,
                max_seqlen=max(seq_lens),
            )
        return

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
        backend=backend,
        max_seqlen=max(seq_lens),
    )

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
        backend=backend,
        max_seqlen=max(seq_lens),
    )
    torch.cuda.synchronize()

    assert torch.equal(packed_output, indexed_output)


def test_prefill_state_indices_read_only_pool_scratch_spans_pool(monkeypatch):
    """SM90 stores every final state, to row state_indices[i] when indexed, so
    the scratch it gets for a read-only pool must span the pool. A compact
    [num_seqs, ...] scratch took writes past its end that only faulted when
    the next page happened to be unmapped."""
    import flashinfer.gdn_prefill as gdn_prefill

    device = torch.device("cuda")
    if get_compute_capability(device)[0] != 9:
        pytest.skip("checks the SM90 dispatch")
    H, D = 16, 128
    seq_lens = [64, 512]
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=4
    )
    slots = [3, 0]
    pool = _make_pool(init_state, slots, 5, 96, init_state.dtype, device)
    state_indices = torch.tensor(slots, dtype=torch.int32, device=device)

    seen = []

    def record(output, output_state, *args, state_indices=None):
        seen.append((tuple(output_state.shape), state_indices))

    monkeypatch.setattr(gdn_prefill, "chunk_gated_delta_rule_sm90", record)
    for initial_state in (pool, None):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            use_cp=False,
            backend="flashinfer",
            max_seqlen=max(seq_lens),
        )
    (pool_shape, pool_indices), (no_pool_shape, no_pool_indices) = seen
    assert pool_shape == tuple(pool.shape) and pool_indices is state_indices
    # Nothing to read or write through: the slot ids are dropped.
    assert no_pool_shape == (len(seq_lens), H, D, D) and no_pool_indices is None


@pytest.mark.parametrize("backend,use_cp", BACKEND_CP_CASES)
def test_prefill_state_indices_none_is_default(backend, use_cp):
    """state_indices=None must reproduce the packed path exactly (default)."""
    _skip_if_not_supported(backend, use_cp)
    device = torch.device("cuda")
    H, D = 16, 128
    seq_lens = [128, 192]
    num_seqs = len(seq_lens)
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=1
    )
    s1 = torch.empty(num_seqs, H, D, D, dtype=init_state.dtype, device=device)
    o1, f1 = _run(
        q, k, v, g, beta, cu_seqlens, init_state.clone(), s1, None, use_cp, backend
    )
    s2 = torch.empty(num_seqs, H, D, D, dtype=init_state.dtype, device=device)
    o2, f2 = _run(
        q, k, v, g, beta, cu_seqlens, init_state.clone(), s2, None, use_cp, backend
    )
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)
    assert torch.equal(f1, f2)


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_pools_of_different_sizes(use_cp):
    """Separate initial and output pools keep their own size, padding and strides."""
    _skip_if_not_supported("flashinfer", use_cp)
    device = torch.device("cuda")
    *inputs, init_state = _make_inputs([256, 512], 8, 128, torch.bfloat16, device, 0)
    packed_output, packed_final = _run(
        *inputs, init_state, torch.empty_like(init_state), None, use_cp, "flashinfer"
    )

    perm = [3, 1]
    in_pool = _make_pool(init_state, perm, 9, 96, init_state.dtype, device)
    out_pool = _make_pool(
        torch.zeros_like(init_state), [], 5, 0, init_state.dtype, device, 2
    )
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    output, _ = _run(*inputs, in_pool, out_pool, idx, use_cp, "flashinfer")
    assert torch.equal(packed_output, output)
    assert torch.equal(packed_final, out_pool[perm])
    assert not out_pool[[0, 2, 4]].any()


def _run_invalid_slot_child(use_cp):
    """Launch with a slot past the pool, in a child: the device assert poisons CUDA."""
    device = torch.device("cuda")
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        [128], 8, 128, torch.bfloat16, device, seed=0
    )
    pool = _make_pool(init_state, [1], 3, 0, torch.float32, device)
    idx = torch.tensor([pool.shape[0]], dtype=torch.int32, device=device)
    _run(q, k, v, g, beta, cu_seqlens, pool, pool, idx, use_cp, "flashinfer")
    torch.cuda.synchronize()


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_out_of_pool_fails_in_isolated_process(use_cp):
    """An id past the pool is caught; unchecked, it silently hits a neighbour tensor."""
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    env["FLASHINFER_CAKE_GDN_VALIDATE_SLOTS"] = "1"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--invalid-slot", str(use_cp)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    combined = completed.stdout + completed.stderr
    # Only a line starting with the marker is a skip.
    skips = [ln for ln in combined.splitlines() if ln.startswith("SKIP:")]
    if skips:
        pytest.skip(skips[-1])
    assert completed.returncode != 0, combined
    assert "_assert_async_cuda_kernel" in combined, combined
    assert "GDN prefill state_indices must contain slots in" in combined, combined


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--invalid-slot":
    # A plain script: the parent turns this marker into the skip.
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
    elif get_compute_capability(torch.device("cuda"))[0] not in (8, 9, 10, 12):
        print("SKIP: no GDN prefill kernel for this device")
    elif (
        is_sm100a_supported(torch.device("cuda"))
        and (int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0) < 13
    ):
        print("SKIP: SM100 GDN prefill requires CUDA 13+")
    else:
        _run_invalid_slot_child(sys.argv[2] == "True")
