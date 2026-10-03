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
    if backend == "cake_gdn" and (major, minor) not in ((10, 0), (10, 3)):
        pytest.skip("cake_gdn prefill requires SM100 or SM103")
    if major not in (9, 10, 12):
        pytest.skip("state_indices GDN prefill path requires SM90, SM100, or SM120")
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
