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


INTEGER_DTYPES = (
    torch.int32,
    torch.int64,
)


def _skip_if_not_supported():
    """Skip where no GDN prefill kernel exists for this device."""
    device = torch.device("cuda")
    major, _ = get_compute_capability(device)
    if major not in (8, 9, 10, 12):
        pytest.skip(
            "state_indices GDN prefill path requires SM8x, SM90, SM100, or SM120"
        )
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
        # This file checks the existing implementation's indexed and packed
        # paths for bitwise identity.  Cake is covered separately against the
        # independent sequential recurrence.
        backend="flashinfer",
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
    _skip_if_not_supported()
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


@pytest.mark.parametrize("index_dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_integer_index_dtypes(index_dtype, use_cp):
    """Sequence and state indices retain their integer dtype across dispatch."""
    _skip_if_not_supported()
    device = torch.device("cuda")
    H, D = 8, 128
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
    )
    torch.cuda.synchronize()

    assert torch.equal(packed_output, indexed_output)
    assert torch.equal(packed_final, indexed_final[slots])


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_preserves_inner_strides(use_cp):
    """Indexed state views use the tensor's actual shape and strides."""
    _skip_if_not_supported()
    device = torch.device("cuda")
    H, D = 8, 128
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
    )
    torch.cuda.synchronize()

    assert torch.equal(packed_output, indexed_output)
    assert torch.equal(packed_final, indexed_final[slots])


def test_prefill_state_indices_requires_output_state_pool():
    """With state_indices set, output_state must be a caller-provided pool: an
    auto-allocated compact [num_seqs, ...] tensor would be indexed out of bounds
    by the pool slot ids, so output_state=None must be rejected."""
    _skip_if_not_supported()
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


def test_prefill_state_indices_rejects_auto_pool_without_initial_state():
    """output_final_state=False does not make the auto-allocation safe.

    The kernel writes each final state to output_state[state_indices[i]]
    whether or not the caller asked for it back -- output_final_state decides
    only whether it is returned. With no initial_state to take a pool shape
    from, the auto-allocation is a compact [num_seqs, ...] tensor, and any slot
    id past num_seqs - 1 writes past its end.

    Nothing about that failure is loud: the caching allocator serves
    sub-allocations out of a much larger block, so the write lands in another
    tensor rather than faulting, and compute-sanitizer reports no error. It has
    to be refused here.
    """
    _skip_if_not_supported()
    device = torch.device("cuda")
    H, D = 16, 128
    seq_lens = [64]
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=6
    )
    # One sequence, so the compact shape is [1, ...] and slot 5 is off the end.
    state_indices = torch.tensor([5], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="explicit output_state pool"):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            output_state=None,
        )


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_without_final_state(use_cp):
    """A state pool can supply initial state without requesting a final state."""
    _skip_if_not_supported()
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
    _skip_if_not_supported()
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


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_initial_state_without_final_state(use_cp):
    """An initial state with no final state asked for.

    The CP path built its initial-state layout out of the *output* state's shape
    and stride, which only exist when a final state was asked for, so this
    combination did not compile at all. It is a public contract: a caller may
    start from a state and want only the output.
    """
    _skip_if_not_supported()
    device = torch.device("cuda")
    H, D = 8, 128
    seq_lens = [256, 512]
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=0
    )
    total = sum(seq_lens)
    out_ref = torch.empty(total, H, D, dtype=q.dtype, device=device)
    ref_state = torch.zeros_like(init_state)
    ref, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        None,
        initial_state=init_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out_ref,
        output_state=ref_state,
        use_cp=use_cp,
    )
    out = torch.empty(total, H, D, dtype=q.dtype, device=device)
    got = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        None,
        initial_state=init_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out,
        use_cp=use_cp,
    )
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=5e-3)


@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_state_indices_pools_of_different_sizes(use_cp):
    """The initial pool and the output pool need not match in size or in stride.

    They are separate tensors indexed by the same `state_indices`, so nothing
    ties their leading dimension together -- and the CP path took the output
    pool's shape for both. The two are given different padding and different
    inner strides as well, because two contiguous pools that differ only in
    their leading dimension still have the same element strides and would not
    catch an address built from the wrong tensor.

    Checked against a packed run rather than against itself: unselected rows
    staying put says nothing about whether the selected ones hold the right
    numbers, or whether the right initial row was read.
    """
    _skip_if_not_supported()
    device = torch.device("cuda")
    H, D = 8, 128
    seq_lens = [256, 512]
    dtype = torch.bfloat16
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, dtype, device, seed=0
    )
    total = sum(seq_lens)

    # One canonical initial state, fp32, fed to both runs. `_make_inputs` hands
    # back a bf16 state on this target; giving the packed run that and the
    # indexed run an fp32 copy makes the two differ in arithmetic as well as in
    # layout, and the comparison stops being about layout at all.
    canonical_init = init_state.float().contiguous()
    packed_out = torch.empty(total, H, D, dtype=q.dtype, device=device)
    packed_state = torch.zeros(
        len(seq_lens), H, D, D, dtype=torch.float32, device=device
    )
    packed, packed_final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        None,
        initial_state=canonical_init,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=packed_out,
        output_state=packed_state,
        use_cp=use_cp,
    )

    perm = [3, 1]
    in_pool = _make_pool(canonical_init, perm, 9, 96, torch.float32, device)
    out_pool = _make_pool(
        torch.zeros_like(canonical_init),
        perm,
        5,
        0,
        torch.float32,
        device,
        inner_stride=2,
    )
    sentinel = torch.arange(
        out_pool.shape[0] * H * D * D, dtype=torch.float32, device=device
    ).reshape(out_pool.shape)
    out_pool.copy_(sentinel)
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    assert in_pool.shape[0] != out_pool.shape[0]
    assert in_pool.stride() != out_pool.stride()

    out = torch.empty(total, H, D, dtype=q.dtype, device=device)
    indexed, indexed_final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        None,
        initial_state=in_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out,
        output_state=out_pool,
        state_indices=idx,
        use_cp=use_cp,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(packed).all() and torch.isfinite(packed_final).all()
    assert torch.isfinite(indexed).all()
    torch.testing.assert_close(indexed, packed, atol=1e-2, rtol=5e-3)
    for i, r in enumerate(perm):
        assert torch.isfinite(out_pool[r]).all()
        torch.testing.assert_close(
            out_pool[r],
            packed_final[i],
            atol=1e-2,
            rtol=5e-3,
            msg=lambda m, r=r, i=i: (
                f"output pool row {r} is not sequence {i}'s final state\n{m}"
            ),
        )
    for r in range(out_pool.shape[0]):
        if r in perm:
            continue
        assert torch.equal(out_pool[r], sentinel[r]), (
            f"output pool row {r} was written although no sequence selected it"
        )


def _run_invalid_slot_child(case_name):
    """Launch the native prefill with one out-of-pool `state_indices` entry.

    Runs in a child process: the bounds check is a device-side assert, which
    poisons the CUDA context for everything that follows it.
    """
    device = torch.device("cuda")
    H, D = 8, 128
    seq_lens = [128]
    q, k, v, g, beta, cu_seqlens, init_state = _make_inputs(
        seq_lens, H, D, torch.bfloat16, device, seed=0
    )
    perm = [1]
    pool = _make_pool(init_state, perm, 3, 0, torch.float32, device)
    idx = torch.tensor(perm, dtype=torch.int32, device=device)
    idx[0] = -1 if case_name == "negative" else int(pool.shape[0])
    _run(q, k, v, g, beta, cu_seqlens, pool, pool, idx, use_cp=False)
    torch.cuda.synchronize()


@pytest.mark.parametrize("case_name", ("negative", "upper"))
def test_prefill_state_indices_out_of_pool_fails_in_isolated_process(case_name):
    """An id outside [0, N_pool) must be caught, not read past the pool.

    The overrun is otherwise silent: the caching allocator carves both pools out
    of a much larger block, so an out-of-range slot lands in a neighbouring
    tensor and neither the kernel nor compute-sanitizer reports anything.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--invalid-slot", case_name],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    combined = completed.stdout + completed.stderr
    # Match the marker at the start of a line. A substring search over the
    # child's whole stdout and stderr turns any unrelated failure whose output
    # happens to contain "SKIP" -- a path, an environment variable, a driver
    # log line -- into a skip, which silently disables this bounds check.
    skips = [ln for ln in combined.splitlines() if ln.startswith("SKIP:")]
    if skips:
        pytest.skip(skips[-1])
    assert completed.returncode != 0, combined
    assert "_assert_async_cuda_kernel" in combined, combined
    assert "GDN prefill state_indices must contain slots in" in combined, combined


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--invalid-slot":
    # Report the skip on stdout rather than through pytest.skip: this runs as a
    # plain script, and the parent turns the marker into the skip.
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
        _run_invalid_slot_child(sys.argv[2])
