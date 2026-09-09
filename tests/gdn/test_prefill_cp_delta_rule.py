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

# ruff: noqa: B008

import os
import random
import math

import pytest
import torch

from .reference_delta_rule import exclusive_cumsum
from . import reference_delta_rule as reference
from flashinfer.utils import (
    get_compute_capability,
    is_sm8x_supported,
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm12x_supported,
)

if torch.cuda.is_available() and is_sm90a_supported(torch.device("cuda")):
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import (
        cp_delta_rule_dsl_sm90 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm90 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm90 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm90 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm90 as cp_delta_rule_t_precompute_dsl,
    )
elif (
    torch.cuda.is_available()
    and is_sm100a_supported(torch.device("cuda"))
    and torch.version.cuda is not None
    and int(torch.version.cuda.split(".")[0]) >= 13
):
    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
        cp_delta_rule_dsl_sm100 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm100 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm100 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm100 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm100 as cp_delta_rule_t_precompute_dsl,
    )
elif torch.cuda.is_available() and is_sm12x_supported(torch.device("cuda")):
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm120 import (
        cp_delta_rule_dsl_sm120 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm120 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm120 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm120 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm120 as cp_delta_rule_t_precompute_dsl,
    )
elif torch.cuda.is_available() and is_sm8x_supported(torch.device("cuda")):
    # Ported one stage at a time. What is not here yet stays None, so the tests
    # for it skip rather than reaching for a name that does not exist -- and a
    # half-ported pipeline is never assembled by accident.
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm80 import (
        cp_delta_rule_dsl_sm80 as cp_delta_rule_dsl,
        cp_delta_rule_fixup_dsl_sm80 as cp_delta_rule_fixup_dsl,
        cp_delta_rule_mn_precompute_dsl_sm80 as cp_delta_rule_mn_precompute_dsl,
        cp_delta_rule_prefill_dsl_sm80 as cp_delta_rule_prefill_dsl,
        cp_delta_rule_t_precompute_dsl_sm80 as cp_delta_rule_t_precompute_dsl,
    )
else:
    cp_delta_rule_dsl = None
    cp_delta_rule_fixup_dsl = None
    cp_delta_rule_mn_precompute_dsl = None
    cp_delta_rule_prefill_dsl = None
    cp_delta_rule_t_precompute_dsl = None

from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
    chunk_bound_host,
    workspace_num_chunks_host,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule


FIXUP_TF32_ATOL = 2e-3
FIXUP_TF32_RTOL = 2e-3
FIXUP_KERNEL_KINDS = ["simt_row4", "simt_row8", "hmma"]
if torch.cuda.is_available() and is_sm8x_supported(torch.device("cuda")):
    # The HMMA fixup hands its math warps extra registers with `setmaxnreg`,
    # which SM8x does not have. It is not built there, so there is no kind to
    # parametrize over rather than a kind that raises.
    FIXUP_KERNEL_KINDS = ["simt_row4", "simt_row8"]


def _skip_if_cp_unsupported(*stages):
    """Skip when this device has no CP path, or none for the stages asked for.

    SM8x is being ported a stage at a time, so a test names the entry points it
    needs and skips when one of them is still None. Naming them is what keeps a
    half-ported pipeline from being assembled: the test for a stage that does
    not exist skips, it does not fail on a missing attribute and it does not
    quietly run against another architecture's.
    """
    device = torch.device("cuda")
    if is_sm100a_supported(device):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip(
                f"SM100 CP GDN prefill requires CUDA 13+, got {torch.version.cuda}"
            )
    elif is_sm8x_supported(device):
        pass
    elif not (is_sm90a_supported(device) or is_sm12x_supported(device)):
        pytest.skip("CP GDN prefill requires SM8x, SM90, SM100, or SM12x")

    for stage in stages or ("cp_delta_rule_dsl",):
        if globals().get(stage) is None:
            pytest.skip(f"{stage} is not ported for this device yet")


def _seed_all(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _make_cu_seqlens(seq_lens, device):
    return torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)


def _make_gates(total_seqlen, num_heads, baseline, device):
    return (
        baseline
        + (1.0 - baseline)
        * torch.rand(total_seqlen, num_heads, dtype=torch.float32, device=device)
    ).contiguous()


@torch.inference_mode()
def _run_cp_kernel_chain(
    q,
    k,
    v,
    alpha,
    beta,
    cu_seqlens,
    total_seqlen,
    max_seqlen,
    cp_chunk_len,
    scale,
    initial_state=None,
):
    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer,
        local_state,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        initial_state=initial_state,
    )

    our_o = torch.empty(
        (total_seqlen, max(q.shape[1], v.shape[1]), q.shape[2]),
        dtype=q.dtype,
        device=q.device,
    )
    our_state = torch.empty(
        cu_seqlens.numel() - 1,
        max(q.shape[1], v.shape[1]),
        q.shape[2],
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    cp_delta_rule_prefill_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        initial_state=initial_state,
    )
    return our_o, our_state


@torch.inference_mode()
def _run_non_cp_prefill(q, k, v, alpha, beta, cu_seqlens, scale, initial_state=None):
    ref_o = torch.empty(
        (q.shape[0], max(q.shape[1], v.shape[1]), q.shape[2]),
        dtype=q.dtype,
        device=q.device,
    )
    ref_state = torch.empty(
        cu_seqlens.numel() - 1,
        max(q.shape[1], v.shape[1]),
        q.shape[2],
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        initial_state,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    return ref_o, ref_state


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("seq_lens", [[128], [192, 64], [2048], [1025], [9999, 6553]])
@pytest.mark.parametrize("gate_baseline", [1.0, 0.9, 0.9995])
@pytest.mark.parametrize("ptr_abi", [False, True], ids=["tensor-abi", "pointer-abi"])
def test_cp_delta_rule_t_precompute(
    qkv_factory,
    dtype,
    seq_lens,
    gate_baseline,
    ptr_abi,
    seed=int(os.environ.get("SEED", "0")),
):
    """Both argument interfaces, each against the reference on its own.

    Comparing the two entries to each other is a weaker claim than comparing
    each to the reference: they share a process-wide workspace, so an
    equivalence test that forgets to snapshot compares a buffer with itself.
    `ptr_abi` is only accepted by the SM80 entry.
    """
    _skip_if_cp_unsupported("cp_delta_rule_t_precompute_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    num_heads = 1
    head_size = 128
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with torch.device(device):
        _, k, _ = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    beta = _make_gates(total_seqlen, num_heads, gate_baseline, device)

    abi = {"_ptr_abi": True} if ptr_abi else {}
    if ptr_abi and "sm80" not in cp_delta_rule_t_precompute_dsl.__name__:
        pytest.skip("the pointer entry is SM80-only")
    our_t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen, **abi
    )
    torch.cuda.synchronize()

    assert our_t.shape == (
        workspace_num_chunks_host(cu_seqlens.cpu(), 64, total_seqlen),
        num_heads,
        64,
        64,
    )
    for seq_idx, _ in enumerate(seq_lens):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        t_start = chunk_bound_host(seq_idx, seq_start, 64)
        ref_t = reference.precompute_blockwise_cp_delta_rule_t(
            k[seq_start:seq_end],
            beta[seq_start:seq_end],
            block_size=64,
            kv_dtype=torch.float32,
            t_dtype=dtype,
        )
        torch.testing.assert_close(
            our_t[t_start : t_start + ref_t.shape[0]], ref_t, atol=5e-3, rtol=5e-3
        )


@torch.inference_mode()
def test_cp_delta_rule_t_precompute_varlen_tail_is_projected(
    qkv_factory,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_t_precompute_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = 1
    head_size = 128
    seq_lens = [96]
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with torch.device(device):
        _, k, _ = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)

    got = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    torch.cuda.synchronize()

    tail = got[1, 0]
    torch.testing.assert_close(
        tail[32:], torch.zeros_like(tail[32:]), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        tail[:, 32:], torch.zeros_like(tail[:, 32:]), atol=0.0, rtol=0.0
    )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("gate_baseline", [0.9, 0.99, 0.9995])
@pytest.mark.parametrize(
    "seq_lens, cp_chunk_len",
    [
        ([64, 192], 64),
        ([128, 200], 128),
        ([192], 192),
        ([1024, 3000], 1024),
        ([96, 64, 192], 128),
    ],
)
@pytest.mark.parametrize("ptr_abi", [False, True], ids=["tensor-abi", "pointer-abi"])
def test_cp_delta_rule_mn_precompute(
    qkv_factory,
    dtype,
    seq_lens,
    cp_chunk_len,
    num_heads,
    gate_baseline,
    ptr_abi,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_mn_precompute_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    head_size = 128
    block_size = 64
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with torch.device(device):
        _, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_heads, gate_baseline, device)

    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    if ptr_abi and "sm80" not in cp_delta_rule_mn_precompute_dsl.__name__:
        pytest.skip("the pointer entry is SM80-only")
    our_transfer, our_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        **({"_ptr_abi": True} if ptr_abi else {}),
    )
    torch.cuda.synchronize()

    assert our_transfer.shape == (
        workspace_num_chunks_host(cu_seqlens.cpu(), cp_chunk_len, total_seqlen),
        num_heads,
        head_size,
        head_size,
    )
    assert our_state.shape == our_transfer.shape

    for seq_idx, seq_len in enumerate(seq_lens):
        seq_start = int(cu_seqlens[seq_idx].item())
        cp_start = chunk_bound_host(seq_idx, seq_start, cp_chunk_len)
        t_start = chunk_bound_host(seq_idx, seq_start, block_size)
        num_cp_chunks = (seq_len + cp_chunk_len - 1) // cp_chunk_len
        for chunk_idx in range(num_cp_chunks):
            chunk_offset = chunk_idx * cp_chunk_len
            chunk_end = min(seq_len, chunk_offset + cp_chunk_len)
            num_t_blocks = (chunk_end - chunk_offset + block_size - 1) // block_size
            t_block_offset = chunk_idx * (cp_chunk_len // block_size)
            slot = cp_start + chunk_idx
            ref_transfer, ref_state = reference.blockwise_cp_delta_rule_pre_transposed(
                k[seq_start + chunk_offset : seq_start + chunk_end],
                v[seq_start + chunk_offset : seq_start + chunk_end],
                alpha[seq_start + chunk_offset : seq_start + chunk_end],
                t[t_start + t_block_offset : t_start + t_block_offset + num_t_blocks],
                block_size=block_size,
                kv_dtype=torch.float32,
            )
            if dtype == torch.bfloat16:
                atol = 5e-3
                rtol = 2e-3
            else:
                atol = 1e-3
                rtol = 5e-4
            torch.testing.assert_close(
                our_transfer[slot].transpose(-1, -2), ref_transfer, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                our_state[slot].transpose(-1, -2), ref_state, atol=atol, rtol=rtol
            )


@torch.inference_mode()
@pytest.mark.parametrize("kernel_kind", FIXUP_KERNEL_KINDS)
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("num_heads", [1, 3])
@pytest.mark.parametrize(
    "seq_lens, cp_chunk_len", [([96, 0, 300], 128), ([1], 1), ([2], 1), ([5], 1)]
)
def test_cp_delta_rule_fixup(
    seq_lens,
    cp_chunk_len,
    num_heads,
    use_initial_state,
    kernel_kind,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_fixup_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    total_seqlen = sum(seq_lens)
    total_cp_chunks = workspace_num_chunks_host(
        cu_seqlens.cpu(), cp_chunk_len, total_seqlen
    )

    local_transfer = (
        torch.randn(
            total_cp_chunks,
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    local_state = (
        torch.randn(
            total_cp_chunks,
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    diag = torch.arange(head_size, device=device)
    local_transfer[:, :, diag, diag] += 0.9
    initial_state = None
    if use_initial_state:
        initial_state = (
            torch.randn(
                len(seq_lens),
                num_heads,
                head_size,
                head_size,
                dtype=torch.float32,
                device=device,
            )
            * 0.03
        )

    our_fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer.contiguous(),
        local_state.contiguous(),
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        initial_state=initial_state,
        _kernel_kind=kernel_kind,
    )
    torch.cuda.synchronize()

    ref_transfers_by_seq = []
    ref_states_by_seq = []
    ref_initial_states_by_seq = []
    ref_seq_indices = []
    for seq_idx, seq_len in enumerate(seq_lens):
        seq_start = int(cu_seqlens[seq_idx].item())
        chunk_start = chunk_bound_host(seq_idx, seq_start, cp_chunk_len)
        num_chunks = (seq_len + cp_chunk_len - 1) // cp_chunk_len
        seq_slots = []
        for chunk_idx in range(num_chunks):
            slot = chunk_start + chunk_idx
            seq_slots.append(slot)
        if seq_slots:
            ref_transfers_by_seq.append(local_transfer[seq_slots])
            ref_states_by_seq.append(local_state[seq_slots])
            if use_initial_state:
                ref_initial_states_by_seq.append(initial_state[seq_idx])
            ref_seq_indices.append(seq_idx)

    _, ref_by_seq = reference.cp_delta_rule_fixup_transposed(
        ref_transfers_by_seq,
        ref_states_by_seq,
        ref_initial_states_by_seq if use_initial_state else None,
    )
    for seq_idx, seq_fixed in zip(ref_seq_indices, ref_by_seq):  # noqa: B905
        seq_start = int(cu_seqlens[seq_idx].item())
        chunk_start = chunk_bound_host(seq_idx, seq_start, cp_chunk_len)
        for chunk_idx in range(seq_fixed.shape[0]):
            torch.testing.assert_close(
                our_fixed_state[chunk_start + chunk_idx],
                seq_fixed[chunk_idx],
                atol=FIXUP_TF32_ATOL,
                rtol=FIXUP_TF32_RTOL,
            )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("chunk_len", [64, 128])
@pytest.mark.parametrize("gate_baseline", [0.9, 0.9995])
def test_cp_delta_rule_prefill_varlen_matches_non_cp_prefill(
    qkv_factory,
    dtype,
    chunk_len,
    gate_baseline,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    num_heads = 1
    head_size = 128
    cp_chunk_len = chunk_len
    seq_lens = [chunk_len, chunk_len]
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_values = [0]
    for seq_len in seq_lens:
        cu_values.append(cu_values[-1] + seq_len)
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int64, device=device)
    scale = 1.0

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    q = q.contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_heads, gate_baseline, device)

    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer, local_state, cu_seqlens, total_seqlen, cp_chunk_len=cp_chunk_len
    )

    our_o = torch.empty_like(q)
    our_state = torch.empty(
        len(seq_lens),
        num_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    cp_delta_rule_prefill_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()

    ref_o = torch.empty_like(q)
    ref_state = torch.empty(
        len(seq_lens),
        num_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


@torch.inference_mode()
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads", [(4, 1, 1), (1, 1, 4)]
)
def test_cp_delta_rule_prefill_varlen_matches_non_cp_prefill_unequal_heads(
    qkv_factory,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_size = 128
    cp_chunk_len = 64
    seq_lens = [96, 64]
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    cu_values = [0]
    for seq_len in seq_lens:
        cu_values.append(cu_values[-1] + seq_len)
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int64, device=device)
    num_sab_heads = max(num_q_heads, num_v_heads)
    scale = 1.0

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    q = q.contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_sab_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_sab_heads, 0.99, device)

    t = cp_delta_rule_t_precompute_dsl(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer, local_state, cu_seqlens, total_seqlen, cp_chunk_len=cp_chunk_len
    )

    our_o = torch.empty(
        total_seqlen, num_sab_heads, head_size, dtype=dtype, device=device
    )
    our_state = torch.empty(
        len(seq_lens),
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    cp_delta_rule_prefill_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()

    ref_o = torch.empty_like(our_o)
    ref_state = torch.empty_like(our_state)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


@torch.inference_mode()
@pytest.mark.parametrize(
    "dtype, seq_lens, cp_chunk_len, num_q_heads, num_k_heads, num_v_heads, gate_baseline, scale",
    [
        (torch.bfloat16, [2048], 1024, 1, 1, 1, 0.99, 1.0),
        (torch.bfloat16, [4096], 2048, 2, 1, 1, 0.9995, 1.0),
        (torch.float16, [2049], 1024, 1, 1, 1, 0.99, "auto"),
        (torch.float16, [8193], 2048, 1, 1, 1, 0.99, "auto"),
        (torch.bfloat16, [1536, 257], 1024, 1, 1, 2, 0.99, 1.0),
    ],
)
def test_cp_delta_rule_kernel_chain_long_small_bh_matches_non_cp_prefill(
    qkv_factory,
    dtype,
    seq_lens,
    cp_chunk_len,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    gate_baseline,
    scale,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    total_seqlen = sum(seq_lens)
    max_seqlen = max(seq_lens)
    num_sab_heads = max(num_q_heads, num_v_heads)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0 / math.sqrt(head_size) if scale == "auto" else scale

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)

    our_o, our_state = _run_cp_kernel_chain(
        q, k, v, alpha, beta, cu_seqlens, total_seqlen, max_seqlen, cp_chunk_len, scale
    )
    torch.cuda.synchronize()
    ref_o, ref_state = _run_non_cp_prefill(q, k, v, alpha, beta, cu_seqlens, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(our_state, ref_state, atol=5e-2, rtol=5e-2)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("seq_lens", [[192, 64], [1025]])
def test_cp_delta_rule_e2e_with_initial_state(
    qkv_factory,
    dtype,
    seq_lens,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0
    dtype = getattr(torch, dtype)
    num_heads = 1

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)
    initial_state = (
        torch.randn(
            len(seq_lens),
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )

    our_o = torch.empty(
        [total_seqlen, num_heads, head_size], dtype=q.dtype, device=q.device
    )
    our_state = torch.empty_like(initial_state)
    cp_delta_rule_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        scale,
        initial_state=initial_state,
        max_seqlen=max(seq_lens),
        cp_chunk_len=128,
    )
    torch.cuda.synchronize()

    ref_o, ref_state = _run_non_cp_prefill(
        q, k, v, alpha, beta, cu_seqlens, scale, initial_state=initial_state
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(our_state, ref_state, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("gate_baseline", [0.9995, 0.99])
@pytest.mark.parametrize("num_heads", [(1, 1, 1), (2, 2, 8), (4, 1, 1), (16, 16, 64)])
@pytest.mark.parametrize("seq_lens", [[192, 64], [2048], [1025], [9999, 65530]])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_cp_delta_rule_e2e(
    qkv_factory,
    dtype,
    seq_lens,
    num_heads,
    gate_baseline,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    head_size = 128
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0
    dtype = getattr(torch, dtype)

    num_q_heads, num_k_heads, num_v_heads = num_heads
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = max(num_heads)

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)
    beta = _make_gates(total_seqlen, num_sab_heads, gate_baseline, device)

    our_o = torch.empty(
        [total_seqlen, num_o_heads, head_size], dtype=q.dtype, device=q.device
    )
    our_state = torch.empty(
        len(seq_lens),
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    our_o.fill_(float("nan"))
    our_state.fill_(float("nan"))
    cp_delta_rule_dsl(
        our_o,
        our_state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        scale,
        max_seqlen=max(seq_lens),
    )
    torch.cuda.synchronize()

    ref_o = torch.empty_like(our_o)
    ref_state = torch.empty_like(our_state)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seqlens,
        True,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    if dtype == torch.bfloat16:
        ref_o = ref_o.to(dtype)
        atol_o = 2e-2
        rtol_o = 2e-2
        atol_state = 1e-2
        rtol_state = 5e-3
    else:
        atol_o = 5e-3
        rtol_o = 5e-3
        atol_state = 1e-3
        rtol_state = 1e-3

    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_state, rtol=rtol_state)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("seq_lens", [[128], [256, 64], [2048]])
def test_cp_delta_rule_public_wrapper_matches_non_cp_prefill(
    qkv_factory,
    dtype,
    seq_lens,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(seed)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)
    head_size = 128
    num_heads = 1
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    scale = 1.0

    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)

    our_o, our_state = chunk_gated_delta_rule(
        q, k, v, alpha, beta, scale, None, True, cu_seqlens, True, use_cp=True
    )
    ref_o, ref_state = chunk_gated_delta_rule(
        q, k, v, alpha, beta, scale, None, True, cu_seqlens, True, use_cp=False
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


@torch.inference_mode()
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_lens", [[128], [256, 64]])
def test_cp_delta_rule_external_state_dtype(
    qkv_factory,
    state_dtype,
    seq_lens,
    seed=int(os.environ.get("SEED", "0")),
):
    _skip_if_cp_unsupported("cp_delta_rule_dsl")
    device = torch.device("cuda")
    _seed_all(seed)
    dtype = torch.bfloat16
    head_size = 128
    num_heads = 1
    num_seqs = len(seq_lens)
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)

    with device:
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype=dtype
        )
        initial_state = (
            torch.randn(num_seqs, num_heads, head_size, head_size) * 0.01
        ).to(state_dtype)
    q = q.contiguous()
    k = torch.nn.functional.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = v.contiguous()
    alpha = _make_gates(total_seqlen, num_heads, 0.99, device)
    beta = _make_gates(total_seqlen, num_heads, 0.99, device)
    our_o = torch.empty_like(q)
    ref_o = torch.empty_like(q)
    our_state = torch.empty_like(initial_state)
    ref_state = torch.empty_like(initial_state)

    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        1.0,
        initial_state,
        True,
        cu_seqlens,
        False,
        output=our_o,
        output_state=our_state,
        use_cp=True,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        1.0,
        initial_state,
        True,
        cu_seqlens,
        False,
        output=ref_o,
        output_state=ref_state,
        use_cp=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(our_state, ref_state, atol=4e-2, rtol=4e-2)


# ─── Regressions for the CP prefill's bookkeeping ─────────────────────────────
# Three properties, one per defect. None of them is a value comparison against a
# reference: they hold whether or not the recurrence itself is right, so they
# keep working while it is being fixed and they fail the moment the bookkeeping
# regresses.


def _cp_stage_inputs(
    seq_lens,
    cp_chunk_len,
    qkv_factory,
    device,
    dtype,
    num_q_heads=1,
    num_k_heads=1,
    num_v_heads=1,
):
    """Stages 1 through 3, and everything stage 4 needs to run.

    Head counts are separate parameters because they arrive at the kernel as
    four adjacent scalars, and an equal-head case cannot tell them apart -- the
    argument shift this file's tests were written for was invisible in exactly
    that way.
    """
    head_size = 128
    num_heads = max(num_q_heads, num_v_heads)
    total_seqlen = sum(seq_lens)
    cu_seqlens = _make_cu_seqlens(seq_lens, device)
    with torch.device(device):
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=dtype
        )
    beta = _make_gates(total_seqlen, num_heads, 0.25, device)
    alpha = _make_gates(total_seqlen, num_heads, 0.9, device)
    k_sab = (
        k
        if num_k_heads == num_heads
        else k.repeat_interleave(num_heads // num_k_heads, dim=1).contiguous()
    )
    max_seqlen = max(seq_lens)
    t = cp_delta_rule_t_precompute_dsl(
        k_sab, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl(
        k_sab,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
    )
    fixed_state = cp_delta_rule_fixup_dsl(
        local_transfer,
        local_state,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
    )
    return dict(
        q=q,
        k=k,
        v=v,
        t=t,
        alpha=alpha,
        fixed_state=fixed_state,
        cu_seqlens=cu_seqlens,
        total_seqlen=total_seqlen,
        max_seqlen=max_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        cp_chunk_len=cp_chunk_len,
        beta=beta,
    )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cp_prefill_leaves_the_fixup_workspace_alone(qkv_factory, dtype):
    """No final state asked for means none written.

    `fixed_state` is the fixup workspace, and on this path the state tensor the
    kernel would write points into it. A store that runs when it was not asked
    for lands on a chunk state another block still has to read, so the property
    is that the workspace comes back byte for byte.
    """
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(0)
    device = torch.device("cuda")
    seq_lens = [64, 192]
    args = _cp_stage_inputs(seq_lens, 64, qkv_factory, device, getattr(torch, dtype))
    before = args["fixed_state"].clone()
    o = torch.empty(
        (args["total_seqlen"], args["num_heads"], args["head_size"]),
        dtype=args["q"].dtype,
        device=device,
    )
    cp_delta_rule_prefill_dsl(
        o,
        None,
        args["q"],
        args["k"],
        args["v"],
        args["t"],
        args["fixed_state"],
        args["alpha"],
        1.0,
        args["cu_seqlens"],
        args["total_seqlen"],
        cp_chunk_len=args["cp_chunk_len"],
        max_seqlen=args["max_seqlen"],
    )
    torch.cuda.synchronize()
    assert torch.equal(args["fixed_state"], before), (
        "the fixup workspace was written while no final state was asked for"
    )


@torch.inference_mode()
# The kernel takes GQA (k == v, q a multiple) and GVA (q == k, v a multiple),
# so those are the two ways the four head counts can differ from each other.
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(1, 1, 1), (4, 1, 1), (1, 1, 4)],
)
def test_cp_prefill_writes_every_value_band(
    qkv_factory, num_q_heads, num_k_heads, num_v_heads
):
    """Both halves of O get written.

    `head_size` is 128 and the fused kernel owns 64 rows per block, so two value
    slices cover it. A grid without that axis runs slice 0 only and leaves the
    other band at whatever the caller passed -- which a value comparison against
    a reference would report as "wrong numbers" rather than "never written".
    Poisoning O separates the two.
    """
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(0)
    device = torch.device("cuda")
    args = _cp_stage_inputs(
        [64],
        64,
        qkv_factory,
        device,
        torch.float16,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    poison = float("nan")
    o = torch.full(
        (args["total_seqlen"], args["num_heads"], args["head_size"]),
        poison,
        dtype=args["q"].dtype,
        device=device,
    )
    state = torch.zeros(
        len(args["cu_seqlens"]) - 1,
        args["num_heads"],
        args["head_size"],
        args["head_size"],
        dtype=torch.float32,
        device=device,
    )
    cp_delta_rule_prefill_dsl(
        o,
        state,
        args["q"],
        args["k"],
        args["v"],
        args["t"],
        args["fixed_state"],
        args["alpha"],
        1.0,
        args["cu_seqlens"],
        args["total_seqlen"],
        cp_chunk_len=args["cp_chunk_len"],
        max_seqlen=args["max_seqlen"],
    )
    torch.cuda.synchronize()
    half = args["head_size"] // 2
    for band, name in ((o[..., :half], "band 0"), (o[..., half:], "band 1")):
        assert not band.isnan().any(), f"{name} of O was never written"


@torch.inference_mode()
def test_cp_prefill_checkpoints_land_in_sequence_order(qkv_factory):
    """Every checkpoint slot holds the state after its own prefix.

    The shape matters. A `cp_chunk_len` equal to the block length runs one block
    per chunk and never reaches the middle-block or last-block checkpoint calls,
    so this is 384 tokens in chunks of 192 with a checkpoint every 64: two
    chunks, three blocks each, and the second chunk goes through both paths.

    Checking that a slot was written catches an index that collapses; checking
    it against the state after that many tokens also catches two that swapped.
    """
    _skip_if_cp_unsupported("cp_delta_rule_prefill_dsl")
    _seed_all(0)
    device = torch.device("cuda")
    cp_chunk_len = 192
    every = 64
    seq_lens = [384]
    args = _cp_stage_inputs(seq_lens, cp_chunk_len, qkv_factory, device, torch.float16)
    per_seq = [length // every for length in seq_lens]
    starts = [0]
    for count in per_seq:
        starts.append(starts[-1] + count)
    checkpoints = torch.full(
        (sum(per_seq), args["num_heads"], args["head_size"], args["head_size"]),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    o = torch.empty(
        (args["total_seqlen"], args["num_heads"], args["head_size"]),
        dtype=args["q"].dtype,
        device=device,
    )
    state = torch.zeros(
        len(seq_lens),
        args["num_heads"],
        args["head_size"],
        args["head_size"],
        dtype=torch.float32,
        device=device,
    )
    cp_delta_rule_prefill_dsl(
        o,
        state,
        args["q"],
        args["k"],
        args["v"],
        args["t"],
        args["fixed_state"],
        args["alpha"],
        1.0,
        args["cu_seqlens"],
        args["total_seqlen"],
        cp_chunk_len=cp_chunk_len,
        max_seqlen=args["max_seqlen"],
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=torch.tensor(starts, dtype=torch.int32, device=device),
        checkpoint_every_n_tokens=every,
    )
    torch.cuda.synchronize()
    for slot in range(checkpoints.shape[0]):
        assert not checkpoints[slot].isnan().any(), (
            f"checkpoint slot {slot} of {checkpoints.shape[0]} was never written"
        )
    for slot in range(checkpoints.shape[0]):
        n = (slot + 1) * every
        _, prefix_state = _run_non_cp_prefill(
            args["q"][:n],
            args["k"][:n],
            args["v"][:n],
            args["alpha"][:n],
            args["beta"][:n],
            _make_cu_seqlens([n], device),
            1.0,
        )
        torch.testing.assert_close(
            checkpoints[slot],
            prefix_state[0],
            atol=5e-3,
            rtol=2e-3,
            msg=lambda m, slot=slot, n=n: (
                f"checkpoint slot {slot} is not the state after {n} tokens\n{m}"
            ),
        )


# --- the pointer-argument entry, SM80 only ---------------------------------


def _sm80_t_entry():
    if get_compute_capability(torch.device("cuda"))[0] != 8:
        pytest.skip("the pointer entry is SM80-only")
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm80 import (
        cp_delta_rule_t_precompute_dsl_sm80,
    )

    return cp_delta_rule_t_precompute_dsl_sm80


def _t_inputs(seq_lens, h_qk, h_v, dtype, device, seed=11):
    torch.manual_seed(seed)
    total = max(sum(seq_lens), 1)
    k = (
        torch.nn.functional.normalize(
            torch.randn(total, h_qk, 128, device=device) * 0.2, dim=-1
        )
        .to(dtype)
        .contiguous()
    )
    beta = (
        0.25 + 0.5 * torch.rand(total, h_v, dtype=torch.float32, device=device)
    ).contiguous()
    cu = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    return k, beta, cu, sum(seq_lens), max(seq_lens)


@torch.inference_mode()
@pytest.mark.parametrize(
    "seq_lens,h_qk,h_v,dtype",
    [
        ([4096], 1, 1, torch.bfloat16),
        ([4096], 1, 1, torch.float16),
        ([2731, 2731, 2730], 1, 1, torch.bfloat16),
        ([2048, 0], 1, 1, torch.bfloat16),
        ([2048] * 4, 4, 4, torch.bfloat16),
        ([8192], 2, 8, torch.bfloat16),
        ([8192], 4, 16, torch.float16),
        ([65], 1, 1, torch.bfloat16),
        ([1] * 8, 1, 1, torch.bfloat16),
    ],
)
def test_t_precompute_pointer_abi_matches_tensor_abi(seq_lens, h_qk, h_v, dtype):
    """The two entries, compared without comparing a buffer with itself.

    Both write the same process-wide workspace, so the first result has to be
    snapshotted before the second call runs -- and the snapshot's storage has
    to be shown to be a different one, or the comparison is `x == x`. The
    workspace's unwritten padding holds whatever the last cell left, including
    NaN, so the comparison is over bytes: `torch.equal` reports a difference
    where the bits agree.
    """
    entry = _sm80_t_entry()
    device = torch.device("cuda")
    k, beta, cu, total, mx = _t_inputs(seq_lens, h_qk, h_v, dtype, device)

    got_tensor = entry(k, beta, cu, total, max_seqlen=mx)
    torch.cuda.synchronize()
    snapshot = got_tensor.clone()
    torch.cuda.synchronize()
    got_ptr = entry(k, beta, cu, total, max_seqlen=mx, _ptr_abi=True)
    torch.cuda.synchronize()

    assert snapshot.data_ptr() != got_ptr.data_ptr(), (
        "the snapshot shares storage with the second result, so this compares "
        "a buffer with itself"
    )
    assert got_tensor.data_ptr() == got_ptr.data_ptr(), (
        "both entries are expected to write the same cached workspace"
    )
    assert torch.equal(
        snapshot.contiguous().view(torch.uint8),
        got_ptr.contiguous().view(torch.uint8),
    ), "the pointer entry produced different bytes"


@torch.inference_mode()
def test_t_precompute_pointer_abi_writes_the_same_blocks():
    """Poison differently before each call, so an unwritten block shows up.

    Byte equality alone cannot separate "both wrote this" from "neither did
    and the leftovers matched". Two different poisons make any block that
    neither entry wrote differ, and any block both wrote agree.
    """
    entry = _sm80_t_entry()
    device = torch.device("cuda")
    k, beta, cu, total, mx = _t_inputs([2048] * 4, 4, 4, torch.bfloat16, device)

    workspace = entry(k, beta, cu, total, max_seqlen=mx)
    torch.cuda.synchronize()

    workspace.fill_(float("inf"))
    torch.cuda.synchronize()
    entry(k, beta, cu, total, max_seqlen=mx)
    torch.cuda.synchronize()
    from_tensor = workspace.clone()
    wrote_tensor = ~from_tensor.isinf()

    workspace.fill_(float("-inf"))
    torch.cuda.synchronize()
    entry(k, beta, cu, total, max_seqlen=mx, _ptr_abi=True)
    torch.cuda.synchronize()
    from_ptr = workspace.clone()
    wrote_ptr = ~from_ptr.isinf()

    assert torch.equal(wrote_tensor, wrote_ptr), (
        "the two entries wrote different sets of elements: "
        f"{int((wrote_tensor ^ wrote_ptr).sum())} disagree"
    )
    assert bool(wrote_tensor.any()), "neither entry wrote anything"
    written = wrote_tensor
    assert torch.equal(
        from_tensor[written].view(torch.uint8), from_ptr[written].view(torch.uint8)
    ), "the entries disagree on the elements they both wrote"


@torch.inference_mode()
@pytest.mark.parametrize("which", ["beta", "cu_seqlens"])
def test_t_precompute_pointer_abi_refuses_a_misaligned_buffer(which):
    """Contiguous is not aligned, and `assumed_align` is a promise, not a check.

    An offset slice of a larger buffer is contiguous at any address, so the
    entry checks the addresses it promises the compiler.
    """
    entry = _sm80_t_entry()
    device = torch.device("cuda")
    k, beta, cu, total, mx = _t_inputs([2048], 1, 1, torch.bfloat16, device)

    if which == "beta":
        wide = torch.empty(beta.numel() + 1, dtype=torch.float32, device=device)
        skewed = wide[1:].view_as(beta)
        skewed.copy_(beta)
        assert skewed.is_contiguous() and skewed.data_ptr() % 16
        args = (k, skewed, cu, total)
    else:
        # int32 `cu_seqlens` is a supported dtype, and one element of it is
        # four bytes -- so an offset slice lands 4 mod 8, which is exactly the
        # promise the entry makes and cannot keep. Slicing an int64 buffer
        # would stay 8-aligned and prove nothing.
        wide = torch.empty(cu.numel() + 1, dtype=torch.int32, device=device)
        skewed = wide[1:]
        skewed.copy_(cu.to(torch.int32))
        assert skewed.is_contiguous() and skewed.data_ptr() % 8 == 4
        args = (k, beta, skewed, total)

    with pytest.raises(RuntimeError, match="alignment"):
        entry(*args, max_seqlen=mx, _ptr_abi=True)


@torch.inference_mode()
def test_t_precompute_pointer_abi_takes_only_the_current_stream():
    """Raw addresses with no owner: the current stream's handle, or nothing.

    Refusing every handle would refuse the composition, which passes the
    current stream's own handle to each stage. So the check is equality, not
    presence.
    """
    entry = _sm80_t_entry()
    import cuda.bindings.driver as cuda_driver

    device = torch.device("cuda")
    k, beta, cu, total, mx = _t_inputs([2048], 1, 1, torch.bfloat16, device)

    other = cuda_driver.CUstream(torch.cuda.Stream().cuda_stream)
    with pytest.raises(RuntimeError, match="current stream only"):
        entry(k, beta, cu, total, max_seqlen=mx, _ptr_abi=True, _stream=other)

    current = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    entry(k, beta, cu, total, max_seqlen=mx, _ptr_abi=True, _stream=current)
    torch.cuda.synchronize()
    # And the tensor entry still accepts any stream.
    entry(k, beta, cu, total, max_seqlen=mx, _stream=other)
    torch.cuda.synchronize()


@torch.inference_mode()
@pytest.mark.parametrize("stage", ["t", "mn"])
def test_pointer_abi_checks_survive_skip_check(stage):
    """`_skip_check=True` must not turn the pointer path's own checks off.

    The composition passes `_skip_check=True` to every stage -- it validated
    shapes once for the whole call -- so a pointer-only invariant placed behind
    that flag is off exactly where it matters. Alignment is a promise the
    compiler cannot verify and a raw pointer has no owner; neither is a shape
    check.
    """
    import cuda.bindings.driver as cuda_driver

    device = torch.device("cuda")
    other = cuda_driver.CUstream(torch.cuda.Stream().cuda_stream)
    if stage == "t":
        entry = _sm80_t_entry()
        k, beta, cu, total, mx = _t_inputs([2048], 1, 1, torch.bfloat16, device)
        wide = torch.empty(beta.numel() + 1, dtype=torch.float32, device=device)
        skewed = wide[1:].view_as(beta)
        skewed.copy_(beta)
        assert skewed.data_ptr() % 16
        bad_align = dict(args=(k, skewed, cu, total), kw={})
        ok_args = (k, beta, cu, total)
        kw = {"max_seqlen": mx}
    else:
        entry = _sm80_mn_entry()
        k, v, t, alpha, cu, total, mx = _mn_inputs([2048], 1, 1, torch.bfloat16, device)
        wide = torch.empty(alpha.numel() + 1, dtype=torch.float32, device=device)
        skewed = wide[1:].view_as(alpha)
        skewed.copy_(alpha)
        assert skewed.data_ptr() % 16
        bad_align = dict(args=(k, v, t, skewed, cu, total), kw={})
        ok_args = (k, v, t, alpha, cu, total)
        kw = {"max_seqlen": mx, "cp_chunk_len": 4096}

    with pytest.raises(RuntimeError, match="alignment"):
        entry(*bad_align["args"], **kw, _ptr_abi=True, _skip_check=True)
    with pytest.raises(RuntimeError, match="current stream only"):
        entry(*ok_args, **kw, _ptr_abi=True, _skip_check=True, _stream=other)


def _sm80_mn_entry():
    if get_compute_capability(torch.device("cuda"))[0] != 8:
        pytest.skip("the pointer entry is SM80-only")
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm80 import (
        cp_delta_rule_mn_precompute_dsl_sm80,
    )

    return cp_delta_rule_mn_precompute_dsl_sm80


def _mn_inputs(seq_lens, h_qk, h_v, dtype, device, seed=11):
    k, beta, cu, total, mx = _t_inputs(seq_lens, h_qk, h_v, dtype, device, seed)
    h = max(h_qk, h_v)
    v = (
        (torch.randn(max(total, 1), h_v, 128, device=device) * 0.2)
        .to(dtype)
        .contiguous()
    )
    alpha = (
        0.9 + 0.1 * torch.rand(max(total, 1), h, dtype=torch.float32, device=device)
    ).contiguous()
    t = _sm80_t_entry()(k, beta, cu, total, max_seqlen=mx)
    torch.cuda.synchronize()
    return k, v, t, alpha, cu, total, mx


@torch.inference_mode()
@pytest.mark.parametrize(
    "seq_lens,h_qk,h_v,dtype",
    [
        ([8192], 1, 1, torch.bfloat16),
        ([2731, 2731, 2730], 1, 1, torch.bfloat16),
        ([2048, 0], 1, 1, torch.bfloat16),
        ([2048] * 4, 4, 4, torch.bfloat16),
        ([8192], 2, 8, torch.float16),
    ],
)
def test_mn_precompute_pointer_abi_matches_tensor_abi(seq_lens, h_qk, h_v, dtype):
    """Both MN outputs, snapshotted before the second entry overwrites them."""
    entry = _sm80_mn_entry()
    device = torch.device("cuda")
    k, v, t, alpha, cu, total, mx = _mn_inputs(seq_lens, h_qk, h_v, dtype, device)

    a_tr, a_st = entry(k, v, t, alpha, cu, total, max_seqlen=mx, cp_chunk_len=4096)
    torch.cuda.synchronize()
    snap_tr, snap_st = a_tr.clone(), a_st.clone()
    torch.cuda.synchronize()
    b_tr, b_st = entry(
        k, v, t, alpha, cu, total, max_seqlen=mx, cp_chunk_len=4096, _ptr_abi=True
    )
    torch.cuda.synchronize()

    for name, snap, got in (("transfer", snap_tr, b_tr), ("state", snap_st, b_st)):
        assert snap.data_ptr() != got.data_ptr(), f"{name}: comparing with itself"
        assert torch.equal(
            snap.contiguous().view(torch.uint8), got.contiguous().view(torch.uint8)
        ), f"{name}: the pointer entry produced different bytes"


@torch.inference_mode()
def test_mn_precompute_pointer_abi_writes_the_same_blocks():
    """Different poison per entry, so an unwritten element cannot pass."""
    entry = _sm80_mn_entry()
    device = torch.device("cuda")
    k, v, t, alpha, cu, total, mx = _mn_inputs([2048] * 4, 4, 4, torch.bfloat16, device)
    tr, st = entry(k, v, t, alpha, cu, total, max_seqlen=mx, cp_chunk_len=4096)
    torch.cuda.synchronize()

    masks = {}
    values = {}
    for label, ptr, poison in (
        ("tensor", False, float("inf")),
        ("pointer", True, float("-inf")),
    ):
        tr.fill_(poison)
        st.fill_(poison)
        torch.cuda.synchronize()
        entry(k, v, t, alpha, cu, total, max_seqlen=mx, cp_chunk_len=4096, _ptr_abi=ptr)
        torch.cuda.synchronize()
        masks[label] = (~tr.isinf(), ~st.isinf())
        values[label] = (tr.clone(), st.clone())

    for i, name in enumerate(("transfer", "state")):
        assert torch.equal(masks["tensor"][i], masks["pointer"][i]), (
            f"{name}: the entries wrote different elements"
        )
        assert bool(masks["tensor"][i].any()), f"{name}: nothing was written"
        written = masks["tensor"][i]
        assert torch.equal(
            values["tensor"][i][written], values["pointer"][i][written]
        ), f"{name}: the entries disagree where both wrote"


@torch.inference_mode()
def test_mn_precompute_pointer_abi_refuses_misaligned_and_a_stream():
    entry = _sm80_mn_entry()
    import cuda.bindings.driver as cuda_driver

    device = torch.device("cuda")
    k, v, t, alpha, cu, total, mx = _mn_inputs([2048], 1, 1, torch.bfloat16, device)

    wide = torch.empty(alpha.numel() + 1, dtype=torch.float32, device=device)
    skewed = wide[1:].view_as(alpha)
    skewed.copy_(alpha)
    assert skewed.is_contiguous() and skewed.data_ptr() % 16
    with pytest.raises(RuntimeError, match="alignment"):
        entry(
            k, v, t, skewed, cu, total, max_seqlen=mx, cp_chunk_len=4096, _ptr_abi=True
        )

    other = cuda_driver.CUstream(torch.cuda.Stream().cuda_stream)
    with pytest.raises(RuntimeError, match="current stream only"):
        entry(
            k,
            v,
            t,
            alpha,
            cu,
            total,
            max_seqlen=mx,
            cp_chunk_len=4096,
            _ptr_abi=True,
            _stream=other,
        )
    current = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    entry(
        k,
        v,
        t,
        alpha,
        cu,
        total,
        max_seqlen=mx,
        cp_chunk_len=4096,
        _ptr_abi=True,
        _stream=current,
    )
    torch.cuda.synchronize()


def _sm80_fixup_entry():
    if get_compute_capability(torch.device("cuda"))[0] != 8:
        pytest.skip("the pointer entry is SM80-only")
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm80 import (
        cp_delta_rule_fixup_dsl_sm80,
    )

    return cp_delta_rule_fixup_dsl_sm80


def _fixup_inputs(seq_lens, heads, device, chunk=1024, seed=11):
    """The MN outputs the fixup consumes, for a batch with an empty sequence."""
    k, v, t, alpha, cu, total, mx = _mn_inputs(
        seq_lens, heads, heads, torch.bfloat16, device, seed
    )
    transfer, state = _sm80_mn_entry()(
        k, v, t, alpha, cu, total, max_seqlen=mx, cp_chunk_len=chunk
    )
    torch.cuda.synchronize()
    return transfer, state, cu, total


def _fixup_initial(
    num_seqs, heads, device, indexed, strided=False, dtype=torch.float32
):
    """A state pool. `dtype` is in the compile key, so it is a test axis.

    vLLM's adapter converts to fp32 before calling, which is why fp32 is the
    default here -- but the public entry accepts fp16 and bf16 pools too, and
    `initial_state_dtype` specializes the kernel on which it got.
    """
    rows = num_seqs + 3 if indexed else num_seqs
    if strided:
        # A pool whose rows are a view of something wider: contiguous nowhere,
        # and its strides are the caller's, which is the case the compile key
        # carries `initial_state_inner_strides` for.
        wide = torch.randn(rows, heads, 128, 136, dtype=dtype, device=device) * 0.05
        return wide[..., :128]
    return (
        torch.randn(rows, heads, 128, 128, dtype=dtype, device=device) * 0.05
    ).contiguous()


@torch.inference_mode()
@pytest.mark.parametrize("kernel_kind", ["simt_row4", "simt_row8"])
@pytest.mark.parametrize(
    "needs_initial,use_indices,strided",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        # `state_indices` without an initial state is a public contract: stage
        # 4 indexes the final state whether or not one came in.
        (False, True, False),
        (True, True, True),
    ],
)
def test_fixup_pointer_abi_matches_tensor_abi(
    kernel_kind, needs_initial, use_indices, strided
):
    """Every specialization the tensor entry has, through both interfaces.

    The pointer entry keeps `needs_initial_state`, `use_state_indices`, the
    state dtype, the pool's inner strides, `rows_per_cta` and the index dtypes
    on the instance, so nothing here is a runtime branch on a nullable
    pointer. The inactive optional slots still get a valid address, and the
    kernel only reads them under `cutlass.const_expr`.
    """
    entry = _sm80_fixup_entry()
    device = torch.device("cuda")
    seq_lens, heads = [2048, 1024, 0, 3000], 4
    transfer, state, cu, total = _fixup_inputs(seq_lens, heads, device)
    initial = (
        _fixup_initial(len(seq_lens), heads, device, use_indices, strided)
        if needs_initial
        else None
    )
    indices = (
        torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
        if use_indices
        else None
    )
    kw = dict(
        initial_state=initial,
        state_indices=indices,
        _kernel_kind=kernel_kind,
        cp_chunk_len=1024,
    )

    got_tensor = entry(transfer, state, cu, total, **kw)
    torch.cuda.synchronize()
    snapshot = got_tensor.clone()
    torch.cuda.synchronize()
    got_ptr = entry(transfer, state, cu, total, **kw, _ptr_abi=True)
    torch.cuda.synchronize()

    assert snapshot.data_ptr() != got_ptr.data_ptr(), "comparing with itself"
    assert torch.equal(
        snapshot.contiguous().view(torch.uint8),
        got_ptr.contiguous().view(torch.uint8),
    ), "the pointer entry produced different bytes"


@torch.inference_mode()
def test_fixup_pointer_abi_writes_the_same_blocks():
    entry = _sm80_fixup_entry()
    device = torch.device("cuda")
    seq_lens, heads = [2048, 1024, 0, 3000], 4
    transfer, state, cu, total = _fixup_inputs(seq_lens, heads, device)
    initial = _fixup_initial(len(seq_lens), heads, device, True)
    indices = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
    kw = dict(initial_state=initial, state_indices=indices, cp_chunk_len=1024)

    out = entry(transfer, state, cu, total, **kw)
    torch.cuda.synchronize()
    masks, values = {}, {}
    for label, ptr, poison in (
        ("tensor", False, float("inf")),
        ("pointer", True, float("-inf")),
    ):
        out.fill_(poison)
        torch.cuda.synchronize()
        entry(transfer, state, cu, total, **kw, _ptr_abi=ptr)
        torch.cuda.synchronize()
        masks[label] = ~out.isinf()
        values[label] = out.clone()
    assert torch.equal(masks["tensor"], masks["pointer"]), (
        "the entries wrote different elements"
    )
    assert bool(masks["tensor"].any()), "nothing was written"
    w = masks["tensor"]
    assert torch.equal(values["tensor"][w], values["pointer"][w])


@torch.inference_mode()
def test_fixup_pointer_abi_checks_survive_skip_check():
    """The composition passes `_skip_check=True`; these checks must remain."""
    import cuda.bindings.driver as cuda_driver

    entry = _sm80_fixup_entry()
    device = torch.device("cuda")
    seq_lens, heads = [2048, 1024], 4
    transfer, state, cu, total = _fixup_inputs(seq_lens, heads, device)

    wide = torch.empty(cu.numel() + 1, dtype=torch.int32, device=device)
    skewed = wide[1:]
    skewed.copy_(cu.to(torch.int32))
    assert skewed.data_ptr() % 8 == 4
    with pytest.raises(RuntimeError, match="alignment"):
        entry(
            transfer,
            state,
            skewed,
            total,
            cp_chunk_len=1024,
            _ptr_abi=True,
            _skip_check=True,
        )

    other = cuda_driver.CUstream(torch.cuda.Stream().cuda_stream)
    with pytest.raises(RuntimeError, match="current stream only"):
        entry(
            transfer,
            state,
            cu,
            total,
            cp_chunk_len=1024,
            _ptr_abi=True,
            _skip_check=True,
            _stream=other,
        )


@torch.inference_mode()
@pytest.mark.parametrize(
    "state_dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_fixup_pointer_abi_covers_every_initial_state_dtype(state_dtype, strided):
    """`initial_state_dtype` is in the compile key, so each is its own kernel.

    The earlier pointer tests only ever built an fp32 pool -- the dtype vLLM's
    adapter converts to -- so the fp16 and bf16 specializations of the pointer
    entry were never compiled, let alone compared. Both pool layouts are
    crossed in, because the indexed strided case is the one whose inner
    strides the compile key carries.

    Three-way: the reference decides correctness, and the two interfaces are
    then required to agree with each other bit for bit.
    """
    entry = _sm80_fixup_entry()
    device = torch.device("cuda")
    seq_lens, heads = [2048, 1024, 0, 3000], 4
    transfer, state, cu, total = _fixup_inputs(seq_lens, heads, device)
    initial = _fixup_initial(
        len(seq_lens), heads, device, True, strided, dtype=state_dtype
    )
    indices = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
    kw = dict(initial_state=initial, state_indices=indices, cp_chunk_len=1024)

    got_tensor = entry(transfer, state, cu, total, **kw)
    torch.cuda.synchronize()
    snapshot = got_tensor.clone()
    torch.cuda.synchronize()
    got_ptr = entry(transfer, state, cu, total, **kw, _ptr_abi=True)
    torch.cuda.synchronize()

    assert snapshot.data_ptr() != got_ptr.data_ptr()
    assert torch.equal(
        snapshot.contiguous().view(torch.uint8),
        got_ptr.contiguous().view(torch.uint8),
    ), f"the interfaces disagree for a {state_dtype} pool"

    # And the result is not merely self-consistent: the fixup's own reference
    # decides it. It takes the per-sequence lists of local results, the way
    # `test_cp_delta_rule_fixup` assembles them.
    transfers_by_seq, states_by_seq, initials_by_seq, seq_indices = [], [], [], []
    for seq_idx in range(len(seq_lens)):
        seq_start = int(cu[seq_idx].item())
        seq_end = int(cu[seq_idx + 1].item())
        num_chunks = max(-(-(seq_end - seq_start) // 1024), 0)
        if num_chunks == 0:
            continue
        chunk_start = chunk_bound_host(seq_idx, seq_start, 1024)
        slots = list(range(chunk_start, chunk_start + num_chunks))
        transfers_by_seq.append(transfer[slots])
        states_by_seq.append(state[slots])
        initials_by_seq.append(initial[int(indices[seq_idx])].float())
        seq_indices.append(seq_idx)

    _, ref_by_seq = reference.cp_delta_rule_fixup_transposed(
        transfers_by_seq, states_by_seq, initials_by_seq
    )
    for seq_idx, seq_fixed in zip(seq_indices, ref_by_seq):  # noqa: B905
        chunk_start = chunk_bound_host(seq_idx, int(cu[seq_idx].item()), 1024)
        for chunk_idx in range(seq_fixed.shape[0]):
            torch.testing.assert_close(
                snapshot[chunk_start + chunk_idx],
                seq_fixed[chunk_idx],
                atol=FIXUP_TF32_ATOL,
                rtol=FIXUP_TF32_RTOL,
            )


@torch.inference_mode()
@pytest.mark.parametrize(
    "needs_initial,store_final,use_indices,checkpoint",
    [
        (i, f, x, c)
        for i in (False, True)
        for f in (False, True)
        for x in (False, True)
        for c in (False, True)
    ],
)
def test_prefill_pointer_abi_matches_tensor_abi(
    needs_initial, store_final, use_indices, checkpoint
):
    """All sixteen crossings of stage 4's optional arguments.

    Each is its own compiled kernel -- `needs_initial_state`,
    `store_final_state`, `use_state_indices` and `needs_checkpointing` are four
    of the thirteen axes in the compile key -- and the pointer entry keeps all
    thirteen on the instance rather than branching on a nullable pointer.

    `store_final_state=False` is the case that also pins a contract: the state
    argument aliases `fixed_state` there, and the kernel never addresses it.
    """
    if get_compute_capability(torch.device("cuda"))[0] != 8:
        pytest.skip("the pointer entry is SM80-only")
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm80 import (
        cp_delta_rule_prefill_dsl_sm80 as prefill,
    )

    device = torch.device("cuda")
    seq_lens, heads = [2048, 1024, 0, 3000], 4
    chunk = 1024
    k, v, t, alpha, cu, total, mx = _mn_inputs(
        seq_lens, heads, heads, torch.bfloat16, device
    )
    transfer, state_ws = _sm80_mn_entry()(
        k, v, t, alpha, cu, total, max_seqlen=mx, cp_chunk_len=chunk
    )
    torch.cuda.synchronize()
    rows = len(seq_lens) + 3 if use_indices else len(seq_lens)
    initial = (
        _fixup_initial(len(seq_lens), heads, device, use_indices)
        if needs_initial
        else None
    )
    indices = (
        torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
        if use_indices
        else None
    )
    fixed = _sm80_fixup_entry()(
        transfer,
        state_ws,
        cu,
        total,
        cp_chunk_len=chunk,
        initial_state=initial,
        state_indices=indices,
    )
    torch.cuda.synchronize()

    q = (
        (torch.randn(total, heads, 128, device=device) * 0.2)
        .to(torch.bfloat16)
        .contiguous()
    )
    out = torch.empty(total, heads, 128, dtype=torch.bfloat16, device=device)
    final = (
        torch.zeros(rows, heads, 128, 128, dtype=torch.bfloat16, device=device)
        if store_final
        else None
    )
    ckpt = starts = None
    if checkpoint:
        per = [max(-(-length // 512), 0) for length in seq_lens]
        starts = torch.tensor(
            [0, *torch.tensor(per).cumsum(0).tolist()],
            dtype=torch.int64,
            device=device,
        )
        ckpt = torch.zeros(
            max(int(starts[-1].item()), 1),
            heads,
            128,
            128,
            dtype=torch.float32,
            device=device,
        )

    kw = dict(
        o=out,
        state=final,
        q=q,
        k=k,
        v=v,
        t=t,
        fixed_state=fixed,
        alpha=alpha,
        scale=1.0 / 128**0.5,
        cu_seqlens=cu,
        total_seqlen=total,
        cp_chunk_len=chunk,
        max_seqlen=mx,
        initial_state=initial,
        state_indices=indices,
        state_checkpoints=ckpt,
        checkpoint_cu_starts=starts,
        checkpoint_every_n_tokens=512 if checkpoint else 0,
    )

    prefill(**kw)
    torch.cuda.synchronize()
    snap_o = out.clone()
    snap_state = final.clone() if final is not None else None
    snap_ckpt = ckpt.clone() if ckpt is not None else None

    out.fill_(0)
    if final is not None:
        final.fill_(0)
    if ckpt is not None:
        ckpt.fill_(0)
    prefill(**kw, _ptr_abi=True)
    torch.cuda.synchronize()

    assert torch.equal(snap_o.view(torch.uint8), out.view(torch.uint8)), (
        "the output differs"
    )
    if snap_state is not None:
        assert torch.equal(snap_state.view(torch.uint8), final.view(torch.uint8)), (
            "the final state differs"
        )
    if snap_ckpt is not None:
        assert torch.equal(snap_ckpt.view(torch.uint8), ckpt.view(torch.uint8)), (
            "the checkpoints differ"
        )


@torch.inference_mode()
def test_the_whole_cp_path_is_byte_identical_through_either_abi():
    """`_ptr_abi` on the composition switches all four entries at once."""
    if get_compute_capability(torch.device("cuda"))[0] != 8:
        pytest.skip("the pointer entry is SM80-only")
    import flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm80 as cp_mod

    device = torch.device("cuda")
    seq_lens, heads = [2048, 1024, 0, 3000], 2
    k, v, t, alpha, cu, total, mx = _mn_inputs(
        seq_lens, heads, heads, torch.bfloat16, device
    )
    q = (
        (torch.randn(total, heads, 128, device=device) * 0.2)
        .to(torch.bfloat16)
        .contiguous()
    )
    beta = (
        0.25 + 0.5 * torch.rand(total, heads, dtype=torch.float32, device=device)
    ).contiguous()
    initial = (
        torch.randn(len(seq_lens), heads, 128, 128, dtype=torch.float32, device=device)
        * 0.05
    ).contiguous()

    def run(ptr):
        out = torch.empty(total, heads, 128, dtype=torch.bfloat16, device=device)
        final = torch.zeros(
            len(seq_lens), heads, 128, 128, dtype=torch.bfloat16, device=device
        )
        cp_mod.cp_delta_rule_dsl_sm80(
            out,
            final,
            q,
            k,
            v,
            alpha,
            beta,
            cu,
            1.0 / 128**0.5,
            initial_state=initial,
            max_seqlen=mx,
            _ptr_abi=ptr,
        )
        torch.cuda.synchronize()
        return out, final

    o_a, s_a = run(False)
    o_a, s_a = o_a.clone(), s_a.clone()
    o_b, s_b = run(True)
    assert torch.equal(o_a.view(torch.uint8), o_b.view(torch.uint8))
    assert torch.equal(s_a.view(torch.uint8), s_b.view(torch.uint8))
