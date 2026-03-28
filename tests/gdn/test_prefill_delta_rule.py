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

import math
import os
import random

import torch
import pytest

from .reference_delta_rule import exclusive_cumsum, blockwise_delta_rule

from flashinfer.utils import get_compute_capability
from flashinfer.gdn_prefill import chunk_gated_delta_rule


def _skip_if_not_sm90():
    """Skip test if not SM90 architecture."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 9:
        pytest.skip(f"GDN prefill requires SM90, but got SM{cc[0]}{cc[1]}")


def _test_prefill_kernel(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: list[int],
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
):
    _skip_if_not_sm90()
    if not alpha and not beta:
        pytest.skip(
            "large diff due to output value amplitude explosion along token dimension"
        )

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_seqs = len(seq_lens)
    total_seqlen = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = max(num_q_heads, num_v_heads)

    dtype = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = torch.device("cuda")
    with device:
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        # l2 norm k to avoid numerical instability
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        cu_seq_lens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64)
        alpha = torch.rand(total_seqlen, num_sab_heads) if alpha else None
        beta = torch.rand(total_seqlen, num_sab_heads) if beta else None

    our_o = torch.empty(
        [total_seqlen, num_o_heads, head_size], dtype=q.dtype, device=q.device
    )
    our_state = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=q.device,
    )
    our_o.fill_(float("nan"))
    our_state.fill_(float("nan"))

    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seq_lens,
        True,
        output=our_o,
        output_state=our_state,
    )

    torch.cuda.synchronize()

    # postprocessing raw output: ref_state is v-last [H,K,V], our_state is k-last [H,V,K], transpose to match
    our_state = our_state.transpose(-1, -2)

    ref_o, ref_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        scale_factor=scale,
        alpha=alpha,
        beta=beta,
        state_dtype=torch.float32,
    )
    ref_o = ref_o.to(q.dtype)
    ref_state = ref_state.to(kv_dtype)

    if dtype == torch.bfloat16:
        ref_o = ref_o.to(dtype)
        atol_o = 1e-2
        rtol_o = 1e-2
        atol_kv = 5e-3
        rtol_kv = 1e-3
    else:
        atol_o = 1e-3
        rtol_o = 1e-3
        atol_kv = 1e-3
        rtol_kv = 1e-4

    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [
        (1, 1, 1),
        (4, 1, 1),
        (3, 3, 3),
        (6, 2, 2),
        (1, 1, 2),
        (2, 2, 4),
        (16, 16, 32),
        (16, 16, 64),
    ],
)
@pytest.mark.parametrize("seq_lens", [[64], [128], [256], [256, 256], [64, 128, 512]])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prefill_kernel_basic(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: list[int],
    scale: float | str,
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_prefill_kernel(
        qkv_factory,
        dtype,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        block_size,
        seq_lens,
        scale,
        alpha,
        beta,
        seed,
    )


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [
        (1, 1, 1),
        (4, 1, 1),
        (3, 3, 3),
        (6, 2, 2),
        (1, 1, 2),
        (2, 2, 4),
        (16, 16, 32),
        (16, 16, 64),
    ],
)
@pytest.mark.parametrize(
    "seq_lens",
    [[31], [61], [91], [121], [251], [511, 501], [31, 63, 93, 123, 150, 500]],
)
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prefill_kernel_nonfull(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: list[int],
    scale: float | str,
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_prefill_kernel(
        qkv_factory,
        dtype,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        block_size,
        seq_lens,
        scale,
        alpha,
        beta,
        seed,
    )


def _test_chunked_prefill(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    block_size: int,
    seq_lens1: list[int],
    seq_lens2: list[int],
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
):
    _skip_if_not_sm90()
    if not alpha and not beta:
        pytest.skip(
            "large diff due to output value amplitude explosion along token dimension"
        )

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_seqs = len(seq_lens1)
    assert num_seqs == len(seq_lens2)
    total_seqlen1 = sum(seq_lens1)
    total_seqlen2 = sum(seq_lens2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = max(num_q_heads, num_v_heads)

    dtype = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = torch.device("cuda")
    with device:
        q1, k1, v1 = qkv_factory(
            seq_lens1, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        q2, k2, v2 = qkv_factory(
            seq_lens2, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        # l2 norm k to avoid numerical instability
        k1 = torch.nn.functional.normalize(k1, p=2.0, dim=-1)
        k2 = torch.nn.functional.normalize(k2, p=2.0, dim=-1)
        cu_seq_lens1 = torch.tensor(exclusive_cumsum(seq_lens1), dtype=torch.int64)
        cu_seq_lens2 = torch.tensor(exclusive_cumsum(seq_lens2), dtype=torch.int64)
        alpha1 = torch.rand(total_seqlen1, num_sab_heads) if alpha else None
        alpha2 = torch.rand(total_seqlen2, num_sab_heads) if alpha else None
        beta1 = torch.rand(total_seqlen1, num_sab_heads) if beta else None
        beta2 = torch.rand(total_seqlen2, num_sab_heads) if beta else None

    our_o1 = torch.empty(
        [total_seqlen1, num_o_heads, head_size], dtype=q1.dtype, device=q1.device
    )
    our_o2 = torch.empty(
        [total_seqlen2, num_o_heads, head_size], dtype=q2.dtype, device=q2.device
    )
    our_state1 = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=q1.device,
    )
    our_state2 = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=q1.device,
    )
    our_o1.fill_(float("nan"))
    our_o2.fill_(float("nan"))
    our_state1.fill_(float("nan"))
    our_state2.fill_(float("nan"))

    chunk_gated_delta_rule(
        q1,
        k1,
        v1,
        alpha1,
        beta1,
        scale,
        None,
        True,
        cu_seq_lens1,
        True,
        output=our_o1,
        output_state=our_state1,
    )
    chunk_gated_delta_rule(
        q2,
        k2,
        v2,
        alpha2,
        beta2,
        scale,
        our_state1,
        True,
        cu_seq_lens2,
        True,
        output=our_o2,
        output_state=our_state2,
    )
    our_state = our_state2

    torch.cuda.synchronize()

    # postprocessing raw output: ref_state is v-last [H,K,V], our_state is k-last [H,V,K], transpose to match
    our_state = our_state.transpose(-1, -2)

    def concat_varlen(t1, cu_seq_lens1, t2, cu_seq_lens2):
        output = []
        for i in range(cu_seq_lens1.size(0) - 1):
            s1 = cu_seq_lens1[i]
            s2 = cu_seq_lens2[i]
            e1 = cu_seq_lens1[i + 1]
            e2 = cu_seq_lens2[i + 1]
            output.append(t1[s1:e1])
            output.append(t2[s2:e2])
        return torch.concat(output)

    cu_seq_lens1 = cu_seq_lens1.cpu()
    cu_seq_lens2 = cu_seq_lens2.cpu()
    our_o = concat_varlen(our_o1, cu_seq_lens1, our_o2, cu_seq_lens2)

    q = concat_varlen(q1, cu_seq_lens1, q2, cu_seq_lens2)
    k = concat_varlen(k1, cu_seq_lens1, k2, cu_seq_lens2)
    v = concat_varlen(v1, cu_seq_lens1, v2, cu_seq_lens2)
    alpha = concat_varlen(alpha1, cu_seq_lens1, alpha2, cu_seq_lens2) if alpha else None
    beta = concat_varlen(beta1, cu_seq_lens1, beta2, cu_seq_lens2) if beta else None

    seq_lens = [a + b for a, b in zip(seq_lens1, seq_lens2, strict=True)]

    ref_o, ref_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        scale_factor=scale,
        alpha=alpha,
        beta=beta,
        state_dtype=torch.float32,
    )
    ref_o = ref_o.to(q.dtype)
    ref_state = ref_state.to(kv_dtype)

    if dtype == torch.bfloat16:
        ref_o = ref_o.to(dtype)
        atol_o = 1e-2
        rtol_o = 1e-2
        atol_kv = 5e-3
        rtol_kv = 1e-3
    else:
        atol_o = 2e-3
        rtol_o = 1e-3
        atol_kv = 1e-3
        rtol_kv = 1e-4

    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(6, 2, 2), (2, 2, 4), (16, 16, 32), (16, 16, 64)],
)
@pytest.mark.parametrize(
    "seq_lens1, seq_lens2",
    list(
        zip(
            [[61], [128], [511, 501], [256, 256], [123, 150, 500], [64, 128, 512]],
            [[128], [61], [256, 256], [511, 501], [64, 128, 512], [123, 150, 500]],
            strict=True,
        )
    ),
)
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_chunked_prefill(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    block_size: int,
    seq_lens1: list[int],
    seq_lens2: list[int],
    scale: float | str,
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    _test_chunked_prefill(
        qkv_factory,
        dtype,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        block_size,
        seq_lens1,
        seq_lens2,
        scale,
        alpha,
        beta,
        seed,
    )


def _test_checkpoint(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_lens: list[int],
    scale: float,
    checkpoint_every_n_tokens: int,
    seed: int | None = None,
):
    """Test state checkpointing by comparing against prefix-based reference runs."""
    _skip_if_not_sm90()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_seqs = len(seq_lens)
    total_seqlen = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = max(num_q_heads, num_v_heads)

    dtype = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        cu_seq_lens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64)
        alpha = torch.rand(total_seqlen, num_sab_heads)
        beta = torch.rand(total_seqlen, num_sab_heads)

    # Compute per-sequence checkpoint counts and cu_starts
    # Only exact multiples; the final partial block state is in output_state
    ckpt_counts = [sl // checkpoint_every_n_tokens for sl in seq_lens]
    total_checkpoints = sum(ckpt_counts)
    ckpt_cu_starts = [0]
    for c in ckpt_counts:
        ckpt_cu_starts.append(ckpt_cu_starts[-1] + c)
    checkpoint_cu_starts = torch.tensor(
        ckpt_cu_starts, dtype=torch.int64, device=device
    )

    # Allocate outputs
    our_o = torch.empty(
        [total_seqlen, num_o_heads, head_size], dtype=dtype, device=device
    )
    our_state = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )
    state_checkpoints = torch.full(
        (total_checkpoints, num_sab_heads, head_size, head_size),
        float("nan"),
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
        cu_seq_lens,
        True,
        output=our_o,
        output_state=our_state,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )
    torch.cuda.synchronize()

    # Verify each checkpoint by running the kernel on prefixes
    seq_offset = exclusive_cumsum(seq_lens)
    for seq_idx in range(num_seqs):
        seq_start = seq_offset[seq_idx]
        seq_len = seq_lens[seq_idx]
        n_ckpts = ckpt_counts[seq_idx]

        for ckpt_idx in range(n_ckpts):
            prefix_len = min((ckpt_idx + 1) * checkpoint_every_n_tokens, seq_len)

            # Run kernel on just this prefix
            prefix_q = q[seq_start : seq_start + prefix_len].contiguous()
            prefix_k = k[seq_start : seq_start + prefix_len].contiguous()
            prefix_v = v[seq_start : seq_start + prefix_len].contiguous()
            prefix_alpha = alpha[seq_start : seq_start + prefix_len].contiguous()
            prefix_beta = beta[seq_start : seq_start + prefix_len].contiguous()
            prefix_cu = torch.tensor([0, prefix_len], dtype=torch.int64, device=device)

            prefix_o = torch.empty(
                [prefix_len, num_o_heads, head_size], dtype=dtype, device=device
            )
            prefix_state = torch.empty(
                (1, num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

            chunk_gated_delta_rule(
                prefix_q,
                prefix_k,
                prefix_v,
                prefix_alpha,
                prefix_beta,
                scale,
                None,
                True,
                prefix_cu,
                True,
                output=prefix_o,
                output_state=prefix_state,
            )
            torch.cuda.synchronize()

            ckpt_global_idx = ckpt_cu_starts[seq_idx] + ckpt_idx
            actual_ckpt = state_checkpoints[ckpt_global_idx]
            expected_ckpt = prefix_state[0]

            torch.testing.assert_close(
                actual_ckpt,
                expected_ckpt,
                atol=1e-3,
                rtol=1e-4,
                msg=f"Checkpoint mismatch: seq={seq_idx}, ckpt={ckpt_idx}",
            )


@pytest.mark.parametrize("checkpoint_every_n_tokens", [64, 128])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(4, 1, 1), (2, 2, 4)],
)
@pytest.mark.parametrize("seq_lens", [[256], [128, 256, 512]])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_checkpoint_correctness(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_lens: list[int],
    checkpoint_every_n_tokens: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale = 1.0 / math.sqrt(head_size)
    _test_checkpoint(
        qkv_factory,
        dtype,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        seq_lens,
        scale,
        checkpoint_every_n_tokens,
        seed,
    )


def test_checkpoint_noop(qkv_factory):
    """Verify that checkpoint_every_n_tokens=0 produces same results as without checkpointing."""
    _skip_if_not_sm90()

    seed = 42
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_lens = [256]
    num_q_heads, num_k_heads, num_v_heads = 4, 1, 1
    head_size = 128
    num_seqs = len(seq_lens)
    total_seqlen = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    scale = 1.0 / math.sqrt(head_size)
    device = torch.device("cuda")

    with device:
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, torch.float16
        )
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        cu_seq_lens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64)
        alpha = torch.rand(total_seqlen, num_sab_heads)
        beta = torch.rand(total_seqlen, num_sab_heads)

    # Run without checkpointing
    o1 = torch.empty(
        [total_seqlen, num_o_heads, head_size], dtype=torch.float16, device=device
    )
    s1 = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
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
        cu_seq_lens,
        True,
        output=o1,
        output_state=s1,
    )

    # Run with checkpoint_every_n_tokens=0 (disabled)
    o2 = torch.empty_like(o1)
    s2 = torch.empty_like(s1)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu_seq_lens,
        True,
        output=o2,
        output_state=s2,
        checkpoint_every_n_tokens=0,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(o1, o2)
    torch.testing.assert_close(s1, s2)


def test_checkpoint_alignment_error():
    """Verify that non-multiple-of-64 checkpoint interval raises ValueError."""
    with pytest.raises(ValueError, match="multiple of the chunk size"):
        chunk_gated_delta_rule(
            torch.empty(1),  # dummy, won't reach kernel
            torch.empty(1),
            torch.empty(1),
            checkpoint_every_n_tokens=100,
        )


def test_checkpoint_negative_interval():
    """Verify that negative checkpoint interval raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        chunk_gated_delta_rule(
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
            checkpoint_every_n_tokens=-1,
        )


def test_checkpoint_missing_tensors():
    """Verify error when checkpoint_every_n_tokens > 0 but tensors are None."""
    with pytest.raises(ValueError, match="must both be provided"):
        chunk_gated_delta_rule(
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
            checkpoint_every_n_tokens=64,
        )


def test_checkpoint_spurious_tensors():
    """Verify error when checkpoint_every_n_tokens == 0 but tensors are provided."""
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="must be None"):
        chunk_gated_delta_rule(
            torch.empty(1, 1, 128, device=device),
            torch.empty(1, 1, 128, device=device),
            torch.empty(1, 1, 128, device=device),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64, device=device),
            state_checkpoints=torch.empty(
                1, 1, 128, 128, dtype=torch.float32, device=device
            ),
            checkpoint_cu_starts=torch.tensor([0, 1], dtype=torch.int64, device=device),
            checkpoint_every_n_tokens=0,
        )


def test_checkpoint_wrong_dtype(qkv_factory):
    """Verify error when state_checkpoints has wrong dtype."""
    _skip_if_not_sm90()
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="float32"):
        chunk_gated_delta_rule(
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int64, device=device),
            state_checkpoints=torch.empty(
                1, 1, 128, 128, dtype=torch.float16, device=device
            ),
            checkpoint_cu_starts=torch.tensor([0, 1], dtype=torch.int64, device=device),
            checkpoint_every_n_tokens=64,
        )


def test_checkpoint_wrong_cu_starts_size(qkv_factory):
    """Verify error when checkpoint_cu_starts has wrong size."""
    _skip_if_not_sm90()
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="elements"):
        chunk_gated_delta_rule(
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int64, device=device),
            state_checkpoints=torch.empty(
                1, 1, 128, 128, dtype=torch.float32, device=device
            ),
            checkpoint_cu_starts=torch.tensor(
                [0, 1, 2], dtype=torch.int64, device=device
            ),
            checkpoint_every_n_tokens=64,
        )
