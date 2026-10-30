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
from contextlib import nullcontext

import torch
import pytest

from .reference_delta_rule import exclusive_cumsum, blockwise_delta_rule

from flashinfer.utils import (
    get_compute_capability,
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm12x_supported,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule


def _skip_if_unsupported():
    """Skip test if not SM8x, SM90, SM100, or SM12x (with CUDA 13+)."""
    device = torch.device("cuda")
    if is_sm100a_supported(device):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip(
                f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}"
            )
    elif (
        is_sm12x_supported(device)
        or is_sm90a_supported(device)
        or get_compute_capability(device)[0] == 8
    ):
        pass  # No additional CUDA version requirement
    else:
        pytest.skip("GDN prefill requires SM8x, SM90, SM100, or SM12x")


def _skip_if_cp_unsupported():
    """Skip test if context parallelism is unsupported."""
    device = torch.device("cuda")
    if is_sm100a_supported(device):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip(
                f"SM100 CP GDN prefill requires CUDA 13+, got {torch.version.cuda}"
            )
        return
    if get_compute_capability(device)[0] == 8:
        return
    if not (is_sm90a_supported(device) or is_sm12x_supported(device)):
        pytest.skip("CP GDN prefill requires SM8x, SM90, SM100, or SM12x")


def _skip_if_not_sm100():
    """Skip test if not SM100 (Blackwell) with CUDA 13+."""
    device = torch.device("cuda")
    if not is_sm100a_supported(device):
        pytest.skip("Requires SM100 (Blackwell)")
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    if cuda_major < 13:
        pytest.skip(f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}")


def _skip_if_not_sm12x():
    device = torch.device("cuda")
    if not is_sm12x_supported(device):
        pytest.skip("Requires SM12x")


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
    use_cp: bool,
    seed: int | None = None,
):
    _skip_if_unsupported()
    if use_cp:
        _skip_if_cp_unsupported()
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
        use_qk_l2norm_in_kernel=False,
        output=our_o,
        output_state=our_state,
        use_cp=use_cp,
        max_seqlen=max(seq_lens),
    )

    torch.cuda.synchronize()

    # Transpose state to match reference layout
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
        atol_o = 2e-3
        rtol_o = 1e-3
        atol_kv = 1e-3
        rtol_kv = 1e-4

    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)
    return our_o, our_state


@torch.inference_mode()
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("seq_lens", [[64], [93]])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sm12x_triangular_tiles_match_dense(
    qkv_factory,
    monkeypatch,
    dtype: str,
    seq_lens: list[int],
    alpha: bool,
    beta: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    _skip_if_not_sm12x()

    import cutlass.cute as cute
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm120 as sm12x

    class DenseMmaDeltaRule(sm12x._FullyFusedDeltaRuleSm120):
        @cute.jit
        def _gemm_upper_triangular_b(self, tiled_mma, t_a, t_b, t_acc, b_shape):
            cute.gemm(tiled_mma, t_acc, t_a, t_b, t_acc)

    def run():
        return _test_prefill_kernel(
            qkv_factory,
            dtype,
            1,
            1,
            1,
            128,
            64,
            seq_lens,
            1.0,
            alpha,
            beta,
            False,
            seed,
        )

    our_o, our_state = run()
    monkeypatch.setattr(sm12x, "_FullyFusedDeltaRuleSm120", DenseMmaDeltaRule)
    monkeypatch.setattr(
        sm12x, "_get_prefill_kernel", sm12x._get_prefill_kernel.__wrapped__
    )
    ref_o, ref_state = run()

    torch.testing.assert_close(our_o, ref_o, atol=0, rtol=0)
    torch.testing.assert_close(our_state, ref_state, atol=0, rtol=0)


@torch.inference_mode()
def test_prefill_block_end_decay(qkv_factory, seed=0):
    _skip_if_unsupported()
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_lens = [64, 111, 192]
    total_seqlen = sum(seq_lens)
    num_heads = 1
    head_size = 128
    dtype = torch.float16
    device = torch.device("cuda")

    with device:
        q, k, v = qkv_factory(
            seq_lens, num_heads, num_heads, num_heads, head_size, dtype
        )
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        alpha = 0.99 + 0.01 * torch.rand(total_seqlen, num_heads)
        beta = 0.99 + 0.01 * torch.rand(total_seqlen, num_heads)
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64)

    our_o = torch.empty_like(q)
    our_state = torch.empty(
        (len(seq_lens), num_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        1.0,
        None,
        True,
        cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=our_o,
        output_state=our_state,
        use_cp=False,
    )

    ref_o, ref_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=alpha,
        beta=beta,
        block_size=64,
        state_dtype=torch.float32,
    )
    torch.testing.assert_close(our_o, ref_o.to(dtype), atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(
        our_state.transpose(-1, -2), ref_state, atol=1e-3, rtol=1e-4
    )


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
@pytest.mark.parametrize("use_cp", [False, True])
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
    use_cp: bool,
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
        use_cp,
        seed,
    )


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
@pytest.mark.parametrize("use_cp", [False, True])
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
    use_cp: bool,
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
        use_cp,
        seed,
    )


@pytest.mark.parametrize("use_cp", [False, True])
@pytest.mark.parametrize(
    "num_q_heads,num_k_heads,num_v_heads", [(1, 1, 1), (16, 16, 64)]
)
@pytest.mark.parametrize("seq_len", [256, 255])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prefill_kernel_zero_length_sequence(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    seq_len: int,
    use_cp: bool,
    scale: float = 0.1,
    seed: int = int(os.environ.get("SEED", "0")),
):
    _skip_if_unsupported()
    if use_cp:
        _skip_if_cp_unsupported()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    head_size = 128
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = max(num_q_heads, num_v_heads)
    dtype = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        q, k, v = qkv_factory(
            [seq_len], num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        alpha = torch.rand(seq_len, num_sab_heads)
        beta = torch.rand(seq_len, num_sab_heads)
        cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int64)
        cu_seq_lens_with_empty = torch.tensor([0, seq_len, seq_len], dtype=torch.int64)

    ref_o = torch.empty(
        [seq_len, num_o_heads, head_size], dtype=q.dtype, device=q.device
    )
    our_o = torch.empty_like(ref_o)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        False,
        cu_seq_lens,
        use_qk_l2norm_in_kernel=False,
        output=ref_o,
    )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        False,
        cu_seq_lens_with_empty,
        use_qk_l2norm_in_kernel=False,
        output=our_o,
        use_cp=use_cp,
        max_seqlen=seq_len,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_o, ref_o, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "state_source", ["caller", "allocated", "allocated_from_initial"]
)
@pytest.mark.parametrize("use_cp", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prefill_zero_length_sequence_state_untouched(
    qkv_factory,
    dtype: str,
    use_cp: bool,
    state_source: str,
    scale: float = 0.1,
    seed: int = int(os.environ.get("SEED", "0")),
):
    _skip_if_unsupported()
    if use_cp:
        _skip_if_cp_unsupported()
    if (
        state_source != "caller"
        and get_compute_capability(torch.device("cuda"))[0] != 8
    ):
        pytest.skip("only SM8x defines the empty row of an allocated state")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_len = 256
    head_size = 128
    num_heads = 1
    sentinel = 123.0
    dtype = getattr(torch, dtype)
    device = torch.device("cuda")

    with device:
        q, k, v = qkv_factory(
            [seq_len], num_heads, num_heads, num_heads, head_size, dtype
        )
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        alpha = torch.rand(seq_len, num_heads)
        beta = torch.rand(seq_len, num_heads)
        cu_seq_lens = torch.tensor([0, seq_len, seq_len], dtype=torch.int64)
        initial_state = None
        if state_source == "allocated_from_initial":
            initial_state = torch.randn(2, num_heads, head_size, head_size) * 0.05

    our_o = torch.empty([seq_len, num_heads, head_size], dtype=q.dtype, device=q.device)
    our_state = torch.empty(
        (2, num_heads, head_size, head_size),
        dtype=torch.float32,
        device=q.device,
    )
    our_state.fill_(sentinel)

    _, out_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        initial_state,
        True,
        cu_seq_lens,
        use_qk_l2norm_in_kernel=False,
        output=our_o,
        output_state=our_state if state_source == "caller" else None,
        use_cp=use_cp,
        max_seqlen=seq_len,
    )
    torch.cuda.synchronize()

    # Nothing writes the empty row: it keeps the caller's buffer, and a buffer
    # allocated here starts as the initial state, or as zeros without one.
    if state_source == "caller":
        want = torch.full_like(our_state[1], sentinel)
    elif initial_state is None:
        want = torch.zeros_like(our_state[1])
    else:
        want = initial_state[1]
    torch.testing.assert_close(
        out_state[1],
        want,
        atol=0,
        rtol=0,
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
    _skip_if_unsupported()
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
        use_qk_l2norm_in_kernel=False,
        output=our_o1,
        output_state=our_state1,
        use_cp=False,
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
        use_qk_l2norm_in_kernel=False,
        output=our_o2,
        output_state=our_state2,
        use_cp=False,
    )
    our_state = our_state2

    torch.cuda.synchronize()

    # Transpose state to match reference layout
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
    use_cp: bool = False,
    cp_chunk_mult: int = 1,
):
    """Test state checkpointing by comparing against prefix-based reference runs."""
    _skip_if_unsupported()
    cp_chunk_len = checkpoint_every_n_tokens * cp_chunk_mult if use_cp else None

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
        use_qk_l2norm_in_kernel=False,
        output=our_o,
        output_state=our_state,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_cp=use_cp,
        _cp_chunk_len=cp_chunk_len,
        max_seqlen=max(seq_lens),
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
                use_qk_l2norm_in_kernel=False,
                output=prefix_o,
                output_state=prefix_state,
                use_cp=use_cp,
                _cp_chunk_len=cp_chunk_len,
            )
            torch.cuda.synchronize()

            ckpt_global_idx = ckpt_cu_starts[seq_idx] + ckpt_idx
            actual_ckpt = state_checkpoints[ckpt_global_idx]
            expected_ckpt = prefix_state[0]

            # A mid-chunk checkpoint is not bit-identical to a prefix run on SM100.
            if cp_chunk_mult > 1:
                torch.testing.assert_close(
                    actual_ckpt, expected_ckpt, atol=4e-2, rtol=4e-2
                )
            else:
                assert torch.equal(actual_ckpt, expected_ckpt), (
                    f"Checkpoint mismatch: seq={seq_idx}, ckpt={ckpt_idx}"
                )


@pytest.mark.parametrize("checkpoint_every_n_tokens", [64, 128])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [(4, 1, 1), (2, 2, 4)],
)
@pytest.mark.parametrize("seq_lens", [[256], [128, 256, 512]])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
# A chunk of three checkpoint intervals stores mid-chunk as well as at its end.
@pytest.mark.parametrize("use_cp, cp_chunk_mult", [(False, 1), (True, 1), (True, 3)])
def test_checkpoint_correctness(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_lens: list[int],
    checkpoint_every_n_tokens: int,
    use_cp: bool,
    cp_chunk_mult: int,
    seed: int = int(os.environ.get("SEED", "0")),
):
    if use_cp:
        _skip_if_cp_unsupported()
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
        use_cp=use_cp,
        cp_chunk_mult=cp_chunk_mult,
    )


def test_checkpoint_noop(qkv_factory):
    """Verify that checkpoint_every_n_tokens=0 produces same results as without checkpointing."""
    _skip_if_unsupported()

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
        use_qk_l2norm_in_kernel=False,
        output=o1,
        output_state=s1,
        use_cp=False,
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
        use_qk_l2norm_in_kernel=False,
        output=o2,
        output_state=s2,
        checkpoint_every_n_tokens=0,
        use_cp=False,
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
    _skip_if_unsupported()
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="state_checkpoints must have dtype"):
        chunk_gated_delta_rule(
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            torch.empty(64, 1, 128, dtype=torch.float16, device=device),
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int64, device=device),
            state_checkpoints=torch.empty(
                1, 1, 128, 128, dtype=torch.int32, device=device
            ),
            checkpoint_cu_starts=torch.tensor([0, 1], dtype=torch.int64, device=device),
            checkpoint_every_n_tokens=64,
        )


@pytest.mark.parametrize("name", ["state_checkpoints", "checkpoint_cu_starts"])
def test_checkpoint_non_contiguous_rejected(name):
    """A strided checkpoint tensor is refused rather than copied or misread."""
    _skip_if_unsupported()
    device = torch.device("cuda")
    H, D = 2, 128
    seq_len, every = 256, 64
    num_checkpoints = seq_len // every
    zeros = torch.zeros(seq_len, H, D, dtype=torch.bfloat16, device=device)
    checkpoint_args = {
        "state_checkpoints": torch.zeros(
            num_checkpoints, H, D, D, dtype=torch.float32, device=device
        ),
        "checkpoint_cu_starts": torch.tensor(
            [0, num_checkpoints], dtype=torch.int64, device=device
        ),
    }
    # The same values, read as every other element of a buffer twice as long.
    checkpoint_args[name] = checkpoint_args[name].repeat_interleave(2, dim=0)[::2]
    assert not checkpoint_args[name].is_contiguous()
    with pytest.raises(RuntimeError, match=f"{name} must be contiguous"):
        chunk_gated_delta_rule(
            zeros,
            zeros,
            zeros,
            cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int64, device=device),
            checkpoint_every_n_tokens=every,
            **checkpoint_args,
        )


def test_checkpoint_wrong_cu_starts_size(qkv_factory):
    """Verify error when checkpoint_cu_starts has wrong size."""
    _skip_if_unsupported()
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


# ---------------------------------------------------------------------------
# State dtype tests
# ---------------------------------------------------------------------------


def _test_prefill_kernel_state_dtype(
    qkv_factory,
    dtype: str,
    state_dtype: torch.dtype,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_lens: list[int],
    scale: float,
    use_cp: bool,
    seed: int | None = None,
):
    _skip_if_unsupported()
    if use_cp:
        _skip_if_cp_unsupported()

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_seqs = len(seq_lens)
    total_seqlen = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

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
        initial_state_ref = (
            torch.randn(
                num_seqs,
                num_sab_heads,
                head_size,
                head_size,
                dtype=torch.float32,
            )
            * 0.01
        ).to(state_dtype)
        initial_state = initial_state_ref.transpose(-1, -2).contiguous()

    our_o = torch.empty(
        [total_seqlen, num_o_heads, head_size], dtype=dtype, device=device
    )
    our_state = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=state_dtype,
        device=device,
    )
    our_o.fill_(float("nan"))
    our_state.zero_()

    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        initial_state,
        True,
        cu_seq_lens,
        use_qk_l2norm_in_kernel=False,
        output=our_o,
        output_state=our_state,
        use_cp=use_cp,
        max_seqlen=max(seq_lens),
    )

    torch.cuda.synchronize()

    # Transpose state to match reference layout [N, H, K, V]
    our_state = our_state.transpose(-1, -2)

    ref_o, ref_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        scale_factor=scale,
        alpha=alpha,
        beta=beta,
        state_dtype=state_dtype,
        initial_state=initial_state_ref,
    )
    ref_o = ref_o.to(dtype)
    ref_state = ref_state.to(state_dtype).float()
    our_state = our_state.float()

    atol_o = 1e-1 if state_dtype != torch.float32 else 5e-2
    rtol_o = 5e-2
    atol_kv = 1e-1
    rtol_kv = 5e-2

    torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)


@pytest.mark.parametrize("scale", ["auto"])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [
        (1, 1, 1),
        (6, 2, 2),
        (16, 16, 32),
    ],
)
@pytest.mark.parametrize("seq_lens", [[64], [256], [256, 256], [64, 128, 512]])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize(
    "state_dtype",
    [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2],
)
@pytest.mark.parametrize("use_cp", [False, True])
def test_prefill_kernel_state_dtype(
    qkv_factory,
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_lens: list[int],
    scale: float | str,
    state_dtype: torch.dtype,
    use_cp: bool,
    seed: int = int(os.environ.get("SEED", "0")),
):
    scale = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    # SM8x refuses an FP8 state up front.
    fp8_refused = state_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and (
        torch.cuda.get_device_capability()[0] == 8
    )
    with (
        pytest.raises(NotImplementedError, match="FP8 state")
        if fp8_refused
        else nullcontext()
    ):
        _test_prefill_kernel_state_dtype(
            qkv_factory,
            dtype,
            state_dtype,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            head_size,
            seq_lens,
            scale,
            use_cp,
            seed=seed,
        )


@pytest.mark.parametrize(
    "cu_seqlens_cpu, max_seqlen, error, match",
    [
        ([0, 192], None, NotImplementedError, "num_seqs \\+ 1"),
        ([0, 200, 192], None, NotImplementedError, "monotonic"),
        ([1, 64, 192], None, NotImplementedError, "monotonic"),
        ([0, 64, 191], None, NotImplementedError, "monotonic"),
        ([0, 64, 192], 96, ValueError, "smaller than the longest"),
    ],
)
def test_sm8x_cu_seqlens_cpu_is_checked_on_its_own(
    cu_seqlens_cpu, max_seqlen, error, match
):
    """A host copy that cannot describe the batch is refused before launch."""
    device = torch.device("cuda")
    if get_compute_capability(device)[0] != 8:
        pytest.skip("only the SM8x prefill routes on cu_seqlens_cpu")
    zeros = torch.zeros(192, 1, 128, dtype=torch.bfloat16, device=device)
    with pytest.raises(error, match=match):
        chunk_gated_delta_rule(
            zeros,
            zeros,
            zeros,
            cu_seqlens=torch.tensor([0, 64, 192], dtype=torch.int64, device=device),
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=max_seqlen,
        )


def test_sm80_v32_matches_v64(qkv_factory, monkeypatch):
    """V32 on its one shape matches V64, which runs without `cu_seqlens_cpu`."""
    device = torch.device("cuda")
    if get_compute_capability(device) != (8, 0):
        pytest.skip("V32 is selected on compute capability 8.0 only")
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm80

    v32_applies = delta_rule_sm80._v32_applies
    taken = []
    monkeypatch.setattr(
        delta_rule_sm80,
        "_v32_applies",
        lambda **kwargs: taken.append(v32_applies(**kwargs)) or taken[-1],
    )
    torch.manual_seed(0)
    seq_len, head_size = 8192, 128
    with device:
        q, k, v = qkv_factory([seq_len], 4, 4, 16, head_size, torch.bfloat16)
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        alpha = torch.rand(seq_len, 16)
        beta = torch.rand(seq_len, 16)
        initial_state = (torch.randn(1, 16, head_size, head_size) * 0.01).to(
            torch.bfloat16
        )
        cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int64)

    runs = [
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            1.0 / math.sqrt(head_size),
            initial_state,
            True,
            cu_seq_lens,
            use_qk_l2norm_in_kernel=False,
            output_state=torch.empty_like(initial_state),
            use_cp=False,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        for cu_seqlens_cpu in ([0, seq_len], None)
    ]
    torch.cuda.synchronize()

    assert taken == [True, False]
    (v32_o, v32_state), (v64_o, v64_state) = runs
    assert torch.equal(v32_o, v64_o)
    assert torch.equal(v32_state, v64_state)
