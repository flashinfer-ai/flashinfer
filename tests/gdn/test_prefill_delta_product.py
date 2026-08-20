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

Note on ``use_qk_l2norm_in_kernel``: as of this writing the prefill path
**ignores it**. ``chunk_gated_delta_rule`` declares and documents the parameter,
but no SM90/SM100/SM120 prefill kernel reads it, and vLLM's bridge normalises
q/k on the host before calling in (see ``fi_chunk_gated_delta_rule``). These
tests therefore pass ``False`` and pre-normalise ``k`` in
``_gen_product_inputs`` -- which is what the algorithm actually needs
(``||k|| == 1`` for the Householder interpretation; q needs nothing). Passing
``True`` behaves identically today but would silently break these tests the day
the flag is implemented.

Decode is the opposite: there ``use_qk_l2norm`` is live and normalises BOTH q
and k, which is why ``test_decode_delta_product._reference`` mirrors it.
"""

from __future__ import annotations

import random

import pytest
import torch

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_product import chunk_gated_delta_product
from flashinfer.utils import (
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm12x_supported,
)

from .reference_delta_product import delta_product
from .reference_delta_rule import delta_rule, exclusive_cumsum


def _skip_if_unsupported():
    """Mirror of test_prefill_delta_rule._skip_if_unsupported."""
    device = torch.device("cuda")
    if is_sm100a_supported(device):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip(
                f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}"
            )
        return
    if is_sm90a_supported(device) or is_sm12x_supported(device):
        return
    pytest.skip("GDN prefill requires SM90, SM100, or SM12x")


def _gen_product_inputs(
    seq_lens,
    num_householder,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    head_size,
    dtype,
    qkv_factory,
    device,
):
    """Build q/k/v/alpha/beta with a householder axis at dim 1 of k and v.

    Reuses ``conftest.gen_qkv`` by generating n_h times as many k/v rows and
    folding the extra rows into the householder axis -- so the element
    distribution matches the existing GDN tests exactly.
    """
    total_seqlen = sum(seq_lens)
    num_sab_heads = max(num_q_heads, num_v_heads)

    with device:
        # one q per real token; n_h keys/values per real token
        q, _, _ = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        _, k, v = qkv_factory(
            [s * num_householder for s in seq_lens],
            num_q_heads,
            num_k_heads,
            num_v_heads,
            head_size,
            dtype,
        )
        k = k.reshape(total_seqlen, num_householder, num_k_heads, head_size)
        v = v.reshape(total_seqlen, num_householder, num_v_heads, head_size)
        # l2 norm k to avoid numerical instability (as the GDN tests do)
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

        alpha = torch.rand(total_seqlen, num_sab_heads)
        beta = torch.rand(total_seqlen, num_householder, num_sab_heads)
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64)

    return q, k, v, alpha, beta, cu_seqlens


@pytest.mark.parametrize(
    "seq_lens",
    [[64], [128], [64, 128, 512]],
    ids=lambda seqlens: "seq_lens=" + ",".join(map(str, seqlens)),
)
@pytest.mark.parametrize("head_size", [128], ids=lambda hs: f"head_size={hs}")
@pytest.mark.parametrize(
    "num_heads",
    [(8, 8, 8), (8, 8, 16)],
    ids=lambda qkv: "num_heads={0}/{1}/{2}".format(*qkv),
)  # (q, k, v) -- GVA
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reference_nh1_equals_delta_rule(
    qkv_factory, seq_lens, head_size, num_heads, dtype, seed=0
):
    """At n_h == 1, DeltaProduct IS DeltaNet. Exact equality -- same fp ops, same order."""
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_q_heads, num_k_heads, num_v_heads = num_heads
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)

    q, k, v, alpha, beta, _ = _gen_product_inputs(
        seq_lens,
        1,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        qkv_factory,
        device,
    )

    prod_o, prod_state = delta_product(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=alpha,
        beta=beta,
        scale_factor=1.0,
    )
    rule_o, rule_state = delta_rule(
        q.float(),
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        seq_lens,
        alpha=alpha,
        beta=beta.squeeze(1),
        scale_factor=1.0,
    )

    torch.testing.assert_close(prod_o, rule_o, atol=0, rtol=0)
    torch.testing.assert_close(prod_state, rule_state, atol=0, rtol=0)


@pytest.mark.parametrize(
    "seq_lens",
    [[64], [256], [64, 128, 512]],
    ids=lambda seqlens: "seq_lens=" + ",".join(map(str, seqlens)),
)
@pytest.mark.parametrize("head_size", [128], ids=lambda hs: f"head_size={hs}")
@pytest.mark.parametrize(
    "num_heads",
    [(8, 8, 8), (8, 8, 16)],
    ids=lambda qkv: "num_heads={0}/{1}/{2}".format(*qkv),
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reference_nh1_matches_kernel(
    qkv_factory, seq_lens, head_size, num_heads, dtype, seed=0
):
    _skip_if_unsupported()
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_q_heads, num_k_heads, num_v_heads = num_heads
    num_o_heads = num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs, total_seqlen = len(seq_lens), sum(seq_lens)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)

    q, k, v, alpha, beta, cu_seqlens = _gen_product_inputs(
        seq_lens,
        1,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        qkv_factory,
        device,
    )

    our_o = torch.full(
        (total_seqlen, num_o_heads, head_size),
        float("nan"),
        dtype=q.dtype,
        device=device,
    )
    our_state = torch.full(
        (num_seqs, num_sab_heads, head_size, head_size),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    chunk_gated_delta_rule(
        q,
        k.squeeze(1),
        v.squeeze(1),
        alpha,
        beta.squeeze(1),
        1.0,  # scale
        None,  # initial_state
        True,  # output_final_state
        cu_seqlens,
        False,  # use_qk_l2norm_in_kernel -- see note below; k is pre-normalised
        output=our_o,
        output_state=our_state,
    )
    torch.cuda.synchronize()
    our_state = our_state.transpose(
        -1, -2
    )  # kernel is [.., V, K]; reference is [.., K, V]

    ref_o, ref_state = delta_product(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=alpha,
        beta=beta,
        scale_factor=1.0,
    )

    if dtype == torch.bfloat16:
        atol_o, rtol_o, atol_kv, rtol_kv = 1e-2, 1e-2, 5e-3, 1e-3
    else:
        atol_o, rtol_o, atol_kv, rtol_kv = 2e-3, 1e-3, 1e-3, 1e-4

    torch.testing.assert_close(our_o, ref_o.to(q.dtype), atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)


@pytest.mark.parametrize(
    "num_householder", [1, 2, 3, 4], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize(
    "seq_lens",
    [[128], [64, 128, 512]],
    ids=lambda seqlens: "seq_lens=" + ",".join(map(str, seqlens)),
)
@pytest.mark.parametrize(
    "num_heads",
    [(8, 8, 8), (8, 8, 16)],
    ids=lambda qkv: "num_heads={0}/{1}/{2}".format(*qkv),
)
def test_reference_equals_expanded_delta_rule(
    qkv_factory, num_householder, seq_lens, num_heads, seed=0
):
    """GDP is GDN on a sequence n_h times longer.

    Gate on the first micro-step of each token (neutral value 1.0 -- alpha here
    is multiplicative, not log-space), query only on the last, then read every
    n_h-th output row.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_q_heads, num_k_heads, num_v_heads = num_heads
    num_sab_heads = max(num_q_heads, num_v_heads)
    head_size, n_h = 128, num_householder
    total_seqlen = sum(seq_lens)
    device = torch.device("cuda")

    q, k, v, alpha, beta, _ = _gen_product_inputs(
        seq_lens,
        n_h,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        torch.float16,
        qkv_factory,
        device,
    )
    q, k, v = q.float(), k.float(), v.float()

    prod_o, prod_state = delta_product(q, k, v, seq_lens, alpha=alpha, beta=beta)

    with device:
        alpha_flat = torch.ones(total_seqlen * n_h, num_sab_heads)
        alpha_flat[0::n_h] = alpha
        q_flat = torch.zeros(total_seqlen * n_h, num_q_heads, head_size)
        q_flat[n_h - 1 :: n_h] = q
    rule_o, rule_state = delta_rule(
        q_flat,
        k.reshape(total_seqlen * n_h, num_k_heads, head_size),
        v.reshape(total_seqlen * n_h, num_v_heads, head_size),
        [s * n_h for s in seq_lens],
        alpha=alpha_flat,
        beta=beta.reshape(total_seqlen * n_h, num_sab_heads),
    )

    torch.testing.assert_close(prod_o, rule_o[n_h - 1 :: n_h], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prod_state, rule_state, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "num_householder", [1, 2, 3, 4], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize(
    "seq_lens",
    [[64], [256], [64, 128, 512]],
    ids=lambda s: "seq_lens=" + ",".join(map(str, s)),
)
@pytest.mark.parametrize(
    "num_heads",
    [(8, 8, 8), (8, 8, 16)],
    ids=lambda qkv: "num_heads={0}/{1}/{2}".format(*qkv),
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prefill_kernel_matches_reference(
    qkv_factory, num_householder, seq_lens, num_heads, dtype, seed=0
):
    """chunk_gated_delta_product == delta_product reference, on real kernels."""
    _skip_if_unsupported()
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_q_heads, num_k_heads, num_v_heads = num_heads
    num_o_heads = num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs, total_seqlen = len(seq_lens), sum(seq_lens)
    head_size, n_h = 128, num_householder
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)

    q, k, v, alpha, beta, cu_seqlens = _gen_product_inputs(
        seq_lens,
        n_h,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        qkv_factory,
        device,
    )

    our_o = torch.full(
        (total_seqlen, num_o_heads, head_size),
        float("nan"),
        dtype=q.dtype,
        device=device,
    )
    our_state = torch.full(
        (num_seqs, num_sab_heads, head_size, head_size),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    chunk_gated_delta_product(
        q,
        k,
        v,
        alpha,
        beta,
        1.0,  # scale
        None,  # initial_state
        True,  # output_final_state
        cu_seqlens,
        False,  # use_qk_l2norm_in_kernel -- see note below; k is pre-normalised
        output=our_o,
        output_state=our_state,
    )
    torch.cuda.synchronize()

    # state shape must not depend on n_h -- this is the whole selling point
    assert our_state.shape == (num_seqs, num_sab_heads, head_size, head_size)
    assert not our_o.isnan().any(), "output buffer left partially unwritten"

    our_state = our_state.transpose(-1, -2)  # kernel [.., V, K] -> ref [.., K, V]

    ref_o, ref_state = delta_product(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=alpha,
        beta=beta,
        scale_factor=1.0,
    )

    if dtype == torch.bfloat16:
        atol_o, rtol_o, atol_kv, rtol_kv = 1e-2, 1e-2, 5e-3, 1e-3
    else:
        atol_o, rtol_o, atol_kv, rtol_kv = 2e-3, 1e-3, 1e-3, 1e-4

    torch.testing.assert_close(our_o, ref_o.to(q.dtype), atol=atol_o, rtol=rtol_o)
    torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)


# --------------------------------------------------------------------------
# The four (pass_output x output_final_state) combinations. Every other test
# passes output= and output_final_state=True, which leaves the wrapper's
# allocate-and-return path and its no-state path completely unexercised --
# and those are exactly where a wrong strided gather or a hardcoded
# output_final_state can hide.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pass_output", [True, False], ids=["output=buf", "output=None"]
)
@pytest.mark.parametrize(
    "output_final_state", [True, False], ids=["final_state", "no_final_state"]
)
@pytest.mark.parametrize(
    "num_householder", [1, 3], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize(
    "num_heads",
    [(8, 8, 8), (8, 8, 16)],
    ids=lambda qkv: "num_heads={0}/{1}/{2}".format(*qkv),
)
def test_prefill_kernel_output_conventions(
    qkv_factory, pass_output, output_final_state, num_householder, num_heads, seed=0
):
    """Return contract and output rows must not depend on how they're requested."""
    _skip_if_unsupported()
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_q_heads, num_k_heads, num_v_heads = num_heads
    num_o_heads = num_sab_heads = max(num_q_heads, num_v_heads)
    seq_lens = [64, 128, 512]
    head_size, n_h = 128, num_householder
    num_seqs, total_seqlen = len(seq_lens), sum(seq_lens)
    device = torch.device("cuda")
    dtype = torch.float16

    q, k, v, alpha, beta, cu_seqlens = _gen_product_inputs(
        seq_lens,
        n_h,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        qkv_factory,
        device,
    )

    our_o = None
    if pass_output:
        our_o = torch.full(
            (total_seqlen, num_o_heads, head_size),
            float("nan"),
            dtype=dtype,
            device=device,
        )

    result = chunk_gated_delta_product(
        q,
        k,
        v,
        alpha,
        beta,
        1.0,  # scale
        None,  # initial_state
        output_final_state,
        cu_seqlens,
        False,  # use_qk_l2norm_in_kernel -- see note below; k is pre-normalised
        output=our_o,
        output_state=None,  # force the wrapper to own the state buffer
    )
    torch.cuda.synchronize()

    # --- return contract mirrors chunk_gated_delta_rule ---
    if output_final_state:
        assert isinstance(result, tuple) and len(result) == 2, (
            f"output_final_state=True must return (output, final_state); "
            f"got {type(result)}"
        )
        got_o, got_state = result
        assert got_state is not None, "final_state requested but None returned"
        assert got_state.shape == (num_seqs, num_sab_heads, head_size, head_size)
    else:
        assert not isinstance(result, tuple), (
            "output_final_state=False must return the output tensor alone, "
            f"got a {type(result)}"
        )
        got_o, got_state = result, None

    if pass_output:
        assert got_o is our_o, "output= was supplied; the same tensor must come back"
    assert got_o.shape == (total_seqlen, num_o_heads, head_size)
    assert not got_o.isnan().any(), "output rows left unwritten"

    ref_o, ref_state = delta_product(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=alpha,
        beta=beta,
        scale_factor=1.0,
    )
    torch.testing.assert_close(got_o, ref_o.to(dtype), atol=2e-3, rtol=1e-3)
    if got_state is not None:
        torch.testing.assert_close(
            got_state.transpose(-1, -2), ref_state, atol=1e-3, rtol=1e-4
        )
