"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Chunked KDA training backward: parity with the reference chunked backward and bit-determinism.

The reference is flash-linear-attention v0.5.2 ``chunk_kda`` (in-kernel Q/K L2 norm,
lower-bound safe gate, sigmoid beta, ``dt_bias``).  The forward save set is taken from
the reference forward; the backward under test is :func:`flashinfer.chunk_kda_backward`.
"""

import pytest
import torch

fla = pytest.importorskip(
    "fla", reason="flash-linear-attention (reference) is not installed"
)
from fla.modules.l2norm import l2norm_fwd  # noqa: E402
from fla.ops.common.gate import fused_beta_sigmoid_fwd  # noqa: E402
from fla.ops.kda import chunk_kda  # noqa: E402
from fla.ops.kda.chunk_fwd import chunk_kda_fwd  # noqa: E402
from fla.ops.utils.index import prepare_chunk_indices  # noqa: E402

from flashinfer import chunk_kda_backward  # noqa: E402

LOWER_BOUND = -5.0
CHUNK = 64
GRADS = ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "dt_bias")


def _requires_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability(0) not in {(10, 0), (10, 3)}:
        pytest.skip("SM100a / SM103a is required")


def _inputs(batch, seq_len, heads, value_heads, *, seed, packed=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev, d = "cuda", 128
    rand = lambda *shape: torch.rand(
        *shape, generator=gen, dtype=torch.float32, device=dev
    )  # noqa: E731
    randn = lambda *shape: torch.randn(
        *shape, generator=gen, dtype=torch.float32, device=dev
    )  # noqa: E731
    total = batch * seq_len if packed else seq_len
    b = 1 if packed else batch
    q = (rand(b, total, heads, d) - 0.5).to(torch.bfloat16)
    k = (rand(b, total, heads, d) - 0.5).to(torch.bfloat16)
    v = (rand(b, total, value_heads, d) - 0.5).to(torch.bfloat16)
    g = (randn(b, total, value_heads, d) * 0.1).to(torch.bfloat16)
    beta = randn(b, total, value_heads).to(torch.bfloat16)
    A_log = torch.log(
        rand(value_heads) + 1.0
    )  # model-init regime: decay rates in [1, 2)
    dt_bias = randn(value_heads * d)
    do = randn(b, total, value_heads, d).to(torch.bfloat16)
    cu = None
    if packed:
        cu = torch.arange(
            0, batch * seq_len + 1, seq_len, dtype=torch.int32, device=dev
        )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        do=do,
        cu_seqlens=cu,
        scale=d**-0.5,
    )


def _reference(inp):
    leaves = {
        n: inp[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
    }
    cu = inp["cu_seqlens"]
    out = chunk_kda(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=inp["scale"],
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        cu_seqlens_cpu=None if cu is None else cu.cpu(),
        A_log=leaves["A_log"],
        dt_bias=leaves["dt_bias"],
    )
    out = out[0] if isinstance(out, tuple) else out
    order = ("q", "k", "v", "beta", "g", "A_log", "dt_bias")
    grads = torch.autograd.grad(out, [leaves[n] for n in order], grad_outputs=inp["do"])
    names = {
        "q": "dq",
        "k": "dk",
        "v": "dv",
        "beta": "dbeta",
        "g": "dg",
        "A_log": "dA_log",
        "dt_bias": "dt_bias",
    }
    return {names[n]: g.detach() for n, g in zip(order, grads, strict=False)}


def _saved(inp):
    with torch.no_grad():
        q_norm, q_rstd = l2norm_fwd(inp["q"])
        k_norm, k_rstd = l2norm_fwd(inp["k"])
        beta = fused_beta_sigmoid_fwd(inp["beta"], 1.0)
        cu = inp["cu_seqlens"]
        chunk_indices = (
            None
            if cu is None
            else prepare_chunk_indices(cu, CHUNK, cu_seqlens_cpu=cu.cpu())
        )
        out = chunk_kda_fwd(
            q=q_norm,
            k=k_norm,
            v=inp["v"],
            g=inp["g"],
            beta=beta,
            scale=inp["scale"],
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu,
            cu_seqlens_cpu=None if cu is None else cu.cpu(),
            chunk_indices=chunk_indices,
            chunk_size=CHUNK,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_gate_in_kernel=True,
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
        )
    return dict(
        q_norm=q_norm,
        k_norm=k_norm,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta=beta,
        Aqk=out[3],
        Akk=out[4],
    )


def _candidate(inp, saved):
    return chunk_kda_backward(
        q_norm=saved["q_norm"],
        k_norm=saved["k_norm"],
        q_rstd=saved["q_rstd"],
        k_rstd=saved["k_rstd"],
        v=inp["v"],
        g=inp["g"],
        beta_logits=inp["beta"],
        beta=saved["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        Aqk=saved["Aqk"],
        Akk=saved["Akk"],
        do=inp["do"],
        scale=inp["scale"],
        lower_bound=LOWER_BOUND,
        cu_seqlens=inp["cu_seqlens"],
    )


SHAPES = [
    pytest.param(1, 256, 2, 2, False, id="b1_t256_h2"),
    pytest.param(1, 512, 2, 4, False, id="b1_t512_h2_hv4_gva"),
    pytest.param(2, 384, 2, 2, False, id="b2_t384_h2"),
    pytest.param(2, 256, 2, 2, True, id="packed_2x256_h2"),
    pytest.param(1, 2048, 16, 16, False, id="b1_t2048_h16", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("batch,seq_len,heads,value_heads,packed", SHAPES)
def test_matches_reference_chunk_kda(batch, seq_len, heads, value_heads, packed):
    _requires_blackwell()
    inp = _inputs(
        batch, seq_len, heads, value_heads, seed=460_000 + seq_len, packed=packed
    )
    ref = _reference(inp)
    got = _candidate(inp, _saved(inp))
    for name in GRADS:
        r, g = ref[name], got[name]
        assert g.shape == r.shape, name
        assert g.dtype == r.dtype, (name, g.dtype, r.dtype)
        torch.testing.assert_close(
            g.float(),
            r.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@pytest.mark.parametrize("batch,seq_len,heads,value_heads,packed", SHAPES[:2])
def test_backward_is_bit_deterministic(batch, seq_len, heads, value_heads, packed):
    _requires_blackwell()
    inp = _inputs(
        batch, seq_len, heads, value_heads, seed=461_000 + seq_len, packed=packed
    )
    saved = _saved(inp)
    first = _candidate(inp, saved)
    torch.cuda.synchronize()
    second = _candidate(inp, saved)
    for name in GRADS:
        assert torch.equal(first[name], second[name]), (
            f"{name} differs between identical backward calls"
        )


def test_rejects_unsupported_shapes():
    _requires_blackwell()
    inp = _inputs(1, 192, 2, 2, seed=1)  # T % 128 != 0
    with pytest.raises(ValueError):
        _candidate(inp, _saved(inp))
