"""Compare multi-head latent attention (MLA) normalization with FlashAttention 2.

Exact FlashAttention 2 (FA2) parity is checked for BF16 Q/K width 192,
V width 128 and a 128-by-64 attention tile. Other geometries retain tolerance-based coverage
in tests/attention/test_varlen.py; bitwise equivalence is not a general
attention contract.
"""

import math

import pytest
import torch
import torch.nn.functional as functional

from b12x.testing.reference.helpers import require_b12x


def _fa2():
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip(
            "Normalization comparison requires vLLM's FlashAttention 2 extension"
        )
    return flash_attn_varlen_func


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "q_lengths,k_lengths",
    [
        ([1], [1]),
        ([63, 65, 129], [127, 129, 257]),
        ([1025], [1025]),
        ([2049], [4097]),
    ],
)
def test_mla_normalization_matches_fa2_under_replay(
    monkeypatch,
    causal,
    q_lengths,
    k_lengths,
):
    from b12x.attention import varlen
    from b12x.attention._shared.contiguous import api as native
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    device = require_b12x()
    flash_attention = _fa2()
    torch.manual_seed(8625)
    q = torch.randn((sum(q_lengths), 3, 192), dtype=torch.bfloat16, device=device)
    device = q.device
    k = torch.randn((sum(k_lengths), 3, 192), dtype=q.dtype, device=device)
    v = torch.randn((sum(k_lengths), 3, 128), dtype=q.dtype, device=device)
    cq = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    ck = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    scale = 1 / math.sqrt(192)
    declaration = varlen.plan(
        q,
        k,
        v,
        cq,
        ck,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        override=varlen.VarlenAttentionConfig(tile_m=128, tile_n=64),
    )

    def prepare(state):
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(
            scratch=scratch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            softmax_scale=scale,
        )
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch,))

    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (declaration.request(name="mla-normalization", prepare_call=prepare),)
        )
        state = require_prepared(declaration, "attention.varlen", device)
        program = state.plan.compiled
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

        def no_compile(*args, **kwargs):
            raise AssertionError("MLA replay must reuse the prepared program")

        monkeypatch.setattr(native, "_compile_varlen_attention", no_compile)
        binding = varlen.bind(
            declaration,
            scratch=scratch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            softmax_scale=scale,
        )
        assert binding.binding.plan.compiled is program
        varlen.run(binding)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual, actual_lse = varlen.run(binding)
        try:
            for amplitude in (1.0, 4.0):
                q.normal_().mul_(amplitude)
                k.normal_()
                v.normal_()
                actual.fill_(float("nan"))
                actual_lse.fill_(float("nan"))
                graph.replay()
                expected, expected_lse = flash_attention(
                    q=q,
                    k=k,
                    v=functional.pad(v, (0, 64)),
                    cu_seqlens_q=cq,
                    cu_seqlens_k=ck,
                    max_seqlen_q=max(q_lengths),
                    max_seqlen_k=max(k_lengths),
                    softmax_scale=scale,
                    causal=causal,
                    return_softmax_lse=True,
                    fa_version=2,
                )
                torch.testing.assert_close(actual, expected[..., :128], rtol=0, atol=0)
                torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
                assert torch.isfinite(actual_lse).all()
                q_start = k_start = 0
                for q_length, k_length in zip(q_lengths, k_lengths, strict=True):
                    rows = (
                        torch.linspace(0, q_length - 1, 9, device=device)
                        .long()
                        .unique()
                    )
                    scores = (
                        torch.einsum(
                            "qhd,khd->hqk",
                            q[q_start + rows].double(),
                            k[k_start : k_start + k_length].double(),
                        )
                        * scale
                    )
                    if causal:
                        mask = torch.arange(k_length, device=device)[None] > (
                            rows[:, None] + k_length - q_length
                        )
                        scores.masked_fill_(mask[None], -torch.inf)
                    truth = torch.einsum(
                        "hqk,khd->qhd",
                        scores.softmax(-1),
                        v[k_start : k_start + k_length].double(),
                    )
                    torch.testing.assert_close(
                        actual[q_start + rows].double(),
                        truth,
                        atol=0.02,
                        rtol=0.01,
                    )
                    torch.testing.assert_close(
                        actual_lse[:, q_start + rows].double(),
                        scores.logsumexp(-1),
                        atol=0.0001,
                        rtol=0.00001,
                    )
                    q_start += q_length
                    k_start += k_length
        finally:
            graph.reset()
