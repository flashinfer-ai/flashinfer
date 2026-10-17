"""Packed GLM prefill contracts, block scales and cached-prefix correctness."""

import math

import pytest
import torch

from flashinfer.prefill import fmha_v2_prefill_glm


def make_inputs(qlens, klens, heads):
    torch.manual_seed(842)
    args = []
    for lengths in (qlens, klens, klens):
        x = (
            (torch.randn(heads, sum(lengths), 256, device="cuda") * 2)
            .to(torch.float8_e4m3fn)
            .transpose(0, 1)
        )
        args.append(x)
    for lengths in (qlens, klens):
        args.append(
            torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0)),
                dtype=torch.int32,
                device="cuda",
            )
        )
    for lengths in (qlens, klens, klens):
        # Deliberately different scales by request, head, and block. A single
        # scalar or a V scale applied only after all PV tiles must fail.
        shape = (len(lengths), heads, (max(lengths) + 63) // 64)
        args.append(torch.rand(shape, device="cuda") * 1.2 + 0.03)
    return args


def reference(args, qlens, klens, causal):
    q, k, v, _, _, sq, sk, sv = args
    outputs, lses, fp8_outputs = [], [], []
    qb = kb = 0
    for b, (nq, nk) in enumerate(zip(qlens, klens, strict=True)):
        rq = torch.arange(nq, device="cuda")
        rk = torch.arange(nk, device="cuda")
        dq = q[qb : qb + nq].float() * sq[b, :, rq // 64].T[..., None]
        dv = v[kb : kb + nk].float() * sv[b, :, rk // 64].T[..., None]
        # Keep QK on raw FP8 values (exactly representable even under TF32),
        # then apply scales. Scaling before matmul introduces different rounding.
        logits = q[qb : qb + nq].float().transpose(0, 1) @ k[
            kb : kb + nk
        ].float().permute(1, 2, 0)
        logits *= (sq[b, :, rq // 64] / 16)[..., None] * sk[b, :, rk // 64][:, None, :]
        if causal:
            logits.masked_fill_(rk[None, :] > rq[:, None] + nk - nq, float("-inf"))
        outputs.append((logits.softmax(-1) @ dv.transpose(0, 1)).transpose(0, 1))
        lses.append(logits.logsumexp(-1).T)
        # Independent blockwise FP8-P oracle. A dense FP32 reference alone
        # cannot distinguish a scale/indexing bug from probability rounding.
        running_max = torch.full_like(logits[..., 0], -float("inf"))
        running_sum = torch.zeros_like(running_max)
        acc = torch.zeros_like(dq.transpose(0, 1))
        prev_v_scale = torch.ones(q.shape[1], device="cuda")
        for start in range(0, nk, 64):
            stop = min(start + 64, nk)
            block = logits[..., start:stop]
            next_max = torch.maximum(running_max, block.amax(-1))
            alpha = (running_max - next_max).exp()
            prob = (block - (next_max[..., None] - math.log(256))).exp()
            next_v_scale = sv[b, :, start // 64]
            acc *= (alpha * (prev_v_scale / next_v_scale)[:, None])[..., None]
            acc += prob.to(torch.float8_e4m3fn).float() @ v[
                kb + start : kb + stop
            ].float().transpose(0, 1)
            running_sum = running_sum * alpha + prob.sum(-1)
            running_max, prev_v_scale = next_max, next_v_scale
        fp8_outputs.append(
            (acc * (prev_v_scale[:, None] / running_sum)[..., None]).transpose(0, 1)
        )
        qb += nq
        kb += nk
    return torch.cat(outputs), torch.cat(lses), torch.cat(fp8_outputs)


@pytest.mark.parametrize(
    "qlens,klens",
    [([1, 31, 129], [1, 33, 257]), ([65, 17], [129, 97]), ([1025, 65], [1537, 129])],
)
@pytest.mark.parametrize("heads", [1, 20])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_glm_packed(qlens, klens, heads, dtype, causal):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    args = make_inputs(qlens, klens, heads)
    out = torch.full(args[0].shape, float("nan"), device="cuda", dtype=dtype)
    stats = torch.full((*out.shape[:2], 2), float("nan"), device="cuda")

    def run():
        return fmha_v2_prefill_glm(
            *args, max(qlens), max(klens), causal=causal, out=out, softmax_stats=stats
        )

    run()
    expected_out, expected_stats = out.clone(), stats.clone()
    for _ in range(3):
        run()
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
    ref, lref, fp8_ref = reference(args, qlens, klens, causal)
    assert (out.float() - ref).norm() / ref.norm() < 0.03
    # Tiny QK/exp rounding differences can cross a discrete FP8-P midpoint,
    # so elementwise comparison to a different GEMM reduction order is brittle.
    # This 0.3% norm budget is 10x tighter than the dense FP32 comparison above
    # and includes the BF16 output's approximately 0.17% rounding RMS.
    assert (out.float() - fp8_ref).norm() / fp8_ref.norm() < 0.003
    lse = stats[..., 0] + (stats[..., 1] / 256).log()
    torch.testing.assert_close(lse, lref, rtol=1e-4, atol=2e-4)

    # Changing scale buffers and input bytes must affect a captured graph.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    args[0].zero_()
    args[-1].mul_(0.5)
    if qlens == [65, 17]:
        # Preserve total lengths and upper bounds, but move a request boundary.
        args[3][1].sub_(1)
        args[4][1].sub_(1)
    graph.replay()
    graph_out, graph_stats = out.clone(), stats.clone()
    run()
    torch.testing.assert_close(out, graph_out, rtol=0, atol=0)
    torch.testing.assert_close(stats, graph_stats, rtol=0, atol=0)
    assert not torch.equal(out, expected_out)
    updated_qlens = [64, 18] if qlens == [65, 17] else qlens
    updated_klens = [128, 98] if qlens == [65, 17] else klens
    # Q=0 makes P exactly 256 before normalization, eliminating FP8-P
    # rounding. This directly verifies V-scale changes and sequence boundaries.
    uniform_ref, uniform_lse, _ = reference(args, updated_qlens, updated_klens, causal)
    torch.testing.assert_close(out.float(), uniform_ref, rtol=0.004, atol=2e-4)
    torch.testing.assert_close(
        stats[..., 0] + (stats[..., 1] / 256).log(), uniform_lse, rtol=1e-5, atol=1e-5
    )


def test_sm120_glm_packed_validation():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    args = make_inputs([65, 17], [129, 97], 20)
    for index, bad in [
        (0, args[0].contiguous()),
        (1, args[1].half()),
        (3, args[3].long()),
        (5, args[5][..., :1]),
    ]:
        invalid = list(args)
        invalid[index] = bad
        with pytest.raises((ValueError, RuntimeError)):
            fmha_v2_prefill_glm(*invalid, 65, 129)
    with pytest.raises((ValueError, RuntimeError)):
        fmha_v2_prefill_glm(
            *args, 65, 129, out=torch.empty(args[0].shape, device="cuda")
        )
