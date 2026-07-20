from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer.experimental.sm12x.norm import mhc

SM12XMHCScratchCaps = mhc.Caps
sm12x_mhc_post = mhc.run_post
sm12x_mhc_post_pre = mhc.run_post_pre
plan_mhc_scratch = mhc.plan

from ..conftest import require_sm12x as require_sm120


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    y_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    y = y.to(residual.dtype if y_dtype is None else y_dtype)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    residual = (
        torch.randn((tokens, 4, hidden_size), generator=gen, dtype=torch.float32).to(
            device
        )
        / 3
    ).to(torch.bfloat16)
    x = (
        torch.randn((tokens, hidden_size), generator=gen, dtype=torch.float32).to(
            device
        )
        / 4
    ).to(torch.bfloat16)
    fn = (
        torch.randn((24, 4 * hidden_size), generator=gen, dtype=torch.float32).to(
            device
        )
        / 64
    )
    scale = torch.randn((3,), generator=gen, dtype=torch.float32).to(device) / 3
    bias = torch.randn((24,), generator=gen, dtype=torch.float32).to(device) / 5
    return (
        residual.contiguous(),
        x.contiguous(),
        fn.contiguous(),
        scale.contiguous(),
        bias.contiguous(),
    )


def _make_mhc_binding(
    *,
    tokens: int,
    hidden_size: int,
    device: torch.device,
    split_k: int = 64,
    expected_m: int | None = None,
):
    max_tokens = max(tokens, expected_m or tokens)
    plan = plan_mhc_scratch(
        SM12XMHCScratchCaps(
            device=device,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            split_k=split_k,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    return plan.bind(
        scratch=scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device),
        post=torch.empty((tokens, 4), dtype=torch.float32, device=device),
        comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
        out=torch.empty((tokens, 4, hidden_size), dtype=torch.bfloat16, device=device),
    )


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_sm12x_mhc_fused_post_pre_match_reference(tokens: int) -> None:
    device = require_sm120()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_450 + tokens,
        device=device,
    )
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post_arg = prev_post.contiguous()
    if tokens == 3:
        prev_post_arg = prev_post_arg.unsqueeze(-1).contiguous()

    residual_cur, post, comb, y = sm12x_mhc_post_pre(
        x,
        residual,
        prev_post_arg,
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        binding=binding,
    )
    torch.cuda.synchronize(device)

    assert (
        residual_cur.untyped_storage().data_ptr()
        == binding.out.untyped_storage().data_ptr()
    )
    assert (
        post.untyped_storage().data_ptr()
        == binding.post_buffer.untyped_storage().data_ptr()
    )
    assert (
        comb.untyped_storage().data_ptr()
        == binding.comb_buffer.untyped_storage().data_ptr()
    )
    assert y.untyped_storage().data_ptr() == binding.y.untyped_storage().data_ptr()

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=8e-3)
    scalar_atol = 2e-5 if tokens >= 8 else 1e-5
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=scalar_atol)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=scalar_atol)


@pytest.mark.parametrize("tokens", [1, 3])
def test_sm12x_mhc_fused_post_pre_with_rmsnorm_match_reference(tokens: int) -> None:
    device = require_sm120()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_470 + tokens,
        device=device,
    )
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(91_471 + tokens)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )

    residual_cur, post, comb, y = sm12x_mhc_post_pre(
        x,
        residual,
        prev_post.contiguous(),
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        binding=binding,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    torch.cuda.synchronize(device)

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    # The fused post_pre kernel (like vLLM's TileLang kernel) computes the
    # RMSNorm variance in fp32 from the collapsed activation -- not from the
    # bf16-rounded activation -- so reference the variance from fp32 y too
    # (matching vllm_y_max == fused_y_max in the benchmark). The activation
    # itself is still bf16 (it is stored bf16 before the norm multiply).
    y_raw_ref_fp32, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        y_dtype=torch.float32,
    )
    rms_scale = torch.rsqrt(y_raw_ref_fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
    y_ref = (
        y_raw_ref_fp32.to(torch.bfloat16).float() * rms_scale * norm_weight.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=4e-5)


def test_sm12x_mhc_fused_post_pre_graph_capture() -> None:
    device = require_sm120()
    tokens = 2
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_460,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post_arg = prev_post.contiguous()
    prev_comb_arg = prev_comb.contiguous()
    # CUDA graph capture requires caller-owned scratch (the partials buffer).
    binding = _make_mhc_binding(
        tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    residual_cur = binding.out
    y = binding.y
    post = binding.post_buffer
    comb = binding.comb_buffer

    def run() -> None:
        sm12x_mhc_post_pre(
            x,
            residual,
            prev_post_arg,
            prev_comb_arg,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            binding=binding,
        )

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=4e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=1e-5)
