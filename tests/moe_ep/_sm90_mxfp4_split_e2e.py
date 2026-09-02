"""Shared independent E2E oracle utilities for split SM90 MXFP4 tests."""

from __future__ import annotations


import torch

from tests.moe_ep.test_moe_ep_sm90_pull_mxfp4_mega_multirank import (
    HIDDEN,
    INTERMEDIATE,
    K64,
    LOCAL_EXPERTS,
    _fast_fp8_mm,
    _prepare_global_humming_operands,
    _quantize_input_per_token,
    _swiglu_sm90_formula,
)


def all_gather_stack(tensor: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered)


def make_hidden(
    rank: int,
    num_tokens: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    return (
        (
            0.75
            * torch.randn(
                num_tokens,
                HIDDEN,
                dtype=torch.float32,
                generator=generator,
            )
            + 0.03125 * (rank + 1)
        )
        .to(torch.bfloat16)
        .cuda()
    )


def global_route_reference(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    raw_global,
    *,
    return_handoff: bool = False,
):
    """Compute all-rank output and optional route-indexed K64 handoff.

    Unlike the Phase-A coverage oracle this handles arbitrary top-k, experts
    with no routes, ``topk_idx=-1``, and zero-token input.
    """
    from tests.moe_ep._sm90_mxfp4_split_reference import (
        quantize_handoff_reference,
    )

    world_size, num_tokens, topk = topk_ids.shape
    fc1, fc1_common, fc2, fc2_common = _prepare_global_humming_operands(raw_global)
    input_fp8, input_scale = _quantize_input_per_token(hidden)
    terms = torch.zeros(
        world_size,
        num_tokens,
        topk,
        HIDDEN,
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    handoff_bytes = None
    handoff_scale = None
    if return_handoff:
        handoff_bytes = torch.zeros(
            world_size,
            num_tokens,
            topk,
            INTERMEDIATE,
            dtype=torch.uint8,
            device=hidden.device,
        )
        handoff_scale = torch.zeros(
            world_size,
            num_tokens,
            topk,
            INTERMEDIATE // K64,
            dtype=torch.float32,
            device=hidden.device,
        )

    for global_expert in range(world_size * LOCAL_EXPERTS):
        routed = (topk_ids == global_expert).nonzero(as_tuple=False)
        if routed.numel() == 0:
            continue
        source_rank, source_token, source_slot = routed.unbind(dim=1)
        target_rank = global_expert // LOCAL_EXPERTS
        local_expert = global_expert % LOCAL_EXPERTS

        fc1_raw = _fast_fp8_mm(
            input_fp8[source_rank, source_token],
            fc1[target_rank, local_expert].transpose(0, 1),
        )
        fc1_output = (
            fc1_raw
            * input_scale[source_rank, source_token]
            * fc1_common[target_rank, local_expert]
        )
        paired = fc1_output.reshape(-1, INTERMEDIATE // 8, 2, 8)
        swiglu = _swiglu_sm90_formula(paired[:, :, 0], paired[:, :, 1]).reshape(
            -1, INTERMEDIATE
        )
        swiglu.mul_(topk_weights[source_rank, source_token, source_slot].unsqueeze(1))
        fc2_input, fc2_scale = quantize_handoff_reference(swiglu)
        if handoff_bytes is not None and handoff_scale is not None:
            handoff_bytes[source_rank, source_token, source_slot] = (
                fc2_input.contiguous().view(torch.uint8)
            )
            handoff_scale[source_rank, source_token, source_slot] = fc2_scale

        fc2_accum = torch.zeros(
            (routed.shape[0], HIDDEN), dtype=torch.float32, device=hidden.device
        )
        for group in range(INTERMEDIATE // K64):
            begin = group * K64
            end = begin + K64
            partial = _fast_fp8_mm(
                fc2_input[:, begin:end].contiguous(),
                fc2[target_rank, local_expert, :, begin:end].transpose(0, 1),
            )
            fc2_accum.add_(partial * fc2_scale[:, group : group + 1])
        output = fc2_accum * fc2_common[target_rank, local_expert]
        terms[source_rank, source_token, source_slot] = output.to(torch.bfloat16)

    reduced = terms.to(torch.float32).sum(dim=2)
    if not return_handoff:
        return reduced
    assert handoff_bytes is not None and handoff_scale is not None
    return reduced, handoff_bytes.view(torch.float8_e4m3fn), handoff_scale


def assert_output_matches(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual).all()
    actual_fp32 = actual.to(torch.float32)
    diff = actual_fp32 - expected
    rel_l2 = diff.norm() / expected.norm().clamp_min(1.0e-6)
    print(
        f"[sm90 mxfp4 split {label}] rel_l2={rel_l2.item():.5g} "
        f"max|d|={diff.abs().max().item() if diff.numel() else 0:.5g} "
        f"amax(ref)={expected.abs().max().item() if expected.numel() else 0:.5g}"
    )
    torch.testing.assert_close(actual_fp32, expected, atol=2.0e-2, rtol=2.0e-2)
    if expected.numel() and expected.norm().item() > 0:
        assert rel_l2.item() < 2.5e-2


def output_digest(tensor: torch.Tensor) -> bytes:
    import hashlib

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy()).digest()


__all__ = [
    "all_gather_stack",
    "assert_output_matches",
    "global_route_reference",
    "make_hidden",
    "output_digest",
]
