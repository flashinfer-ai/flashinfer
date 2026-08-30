# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from flashinfer.jit.cake_megamoe_topk_reduce import (
    run_cake_megamoe_topk_reduce,
)
from flashinfer.utils import is_sm100a_supported


_ATOL = 1e-2
_RTOL = 1e-2
_HIDDEN_SIZE = 4096
_TOP_K = 6
_TAIL_SENTINEL = -57.5

_is_exact_sm100a = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (10, 0)
    and is_sm100a_supported(torch.device("cuda"))
)

pytestmark = [
    pytest.mark.arch_blackwell,
    pytest.mark.skipif(
        not _is_exact_sm100a,
        reason="frozen reducer requires exact SM100a and CUDA 12.8+",
    ),
]


def _partials(capacity: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        capacity,
        _TOP_K,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()


def _ordered_reference(partials: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Accumulate K=0..5 in FP32, preserving the reducer's exact semantics."""

    if num_tokens == 0:
        return torch.empty(
            (0, _HIDDEN_SIZE), dtype=torch.bfloat16, device=partials.device
        )
    acc = partials[:num_tokens, 0].float()
    for topk_idx in range(1, _TOP_K):
        acc = acc + partials[:num_tokens, topk_idx].float()
    return acc.to(torch.bfloat16)


def _sentinel_output(capacity: int) -> torch.Tensor:
    return torch.full(
        (capacity, _HIDDEN_SIZE),
        _TAIL_SENTINEL,
        dtype=torch.bfloat16,
        device="cuda",
    )


def _assert_prefix_and_tail(
    out: torch.Tensor,
    expected_prefix: torch.Tensor,
    num_tokens: int,
) -> None:
    torch.testing.assert_close(
        out[:num_tokens], expected_prefix, atol=_ATOL, rtol=_RTOL
    )
    if num_tokens < out.shape[0]:
        expected_tail = torch.full_like(out[num_tokens:], _TAIL_SENTINEL)
        assert torch.equal(out[num_tokens:], expected_tail)


@pytest.mark.parametrize(
    ("num_tokens", "capacity"),
    [
        (0, 256),
        (1, 256),
        (8, 256),
        (255, 256),
        (256, 256),
        (1, 4096),
        (257, 4096),
        (4096, 4096),
    ],
)
def test_reducer_matches_ordered_bf16_reference_and_preserves_tail(
    num_tokens: int,
    capacity: int,
) -> None:
    partials = _partials(capacity, seed=capacity + num_tokens)
    out = _sentinel_output(capacity)
    before = out.clone() if num_tokens == 0 else None
    expected = _ordered_reference(partials, num_tokens)

    run_cake_megamoe_topk_reduce(partials, out, num_tokens)
    torch.cuda.synchronize()

    _assert_prefix_and_tail(out, expected, num_tokens)
    if before is not None:
        assert torch.equal(out, before), "T=0 must be a host-side no-op"


def test_reducer_launches_on_non_default_stream() -> None:
    capacity, num_tokens = 256, 37
    partials = _partials(capacity, seed=2026)
    out = _sentinel_output(capacity)
    expected = _ordered_reference(partials, num_tokens)

    # Resolve and compile the JIT module before exercising the steady-state
    # stream contract.
    run_cake_megamoe_topk_reduce(partials, out, 1)
    torch.cuda.synchronize()
    out.fill_(_TAIL_SENTINEL)

    launch_stream = torch.cuda.Stream()
    launch_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(launch_stream):
        run_cake_megamoe_topk_reduce(partials, out, num_tokens)
    torch.cuda.current_stream().wait_stream(launch_stream)

    _assert_prefix_and_tail(out, expected, num_tokens)


def test_reducer_cuda_graph_capture_and_replay_tracks_input_updates() -> None:
    capacity, num_tokens = 256, 53
    partials = _partials(capacity, seed=3105)
    out = _sentinel_output(capacity)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        # JIT compilation and module loading are deliberately outside capture.
        run_cake_megamoe_topk_reduce(partials, out, num_tokens)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    out.fill_(_TAIL_SENTINEL)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        run_cake_megamoe_topk_reduce(partials, out, num_tokens)

    torch.cuda.synchronize()
    out.fill_(_TAIL_SENTINEL)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_prefix_and_tail(
        out, _ordered_reference(partials, num_tokens), num_tokens
    )

    updated = _partials(capacity, seed=3106)
    partials.copy_(updated)
    out.fill_(_TAIL_SENTINEL)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_prefix_and_tail(
        out, _ordered_reference(updated, num_tokens), num_tokens
    )
