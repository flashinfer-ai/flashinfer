# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frost's three routing operands agree with an independent stable CPU sort."""

import pytest
import torch

from flashinfer.fused_moe.cute_dsl import moe_utils


def _require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (
        10,
        12,
    ):
        pytest.skip("SM100 or SM120 family required")


def _reference(x, ids, experts):
    flat = ids.cpu().flatten().long()
    order = torch.argsort(flat, stable=True)
    inverse = torch.empty_like(flat, dtype=torch.int32)
    inverse[order] = torch.arange(flat.numel(), dtype=torch.int32)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            torch.bincount(flat, minlength=experts).cumsum(0).int(),
        )
    )
    routed = x.cpu()[order // ids.shape[1]]
    return routed.cuda(), inverse.cuda(), offsets.cuda()


@pytest.mark.parametrize(
    "tokens,experts,top_k,hidden",
    [
        (1, 128, 8, 2048),
        (2, 128, 8, 2048),
        (4, 128, 8, 2048),
        (8, 128, 8, 2048),
        (16, 128, 8, 2048),
        (32, 128, 8, 2048),
        (64, 128, 8, 2048),
        (1, 8, 1, 256),
        (7, 16, 3, 520),
        (2, 128, 64, 2048),
        (8, 128, 64, 2048),
        (64, 64, 1, 4096),
        (3, 128, 8, 8),
        (1, 128, 8, 16384),
        (3, 4096, 1, 8),
    ],
)
def test_small_route_live_capture(tokens, experts, top_k, hidden):
    _require_blackwell()
    x = (
        (torch.arange(tokens * hidden, device="cuda") % 31 - 15)
        .to(torch.bfloat16)
        .reshape(tokens, hidden)
    )
    generator = torch.Generator().manual_seed(123)
    ids_cpu = torch.stack(
        [torch.randperm(experts, generator=generator)[:top_k] for _ in range(tokens)]
    ).int()
    ids = ids_cpu.cuda()
    routed = torch.empty(tokens * top_k, hidden, dtype=torch.bfloat16, device="cuda")
    inverse = torch.empty(tokens * top_k, dtype=torch.int32, device="cuda")
    offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")

    def run():
        assert moe_utils._try_moe_route_permute_small(
            x, ids, routed, inverse, offsets, experts
        )

    def check():
        for actual, expected in zip(
            (routed, inverse, offsets), _reference(x, ids, experts), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            run()
        # Each replay changes live inputs. Concentrated routes exercise leading,
        # interior and trailing empty experts, including the final E+1 offset.
        patterns = (
            (ids_cpu + 2) % experts,
            torch.arange(top_k).expand(tokens, -1),
            torch.arange(experts - top_k, experts).expand(tokens, -1),
        )
        for pattern in patterns:
            x.neg_()
            ids.copy_(pattern)
            expected = _reference(x, ids, experts)
            assert not torch.equal(routed, expected[0]), "Stale replay must fail"
            routed.fill_(float("nan"))
            inverse.fill_(-7)
            offsets.fill_(-7)
            graph.replay()
            for actual, target in zip(
                (routed, inverse, offsets), expected, strict=True
            ):
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(side)


@pytest.mark.parametrize(
    "bad",
    [
        "cpu",
        "dtype",
        "indices",
        "noncontiguous",
        "alignment",
        "offsets",
        "inverse",
        "experts",
        "rows",
        "alias",
        "partial_alias",
    ],
)
def test_small_route_declines_before_jit(bad, monkeypatch):
    _require_blackwell()
    x = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(8, 8, device="cuda", dtype=torch.int32)
    routed = torch.empty(64, 64, device="cuda", dtype=torch.bfloat16)
    inverse = torch.empty(64, device="cuda", dtype=torch.int32)
    offsets = torch.empty(129, device="cuda", dtype=torch.int32)
    experts = 128
    if bad == "cpu":
        x = x.cpu()
    elif bad == "dtype":
        x = x.float()
    elif bad == "indices":
        ids = ids.long()
    elif bad == "noncontiguous":
        x = torch.zeros(8, 128, device="cuda", dtype=x.dtype)[:, ::2]
    elif bad == "alignment":
        x = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view_as(x)
    elif bad == "offsets":
        offsets = offsets[:-1]
    elif bad == "inverse":
        inverse = inverse[:-1]
    elif bad == "experts":
        experts = 4097
    elif bad == "rows":
        x = torch.zeros(65, 64, device="cuda", dtype=x.dtype)
        ids = torch.zeros(65, 8, device="cuda", dtype=ids.dtype)
        routed = torch.empty(520, 64, device="cuda", dtype=routed.dtype)
        inverse = torch.empty(520, device="cuda", dtype=inverse.dtype)
    elif bad == "alias":
        ids = ids[:, :1].contiguous()
        inverse = inverse[:8]
        routed = x
    elif bad == "partial_alias":
        storage = torch.empty(routed.numel() + 8, device="cuda", dtype=x.dtype)
        x = storage[: x.numel()].view_as(x)
        routed = storage[8:].view_as(routed)

    def no_launch():
        raise AssertionError("Declined metadata must not build or launch a kernel")

    monkeypatch.setattr(moe_utils, "_get_moe_utils_module", no_launch)
    assert not moe_utils._try_moe_route_permute_small(
        x, ids, routed, inverse, offsets, experts
    )
