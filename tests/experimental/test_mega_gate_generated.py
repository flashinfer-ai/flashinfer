# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""Routing values, physical mapping, held-out token counts, current-stream and changed-input replay."""

import pytest
import torch
import torch.nn.functional as F
from flashinfer.experimental.deepgemm_mega_gate import mega_gate as _runtime
from flashinfer.mega_gate import prepare_mega_gate

K, E, TOPK = (
    _runtime.EXPORTED["K"],
    _runtime.EXPORTED["E"],
    _runtime.EXPORTED["num_topk"],
)
EXPORTED_ROWS = [
    (1, False, True),
    (3, False, True),
    (16, False, True),
    (128, False, True),
    (512, False, True),
    (1024, False, True),
    (2048, False, True),
    (4096, False, True),
    (8192, False, True),
    (16, True, True),
    (16, False, False),
    (1, False, False),
]
HELD_OUT_TOKEN_COUNTS = (2, 7, 33, 100, 400, 777, 3000, 5000)


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_facts(torch.cuda.current_device())[0]
    except RuntimeError as error:
        pytest.skip(str(error))


def _mapping(physical):
    counts = torch.full((E,), 2, dtype=torch.int32, device="cuda") if physical else None
    mapping = (
        torch.stack(
            (
                torch.arange(E, device="cuda", dtype=torch.int32),
                torch.arange(E, device="cuda", dtype=torch.int32) + E,
            ),
            1,
        )
        if physical
        else None
    )
    return mapping, counts


def test_every_token_count_selects_an_exported_template():
    for sms in (148, 152):
        for deterministic in (False, True):
            for M in (*range(1, 4097), 8192, 16384, 65536, 1 << 20):
                index, launch_ctas, workers = _runtime.select_template(
                    M, sms, deterministic=deterministic
                )
                template = _runtime.TEMPLATES[index]
                assert template["program"] in _runtime.PROGRAMS
                if template["static_tokens"]:
                    assert _runtime._tile_geometry(
                        template, template["static_tokens"]
                    ) == _runtime._tile_geometry(template, M)
                assert (
                    launch_ctas
                    % (
                        template["num_mma_ctas"]
                        * template["num_expert_groups"]
                        * template["num_split_k"]
                    )
                    == 0
                )
                assert workers >= 1 and launch_ctas >= 1
                assert (not deterministic) or template["num_split_k"] == 1
    for M, deterministic, physical in EXPORTED_ROWS:
        unmapped = deterministic or not physical
        index_100, _, _ = _runtime.select_template(
            M,
            148,
            has_physical_map=physical,
            unmapped_output=unmapped,
            deterministic=deterministic,
        )
        index_103, _, _ = _runtime.select_template(
            M,
            152,
            has_physical_map=physical,
            unmapped_output=unmapped,
            deterministic=deterministic,
        )
        assert index_100 == index_103
        exact = _runtime.TEMPLATES[index_100]["static_tokens"]
        # The small physical-map production rows and the one-tile logical-output rows run their exact-shape
        # program (one 16-token tile serves M 2..16, one program per route); every other row runs the runtime program.
        if physical and not deterministic:
            static = {1: 1, 3: 16, 16: 16, 128: 128, 512: 512}
        elif not physical and not deterministic:
            static = {m: 16 for m in range(2, 17)}
        else:
            static = {}
        assert exact == static.get(M, 0)


@pytest.mark.parametrize("M,deterministic,physical", EXPORTED_ROWS)
def test_routing_values_stream_and_replay(M, deterministic, physical):
    _skip_unless_exported()
    x = torch.ones((M, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((E, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.arange(E, dtype=torch.float32, device="cuda")
    mapping, counts = _mapping(physical)
    unmapped = (
        torch.empty((M, TOPK), dtype=torch.int64, device="cuda")
        if deterministic or not physical
        else None
    )
    ep_rank = 7 if deterministic else 0
    plan = prepare_mega_gate(
        x,
        weight,
        TOPK,
        bias=bias,
        to_physical_map=mapping,
        logical_count=counts,
        unmapped_topk_idx=unmapped,
        ep_rank=ep_rank,
        deterministic=deterministic,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for changed in (False, True, False):
        with torch.cuda.stream(stream):
            bias.copy_(
                torch.arange(E, device="cuda", dtype=torch.float32).flip(0)
                if changed
                else torch.arange(E, device="cuda", dtype=torch.float32)
            )
            plan.outputs[0].fill_(-1)
            plan.outputs[1].fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        logical = (
            torch.arange(TOPK, device="cuda", dtype=torch.int64)
            if changed
            else torch.arange(E - 1, E - TOPK - 1, -1, device="cuda")
        )
        expected = logical[None].expand(M, -1)
        if physical:
            duplicate = ((ep_rank + torch.arange(M, device="cuda") * 23333) % 2)[
                :, None
            ]
            expected = expected + duplicate * E
        assert torch.equal(plan.outputs[0], expected)
        torch.testing.assert_close(
            plan.outputs[1],
            torch.full_like(plan.outputs[1], 0.25),
            atol=1e-5,
            rtol=1e-5,
        )
        if unmapped is not None:
            assert torch.equal(unmapped, logical[None].expand(M, -1))


def _reference(x, weight, bias):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        logits = x.float() @ weight.float().T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    scores = F.softplus(logits).sqrt()
    return scores, scores + bias


def _check_routing(
    indices, weights, unmapped, *, scores, ranking, mapping, counts, ep_rank
):
    M = scores.shape[0]
    if unmapped is not None:
        logical = unmapped
    elif mapping is not None:
        logical = torch.where(indices < E, indices, indices - E)
    else:
        logical = indices
    assert bool(((logical >= 0) & (logical < E)).all())
    selected = scores.gather(1, logical)
    expected_weights = selected / (selected.sum(1, keepdim=True) + 1e-20) * 1.5
    torch.testing.assert_close(weights, expected_weights, atol=1e-4, rtol=1e-4)
    chosen = ranking.gather(1, logical)
    omitted = ranking.scatter(1, logical, float("-inf")).amax(1)
    crossing = omitted - chosen.amin(1)
    assert bool((crossing <= 1e-5 + 1e-5 * ranking.abs().amax(1)).all())
    if mapping is not None:
        tokens = torch.arange(M, device=indices.device, dtype=torch.int64)
        duplicate = (ep_rank + tokens[:, None] * 23333) % counts[logical].to(
            torch.int64
        )
        assert torch.equal(indices, mapping[logical, duplicate].to(torch.int64))
    else:
        assert torch.equal(indices, logical)


@pytest.mark.parametrize("M", HELD_OUT_TOKEN_COUNTS)
@pytest.mark.parametrize(
    "deterministic,physical",
    [(False, True), (True, True), (False, False)],
    ids=["physical", "deterministic_unmapped", "logical_unmapped"],
)
def test_held_out_token_counts_match_reference(M, deterministic, physical):
    _skip_unless_exported()
    generator = torch.Generator(device="cuda").manual_seed(1000 + M)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = torch.randn(
        (E, K), dtype=torch.bfloat16, device="cuda", generator=generator
    ).mul_(K**-0.5)
    bias = torch.randn(
        (E,), dtype=torch.float32, device="cuda", generator=generator
    ).mul_(0.1)
    mapping, counts = _mapping(physical)
    unmapped = (
        torch.full((M, TOPK), -2, dtype=torch.int64, device="cuda")
        if deterministic or not physical
        else None
    )
    ep_rank = 3
    plan = prepare_mega_gate(
        x,
        weight,
        TOPK,
        bias=bias,
        to_physical_map=mapping,
        logical_count=counts,
        unmapped_topk_idx=unmapped,
        ep_rank=ep_rank,
        deterministic=deterministic,
    )
    assert M not in {row[0] for row in EXPORTED_ROWS}
    indices, weights = plan.run()
    torch.cuda.synchronize()
    scores, ranking = _reference(x, weight, bias)
    _check_routing(
        indices,
        weights,
        unmapped,
        scores=scores,
        ranking=ranking,
        mapping=mapping,
        counts=counts,
        ep_rank=ep_rank,
    )
    if deterministic:
        snapshot = (indices.clone(), weights.clone(), unmapped.clone())
        for _ in range(3):
            plan.run()
        torch.cuda.synchronize()
        assert all(
            torch.equal(a, b)
            for a, b in zip((indices, weights, unmapped), snapshot, strict=True)
        )
