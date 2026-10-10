"""GLM C4 pooling across chunk boundaries and prepared compiled replay."""

import pytest
import torch

from b12x.attention.dsa_indexer import glm_pool
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


def _quantize(source):
    value = source.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        view = value.reshape(*value.shape[:-1], -1, 2, stride)
        a, b = view.unbind(-2)
        value = torch.stack((a + b, a - b), dim=-2).reshape_as(value)
    value = (value * 128**-0.5).to(torch.bfloat16).float()
    scale = torch.exp2(
        torch.ceil(torch.log2(value.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    quantized = (value / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale


@pytest.mark.parametrize("rows", [32, 2080])
@torch.no_grad()
def test_rotated_query_prepared_graph(rows):
    device = require_b12x()
    source = torch.randn((rows, 128), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows, device=device)
    initial_weights = torch.randn(rows, device=device)
    weights = initial_weights.clone()
    plan = glm_pool.plan_queries(source, output, scales, weights)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="query",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.quantize(source, output, scales, weights)
                    ),
                ),
            )
        )

        def forward():
            weights.copy_(initial_weights)
            glm_pool.quantize_queries(source, output, scales, weights, plan=plan)

        call = torch.compile(forward, fullgraph=True, dynamic=False)
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        session.freeze()
        for _ in range(3):
            source.normal_()
            initial_weights.normal_()
            expected, expected_scales = _quantize(source)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
            torch.testing.assert_close(
                weights, initial_weights * expected_scales / 64, rtol=0, atol=0
            )
        graph.reset()


@torch.no_grad()
def test_pool_state_survives_repeated_chunk_boundaries():
    device = require_b12x()
    chunk, chunks = 65, 9
    rows = chunk * chunks
    key = torch.randn((rows, 128), dtype=torch.bfloat16, device=device)
    gate = torch.randn_like(key)
    ape = torch.randn((4, 128), dtype=torch.bfloat16, device=device)
    positions = torch.arange(rows, dtype=torch.int64, device=device) + 3
    pages = torch.tensor([2, 0, 3, 1], dtype=torch.int64, device=device)
    slots = pages[positions // 256] * 256 + positions % 256
    state_slots = torch.tensor([1], dtype=torch.int64, device=device)
    full_starts = torch.tensor([0, rows], dtype=torch.int32, device=device)
    starts = torch.tensor([0, chunk], dtype=torch.int32, device=device)
    cache = torch.zeros((4, 64, 132), dtype=torch.uint8, device=device)
    initial_tail = torch.randn((2, 2, 4, 128), dtype=torch.bfloat16, device=device)
    tail = initial_tail.clone()
    expected_cache, expected_tail = cache.clone(), tail.clone()
    keys, gates = torch.empty_like(key[:chunk]), torch.empty_like(gate[:chunk])
    token_slots, token_positions = (
        torch.empty_like(slots[:chunk]),
        torch.empty_like(positions[:chunk]),
    )
    keys.copy_(key[:chunk])
    gates.copy_(gate[:chunk])
    token_slots.copy_(slots[:chunk])
    token_positions.copy_(positions[:chunk])
    full_arguments = (
        expected_cache,
        expected_tail,
        state_slots,
        full_starts,
        key,
        gate,
        ape,
        slots,
        positions,
    )
    arguments = (
        cache,
        tail,
        state_slots,
        starts,
        keys,
        gates,
        ape,
        token_slots,
        token_positions,
    )
    full_plan = glm_pool.plan_pools(*full_arguments, mode="decode")
    chunk_plan = glm_pool.plan_pools(*arguments, mode="prefill")
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                plan.request(
                    name=name,
                    prepare_call=lambda state, args=args: PreparedCall(
                        run=lambda: state.update(*args)
                    ),
                )
                for name, plan, args in (
                    ("full", full_plan, full_arguments),
                    ("chunk", chunk_plan, arguments),
                )
            )
        )
        compiled = torch.compile(
            lambda: glm_pool.update_pools(*arguments, plan=chunk_plan),
            fullgraph=True,
            dynamic=False,
        )
        compiled()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        expected_cache.zero_()
        expected_tail.copy_(initial_tail)
        glm_pool.update_pools(*full_arguments, plan=full_plan)
        cache.zero_()
        tail.copy_(initial_tail)
        for start in range(0, rows, chunk):
            keys.copy_(key[start : start + chunk])
            gates.copy_(gate[start : start + chunk])
            token_slots.copy_(slots[start : start + chunk])
            token_positions.copy_(positions[start : start + chunk])
            graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(tail, expected_tail, rtol=0, atol=0)
        graph.reset()


def _reference(cache, tail, starts, key, gate, ape, slots, positions):
    expected_cache, expected_tail = cache.clone(), tail.clone()
    keys_out = expected_cache.view(cache.shape[0], -1)[:, : 64 * 128].view(-1, 64, 128)
    scales_out = expected_cache.view(torch.float32).view(cache.shape[0], -1)[
        :, 64 * 32 :
    ]
    for request, state_slot in enumerate((2, 1)):
        start, end = starts[request : request + 2].cpu().tolist()
        first_position = int(positions[start].item())
        prefix = first_position % 4
        previous = [
            (first_position - prefix + i) % tail.shape[2] for i in range(prefix)
        ]
        keys = torch.cat((tail[state_slot, 0, previous], key[start:end]))
        gates = torch.cat((tail[state_slot, 1, previous], gate[start:end]))
        count = keys.shape[0] // 4
        values = keys[: count * 4].reshape(count, 4, 128).float()
        scores = gates[: count * 4].reshape(count, 4, 128).float() + ape.float()
        probabilities = torch.softmax(scores, dim=1)
        pooled = (values * probabilities).sum(1).to(torch.bfloat16)
        quantized, scales = _quantize(pooled)
        completion = torch.arange(count, device=key.device) * 4 + (3 - prefix) + start
        locations = slots[completion]
        pages, offsets = locations // 256, locations % 256 // 4
        keys_out[pages, offsets] = quantized.view(torch.uint8)
        scales_out[pages, offsets] = scales
        for row in range(max(start, end - tail.shape[2]), end):
            slot = int(positions[row].item()) % tail.shape[2]
            expected_tail[state_slot, 0, slot] = key[row]
            expected_tail[state_slot, 1, slot] = gate[row]
    return expected_cache, expected_tail


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("rows,ring", [(8, 4), (65, 8), (4096, 4)])
@torch.no_grad()
def test_chunked_pool_production_compiled_graph(mode, rows, ring):
    device = require_b12x()
    cache = torch.zeros((64, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.randn((4, 2, ring, 128), dtype=torch.bfloat16, device=device)
    initial_tail = tail.clone()
    state_slots = torch.tensor([2, 1, -1], dtype=torch.int64, device=device)
    split = rows - rows // 3
    starts = torch.tensor([0, split, rows, rows], dtype=torch.int32, device=device)
    key = torch.randn((rows, 128), dtype=torch.bfloat16, device=device)
    gate = torch.randn_like(key)
    ape = torch.randn((4, 128), dtype=torch.bfloat16, device=device)
    positions = torch.cat(
        (
            torch.arange(split, device=device) + 3,
            torch.arange(rows - split, device=device) + 255,
        )
    )
    table = torch.randperm(64, device=device).reshape(2, 32)
    requests = torch.cat(
        (
            torch.zeros(split, device=device, dtype=torch.int64),
            torch.ones(rows - split, device=device, dtype=torch.int64),
        )
    )
    slots = table[requests, positions // 256] * 256 + positions % 256
    arguments = (cache, tail, state_slots, starts, key, gate, ape, slots, positions)
    plan = glm_pool.plan_pools(*arguments, mode=mode)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="pool",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.update(*arguments)
                    ),
                ),
            )
        )
        call = torch.compile(
            lambda: glm_pool.update_pools(*arguments, plan=plan),
            fullgraph=True,
            dynamic=False,
        )
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        session.freeze()
        for _ in range(2):
            key.normal_()
            gate.normal_()
            cache.zero_()
            tail.copy_(initial_tail)
            expected_cache, expected_tail = _reference(
                *arguments[:2], starts, key, gate, ape, slots, positions
            )
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for field in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[field] == after[field]
            torch.testing.assert_close(tail, expected_tail, rtol=0, atol=0)
            torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        graph.reset()
