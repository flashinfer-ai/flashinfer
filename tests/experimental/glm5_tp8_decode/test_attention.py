# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Eight-process attention and indexer tests on one node with eight B300 GPUs."""

from dataclasses import replace

import pytest
import torch

from . import _api, data
from .moe_reference import bitwise
from ._attention_case import AttentionCase
from ._attention_checks import check_attention

pytestmark = pytest.mark.node_exclusive


@pytest.mark.parametrize(
    "past", [0, 31, 63, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 4096]
)
@pytest.mark.parametrize("seed", [1, 7])
def test_attention_causal_boundaries(collective, collective_watchdog, past, seed):
    case = AttentionCase(collective, capacity=max(4096, past + 8), seed=seed)
    try:
        case.run(4, "full", past, seed)
        case.run(1, "shared", past, seed)
        case.capture()
        for q_len, mode in ((4, "full"), (1, "shared"), (4, "shared")):
            expected = case.run(q_len, mode, past, seed + 101, graph=True)
            if mode == "full":
                expected_ids = case.full_selection(past)
            repeated = case.run(q_len, mode, past, seed + 101, graph=True)
            if mode == "full":
                repeated_ids = case.full_selection(past)
                assert torch.equal(
                    expected_ids.sort(dim=-1).values,
                    repeated_ids.sort(dim=-1).values,
                ), "same-input full replay changed the exact top-k set"
            else:
                assert bitwise(repeated, expected), (
                    "same-input fixed-selection replay changed attention output"
                )
        # The bound device cur_pos changes without recapture, including crossing
        # the dense/top-2048 boundary in both directions.
        changed_past = past + 1 if past % 2 == 0 else max(0, past - 1)
        case.run(4, "full", changed_past, seed + 211, graph=True)
        case.run(1, "shared", changed_past, seed + 307, graph=True)
    finally:
        case.close()


@pytest.mark.parametrize(
    "past,seed",
    [
        (16384, 1),
        (65536, 7),
        (131072, 1),
        (196608, 7),
        (4096, 9001),
        (65536, 9001),
        (196608, 9001),
        (4096, 9101),
        (65536, 9101),
        (196608, 9101),
    ],
)
def test_attention_long_and_concentrated_contexts(
    collective, collective_watchdog, past, seed
):
    case = AttentionCase(collective, capacity=past + 8, seed=seed)
    try:
        case.run(4, "full", past, seed)
        case.run(1, "shared", past, seed)
        case.capture()
        case.run(4, "full", past + 1, seed + 101, graph=True)
        case.run(1, "shared", past + 1, seed + 211, graph=True)
        case.run(4, "shared", past, seed + 307, graph=True)
    finally:
        case.close()


def test_attention_changed_selection_same_input_and_graph(
    collective, collective_watchdog
):
    case = AttentionCase(collective)
    try:
        case.run(4, "full", 4096, 1)
        case.run(1, "shared", 4096, 1)
        case.capture()
        for q_len in (1, 4):
            first = case.run(q_len, "shared", 4096, 101, graph=True, selection_seed=7)
            second = case.run(q_len, "shared", 4096, 101, graph=True, selection_seed=23)
            assert not bitwise(first, second), (
                "selection change did not affect the chosen fixture"
            )
    finally:
        case.close()


def test_attention_independent_workspaces_and_request_reset(
    collective, collective_watchdog
):
    first = AttentionCase(collective)
    second = None
    try:
        first.run(4, "full", 2048, 1)
        first.run(1, "shared", 4096, 7)
        first.capture()
        second = AttentionCase(collective, layer=first.layer, packed=first.packed)
        second.run(1, "shared", 31, 23)
        second.run(4, "shared", 2049, 31)
        first.run(4, "full", 2047, 41, graph=True)
        before = {
            q: int(first.plans[(q, "shared")]._diagnostics["generation"]())
            for q in (1, 4)
        }
        first.ws.reset_request()
        assert {
            q: int(first.plans[(q, "shared")]._diagnostics["generation"]())
            for q in (1, 4)
        } == before
        first.run(1, "shared", 0, 43, graph=True)
        first.run(4, "full", 31, 53, graph=True)
        dead = second.plans[(4, "shared")]
        second.close()
        second = None
        with pytest.raises(RuntimeError, match="destroy|closed|invalid"):
            dead()
        second = AttentionCase(collective, layer=first.layer, packed=first.packed)
        second.run(4, "full", 4096, 59)
        first.run(1, "shared", 2048, 61, graph=True)
    finally:
        if second is not None:
            second.close()
        first.close()


def test_attention_rank_local_prepare_failure_is_collective(
    collective, collective_watchdog
):
    case = AttentionCase(collective)
    try:
        key = (4, "full")
        before = {
            q: int(case.plans[(q, "shared")]._diagnostics["generation"]())
            for q in (1, 4)
        }
        bad_caches = dict(case.caches)
        if collective.rank == 3:
            bad_caches["kv_cache"] = bad_caches["kv_cache"][:, :-1]
        with pytest.raises(
            (ValueError, RuntimeError), match="rank 3|cache|shape|contiguous"
        ):
            _api.prepare_attention(
                case.x[key],
                case.packed,
                bad_caches,
                case.rope,
                case.position,
                case.out[key],
                case.ws,
                indexer="full",
            )
        # Packed weights contain pointer-only CUDA descriptors after prepare.
        # A malformed tensor on either the indexer or an MLA rank must be
        # rejected collectively before any peer can enter the kernel.
        for bad_rank, field in ((0, "q_a_indexer_wk"), (3, "q_gamma")):
            weights = case.packed
            if collective.rank == bad_rank:
                tensors = dict(weights.tensors)
                tensors[field] = tensors[field].reshape(-1)[:16]
                weights = replace(weights, tensors=tensors)
            with pytest.raises((ValueError, RuntimeError), match=field):
                _api.prepare_attention(
                    case.x[key],
                    weights,
                    case.caches,
                    case.rope,
                    case.position,
                    case.out[key],
                    case.ws,
                    indexer="full",
                )
        assert {
            q: int(case.plans[(q, "shared")]._diagnostics["generation"]())
            for q in (1, 4)
        } == before
        case.run(4, "full", 2048, 1)
        case.run(1, "shared", 2049, 7)
    finally:
        case.close()


@pytest.mark.parametrize("past", [1024, 2044, 2045, 4096])
@pytest.mark.parametrize("pattern", ["FFFF", "FSSS"])
def test_attention_only_chain_delayed_rank_zero(
    collective, collective_watchdog, past, pattern
):
    case = AttentionCase(collective)
    try:
        # Warm the delay kernel before any peer-waiting grid can be resident.
        torch.cuda._sleep(1)
        collective.sync()
        case.run(4, "full", past, 7)
        baseline_full_ids = case.full_selection(past)
        expected_shared = case.run(4, "shared", past, 7)
        fixed_shared_ids = data.make_selection(4, past, 7)
        expected_input = data.make_attention_input(4, 7, collective.device)
        case.capture()
        for delayed in (False, True):
            initial = {name: tensor.clone() for name, tensor in case.caches.items()}
            collective.sync()
            with torch.cuda.stream(case.stream):
                for mode in ("full", "shared"):
                    case.out_storage[(4, mode)].fill_(17)
                if collective.rank == 0:
                    case.selection[(4, "shared")].copy_(fixed_shared_ids)
                for index, kind in enumerate(pattern):
                    if delayed and collective.rank == 0 and index == 1:
                        torch.cuda._sleep(40_000_000)
                    mode = "full" if kind == "F" else "shared"
                    case.graphs[(4, mode)].replay()
                    case.counts[4] += 1
            collective.sync()

            def guards():
                for kind in set(pattern):
                    mode = "full" if kind == "F" else "shared"
                    assert bitwise(case.x[(4, mode)], expected_input)
                    assert bool((case.out_storage[(4, mode)][4:] == 17).all())
                assert case.position.item() == past
                assert bitwise(case.rope, case.rope_snapshot)
                if collective.rank == 0:
                    assert bitwise(
                        case.selection[(4, "shared")].cpu(), fixed_shared_ids
                    )
                return int(case.plans[(4, "full")]._diagnostics["generation"]())

            generation = collective.checked(guards)
            assert collective.gather(generation) == [case.counts[4]] * 8
            # Snapshot every observed output before reconstructing any call.
            # Shared mode does not overwrite the full indexer's IDs or logits.
            observed = {
                kind: case.out[(4, "full" if kind == "F" else "shared")].cpu().clone()
                for kind in set(pattern)
            }
            observed = collective.gather(observed)
            full_ids = case.full_selection(past)
            assert torch.equal(
                full_ids.sort(dim=-1).values,
                baseline_full_ids.sort(dim=-1).values,
            ), "attention chain changed the exact full-indexer top-k set"
            for kind in set(pattern):
                assert all(
                    bitwise(value[kind], observed[0][kind]) for value in observed
                ), f"attention chain {kind} outputs differ across ranks"

            # Current diagnostics/records describe the final call. Run every
            # stage, selection, cache and exact-wire check on that call before
            # another launch changes its generation or scratch.
            mode = "full" if pattern[-1] == "F" else "shared"
            key = (4, mode)
            check_attention(
                collective,
                case.plans[key],
                case.weights,
                case.caches,
                initial,
                case.x[key],
                case.out[key],
                case.rope,
                past,
                indexer=mode,
                selection=case.selection[key],
            )
            if "S" in pattern:
                assert all(
                    bitwise(value["S"], expected_shared) for value in observed
                ), "fixed-selection chain output differs from validated shared replay"

            # Full indexing may publish the same set in another slot order,
            # changing legal FP32 reduction rounding. Reconstruct with this
            # call's actual order through independently validated shared mode,
            # then require exact output equality for every observed rank.
            reconstructed = case.run(
                4,
                "shared",
                past,
                7,
                graph=True,
                explicit_selection=full_ids,
            )
            assert all(bitwise(value["F"], reconstructed) for value in observed), (
                "full chain output differs from validated same-order reconstruction"
            )
    finally:
        case.close()
