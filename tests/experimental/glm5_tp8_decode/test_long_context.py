# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Focused public-API regression for 512K/1M on eight B300 GPUs.

test_selector.py feeds adversarial scores to the top-k selector directly.
These cases exercise planning, fixed-address replay, scratch reuse, high-ID
transport and all per-call numerical and wire checks at the extended sizes.
"""

import pytest
import torch

from . import _api
from ._attention_case import AttentionCase


pytestmark = pytest.mark.node_exclusive

_CAPACITY = 1_052_672
_HALF_M = 524_288
_ONE_M = 1_048_576
_LAST_Q4_PAST = _CAPACITY - 4


def _tail_selection(q_len, past):
    """Legal accepted rows whose IDs all need more than 18 bits."""
    selected = torch.stack(
        [
            torch.arange(past + query + 1 - 2048, past + query + 1, dtype=torch.int32)
            for query in range(q_len)
        ]
    )
    assert int(selected.min()) >= 1 << 18
    return selected


@pytest.mark.parametrize("weight_seed", [1, 9001, 9101])
def test_long_context_public_attention_replay_and_transport(
    collective, collective_watchdog, weight_seed
):
    # Use the complete per-stage reference with the same tolerances as the
    # short-context tests, not an output-only comparison. The fixture also
    # checks causal top-k, cache guards, current generations and exact TP exits.
    case = AttentionCase(collective, capacity=_CAPACITY, seed=weight_seed)
    try:
        if weight_seed == 1:
            pasts = (
                202756,
                202757,
                202756,  # default -> long-context -> default dispatch
                _HALF_M - 4,
                _HALF_M - 3,
                _HALF_M,
                _ONE_M - 4,
                _ONE_M - 3,
                _ONE_M,
                _LAST_Q4_PAST,
                31,  # small-length dispatch after the largest workspace use
            )
        else:
            # Both concentrated indexer-cache families reach each long
            # allocation boundary.
            pasts = (202757, _HALF_M, _ONE_M, _LAST_Q4_PAST)
        for index, past in enumerate(pasts):
            input_seed = weight_seed + 101 * index
            case.run(4, "full", past, input_seed, graph=bool(case.graphs))
            if not case.graphs:
                case.capture()
            if past == _ONE_M:
                before = case.full_selection(past)
                case.run(4, "full", past, input_seed, graph=True)
                after = case.full_selection(past)
                assert torch.equal(
                    before.sort(dim=-1).values, after.sort(dim=-1).values
                ), "same-input long-context replay changed the exact top-k set"

        # Padding is storage capacity, not permission to execute query rows
        # past the public logical limit. All ranks must reject before launch.
        before = {
            q: int(case.plans[(q, "shared")]._diagnostics["generation"]())
            for q in (1, 4)
        }
        for q_len, past in ((4, _CAPACITY - 3), (1, _CAPACITY)):
            mode = "full" if q_len == 4 else "shared"
            key = (q_len, mode)
            case.position.fill_(past)
            collective.sync()
            with pytest.raises(ValueError, match="cur_pos|query rows|max_seq_len"):
                _api.prepare_attention(
                    case.x[key],
                    case.packed,
                    case.caches,
                    case.rope,
                    case.position,
                    case.out[key],
                    case.ws,
                    indexer=mode,
                    selection=case.selection[key],
                )
        assert {
            q: int(case.plans[(q, "shared")]._diagnostics["generation"]())
            for q in (1, 4)
        } == before

        # The same captured q1/q4 plans must consume full-width IDs through LL
        # publication and address the last cache row correctly. Every call runs
        # the stage and exact-wire reference checks.
        for index, (q_len, past) in enumerate(
            (
                (1, _ONE_M),
                (4, _ONE_M),
                (1, _CAPACITY - 1),
                (4, _LAST_Q4_PAST),
            )
        ):
            case.run(
                q_len,
                "shared",
                past,
                weight_seed + 501 + index * 17,
                graph=True,
                explicit_selection=_tail_selection(q_len, past),
            )
    finally:
        case.close()
