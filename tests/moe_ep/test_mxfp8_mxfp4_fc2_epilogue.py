# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused host tests for the FC2 route-store and combine contract."""

import inspect

import pytest

pytest.importorskip("cutlass")

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.fc2_epilogue import (  # noqa: E402
    DeterministicTopKReducer,
    FC2_EPILOGUE_TILE,
    Fc2RouteStore,
    REDUCE_HIDDEN_PER_THREAD,
    validate_fc2_contract,
)


def test_fc2_shape_contract() -> None:
    validate_fc2_contract(hidden=7168, num_topk=8)
    assert FC2_EPILOGUE_TILE == 32
    assert REDUCE_HIDDEN_PER_THREAD == 8

    with pytest.raises(ValueError, match="positive multiple of 32"):
        validate_fc2_contract(hidden=33, num_topk=8)
    with pytest.raises(ValueError, match="num_topk must be positive"):
        validate_fc2_contract(hidden=32, num_topk=0)


def test_route_store_exposes_the_same_local_and_peer_slot_layout() -> None:
    tensor = object()
    direct = Fc2RouteStore(tensor=tensor)
    peer = Fc2RouteStore(tensor=tensor, peer_rank_ptr_mapper=object())
    assert not direct.is_peer_store
    assert peer.is_peer_store


def test_fc2_store_interface_cannot_accept_a_routing_score() -> None:
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.fc2_store import (
        stage_native_bf16,
        store_staged_vector,
    )

    for function in (stage_native_bf16, store_staged_vector):
        parameters = inspect.signature(function).parameters
        assert "route_score" not in parameters
        assert "topk_score" not in parameters
    route = inspect.signature(Fc2RouteStore.resolve_route_row).parameters
    assert "src_topk_idx" in route
    assert "route_score" not in route


def test_external_reducer_has_fixed_slot_geometry() -> None:
    reducer = DeterministicTopKReducer(hidden=256, num_topk=8)
    assert reducer.hidden_tiles == 256 // REDUCE_HIDDEN_PER_THREAD
    assert reducer.num_topk == 8
