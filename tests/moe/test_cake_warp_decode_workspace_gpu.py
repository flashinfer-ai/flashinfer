"""Workspace-receipt lifecycle and graph-mutation tests on one parity geometry."""

from __future__ import annotations

import pytest
import torch

from flashinfer.fused_moe import SwiGLU
from tests.moe.cake_warp_decode_reference import (
    TARGETS,
    Geometry,
    check_graph_replay_after_mutation,
    check_reprepare_and_release_without_stream_sync,
    check_same_address_receipt_generations,
    prepare_fixture,
)

# One parameterised parity geometry: it reaches the activation-parameter FFI and
# the parameter-mutation replays on exact SM100 and SM103. The clamped geometry's
# rows live in test_cake_warp_decode_clamped_gpu.py. Token counts are its
# schedule / route-packer boundaries.
_GEOMETRY = Geometry(
    "e128_i3072_k4_swiglu_oa",
    6144,
    3072,
    128,
    4,
    (1, 2, 32),
    SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
)


@pytest.fixture(scope="module")
def workspace_fixture():
    if not torch.cuda.is_available():
        pytest.skip("Cake warp decode workspace tests require a CUDA device")
    capability = torch.cuda.get_device_capability(torch.device("cuda"))
    if capability not in TARGETS:
        pytest.skip(
            "Cake warp decode workspace tests require exact SM100 or SM103, "
            f"got SM{capability[0]}{capability[1]}"
        )
    fixture = prepare_fixture(_GEOMETRY, seed=20260831)
    yield fixture
    torch.cuda.synchronize()
    del fixture


@pytest.mark.gpu
def test_same_address_receipt_generations(workspace_fixture):
    check_same_address_receipt_generations(workspace_fixture)


@pytest.mark.gpu
def test_reprepare_and_release_without_caller_stream_sync(workspace_fixture):
    check_reprepare_and_release_without_stream_sync(workspace_fixture)


@pytest.mark.gpu
@pytest.mark.parametrize("num_tokens", _GEOMETRY.token_counts)
def test_graph_replay_follows_mutations_and_in_flight_release(
    workspace_fixture, num_tokens
):
    check_graph_replay_after_mutation(workspace_fixture, num_tokens)
