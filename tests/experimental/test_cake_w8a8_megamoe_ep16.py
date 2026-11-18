"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Tests for the experimental Cake W8A8 MXFP8 MegaMoE EP16 backend (SM103).
#
# Host tests run without a GPU (``-k "not test_ep16"``). The 16-rank tests
# (``-k test_ep16``) need torchrun with exactly 16 ranks on compute capability
# 10.3 devices in one NVLink domain:
#
#   torchrun --nnodes 4 --nproc_per_node 4 --rdzv_backend c10d \
#       --rdzv_endpoint $HOST:29500 -m pytest tests/experimental/test_cake_w8a8_megamoe_ep16.py -k test_ep16

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import traceback
import warnings
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake_w8a8_megamoe_ep16 import cake_jit, cake_plan
from flashinfer.experimental.cake_w8a8_megamoe_ep16.cake_backend import (
    KERNEL_NAME,
    STAGE_ARGS,
    STAGE_ENTRIES,
    SYMMETRIC_BUFFERS,
    CakeWorkspace,
    Sm103Mxfp8Mxfp8CakeMegaKernelBackend,
    build_stage_calls,
    routing_overflow,
)
from flashinfer.experimental.cake_w8a8_megamoe_ep16.cake_config import (
    Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig,
)
from flashinfer.experimental.cake_w8a8_megamoe_ep16.cake_staging import (
    MASKED_ROUTE,
    resolve_bound,
    stage_inputs,
)
from flashinfer.experimental.cake_w8a8_megamoe_ep16.cake_symmetric import (
    SymmetricBuffer,
    as_int64_image,
    check_self_entry,
    peer_table,
    peer_table_images,
    peer_table_shape,
)
from flashinfer.experimental.cake_w8a8_megamoe_ep16.cake_weights import (
    QUANTIZE_ENTRY,
    SCALE_MODE_W2_SWAP,
    SCALE_MODE_W13,
    W2_FP8_LAYOUT,
    W2_KTILE_ROWS,
    W2_SHAPE,
    W13_FP8_LAYOUT,
    W13_KTILE_ROWS,
    W13_SHAPE,
    TransformedCakeW8a8MegaWeights,
    launch_quantize_weights,
    pack_scales_cp,
    reference_interleave_gate_up_128,
    reference_ktile_major_payload,
    reference_quantize_mxfp8_block32,
    reference_row_major_payload,
    unpack_scales_cp,
    validate_canonical_weights,
    validate_transformed_mega_weights,
)
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpArchError,
    MoEEpConfigError,
    MoEEpMegaLayer,
    MoEEpTensors,
    MoEWeightPack,
    PrequantizedMoEWeights,
    UnquantizedMoEWeights,
)
from flashinfer.moe_ep import (
    Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig as ExportedConfig,
)
from flashinfer.moe_ep import (
    TransformedCakeW8a8MegaWeights as ExportedWeights,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "flashinfer" / "experimental" / "cake_w8a8_megamoe_ep16"
CSRC_DIR = PACKAGE_DIR / "csrc" / "cake_w8a8_megamoe_ep16"
MANIFEST_PATH = CSRC_DIR / cake_jit.MANIFEST_NAME
HAND_WRITTEN_FILES = (
    *sorted(PACKAGE_DIR.glob("*.py")),
    *sorted(CSRC_DIR.glob("*_binding.cu")),
    PACKAGE_DIR / "README.md",
    REPO_ROOT
    / "flashinfer"
    / "moe_ep"
    / "backends"
    / "mega"
    / "kernel"
    / "sm103"
    / "__init__.py",
    Path(__file__).resolve(),
    REPO_ROOT / "examples" / "experimental" / "cake_w8a8_megamoe_ep16.py",
    REPO_ROOT / "benchmarks" / "moe_ep" / "bench_cake_w8a8_megamoe_ep16.py",
)
# The complete frozen build/route table the shipped programs were generated
# from (manifest ``frozen_build_knobs``): the pin of route
# ``w8a8_megamoe_ep16_staged_v4``. A regenerated drop must reproduce it exactly.
FROZEN_BUILD_KNOBS = {
    "INPUT_TOKEN_READINESS": "0",
    "PDL_CHAIN": "1",
    "PDL_PREFETCH_KSTEPS": "0",
    "SWAP_ROUTE_N": "32",
    "FC1_SWAP_STAGES": "4",
    "FC2_SWAP_STAGES": "8",
    "FI_SWIGLU_EX2_RCP": "0",
    "STRICT_FP32_SWIGLU_STAGING": "0",
    "WEIGHT_KTILE_MAJOR": "1",
    "SOURCE_PREQUANT": "1",
    "FC2_TOKEN_BACK_PUSH": "1",
    "PREPROCESS_VEC128": "0",
    "SWAP_FC12_ONE_LAUNCH": "0",
    "SINGLE_LAUNCH_MEGA": "0",
    "BALANCED_DIRECT_DISPATCH": "0",
    "BALANCED_SHARED_FC1_INPUT": "0",
    "BALANCED_OWNER_REDUCE": "0",
    "HOTSET_OWNER_REDUCE": "0",
    "FC12_INDEPENDENT_COMPLETION": "0",
    "FC12_G1_M128": "0",
    "FINE_GRAINED_TOPK_REDUCE": "0",
    "PUBLISH_FOLDED_RESET": "1",
    "COMBINE_SLICES": "3",
    "PUSH_DISPATCH": "1",
    "PUSH_WAIT_TIMEOUT_CYCLES": "0",
    "FC2_COMPLETION_COUNTERS": "1",
}


def _fleet(**overrides) -> FleetParams:
    values = dict(num_experts=512, max_tokens_per_rank=64, token_hidden_size=3072)
    values.update(overrides)
    return FleetParams(**values)


def _bootstrap(world_size: int = 16, rank: int = 0, **overrides) -> BootstrapConfig:
    return BootstrapConfig(world_size=world_size, rank=rank, **overrides)


# ---------------------------------------------------------------------------
# Config and registry.
# ---------------------------------------------------------------------------
def test_config_defaults() -> None:
    config = Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig()
    assert config.kernel_name == KERNEL_NAME == "sm103_mxfp8_mxfp8_bf16_cake"
    assert config.intermediate_size == 5120
    assert config.top_k == 8
    assert config.fast_math is True
    assert config.knobs is None
    assert config.max_routes_per_expert == 32
    assert ExportedConfig is Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig
    assert ExportedWeights is TransformedCakeW8a8MegaWeights


@pytest.mark.parametrize(
    "field,value",
    (
        ("intermediate_size", 2048),
        ("intermediate_size", True),
        ("top_k", 6),
        ("kernel_name", "sm100_mxfp8_mxfp8_bf16_cake"),
        ("knobs", {}),
        ("knobs", "auto"),
        ("max_routes_per_expert", 64),
    ),
)
def test_config_rejects_other_values(field, value) -> None:
    with pytest.raises(ValueError):
        Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig(**{field: value})


def test_config_rejects_fast_math_false() -> None:
    with pytest.raises(NotImplementedError):
        Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig(fast_math=False)


def test_registry_resolves_kernel_name_and_warns_once() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = create_mega_kernel(Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig())
        create_mega_kernel(Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig())
    assert isinstance(backend, Sm103Mxfp8Mxfp8CakeMegaKernelBackend)
    assert backend.kernel_name() == KERNEL_NAME
    assert backend.supports_output_view
    assert backend.runtime_requirements(_bootstrap()) == frozenset(
        {"torch_dist", "nvshmem"}
    )
    experimental = [w for w in caught if issubclass(w.category, ExperimentalWarning)]
    # Warned at most once per process for this (API, backend) pair.
    assert len(experimental) <= 1
    with pytest.raises(TypeError):
        Sm103Mxfp8Mxfp8CakeMegaKernelBackend(SimpleNamespace(kernel_name=KERNEL_NAME))


# ---------------------------------------------------------------------------
# validate_init guards (capability and world monkeypatched; no GPU needed).
# ---------------------------------------------------------------------------
@pytest.fixture
def sm103_host(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.3a")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    backend = Sm103Mxfp8Mxfp8CakeMegaKernelBackend(
        Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig()
    )
    return backend


def test_validate_init_accepts_the_contract(sm103_host) -> None:
    sm103_host.validate_init(_bootstrap(), _fleet())
    sm103_host.validate_init(_bootstrap(), _fleet(max_tokens_per_rank=1))


@pytest.mark.parametrize("capability", ((10, 0), (12, 0), (9, 0), (10, 7)))
def test_validate_init_requires_exact_sm103(
    sm103_host, monkeypatch, capability
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    with pytest.raises(MoEEpArchError, match="requires sm_103"):
        sm103_host.validate_init(_bootstrap(), _fleet())


def test_validate_init_requires_sm103a_build_target(sm103_host, monkeypatch) -> None:
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a 12.0f")
    with pytest.raises(MoEEpArchError, match="FLASHINFER_CUDA_ARCH_LIST"):
        sm103_host.validate_init(_bootstrap(), _fleet())


@pytest.mark.parametrize(
    "bootstrap,fleet,reason",
    (
        (_bootstrap(world_size=8), _fleet(), "EP world size 16"),
        (_bootstrap(world_size=32, rank=0), _fleet(), "EP world size 16"),
        (_bootstrap(), _fleet(num_experts=256), "512 experts"),
        (_bootstrap(), _fleet(token_hidden_size=4096), "token_hidden_size 3072"),
        (_bootstrap(), _fleet(max_tokens_per_rank=128), "max_tokens_per_rank in"),
        (_bootstrap(), _fleet(dtype_bytes=1), "dtype_bytes=2"),
    ),
)
def test_validate_init_rejects_geometry_with_context(
    sm103_host, bootstrap, fleet, reason
) -> None:
    with pytest.raises(MoEEpConfigError) as info:
        sm103_host.validate_init(bootstrap, fleet)
    message = str(info.value)
    assert "unsupported configuration" in message
    assert reason in message
    for key in (
        "world_size=",
        "num_experts=",
        "hidden=",
        "intermediate=",
        "top_k=",
        "max_tokens_per_rank=",
    ):
        assert key in message


def test_validate_init_rejects_non_default_stream(sm103_host) -> None:
    with pytest.raises(MoEEpConfigError, match="stream must be 0"):
        sm103_host.validate_init(_bootstrap(stream=7), _fleet())


def test_validate_init_checks_collective_agreement(sm103_host, monkeypatch) -> None:
    proposals_seen = []

    def all_gather_object(output, proposal, *, group):
        proposals_seen.append(proposal)
        output[:] = [proposal] * 15 + [("other",)]

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    sm103_host._ep_bootstrap = _bootstrap()
    sm103_host._ep_rank, sm103_host._ep_world_size = 0, 16
    sm103_host._ep_comm_group = object()
    with pytest.raises(MoEEpConfigError, match="same sm103 Cake backend"):
        sm103_host.validate_init(_bootstrap(), _fleet())
    assert len(proposals_seen) == 1 and proposals_seen[0][0] == KERNEL_NAME


# ---------------------------------------------------------------------------
# Staging.
# ---------------------------------------------------------------------------
def test_resolve_bound_rules() -> None:
    assert resolve_bound(16, None, 64) == 64
    assert resolve_bound(16, 16, 64) == 16
    assert resolve_bound(0, 1, 64) == 1
    with pytest.raises(MoEEpConfigError, match="below this rank"):
        resolve_bound(17, 16, 64)
    with pytest.raises(MoEEpConfigError, match="must be in"):
        resolve_bound(1, 65, 64)
    with pytest.raises(MoEEpConfigError, match="must be in"):
        resolve_bound(1, 0, 64)
    with pytest.raises(MoEEpConfigError, match="int or None"):
        resolve_bound(1, True, 64)
    with pytest.raises(MoEEpConfigError, match="capacity"):
        resolve_bound(1, None, 65)


def test_stage_inputs_pads_to_the_bound() -> None:
    hidden_out = torch.full((64, 3072), 7.0, dtype=torch.bfloat16)
    ids_out = torch.full((64, 8), 9, dtype=torch.int64)
    weights_out = torch.full((64, 8), 3.0, dtype=torch.float32)
    n, bound = 5, 16
    hidden = torch.randn((n, 3072)).to(torch.bfloat16)
    ids = torch.arange(n * 8, dtype=torch.int64).view(n, 8) % 512
    weights = torch.rand((n, 8), dtype=torch.float32)
    assert (
        stage_inputs(
            hidden,
            ids,
            weights,
            hidden_out=hidden_out,
            ids_out=ids_out,
            weights_out=weights_out,
            bound=bound,
        )
        == n
    )
    assert torch.equal(hidden_out[:n], hidden)
    assert torch.equal(ids_out[:n], ids)
    assert torch.equal(weights_out[:n], weights)
    assert torch.count_nonzero(hidden_out[n:bound]) == 0
    assert torch.all(ids_out[n:bound] == MASKED_ROUTE)
    assert torch.count_nonzero(weights_out[n:bound]) == 0
    # Rows at and above the bound are untouched.
    assert torch.all(hidden_out[bound:] == 7.0)
    assert torch.all(ids_out[bound:] == 9)
    with pytest.raises(MoEEpConfigError):
        stage_inputs(
            hidden,
            ids,
            weights,
            hidden_out=hidden_out,
            ids_out=ids_out,
            weights_out=weights_out,
            bound=4,
        )


def test_staging_performs_no_host_memset() -> None:
    """The publish program zeroes the combine read window in-kernel; the
    staging module exposes no host-side clear and the backend calls none."""

    from flashinfer.experimental.cake_w8a8_megamoe_ep16 import (
        cake_backend,
        cake_staging,
    )

    assert not hasattr(cake_staging, "clear_combine_window")
    assert not hasattr(cake_backend, "clear_combine_window")
    assert "clear_combine_window" not in cake_staging.__all__
    source = inspect.getsource(
        cake_backend.Sm103Mxfp8Mxfp8CakeMegaKernelBackend.compute
    )
    assert "zero_(" not in source and "fill_(" not in source


def test_stage_inputs_zero_tokens_is_all_padding() -> None:
    hidden_out = torch.ones((64, 3072), dtype=torch.bfloat16)
    ids_out = torch.zeros((64, 8), dtype=torch.int64)
    weights_out = torch.ones((64, 8), dtype=torch.float32)
    stage_inputs(
        torch.empty((0, 3072), dtype=torch.bfloat16),
        torch.empty((0, 8), dtype=torch.int64),
        torch.empty((0, 8), dtype=torch.float32),
        hidden_out=hidden_out,
        ids_out=ids_out,
        weights_out=weights_out,
        bound=8,
    )
    assert torch.count_nonzero(hidden_out[:8]) == 0
    assert torch.all(ids_out[:8] == MASKED_ROUTE)
    assert torch.all(hidden_out[8:] == 1.0)


# ---------------------------------------------------------------------------
# Plan: grids, tile maps, sizes.
# ---------------------------------------------------------------------------
@pytest.fixture
def plan_device(monkeypatch):
    props = SimpleNamespace(multi_processor_count=148)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    cake_plan.sm_count.cache_clear()
    cake_plan.persistent_grid.cache_clear()
    cake_plan.quantize_grid.cache_clear()
    yield 0
    cake_plan.sm_count.cache_clear()
    cake_plan.persistent_grid.cache_clear()
    cake_plan.quantize_grid.cache_clear()


@pytest.mark.parametrize("tokens", (1, 16, 33, 64))
def test_stage_grids(plan_device, tokens) -> None:
    grids = cake_plan.stage_grids(tokens, plan_device)
    assert grids.publish_push == tokens
    assert grids.local_pack == 16 * tokens
    assert grids.fc1 == grids.fc2 == 128  # 148 SMs -> 128 (common tile divisor)
    assert grids.combine == 3 * tokens == cake_plan.COMBINE_CTAS_PER_TOKEN * tokens
    assert list(grids.as_dict()) == list(cake_plan.STAGES)
    assert len(grids.as_dict()) == 5
    assert "route_reset" not in grids.as_dict()
    assert "partial_barrier" not in grids.as_dict()
    assert not hasattr(grids, "partial_barrier")


@pytest.mark.parametrize(
    "sms,grid",
    (
        (132, 128),  # manifest persistent_grid_by_sm_count
        (148, 128),
        (152, 128),
        (160, 128),
        (153, 128),  # odd count: 128 lies in [115, 153]
        (256, 256),  # the SM count itself divides 1280 and 768
        (64, 64),
        (100, 100),  # no divisor of gcd(1280, 768) = 256 in [75, 100]: even SM count
        (99, 98),
        (3, 2),
        (2, 2),
        (1, 1),
    ),
)
def test_persistent_grid_is_the_largest_common_tile_divisor(sms, grid) -> None:
    assert cake_plan.persistent_grid_for_sm_count(sms) == grid
    # Independent restatement of the manifest rule: the largest g in
    # [ceil(3 * sms / 4), sms] dividing both 1280 FC1 and 768 FC2 tiles (no CTA
    # runs a partial last wave), else the even SM count.
    lowest = (3 * sms + 3) // 4
    divisors = [g for g in range(lowest, sms + 1) if 1280 % g == 0 and 768 % g == 0]
    assert grid == (max(divisors) if divisors else sms & ~1)
    with pytest.raises(RuntimeError):
        cake_plan.persistent_grid_for_sm_count(0)


def test_persistent_grid_follows_the_device_sm_count(plan_device, monkeypatch) -> None:
    assert cake_plan.persistent_grid(plan_device) == 128
    assert cake_plan.quantize_grid(plan_device) == 148 * 4
    cake_plan.sm_count.cache_clear()
    cake_plan.persistent_grid.cache_clear()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=256),
    )
    assert cake_plan.persistent_grid(1) == 256


@pytest.mark.parametrize("tokens", (0, 65, 16.0, True))
def test_stage_grids_reject_out_of_contract_tokens(plan_device, tokens) -> None:
    with pytest.raises((ValueError, TypeError)):
        cake_plan.stage_grids(tokens, plan_device)


def test_static_plan_is_the_full_envelope() -> None:
    plan = cake_plan.static_plan("cpu")
    assert plan.fc1_grid_m == 32 * 40 == 1280
    assert plan.fc2_grid_m == 32 * 24 == 768
    assert torch.equal(
        plan.fc1_tile_expert, torch.arange(32, dtype=torch.int32).repeat_interleave(40)
    )
    assert torch.equal(
        plan.fc1_tile_m_local, torch.arange(0, 5120, 128, dtype=torch.int32).repeat(32)
    )
    assert torch.equal(
        plan.fc2_tile_expert, torch.arange(32, dtype=torch.int32).repeat_interleave(24)
    )
    assert torch.equal(
        plan.fc2_tile_m_local, torch.arange(0, 3072, 128, dtype=torch.int32).repeat(32)
    )
    assert torch.equal(
        plan.expert_row_offsets, torch.arange(0, 4096, 128, dtype=torch.int32)
    )
    assert torch.all(plan.ones == 1) and torch.all(plan.zeros == 0)
    assert torch.all(torch.isinf(plan.infinities)) and plan.infinities.numel() == 32


def test_scale_byte_sizes() -> None:
    assert cake_plan.W13_SCALE_BYTES == 31_457_280 == 32 * 10240 * 3072 // 32
    assert cake_plan.W2_SCALE_BYTES == 15_728_640 == 32 * 3072 * 5120 // 32
    assert cake_plan.FC1_INPUT_SCALE_BYTES == 393_216
    assert cake_plan.FC2_INPUT_SCALE_BYTES == 655_360
    assert cake_plan.PG_FLAG_WORDS == 2 + 2 * 2 * 16 == 66
    with pytest.raises(ValueError):
        cake_plan.scale_bytes(100, 3072)


# ---------------------------------------------------------------------------
# Peer tables.
# ---------------------------------------------------------------------------
def test_peer_table_is_exactly_world_size_and_images_preserve_bits() -> None:
    active = (1, 2**63 - 1, 2**63, 2**64 - 1, *range(5, 17))
    table = peer_table(active)
    assert table == active and len(table) == 16
    images = peer_table_images(table)
    assert images == (1, 2**63 - 1, -(2**63), -1, *range(5, 17))
    assert len(images) == 16 and all(isinstance(v, int) for v in images)
    # No zero padding crosses the FFI; a padded table is rejected.
    with pytest.raises(ValueError, match="exactly 16"):
        peer_table_images(table + (0,) * 16)
    assert as_int64_image(2**64 - 1) == -1


@pytest.mark.parametrize(
    "pointers",
    (
        None,
        (1,) * 15,
        (1,) * 17,
        (0,) + (1,) * 15,
        (-1,) + (1,) * 15,
        (2**64,) + (1,) * 15,
        (True,) + (1,) * 15,
        ("1",) + (1,) * 15,
    ),
)
def test_peer_table_rejects_invalid_tables(pointers) -> None:
    with pytest.raises(ValueError):
        peer_table(pointers)


def test_peer_table_shape_is_a_prebuilt_tuple_of_images() -> None:
    import tvm_ffi

    table = peer_table(tuple(range(0x7F00_0000_0000, 0x7F00_0000_0000 + 16)))
    shape = peer_table_shape(table)
    assert isinstance(shape, tvm_ffi.Shape) and isinstance(shape, tuple)
    assert len(shape) == 16 and tuple(shape) == peer_table_images(table)
    assert _python_arg_kind(shape) == "peer_table"
    with pytest.raises(ValueError, match="exactly 16"):
        peer_table_shape(table[:15])


# ---------------------------------------------------------------------------
# Stage call interface shape (host-only).
# ---------------------------------------------------------------------------
PEERS = tuple(range(0x1000, 0x1000 + 16 * 0x100, 0x100))  # 16 non-null images
FAKE_DEVICE_INDEX = 2


def test_symmetric_self_entry_is_asserted_against_the_local_pointer() -> None:
    """``check_self_entry`` (called once per symmetric buffer at workspace
    allocation) accepts a peer table whose slot ``rank`` is the buffer's own
    base address and rejects any other table or an out-of-range rank: the
    programs poll or read a symmetric buffer through the local pointer while
    the peers write it through the table, so a mismatch would be a silent
    hang (manifest ``peer_table_rule`` / ``counter_layout``)."""

    local = torch.zeros(2, dtype=torch.uint32)
    table = list(PEERS)
    table[3] = int(local.data_ptr())
    buffer = SymmetricBuffer(
        tensor=local,
        peer_pointers=peer_table(table),
        peer_images=peer_table_images(table),
        peers=torch.tensor(peer_table_images(table), dtype=torch.int64),
    )
    check_self_entry(buffer, rank=3, name="fc2_completion")
    with pytest.raises(RuntimeError, match="symmetric self entry"):
        check_self_entry(buffer, rank=4, name="fc2_completion")
    for rank in (-1, 16):
        with pytest.raises(RuntimeError, match="has no entry for rank"):
            check_self_entry(buffer, rank=rank, name="fc2_completion")
    # The allocation path asserts it for every symmetric buffer with the
    # process-group rank, before the flag pad (bound as a table only).
    source = inspect.getsource(Sm103Mxfp8Mxfp8CakeMegaKernelBackend._allocate_workspace)
    assert "for name, buffer in symmetric.items():" in source
    assert "check_self_entry(buffer, rank=rank, name=name)" in source
    assert source.index("check_self_entry(") < source.index("allocate_flag_pad(")


def _cpu_workspace(
    *, rank: int = 3, device: torch.device | None = None, capacity: int = 64
) -> CakeWorkspace:
    """A real ``CakeWorkspace`` over CPU tensors (no NVSHMEM, no JIT module);
    the symmetric buffers are stand-ins with 16-entry peer tables."""

    cpu = torch.device("cpu")

    def sym(shape, dtype):
        return SimpleNamespace(
            local=torch.zeros(shape, dtype=dtype),
            peer_images=PEERS,
            peers=torch.tensor(PEERS),
        )

    return CakeWorkspace(
        device=cpu if device is None else device,
        group=None,
        rank=rank,
        world_size=16,
        capacity=capacity,
        module=None,
        entries={stage: SimpleNamespace(name=stage) for stage in cake_plan.STAGES},
        use_torch_stream=contextlib.nullcontext,
        plan=cake_plan.static_plan(cpu),
        push_inbox=sym((16, 64, 3232), torch.uint8),
        push_flags=sym((16, 64), torch.uint32),
        local_terms=sym((1024, 8, 3072), torch.bfloat16),
        fc2_completion=sym((2,), torch.uint32),
        pg_flags=sym((cake_plan.PG_FLAG_WORDS,), torch.uint32),
        push_generation=torch.zeros(2, dtype=torch.uint32),
        recv_weights=torch.zeros(8192),
        expert_counts=torch.zeros(32, dtype=torch.int32),
        route_map=torch.zeros(4096, dtype=torch.int32),
        inverse_rows=torch.zeros(8192, dtype=torch.int32),
        overflow=torch.zeros(1, dtype=torch.int32),
        fc1_input_fp8=torch.zeros((32, 128, 3072), dtype=torch.float8_e4m3fn),
        fc1_input_scales=torch.zeros(393216, dtype=torch.uint8),
        fc2_input_fp8=torch.zeros((32, 128, 5120), dtype=torch.float8_e4m3fn),
        fc2_input_scales=torch.zeros(655360, dtype=torch.uint8),
        hidden=torch.zeros((64, 3072), dtype=torch.bfloat16),
        ids=torch.full((64, 8), -1, dtype=torch.int64),
        weights=torch.zeros((64, 8), dtype=torch.float32),
        output=torch.zeros((64, 3072), dtype=torch.bfloat16),
    )


def _fake_workspace() -> CakeWorkspace:
    """CPU workspace that reports a CUDA device index (no binding receives it:
    every program has a tensor argument that selects the device)."""

    return _cpu_workspace(device=torch.device("cuda", FAKE_DEVICE_INDEX))


def _fake_weights() -> TransformedCakeW8a8MegaWeights:
    return TransformedCakeW8a8MegaWeights(
        w13=torch.zeros(W13_SHAPE, dtype=torch.float8_e4m3fn),
        w13_scales=torch.zeros(cake_plan.W13_SCALE_BYTES, dtype=torch.uint8),
        w2=torch.zeros(W2_SHAPE, dtype=torch.float8_e4m3fn),
        w2_scales_swap=torch.zeros(cake_plan.W2_SCALE_BYTES, dtype=torch.uint8),
        w13_layout=W13_FP8_LAYOUT,
        w2_layout=W2_FP8_LAYOUT,
    )


def _quantize_call_args() -> tuple:
    """The positional tuple ``launch_quantize_weights`` hands to its binding."""

    recorded: list[tuple] = []
    module = SimpleNamespace(**{QUANTIZE_ENTRY: lambda *args: recorded.append(args)})
    launch_quantize_weights(
        module,
        source=torch.zeros((32, 256, 128), dtype=torch.bfloat16),
        quantized=torch.zeros((32, 256, 128), dtype=torch.float8_e4m3fn),
        scales_cp=torch.zeros(32 * 256 * 128 // 32, dtype=torch.uint8),
        rows=256,
        columns=128,
        gate_width=64,
        interleave_gate_up=1,
        scale_tile_n256_k128=SCALE_MODE_W13,
        ktile_major_rows=W13_KTILE_ROWS,
        grid_x=4,
    )
    assert len(recorded) == 1
    return recorded[0]


def test_quantize_launch_passes_the_payload_placement_scalar() -> None:
    """experts, rows, columns, gate_width, interleave, scale mode,
    ktile_major_rows, grid_x follow the three tensors (10 device args + grid)."""

    args = _quantize_call_args()
    assert len(args) == 11
    assert all(isinstance(t, torch.Tensor) for t in args[:3])
    assert args[3:] == (32, 256, 128, 64, 1, SCALE_MODE_W13, 256, 4)
    assert (W13_KTILE_ROWS, W2_KTILE_ROWS) == (256, 128)
    assert W13_FP8_LAYOUT == "ktile_major_r256_k128"
    assert W2_FP8_LAYOUT == "ktile_major_r128_k128"


def _stage_call_args(plan_device) -> dict[str, tuple]:
    """Python positional tuples of all six bindings, by stage."""

    grids = cake_plan.stage_grids(16, plan_device)
    calls = build_stage_calls(_fake_workspace(), _fake_weights(), grids)
    by_name = {entry.name: args for entry, args in calls}
    by_name["quantize_weights"] = _quantize_call_args()
    return by_name


def test_stage_calls_follow_the_documented_interface(plan_device) -> None:
    grids = cake_plan.stage_grids(16, plan_device)
    calls = build_stage_calls(_fake_workspace(), _fake_weights(), grids)
    assert [entry.name for entry, _ in calls] == list(cake_plan.STAGES)
    assert set(STAGE_ENTRIES) == set(cake_plan.STAGES)
    assert all(STAGE_ENTRIES[stage] == stage for stage in cake_plan.STAGES)
    assert set(STAGE_ARGS) == set(cake_plan.STAGES)
    lengths = {entry.name: len(args) for entry, args in calls}
    assert lengths == {
        "publish_push": 17,
        "local_pack": 19,
        "fc1": 21,
        "fc2": 21,
        "combine": 11,
    }
    assert "partial_barrier" not in lengths
    by_name = {entry.name: args for entry, args in calls}
    publish, pack = by_name["publish_push"], by_name["local_pack"]
    fc2, combine = by_name["fc2"], by_name["combine"]
    ws_terms = fc2[16]
    # Every call ends with its grid; scalars carry the bound-dependent values.
    # publish_push: tokens, pg_world, pg_rank, pg_flags, inbox, inbox peers,
    # flags, flags peers, grid_x = tokens.
    assert publish[8:12] == (16, 16, 3, PEERS) and publish[-1] == 16
    # publish_push zeroes the combine window and resets the routing state, so
    # it receives the term buffer (the same symmetric local FC2 pushes into)
    # and the three routing-state buffers local_pack reads and writes; the two
    # push programs share the inbox, the flags and the generation words.
    assert publish[3] is ws_terms
    assert tuple(publish[3].shape) == (1024, 8, 3072)
    assert publish[3].dtype == torch.bfloat16
    assert publish[4] is pack[1]  # expert_counts
    assert publish[5] is pack[2]  # route_map
    assert publish[6] is pack[4]  # overflow
    assert publish[7] is pack[7]  # push_generation
    assert publish[12] is pack[14]  # push_inbox (symmetric local)
    assert publish[14] is pack[16]  # push_flags (symmetric local)
    assert tuple(publish[4].shape) == (32,)
    assert tuple(publish[5].shape) == (4096,)
    assert tuple(publish[6].shape) == (1,)
    assert tuple(publish[7].shape) == (2,) and publish[7].dtype == torch.uint32
    assert tuple(publish[12].shape) == (16, 64, 3232)
    assert publish[12].dtype == torch.uint8
    assert tuple(publish[14].shape) == (16, 64) and publish[14].dtype == torch.uint32
    # local_pack: tokens_per_rank, active_rows_per_expert, scale_tile_k128,
    # pg_world, pg_rank, pg_flags, ..., grid_x = 16 * tokens.
    assert pack[8:14] == (16, 32, 1, 16, 3, PEERS) and pack[-1] == 256
    # The two routing constants local_pack is bound with are the manifest's
    # local_pack_scalar_rule values (the exported fc1 fixes them).
    rule = json.loads(MANIFEST_PATH.read_text())["route_contract"]["push_dispatch"][
        "local_pack_scalar_rule"
    ]
    scalars = {
        key: int(value)
        for key, value in re.findall(
            r"(active_rows_per_expert|scale_tile_k128) = (\d+)", rule
        )
    }
    assert scalars == {"active_rows_per_expert": 32, "scale_tile_k128": 1}
    assert (
        pack[9] == cake_plan.ACTIVE_ROWS_PER_EXPERT == scalars["active_rows_per_expert"]
    )
    assert pack[10] == cake_plan.SCALE_TILE_K128 == scalars["scale_tile_k128"]
    assert pack[8] == grids.tokens_per_rank == 16  # this forward's bound
    assert by_name["fc1"][-6:] == (10240, 24, 1280, 1, 128, 128)
    assert fc2[9:13] == (128, 3072, 40, 768) and fc2[-1] == 128
    # fc2: ..., remote_pg_world, remote_pg_rank, remote_pg_flags, remote_terms,
    # remote_terms_peers, fc2_completion (symmetric local u32 [2]),
    # fc2_completion_peers, grid_x = persistent grid.
    assert fc2[13:16] == (16, 3, PEERS)
    completion = fc2[18]
    assert isinstance(completion, torch.Tensor)
    assert tuple(completion.shape) == (2,) and completion.dtype == torch.uint32
    assert fc2[19] == PEERS
    # combine: local_terms, output, push_generation, tokens_per_rank,
    # source_rank, pg_world, pg_rank, pg_flags, fc2_completion,
    # fc2_completion_peers, grid_x = 3 * tokens.
    assert combine[0] is ws_terms
    assert tuple(combine[1].shape) == (64, 3072) and combine[1].dtype == torch.bfloat16
    assert combine[2] is publish[7]  # push_generation, the word local_pack advances
    assert combine[3:8] == (16, 3, 16, 3, PEERS)
    assert combine[8] is completion  # the same counter fc2 signals into
    assert combine[9] == PEERS and combine[-1] == 48
    # Peer tables are host tuples of exactly 16 signed images; symmetric locals precede them.
    assert publish[13] == PEERS and publish[15] == PEERS
    assert pack[15] == PEERS and pack[17] == PEERS
    assert fc2[17] == PEERS and fc2[16].dtype == torch.bfloat16
    assert all(len(arg) == 16 for arg in publish if isinstance(arg, tuple))
    assert all(len(arg) == 16 for arg in pack if isinstance(arg, tuple))
    assert all(len(arg) == 16 for arg in fc2 if isinstance(arg, tuple))
    assert all(len(arg) == 16 for arg in combine if isinstance(arg, tuple))
    # No binding receives the device index: every program has a tensor argument
    # (the retired barrier binding was the only one that took it).
    for args in by_name.values():
        assert isinstance(args[0], torch.Tensor), args[0]
    # TMA scale views reshape the flat planes without copies.
    assert tuple(by_name["fc1"][2].shape) == (32, 960, 32, 32)
    assert tuple(by_name["fc1"][3].shape) == (32, 24, 16, 32)
    assert tuple(by_name["fc2"][2].shape) == (32, 960, 16, 32)
    assert tuple(by_name["fc2"][3].shape) == (32, 40, 16, 32)


def test_workspace_buffers_follow_the_push_dispatch_contract() -> None:
    """The workspace's symmetric buffers are exactly the route's inbox, flags,
    term buffer and fc2 completion counter with the contract shapes and byte
    sizes; the pull-route publications are gone; ``push_generation`` is
    rank-local."""

    fields = {f.name for f in dataclasses.fields(CakeWorkspace)}
    assert set(SYMMETRIC_BUFFERS) == {
        "push_inbox",
        "push_flags",
        "local_terms",
        "fc2_completion",
    }
    assert set(SYMMETRIC_BUFFERS) | {"pg_flags", "push_generation"} <= fields
    for retired in (
        "hidden_states_mxfp8",
        "hidden_scales_mxfp8",
        "topk_ids_sym",
        "topk_weights_sym",
    ):
        assert retired not in fields, retired

    def nbytes(shape, dtype) -> int:
        return math.prod(shape) * dtype.itemsize

    assert SYMMETRIC_BUFFERS["push_inbox"] == ((16, 64, 3232), torch.uint8)
    assert nbytes(*SYMMETRIC_BUFFERS["push_inbox"]) == cake_plan.PUSH_INBOX_BYTES
    assert cake_plan.PUSH_INBOX_BYTES == 3_309_568
    assert SYMMETRIC_BUFFERS["push_flags"] == ((16, 64), torch.uint32)
    assert nbytes(*SYMMETRIC_BUFFERS["push_flags"]) == cake_plan.PUSH_FLAGS_BYTES
    assert cake_plan.PUSH_FLAGS_BYTES == 4096
    assert SYMMETRIC_BUFFERS["local_terms"] == ((1024, 8, 3072), torch.bfloat16)
    # One symmetric u32 [2] counter per rank (8 bytes): word 0 = done counter
    # (16 adds per forward, never reset), word 1 = fc2 CTA arrivals (back to 0).
    assert SYMMETRIC_BUFFERS["fc2_completion"] == ((2,), torch.uint32)
    assert (
        nbytes(*SYMMETRIC_BUFFERS["fc2_completion"]) == cake_plan.FC2_COMPLETION_BYTES
    )
    assert cake_plan.FC2_COMPLETION_WORDS == 2 and cake_plan.FC2_COMPLETION_BYTES == 8
    assert cake_plan.FC2_DONE_WORD == 0 and cake_plan.FC2_ARRIVALS_WORD == 1
    assert cake_plan.PUSH_GENERATION_WORDS == 2
    # One slot: contiguous fields covering exactly the 3232 bytes, a multiple
    # of the 16-byte bulk-copy granule, ids+weights tail at 3168.
    assert cake_plan.PUSH_SLOT_LAYOUT == {
        "row_fp8_e4m3": (0, 3072),
        "scales_ue8m0": (3072, 96),
        "ids_i32": (3168, 32),
        "weights_f32": (3200, 32),
    }
    spans = list(cake_plan.PUSH_SLOT_LAYOUT.values())
    assert spans[0][0] == 0
    assert all(a[0] + a[1] == b[0] for a, b in zip(spans[:-1], spans[1:], strict=True))
    assert spans[-1][0] + spans[-1][1] == cake_plan.PUSH_SLOT_BYTES == 3232
    assert cake_plan.PUSH_SLOT_BYTES % 16 == 0
    assert cake_plan.PUSH_SLOT_LAYOUT["ids_i32"][0] % 16 == 0
    # The CPU stand-in workspace mirrors the shapes and dtypes.
    ws = _cpu_workspace()
    for name, (shape, dtype) in SYMMETRIC_BUFFERS.items():
        tensor = getattr(ws, name).local
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype, name
    assert tuple(ws.push_generation.shape) == (2,)
    assert ws.push_generation.dtype == torch.uint32
    assert tuple(ws.fc2_completion.local.shape) == (2,)
    assert ws.fc2_completion.local.dtype == torch.uint32


# ---------------------------------------------------------------------------
# Manifest <-> Python argument tuples <-> hand-written binding signatures.
#
# Device-argument kinds map to binding parameters as: buffer / tma ->
# tvm::ffi::TensorView, parameter / process_group_scalar -> int64_t,
# peer_table -> tvm::ffi::Shape (exactly 16 entries). A program without any
# tensor argument would take int64_t device_index first (none in this route:
# the retired barrier was the only one); every binding ends with int64_t grid_x.
# ---------------------------------------------------------------------------
_KIND_OF_DEVICE_ARGUMENT = {
    "buffer": "tensor",
    "tma": "tensor",
    "parameter": "scalar",
    "process_group_scalar": "scalar",
    "peer_table": "peer_table",
}


def _manifest_programs() -> dict[str, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {program["stage"]: program for program in manifest["programs"]}


def _expected_binding_kinds(program: dict) -> list[str]:
    device_arguments = program["arguments"]["device_arguments"]
    kinds = []
    if not any(a["kind"] in ("buffer", "tma") for a in device_arguments):
        kinds.append("scalar")  # device_index
    kinds.extend(_KIND_OF_DEVICE_ARGUMENT[a["kind"]] for a in device_arguments)
    kinds.append("scalar")  # grid_x
    return kinds


def _python_arg_kind(arg) -> str:
    if isinstance(arg, torch.Tensor):
        return "tensor"
    if isinstance(arg, tuple):
        assert len(arg) == 16 and all(
            isinstance(v, int) and not isinstance(v, bool) and v != 0 for v in arg
        ), arg
        return "peer_table"
    if isinstance(arg, int) and not isinstance(arg, bool):
        return "scalar"
    return f"unsupported:{type(arg).__name__}"


def test_manifest_matches_the_route_contract() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["arch"] == cake_jit.TARGET_ARCH
    assert (
        manifest["cake"]["route_id"]
        == cake_plan.ROUTE_ID
        == ("w8a8_megamoe_ep16_staged_v4")
    )
    assert manifest["forward_stages"] == list(cake_plan.STAGES)
    assert manifest["forward_stages"] == [
        "publish_push",
        "local_pack",
        "fc1",
        "fc2",
        "combine",
    ]
    stages = {program["stage"] for program in manifest["programs"]}
    for retired in (
        "route_reset",
        "publish_inputs",
        "input_barrier",
        "dispatch",
        "partial_barrier",
    ):
        assert retired not in manifest["grid_formulas"], retired
        assert retired not in stages, retired
        assert retired not in manifest["forward_stages"], retired
    assert stages == set(cake_plan.STAGES) | {"quantize_weights"}
    assert manifest["setup_stages"] == ["quantize_weights"]
    # The frozen knob table is the route's pin: the complete table, not only
    # the knobs this package reads, so a regenerated drop with another route
    # fails here before anything is built.
    assert manifest["frozen_build_knobs"] == FROZEN_BUILD_KNOBS
    assert manifest["frozen_build_knobs"]["PUSH_DISPATCH"] == "1"
    # "0" = the untimed flag wait (the shipped program text).
    assert manifest["frozen_build_knobs"]["PUSH_WAIT_TIMEOUT_CYCLES"] == "0"
    # "1" = the counted combine (fc2 completion counters, no partial barrier).
    assert manifest["frozen_build_knobs"]["FC2_COMPLETION_COUNTERS"] == "1"
    note = manifest["forward_launch_order_note"]
    assert "five forward programs" in note
    assert "no input barrier and no partial barrier" in note
    assert "fc1, fc2 and combine (launch.use_pdl)" in note
    contract = manifest["route_contract"]
    assert contract["world_size"] == cake_plan.WORLD_SIZE
    assert contract["local_experts"] == cake_plan.LOCAL_EXPERTS
    assert contract["hidden"] == cake_plan.HIDDEN
    assert contract["intermediate"] == cake_plan.INTERMEDIATE
    assert contract["max_tokens_per_rank"] == cake_plan.MAX_TOKENS_PER_RANK
    assert contract["swap_route_n"] == cake_plan.ACTIVE_ROWS_PER_EXPERT == 32
    assert contract["fc1_tiles"] == cake_plan.FC1_SWAP_GRID_M == 1280
    assert contract["fc2_tiles"] == cake_plan.FC2_SWAP_GRID_M == 768
    assert contract["combine_ctas_per_token"] == cake_plan.COMBINE_CTAS_PER_TOKEN == 3
    # k-tile-major FP8 weight payloads: layout tags, quantize scalar, A tensor maps.
    assert contract["w13_fp8_layout"] == W13_FP8_LAYOUT
    assert contract["w2_fp8_layout"] == W2_FP8_LAYOUT
    rule = contract["weight_layout_rule"]
    assert "(((e * (rows // R) + r // R) * (K // 128) + k // 128) * R + r % R)" in rule
    assert "permute(0, 1, 3, 2, 4)" in rule
    programs_by_stage = _manifest_programs()
    quantize = programs_by_stage["quantize_weights"]
    assert quantize["arguments"]["device_arguments"][-1]["name"] == "ktile_major_rows"
    assert (
        f"bind {W13_KTILE_ROWS} for the w13 launch"
        in quantize["weight_layout_contract"]
    )
    assert f"{W2_KTILE_ROWS} for the w2 launch" in quantize["weight_layout_contract"]
    for stage, box_rows, k in (
        ("fc1", W13_KTILE_ROWS, 3072),
        ("fc2", W2_KTILE_ROWS, 5120),
    ):
        a = programs_by_stage[stage]["arguments"]["device_arguments"][0]
        assert a["name"] == "A" and a["kind"] == "tma"
        tma = a["tma"]
        assert tma["global_dim"] == [
            "128",
            str(box_rows),
            "lm.axis[-1] // 128",
            f"lm.axis.outer(2) * lm.axis[-2] // {box_rows}",
        ]
        assert tma["global_strides_elements"] == [
            "128",
            str(box_rows * 128),
            f"{box_rows} * lm.axis[-1]",
        ]
        assert tma["box_dim"] == ["128", str(box_rows), "1", "1"]
        assert tma["checks"] == [
            "lm.axis[-1] % 128 == 0",
            f"lm.axis.outer(2) * lm.axis[-2] % {box_rows} == 0",
        ]
        assert tma["swizzle"] == "Swizzle.SZ_128B" and tma["l2_promotion"] == "none"
        assert k % 128 == 0
        # B stays the row-major activation view.
        b = programs_by_stage[stage]["arguments"]["device_arguments"][1]["tma"]
        assert b["global_dim"][1] == "lm.axis[-2]" and b["box_dim"] == [
            "128",
            "32",
            "1",
            "1",
        ]
    assert manifest["grid_formulas"] == {
        "publish_push": "(tokens_per_rank, 1, 1)",
        "local_pack": "(16 * tokens_per_rank, 1, 1)",
        "fc1": "(persistent_grid, 1, 1)",
        "fc2": "(persistent_grid, 1, 1)",
        "combine": "(3 * tokens_per_rank, 1, 1)",
        "quantize_weights": "(4 * sm_count, 1, 1)",
    }
    assert list(manifest["grid_formulas"])[:5] == list(cake_plan.STAGES)
    # The host persistent-grid rule reproduces the manifest's table.
    table = manifest["persistent_grid_by_sm_count"]
    assert table and all(
        cake_plan.persistent_grid_for_sm_count(int(sms)) == grid
        for sms, grid in table.items()
    )
    # Programmatic dependent launch: the launch attribute belongs to exactly the
    # three dependents (fc1, fc2, combine); local_pack alone sets a
    # shared-memory carveout.
    knobs = manifest["frozen_build_knobs"]
    assert knobs["PDL_CHAIN"] == "1" and knobs["PDL_PREFETCH_KSTEPS"] == "0"
    assert cake_plan.PDL_EDGES == (
        ("local_pack", "fc1"),
        ("fc1", "fc2"),
        ("fc2", "combine"),
    )
    assert cake_plan.PDL_STAGES == ("fc1", "fc2", "combine")
    assert cake_plan.PREFERRED_SHARED_MEMORY_CARVEOUT == {"local_pack": 100}
    assert contract["pdl_edges"] == [list(edge) for edge in cake_plan.PDL_EDGES]
    assert contract["pdl_programs"] == {
        "local_pack": ["launch_dependents"],
        "fc1": ["wait", "launch_dependents"],
        "fc2": ["wait", "launch_dependents"],
        "combine": ["wait"],
    }
    assert "publish_push contains no griddepcontrol instruction" in contract["pdl_rule"]
    for primary, dependent in contract["pdl_edges"]:
        assert "launch_dependents" in contract["pdl_programs"][primary]
        assert "wait" in contract["pdl_programs"][dependent]
    assert "cudaLaunchAttributeProgrammaticStreamSerialization" in contract["pdl_rule"]
    assert "programmaticStreamSerializationAllowed = 1" in contract["pdl_rule"]
    assert "Launch fc1, fc2 and combine" in contract["pdl_rule"]
    assert (
        "combine executes griddepcontrol.wait in every thread before its first "
        "global read and only then polls the local fc2 completion counter"
    ) in contract["pdl_rule"]
    assert (
        "three attributed launches become programmatic graph edges"
        in (contract["pdl_rule"])
    )
    assert contract["function_attributes"] == {
        stage: {"preferred_shared_memory_carveout": value}
        for stage, value in cake_plan.PREFERRED_SHARED_MEMORY_CARVEOUT.items()
    }
    assert (
        "cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100)"
        in contract["function_attribute_rule"]
    )
    for stage, program in programs_by_stage.items():
        launch = program["launch"]
        assert launch["use_pdl"] is (stage in cake_plan.PDL_STAGES), stage
        carveout = cake_plan.PREFERRED_SHARED_MEMORY_CARVEOUT.get(stage)
        assert launch["preferred_shared_memory_carveout"] == carveout, stage
        attributes = (
            []
            if carveout is None
            else [{"kind": "preferred_shared_memory_carveout", "value": carveout}]
        )
        assert launch["function_attributes"] == attributes, stage
        assert program.get("pdl_instructions") == contract["pdl_programs"].get(stage), (
            stage
        )
    # publish_push owns the reset of the routing state and the combine window
    # and pushes the slots; local_pack waits on its local flags.
    publish = programs_by_stage["publish_push"]
    reset_contract = publish["state_reset_contract"]
    for fact in (
        "overflow[0] = 0",
        "expert_counts[0:32] = 0",
        "route_map[0:4096] = -1",
    ):
        assert fact in reset_contract
    assert "no route_reset program" in reset_contract
    assert [a["name"] for a in publish["arguments"]["device_arguments"]] == [
        "source_hidden",
        "source_ids",
        "source_weights",
        "local_terms",
        "expert_counts",
        "route_map",
        "overflow",
        "push_generation",
        "tokens",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "push_inbox",
        "push_inbox_peers",
        "push_flags",
        "push_flags_peers",
    ]
    local_pack = programs_by_stage["local_pack"]
    assert [a["name"] for a in local_pack["arguments"]["device_arguments"]] == [
        "recv_weights",
        "expert_counts",
        "route_map",
        "inverse_rows",
        "overflow",
        "quantized",
        "scales_cp",
        "push_generation",
        "tokens_per_rank",
        "active_rows_per_expert",
        "scale_tile_k128",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "push_inbox",
        "push_inbox_peers",
        "push_flags",
        "push_flags_peers",
    ]
    assert "zero-initialized once" in publish["push_contract"]
    assert "no __restrict__" in publish["push_contract"]
    assert (
        "device synchronize, then a process-group barrier" in publish["push_contract"]
    )
    assert "waits only on its LOCAL push_flags" in local_pack["push_wait_contract"]
    assert "untimed ld.acquire.sys.global.u32 poll" in local_pack["push_wait_contract"]
    assert "No input barrier precedes this program" in local_pack["push_wait_contract"]
    assert "no ld.global.nc" in local_pack["inbox_read_contract"]
    assert (
        "active_rows_per_expert = 32 and scale_tile_k128 = 1"
        in (local_pack["scalar_binding_contract"])
    )
    # The local push_inbox / push_flags pointers are plain (peers write them
    # while the kernel runs); every other buffer pointer keeps __restrict__.
    for program in (publish, local_pack):
        for argument in program["arguments"]["device_arguments"]:
            if argument["kind"] != "buffer":
                continue
            plain = argument["name"] in ("push_inbox", "push_flags")
            assert ("__restrict__" not in argument["c_type"]) is plain, argument
        assert "__restrict__ push_inbox" not in program["declaration"]
        assert "__restrict__ push_flags" not in program["declaration"]
    for program in (publish, local_pack):
        by_name = {a["name"]: a for a in program["arguments"]["device_arguments"]}
        for name, shape in (
            ("push_inbox", ["16", "64", "3232"]),
            ("push_flags", ["16", "64"]),
        ):
            assert by_name[name]["symmetric_memory"] == name
            assert by_name[name]["peers_arg"] == f"{name}_peers"
            assert by_name[name]["shape"] == shape
            assert by_name[f"{name}_peers"]["kind"] == "peer_table"
        assert by_name["push_inbox"]["dtype"] == "u8"
        assert by_name["push_flags"]["dtype"] == "u32"
        assert by_name["push_generation"]["dtype"] == "u32"
    # Push-dispatch buffer contract: symmetric inbox and flags, rank-local
    # generation words; the host constants restate it.
    push = contract["push_dispatch"]
    assert push["slot_bytes"] == cake_plan.PUSH_SLOT_BYTES
    assert push["slot_layout"] == {
        name: list(span) for name, span in cake_plan.PUSH_SLOT_LAYOUT.items()
    }
    assert push["inbox_shape"] == list(cake_plan.PUSH_INBOX_SHAPE)
    assert push["inbox_bytes_per_rank"] == cake_plan.PUSH_INBOX_BYTES
    assert push["flags_shape"] == list(cake_plan.PUSH_FLAGS_SHAPE)
    assert push["flags_bytes_per_rank"] == cake_plan.PUSH_FLAGS_BYTES
    assert push["generation_words"] == cake_plan.PUSH_GENERATION_WORDS
    assert "(source_rank * 64 + source_token) * 3232" in push["slot_offset_rule"]
    assert "same number of forwards" in push["generation_rule"]
    # The reuse argument no longer rests on a partial barrier: a source's
    # publish_push(g + 1) follows its combine(g), which waited for every
    # rank's fc2(g) signal, hence for every local_pack(g).
    assert "partial_barrier must stay" not in push["reuse_rule"]
    assert "no partial_barrier program" in push["reuse_rule"]
    assert (
        "publish_push(g + 1) is stream-ordered after S's combine(g), which waited "
        "until fc2_completion[0] on S reached 16 * (g + 1)"
    ) in push["reuse_rule"]
    assert "ld.acquire.sys.global.u32, UNTIMED" in push["wait_rule"]
    assert "never traps" in push["wait_rule"]
    assert "PUSH_WAIT_TIMEOUT_CYCLES" in push["wait_timeout_knob"]
    assert "0 = untimed" in push["wait_timeout_knob"]
    assert (
        "never the non-coherent read-only ld.global.nc path" in push["inbox_read_rule"]
    )
    assert "declared without __restrict__" in push["inbox_read_rule"]
    assert (
        "device synchronize, then a process-group barrier"
        in (push["initialization_rule"])
    )
    assert "FlashInfer workspace allocation perform" in push["initialization_rule"]
    assert "CakePeerPointerTable<T, 32>" in push["peer_table_rule"]
    assert "entries [0, pg_world) = 16 are used" in push["peer_table_rule"]
    # Symmetric self entry: on every rank the table's own entry is the local
    # pointer bound beside it; the bindings check it per launch and the
    # workspace asserts it once at allocation (cake_symmetric.check_self_entry).
    assert (
        "push_inbox_peers[pg_rank] == push_inbox and push_flags_peers[pg_rank] == "
        "push_flags (publish_push and local_pack)"
    ) in push["peer_table_rule"]
    assert "a mismatch is a silent hang" in push["peer_table_rule"]
    assert "check it per launch with TVM_FFI_CHECK" in push["peer_table_rule"]
    assert (
        "the workspace must assert it once at allocation" in (push["peer_table_rule"])
    )
    assert (
        "pg_flags has no local pointer: no self-entry check"
        in (push["peer_table_rule"])
    )
    assert "active_rows_per_expert = 32" in push["local_pack_scalar_rule"]
    assert "scale_tile_k128 = 1" in push["local_pack_scalar_rule"]
    assert (
        "tokens_per_rank = this forward's token count"
        in (push["local_pack_scalar_rule"])
    )
    assert "publish_push zeroes rows" in contract["combine_window_rule"]
    assert (
        "waited for every rank's fc2 completion signal"
        in (contract["combine_window_rule"])
    )
    assert "local_pack flags more in `overflow`" in contract["routing_capacity"]
    # FC2 completion counters: the host constants restate the counter layout,
    # and the signal / wait / generation / initialization / PDL / forward
    # rules carry the facts the bindings and the workspace rely on.
    completion = contract["fc2_completion"]
    assert set(completion) == {
        "counter_words",
        "counter_bytes_per_rank",
        "done_word",
        "arrivals_word",
        "counter_layout",
        "signal_rule",
        "wait_rule",
        "generation_rule",
        "initialization_rule",
        "pdl_rule",
        "forward_rule",
    }
    assert completion["counter_words"] == cake_plan.FC2_COMPLETION_WORDS == 2
    assert completion["counter_bytes_per_rank"] == cake_plan.FC2_COMPLETION_BYTES == 8
    assert completion["done_word"] == cake_plan.FC2_DONE_WORD == 0
    assert completion["arrivals_word"] == cake_plan.FC2_ARRIVALS_WORD == 1
    layout = completion["counter_layout"]
    assert "symmetric u32[2] every rank owns (8 bytes, zero-initialized once" in layout
    assert "monotonic, 16 per forward, never reset" in layout
    assert "rank-LOCAL CTA arrival counter" in layout
    assert "plain local pointer fc2_completion (no __restrict__)" in layout
    assert "CakePeerPointerTable<unsigned int, 32> fc2_completion_peers" in layout
    assert "entries [0, 16) used" in layout
    assert (
        "fc2_completion_peers[remote_pg_rank] == fc2_completion (fc2) and "
        "fc2_completion_peers[pg_rank] == fc2_completion (combine)"
    ) in layout
    assert "remote_terms_peers[remote_pg_rank] == remote_terms (fc2)" in layout
    assert "the workspace must assert it once at allocation" in layout
    signal = completion["signal_rule"]
    assert "atom.acq_rel.gpu.global.add.u32 +1 on fc2_completion[1]" in signal
    assert "stores fc2_completion[1] = 0" in signal
    assert "then the warp issues ONE red.release.sys.global.add.u32 +1" in signal
    assert "16 adds in one warp-wide instruction" in signal
    # The reduction's .release is the signal's one system-scope release; no
    # separate fence.release.sys precedes it (it lowered to a second
    # MEMBAR.ALL.SYS that drained nothing new).
    assert "No separate fence.release.sys precedes the reduction" in signal
    assert "its .release qualifier is the signal's one system-scope release" in (signal)
    assert (
        "Exactly one system fence (red.release.sys lowers to one MEMBAR.ALL.SYS + "
        "one REDG) and 16 remote adds per rank per forward"
    ) in signal
    assert "no explicit fence in front of the reduction" in signal
    assert "ONE fence.release.sys" not in signal
    wait = completion["wait_rule"]
    assert "generation = push_generation[0] (== g + 1 during forward g)" in wait
    assert (
        "polls its LOCAL fc2_completion[0] UNTIMED with ld.acquire.sys.global.u32"
        in (wait)
    )
    assert "(observed - 16 * generation) < 2^31 unsigned" in wait
    assert "never traps" in wait
    assert "then fence.acquire.sys, then bar.sync" in wait
    assert "explicit coherent ld.global.v4.b32 loads" in wait
    assert "plain (no __restrict__) local_terms pointer, never ld.global.nc" in wait
    generation_rule = completion["generation_rule"]
    assert "the last CTA of local_pack(g) stores g + 1" in generation_rule
    assert "fc2_completion[0] equals exactly 16 * (g + 1) when combine(g) passes" in (
        generation_rule
    )
    assert "every 2^28 forwards" in generation_rule
    assert "Every rank must execute the same number of forwards" in generation_rule
    initialization = completion["initialization_rule"]
    assert "zero-initialized exactly once, at allocation, on every rank" in (
        initialization
    )
    assert "together with push_flags / push_generation" in initialization
    assert "the host never writes it afterwards" in initialization
    assert "device synchronize, then a process-group barrier" in initialization
    pdl = completion["pdl_rule"]
    assert (
        "fc2 executes griddepcontrol.launch_dependents when a CTA starts its last tile"
        in (pdl)
    )
    assert (
        "combine is launched with cudaLaunchAttributeProgrammaticStreamSerialization"
        in (pdl)
    )
    assert "128 threads, no shared memory, at most 3 * 64 = 192 per rank" in pdl
    assert "can neither starve nor deadlock it" in pdl
    assert completion["forward_rule"].startswith(
        "forward = publish_push, local_pack, fc1, fc2, combine (five launches, no "
        "partial_barrier program)"
    )
    programs = _manifest_programs()
    for stage, program in programs.items():
        assert program["file"] == f"cake_w8a8_megamoe_ep16_{stage}_kernel.cu"
        assert (CSRC_DIR / program["file"]).is_file()
        for argument in program["arguments"]["device_arguments"]:
            if argument["kind"] == "peer_table":
                assert argument["capacity"] == cake_plan.PEER_TABLE_CAPACITY
        for group in program["arguments"]["process_groups"]:
            assert group["world_size"] == cake_plan.WORLD_SIZE
        assert program["compile"]["receipt_options"] == ["--use_fast_math"]
    assert cake_jit.USE_FAST_MATH is True and cake_jit.NUMERICS_FLAGS == ()
    # fc2 signals, combine waits: both bind the same symmetric counter and its
    # peer table as their two trailing device arguments; the counter pointers
    # are plain (peers add into word 0 while fc2 runs; the combine polls it)
    # and so is combine's local_terms (remote fc2 stores land in it).
    fc2, combine = programs["fc2"], programs["combine"]
    fc2_names = [a["name"] for a in fc2["arguments"]["device_arguments"]]
    assert fc2_names[-7:] == [
        "remote_pg_world",
        "remote_pg_rank",
        "remote_pg_flags",
        "remote_terms",
        "remote_terms_peers",
        "fc2_completion",
        "fc2_completion_peers",
    ]
    assert [a["name"] for a in combine["arguments"]["device_arguments"]] == [
        "local_terms",
        "output",
        "push_generation",
        "tokens_per_rank",
        "source_rank",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "fc2_completion",
        "fc2_completion_peers",
    ]
    for program, process_group in ((fc2, "remote_pg"), (combine, "pg")):
        by_name = {a["name"]: a for a in program["arguments"]["device_arguments"]}
        counter = by_name["fc2_completion"]
        assert counter["kind"] == "buffer" and counter["dtype"] == "u32"
        assert counter["shape"] == ["2"]
        assert counter["symmetric_memory"] == "fc2_completion"
        assert counter["peers_arg"] == "fc2_completion_peers"
        assert counter["c_type"] == "unsigned int*"  # no __restrict__
        assert by_name["fc2_completion_peers"]["kind"] == "peer_table"
        assert by_name["fc2_completion_peers"]["element_dtype"] == "u32"
        assert by_name["fc2_completion_peers"]["owner"] == {
            "role": "peers",
            "symmetric_memory": "fc2_completion",
        }
        assert "__restrict__ fc2_completion" not in program["declaration"]
        assert "unsigned int* fc2_completion, " in program["declaration"]
        symmetric = {s["name"]: s for s in program["arguments"]["symmetric_memories"]}
        assert symmetric["fc2_completion"] == {
            "name": "fc2_completion",
            "dtype": "u32",
            "shape": ["2"],
            "local_arg": "fc2_completion",
            "peers_arg": "fc2_completion_peers",
            "process_group": process_group,
        }
        assert [g["name"] for g in program["arguments"]["process_groups"]] == [
            process_group
        ]
    assert fc2["arguments"]["peer_table_argument_count"] == 3
    assert combine["arguments"]["peer_table_argument_count"] == 2
    combine_by_name = {a["name"]: a for a in combine["arguments"]["device_arguments"]}
    assert combine_by_name["local_terms"]["c_type"] == "__nv_bfloat16*"
    assert combine_by_name["output"]["c_type"] == "__nv_bfloat16* __restrict__"
    assert combine_by_name["push_generation"]["c_type"] == "unsigned int* __restrict__"
    assert combine_by_name["push_generation"]["dtype"] == "u32"
    for name in ("tokens_per_rank", "source_rank"):
        assert combine_by_name[name]["kind"] == "parameter"
    for name in ("pg_world", "pg_rank"):
        assert combine_by_name[name]["kind"] == "process_group_scalar"
    assert "__restrict__ local_terms" not in combine["declaration"]
    assert "__restrict__ remote_terms" in fc2["declaration"]  # fc2 keeps its own
    signal_contract = fc2["completion_signal_contract"]
    assert "plain local pointer of the symmetric u32[2] counter, no __restrict__" in (
        signal_contract
    )
    assert (
        "fc2_completion_peers[remote_pg_rank] == fc2_completion, checked by the "
        "reference binding"
    ) in signal_contract
    assert (
        "the last CTA of the grid issues one warp-wide red.release.sys.global.add.u32 +1"
        in (signal_contract)
    )
    assert (
        "its .release is the signal's one system fence, no separate fence.release.sys"
        in (signal_contract)
    )
    assert "fence.release.sys and one warp-wide" not in signal_contract
    assert "No partial_barrier program follows; combine waits on the counter" in (
        signal_contract
    )
    wait_contract = combine["completion_wait_contract"]
    assert "bind the same fc2_completion / fc2_completion_peers as fc2" in wait_contract
    assert (
        "fc2_completion_peers[pg_rank] == fc2_completion, checked by the reference "
        "binding"
    ) in wait_contract
    assert "push_generation (the rank-local u32[2] of publish_push / local_pack)" in (
        wait_contract
    )
    assert (
        "launched with the programmatic-stream-serialization attribute behind fc2"
        in (wait_contract)
    )
    assert (
        "fc2_completion[0] >= 16 * push_generation[0] (wrap-safe modular compare)"
        in (wait_contract)
    )
    # The combine's launch is the third PDL dependent: 128 threads, no shared
    # memory, grid 3 * tokens, use_pdl with exactly the wait instruction; fc2
    # keeps its launch (192 threads, 181248 B) and gains launch_dependents.
    assert combine["launch"]["use_pdl"] is True
    assert combine["pdl_instructions"] == ["wait"]
    assert combine["launch"]["threads_per_block"] == 128
    assert combine["launch"]["dynamic_smem_bytes"] == 0
    assert combine["launch"]["function_attributes"] == []
    assert combine["grid_formula"] == "(3 * tokens_per_rank, 1, 1)"
    assert combine["forward_index"] == 4
    assert fc2["launch"]["use_pdl"] is True
    assert fc2["pdl_instructions"] == ["wait", "launch_dependents"]
    assert fc2["launch"]["threads_per_block"] == 192
    assert fc2["launch"]["dynamic_smem_bytes"] == 181248
    assert fc2["forward_index"] == 3
    # The partial barrier left no trace in the program list.
    assert manifest["public_material_check"]["result"] == "clean"
    assert not any(
        "partial_barrier" in name
        for name in manifest["public_material_check"]["files_scanned"]
    )


def test_python_arg_tuples_match_the_manifest_device_arguments(plan_device) -> None:
    programs = _manifest_programs()
    by_stage = _stage_call_args(plan_device)
    assert set(by_stage) == set(programs)
    for stage, args in by_stage.items():
        expected = _expected_binding_kinds(programs[stage])
        actual = [_python_arg_kind(arg) for arg in args]
        assert actual == expected, (stage, actual, expected)


def _binding_run_parameters(text: str) -> list[str]:
    """Parameter declarations of ``void Run(...)`` in a binding translation unit."""

    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    match = re.search(r"\bvoid\s+Run\s*\(", text)
    assert match, "binding has no `void Run(`"
    depth, index = 1, match.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        index += 1
    body = text[match.end() : index - 1]
    parameters, current, depth = [], "", 0
    for char in body:
        if char in "<(":
            depth += 1
        elif char in ">)":
            depth -= 1
        if char == "," and depth == 0:
            parameters.append(current)
            current = ""
        else:
            current += char
    parameters.append(current)
    return [" ".join(p.split()) for p in parameters if p.strip()]


def _binding_parameter_kind(declaration: str) -> str:
    if "TensorView" in declaration:
        return "tensor"
    if "Shape" in declaration:
        return "peer_table"
    if "int64_t" in declaration:
        return "scalar"
    return f"unsupported:{declaration}"


def test_binding_sources_match_manifest_and_python_args(plan_device) -> None:
    programs = _manifest_programs()
    binding_paths = {
        stage: CSRC_DIR / f"cake_w8a8_megamoe_ep16_{stage}_binding.cu"
        for stage in programs
    }
    missing = sorted(
        stage for stage, path in binding_paths.items() if not path.is_file()
    )
    if missing:
        pytest.skip(f"hand-written bindings not present yet for: {missing}")
    by_stage = _stage_call_args(plan_device)
    for stage, path in binding_paths.items():
        text = path.read_text()
        exports = re.findall(r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*(\w+)\s*,", text)
        assert exports == [stage], (path.name, exports)
        assert STAGE_ENTRIES.get(stage, QUANTIZE_ENTRY) == stage
        # The binding repeats the manifest declaration verbatim and launches
        # the program's block size.
        assert programs[stage]["declaration"] in text, path.name
        block = re.findall(r"constexpr unsigned kBlockThreads = (\d+);", text)
        assert [int(b) for b in block] == [
            programs[stage]["launch"]["threads_per_block"]
        ], path.name
        if stage in ("fc1", "fc2"):
            # A is encoded over the k-tile-major weight payload with the
            # manifest's box height; B over the row-major activation rows.
            device_arguments = programs[stage]["arguments"]["device_arguments"]
            a_rows = int(device_arguments[0]["tma"]["box_dim"][1])
            b_rows = int(device_arguments[1]["tma"]["box_dim"][1])
            constant = {"fc1": "kW13BoxRows", "fc2": "kW2BoxRows"}[stage]
            assert f"constexpr cuuint32_t {constant} = {a_rows};" in text, path.name
            assert f'encode_weight(A, {constant}, "A")' in text, path.name
            assert f'encode_operand(B, {b_rows}, "B")' in text, path.name
            assert "encode_operand(A," not in text, path.name
        # The programmatic-stream-serialization launch attribute sits on exactly
        # the PDL dependents; the carveout function attribute on exactly the
        # program the manifest names (local_pack), with the manifest's value.
        launch = programs[stage]["launch"]
        attribute_lines = (
            "cudaLaunchAttribute attrs[1]{};",
            "attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;",
            "attrs[0].val.programmaticStreamSerializationAllowed = 1;",
            "config.attrs = attrs;",
            "config.numAttrs = 1;",
        )
        if launch["use_pdl"]:
            assert stage in cake_plan.PDL_STAGES, path.name
            for line in attribute_lines:
                assert line in text, (path.name, line)
        else:
            assert stage not in cake_plan.PDL_STAGES, path.name
            assert "cudaLaunchAttribute" not in text, path.name
            assert "config.attrs" not in text and "numAttrs" not in text, path.name
        carveout = re.findall(
            r"constexpr int kPreferredSharedMemoryCarveout = (\d+);", text
        )
        if launch["preferred_shared_memory_carveout"] is None:
            assert not carveout, path.name
            assert "PreferredSharedMemoryCarveout" not in text, path.name
        else:
            assert [int(c) for c in carveout] == [
                launch["preferred_shared_memory_carveout"]
            ], path.name
            assert (
                "cudaFuncAttributePreferredSharedMemoryCarveout, "
                "kPreferredSharedMemoryCarveout));"
            ) in text, path.name
            assert "cudaFuncSetAttribute(" in text, path.name
        parameters = _binding_run_parameters(text)
        kinds = [_binding_parameter_kind(p) for p in parameters]
        expected = _expected_binding_kinds(programs[stage])
        assert kinds == expected, (path.name, parameters, expected)
        assert parameters[-1].endswith("grid_x"), (path.name, parameters[-1])
        if expected[0] == "scalar" and not any(
            a["kind"] in ("buffer", "tma")
            for a in programs[stage]["arguments"]["device_arguments"]
        ):
            assert "device_index" in parameters[0], (path.name, parameters[0])
        assert len(by_stage[stage]) == len(parameters), (stage, by_stage[stage])


# ---------------------------------------------------------------------------
# Weights (host side).
# ---------------------------------------------------------------------------
def test_quantize_reference_rule() -> None:
    weight = torch.zeros((1, 3, 64), dtype=torch.float32)
    weight[0, 0, :32] = 448.0  # amax / 448 == 1 -> code 127, exact
    weight[0, 0, 32:] = 449.0  # just above -> next power of two, code 128
    weight[0, 1, :32] = 0.0  # zero block -> amax floor 1e-7 -> code 95
    weight[0, 1, 32:] = 1.0  # 1/448 -> 2^-8 (ceil) -> code 119
    weight[0, 2, :32] = 1e-9  # below the floor -> code 95, value 1e-9 * 2^32
    weight[0, 2, 32:] = 3.0e38  # near float32 max -> 2^120 -> code 247
    values, codes = reference_quantize_mxfp8_block32(weight)
    assert values.dtype == torch.float8_e4m3fn and tuple(values.shape) == (1, 3, 64)
    assert codes.tolist() == [[[127, 128], [95, 119], [95, 247]]]
    assert float(values[0, 0, 0]) == 448.0
    assert float(values[0, 0, 32]) == 224.0  # 449 / 2 rounded to e4m3
    assert torch.count_nonzero(values[0, 1, :32].float()) == 0
    assert float(values[0, 2, 0]) == pytest.approx(1e-9 * 2**32, rel=0.07)
    with pytest.raises(ValueError):
        reference_quantize_mxfp8_block32(torch.zeros((1, 33)))


def _program_source_row(output_row: int, gate_width: int) -> int:
    """Mode-1 ``interleave_gate_up`` row rule of the quantization program."""

    chunk, within = divmod(output_row, 256)
    if within < 128:
        return gate_width + chunk * 128 + within
    return chunk * 128 + within - 128


def test_interleave_gate_up_128_matches_the_program_row_rule() -> None:
    w13 = torch.arange(1024, dtype=torch.float32).view(1, 1024, 1)
    out = reference_interleave_gate_up_128(w13)
    assert torch.equal(out.flatten().sort().values, w13.flatten())
    # Chunk 0: up rows 0..127 (canonical 512..639), then gate rows 0..127.
    assert torch.equal(out[0, :128, 0], torch.arange(512, 640, dtype=torch.float32))
    assert torch.equal(out[0, 128:256, 0], torch.arange(0, 128, dtype=torch.float32))
    expected = [_program_source_row(row, gate_width=512) for row in range(1024)]
    assert out[0, :, 0].tolist() == expected


def _program_cp_index(
    mode: int, expert: int, row: int, block: int, rows: int, columns: int
) -> int:
    """The quantization program's ``cp_index`` for one scale byte (modes 1 and 2)."""

    tile_k, u = divmod(block, 4)
    tiles_k = columns // 128
    if mode == SCALE_MODE_W13:
        tile_m, row_in_tile = divmod(row, 256)
        d, row_in_128 = divmod(row_in_tile, 128)
        a, row32 = divmod(row_in_128, 32)
        cp_in_tile = d * 512 + row32 * 16 + a * 4 + u
        tiles_m, tile_bytes = rows // 256, 1024
    else:
        tile_m, row_in_tile = divmod(row, 128)
        a, row32 = divmod(row_in_tile, 32)
        cp_in_tile = row32 * 16 + a * 4 + u
        tiles_m, tile_bytes = rows // 128, 512
    return (
        expert * tiles_m * tiles_k * tile_bytes
        + (tile_m * tiles_k + tile_k) * tile_bytes
        + cp_in_tile
    )


@pytest.mark.parametrize(
    "mode,rows,columns",
    ((SCALE_MODE_W13, 512, 256), (SCALE_MODE_W2_SWAP, 256, 384)),
)
def test_pack_scales_cp_matches_the_program_index_rule(mode, rows, columns) -> None:
    experts, blocks = 2, columns // 32
    codes = torch.arange(experts * rows * blocks, dtype=torch.int64).view(
        experts, rows, blocks
    )
    packed = pack_scales_cp(codes, mode=mode)
    assert packed.numel() == experts * rows * columns // 32
    expected = torch.empty_like(packed)
    for expert in range(experts):
        for row in range(rows):
            for block in range(blocks):
                index = _program_cp_index(mode, expert, row, block, rows, columns)
                expected[index] = codes[expert, row, block]
    assert torch.equal(packed, expected)
    assert torch.equal(
        unpack_scales_cp(packed, mode=mode, rows=rows, columns=columns), codes
    )


@pytest.mark.parametrize(
    "mode,shape,size",
    (
        (SCALE_MODE_W13, W13_SHAPE, cake_plan.W13_SCALE_BYTES),
        (SCALE_MODE_W2_SWAP, W2_SHAPE, cake_plan.W2_SCALE_BYTES),
    ),
)
def test_pack_scales_cp_is_a_bijection_of_the_contract_planes(
    mode, shape, size
) -> None:
    experts, rows, columns = shape
    codes = torch.randint(0, 255, (experts, rows, columns // 32), dtype=torch.uint8)
    packed = pack_scales_cp(codes, mode=mode)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (size,)
    assert size == cake_plan.scale_bytes(rows, columns)
    assert torch.equal(
        unpack_scales_cp(packed, mode=mode, rows=rows, columns=columns), codes
    )
    assert torch.equal(
        torch.bincount(packed.int(), minlength=256),
        torch.bincount(codes.flatten().int(), minlength=256),
    )


def test_pack_scales_cp_rejects_other_modes_and_shapes() -> None:
    codes = torch.zeros((1, 256, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="mode"):
        pack_scales_cp(codes, mode=0)
    with pytest.raises(ValueError, match="rows"):
        pack_scales_cp(torch.zeros((1, 128, 4), dtype=torch.uint8), mode=SCALE_MODE_W13)
    with pytest.raises(ValueError, match="blocks"):
        pack_scales_cp(torch.zeros((1, 256, 3), dtype=torch.uint8), mode=SCALE_MODE_W13)
    with pytest.raises(ValueError):
        unpack_scales_cp(
            torch.zeros(1023, dtype=torch.uint8),
            mode=SCALE_MODE_W13,
            rows=256,
            columns=128,
        )


def test_weight_pack_validation() -> None:
    cpu_w13 = (
        torch.zeros(W13_SHAPE[1:], dtype=torch.bfloat16).unsqueeze(0).expand(32, -1, -1)
    )
    with pytest.raises(NotImplementedError):
        validate_canonical_weights(
            PrequantizedMoEWeights(
                w13=cpu_w13,
                w2=cpu_w13,
                w13_scale=torch.zeros(1),
                w2_scale=torch.zeros(1),
            )
        )
    with pytest.raises(TypeError):
        validate_canonical_weights(SimpleNamespace(w13=cpu_w13, w2=cpu_w13))
    with pytest.raises(MoEEpConfigError, match="shape"):
        validate_canonical_weights(
            UnquantizedMoEWeights(
                w13=torch.zeros((32, 8, 8), dtype=torch.bfloat16),
                w2=torch.zeros((32, 8, 8), dtype=torch.bfloat16),
            )
        )
    assert isinstance(MoEWeightPack(cpu_w13, cpu_w13), UnquantizedMoEWeights)
    with pytest.raises(MoEEpConfigError, match="TransformedCakeW8a8MegaWeights"):
        validate_transformed_mega_weights(("w13", "w2"))
    # The layout tags are checked before the tensors: a row-major or otherwise
    # tagged payload is rejected on the host, whatever the tensors' device.
    tagged = _fake_weights()
    assert (tagged.w13_layout, tagged.w2_layout) == (W13_FP8_LAYOUT, W2_FP8_LAYOUT)
    for field, value in (
        ("w13_layout", "row_major"),
        ("w13_layout", W2_FP8_LAYOUT),
        ("w2_layout", "row_major"),
        ("w2_layout", W13_FP8_LAYOUT),
        ("w2_layout", None),
    ):
        with pytest.raises(MoEEpConfigError, match=f"{field} must be"):
            validate_transformed_mega_weights(
                dataclasses.replace(tagged, **{field: value})
            )
    # Untagged construction is a TypeError: the fields have no default.
    with pytest.raises(TypeError):
        TransformedCakeW8a8MegaWeights(  # type: ignore[call-arg]
            w13=tagged.w13,
            w13_scales=tagged.w13_scales,
            w2=tagged.w2,
            w2_scales_swap=tagged.w2_scales_swap,
        )
    # Correctly tagged CPU stand-ins reach the tensor checks (CUDA required).
    with pytest.raises(MoEEpConfigError, match="CUDA tensor"):
        validate_transformed_mega_weights(tagged)


@pytest.mark.parametrize(
    "box_rows,rows,columns",
    ((W13_KTILE_ROWS, 512, 256), (W2_KTILE_ROWS, 256, 384)),
)
def test_ktile_major_payload_reference_matches_the_byte_formula(
    box_rows, rows, columns
) -> None:
    """The pure-Torch permutation equals the manifest's byte placement rule
    (((e * (rows / R) + r / R) * (K / 128) + k / 128) * R + r % R) * 128 + k % 128
    and round-trips through its inverse."""

    experts = 2
    q = torch.arange(experts * rows * columns, dtype=torch.int32).view(
        experts, rows, columns
    )
    payload = reference_ktile_major_payload(q, box_rows=box_rows)
    assert tuple(payload.shape) == (experts, rows, columns)
    assert payload.dtype == q.dtype and payload.is_contiguous()
    e, r, k = torch.meshgrid(
        torch.arange(experts), torch.arange(rows), torch.arange(columns), indexing="ij"
    )
    index = (
        ((e * (rows // box_rows) + r // box_rows) * (columns // 128) + k // 128)
        * box_rows
        + r % box_rows
    ) * 128 + k % 128
    assert torch.equal(payload.flatten()[index], q)
    # A permutation: every byte lands exactly once.
    assert torch.equal(payload.flatten().sort().values, q.flatten())
    # One R x 128 block is contiguous in the payload.
    assert torch.equal(
        payload.flatten()[: box_rows * 128].view(box_rows, 128), q[0, :box_rows, :128]
    )
    assert torch.equal(reference_row_major_payload(payload, box_rows=box_rows), q)
    for bad in (q[:, : rows - 1], q[..., : columns - 1], q[0]):
        with pytest.raises(ValueError):
            reference_ktile_major_payload(bad, box_rows=box_rows)
    with pytest.raises(ValueError):
        reference_ktile_major_payload(q, box_rows=rows + box_rows)
    # The contract weights tile exactly under their box heights.
    assert W13_SHAPE[1] % W13_KTILE_ROWS == 0 and W13_SHAPE[2] % 128 == 0
    assert W2_SHAPE[1] % W2_KTILE_ROWS == 0 and W2_SHAPE[2] % 128 == 0


# ---------------------------------------------------------------------------
# JIT spec.
# ---------------------------------------------------------------------------
def _fake_csrc(tmp_path: Path, stages=("publish_push", "combine")) -> Path:
    root = tmp_path / "csrc" / "cake_w8a8_megamoe_ep16"
    root.mkdir(parents=True)
    for stage in stages:
        (root / f"cake_w8a8_megamoe_ep16_{stage}_kernel.cu").write_text(f"// {stage}\n")
        (root / f"cake_w8a8_megamoe_ep16_{stage}_binding.cu").write_text(
            f"// {stage} binding\n"
        )
    return root


def test_source_discovery_pairs_kernels_and_bindings(tmp_path) -> None:
    root = _fake_csrc(tmp_path)
    sources = cake_jit.discover_sources(root)
    assert [p.name for p in sources] == [
        "cake_w8a8_megamoe_ep16_combine_kernel.cu",
        "cake_w8a8_megamoe_ep16_publish_push_kernel.cu",
        "cake_w8a8_megamoe_ep16_combine_binding.cu",
        "cake_w8a8_megamoe_ep16_publish_push_binding.cu",
    ]
    (root / "cake_w8a8_megamoe_ep16_fc1_kernel.cu").write_text("// unpaired\n")
    with pytest.raises(FileNotFoundError, match="pair up"):
        cake_jit.discover_sources(root)
    with pytest.raises(FileNotFoundError):
        cake_jit.discover_sources(tmp_path / "missing")
    assert cake_jit.available(tmp_path / "missing") is False


def test_module_name_is_target_plus_content_digest(tmp_path) -> None:
    root = _fake_csrc(tmp_path)
    sources = cake_jit.discover_sources(root)
    name = cake_jit.module_name(sources)
    assert re.fullmatch(r"cake_w8a8_megamoe_ep16_sm_103a_[0-9a-f]{20}", name)
    (root / "cake_w8a8_megamoe_ep16_combine_kernel.cu").write_text("// changed\n")
    assert cake_jit.module_name(cake_jit.discover_sources(root)) != name


@pytest.mark.parametrize(
    "arch_list,expected",
    (
        ("10.3a", ((10, 3),)),
        ("10.0a 10.3a", ((10, 3),)),
        ("10.0a", ()),
        ("9.0a 12.0f", ()),
    ),
)
def test_loader_targets_follow_flashinfer_cuda_arch_list(
    monkeypatch, arch_list, expected
) -> None:
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", arch_list)
    assert cake_jit.supported_capabilities() == expected
    if expected:
        assert cake_jit.require_supported_capability((10, 3)) == (10, 3)
    else:
        with pytest.raises(RuntimeError, match="FLASHINFER_CUDA_ARCH_LIST"):
            cake_jit.require_supported_capability((10, 3))
    with pytest.raises(RuntimeError, match="requires compute capability 10.3"):
        cake_jit.require_supported_capability((10, 0))


def test_jit_spec_flags_and_sources(monkeypatch, tmp_path) -> None:
    from flashinfer.jit.core import sm103a_nvcc_flags

    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.3a")
    root = _fake_csrc(tmp_path)
    spec = cake_jit.gen_cake_w8a8_megamoe_ep16_spec(root)
    assert spec.name == cake_jit.module_name(cake_jit.discover_sources(root))
    assert spec.name.startswith("cake_w8a8_megamoe_ep16_sm_103a_")
    assert [Path(s).name for s in spec.sources] == [
        p.name for p in cake_jit.discover_sources(root)
    ]
    for flag in sm103a_nvcc_flags:
        assert flag in spec.extra_cuda_cflags
    assert any("compute_103a" in flag for flag in spec.extra_cuda_cflags)
    assert not any("compute_100a" in flag for flag in spec.extra_cuda_cflags)
    assert "-lcuda" in spec.extra_ldflags
    assert cake_jit.USE_FAST_MATH is True
    assert cake_jit.available(root) is True
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a")
    assert cake_jit.available(root) is False
    with pytest.raises(RuntimeError, match="FLASHINFER_CUDA_ARCH_LIST"):
        cake_jit.gen_cake_w8a8_megamoe_ep16_spec(root)


def test_jit_resolves_flashinfer_headers() -> None:
    header_dirs = cake_jit._get_flashinfer_header_dirs()
    assert any((path / "tvm_ffi_utils.h").is_file() for path in header_dirs)


# ---------------------------------------------------------------------------
# License headers and hook hygiene of the hand-written files.
# ---------------------------------------------------------------------------
PY_LICENSE_HEAD = '"""\nCopyright (c) 2026 by FlashInfer team.\n'
CU_LICENSE_HEAD = "/*\n * Copyright (c) 2026 by FlashInfer team.\n"


@pytest.mark.parametrize("path", HAND_WRITTEN_FILES, ids=lambda p: p.name)
def test_hand_written_files_carry_the_license_header(path: Path) -> None:
    assert path.is_file(), path
    text = path.read_text()
    if path.suffix == ".py":
        assert text.startswith(PY_LICENSE_HEAD), path
    elif path.suffix == ".cu":
        assert text.startswith(CU_LICENSE_HEAD), path
    assert text.endswith("\n") and "\t" not in text and "\r" not in text, path
    assert not any(line != line.rstrip() for line in text.splitlines()), path


def test_new_file_names_start_with_cake() -> None:
    for path in PACKAGE_DIR.glob("*"):
        if path.name in ("__init__.py", "README.md", "csrc", "__pycache__"):
            continue
        assert path.name.startswith("cake_"), path


def test_core_touch_points_are_explicit() -> None:
    kernel_init = (
        REPO_ROOT / "flashinfer/moe_ep/backends/mega/kernel/__init__.py"
    ).read_text()
    assert "sm103" in kernel_init
    sm103_init = (
        REPO_ROOT / "flashinfer/moe_ep/backends/mega/kernel/sm103/__init__.py"
    ).read_text()
    assert "flashinfer.experimental.cake_w8a8_megamoe_ep16" in sm103_init
    aot = (REPO_ROOT / "flashinfer/aot.py").read_text()
    assert "cake_w8a8_megamoe_ep16" not in aot
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert '"flashinfer.experimental.cake_w8a8_megamoe_ep16"' in pyproject
    # The generated programs are clang-format fixed points under the repository
    # style, so the hooks run on them unchanged: no exclusion is registered.
    precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    assert "cake_w8a8_megamoe_ep16" not in precommit
    # User-facing material reaches the overflow flag through the public helper.
    for path in (
        PACKAGE_DIR / "README.md",
        REPO_ROOT / "examples" / "experimental" / "cake_w8a8_megamoe_ep16.py",
        REPO_ROOT / "benchmarks" / "moe_ep" / "bench_cake_w8a8_megamoe_ep16.py",
    ):
        text = path.read_text()
        assert "routing_overflow(" in text, path
        assert "._kernel" not in text and "._workspace" not in text, path


def test_core_import_keeps_the_experimental_surface_lightweight() -> None:
    """``import flashinfer.moe_ep`` registers the backend without loading the
    JIT loader (nor tvm-ffi, unless core itself already loaded it)."""

    # Static: no module-level tvm-ffi / NVSHMEM / JIT import anywhere on the
    # import path of the package (``cake_jit`` is the JIT loader itself and is
    # only imported inside functions).
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name == "cake_jit.py":
            continue
        for line in path.read_text().splitlines():
            match = re.match(r"(?:import|from)\s+([\w.]+)", line)
            if match is None:
                continue  # not a module-level import statement
            parts = set(match.group(1).split("."))
            assert not parts & {"tvm_ffi", "nvshmem", "cake_jit", "jit"}, (path, line)
    script = textwrap.dedent(
        """
        import json, sys
        import flashinfer
        before = set(sys.modules)
        import flashinfer.moe_ep
        from flashinfer.moe_ep.core.kernel.registry import _available_mega_kernels
        prefix = "flashinfer.experimental.cake_w8a8_megamoe_ep16"
        print(json.dumps({
            "tvm_ffi_before": "tvm_ffi" in before,
            "tvm_ffi_after": "tvm_ffi" in sys.modules,
            "nvshmem_before": any(m.split(".")[0] == "nvshmem" for m in before),
            "nvshmem_after": any(m.split(".")[0] == "nvshmem" for m in sys.modules),
            "cake_modules": sorted(m for m in sys.modules if m.startswith(prefix)),
            "registered": "sm103_mxfp8_mxfp8_bf16_cake" in _available_mega_kernels(),
        }))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "FLASHINFER_DISABLE_VERSION_CHECK": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    prefix = "flashinfer.experimental.cake_w8a8_megamoe_ep16"
    assert report["registered"] is True
    assert f"{prefix}.cake_backend" in report["cake_modules"]
    assert f"{prefix}.cake_jit" not in report["cake_modules"], report
    if not report["tvm_ffi_before"]:
        assert not report["tvm_ffi_after"], report
    if not report["nvshmem_before"]:
        assert not report["nvshmem_after"], report


# ---------------------------------------------------------------------------
# Backend host paths on a CPU workspace: the five-launch forward with no
# host memset, launch-plan release, the public overflow helper.
# ---------------------------------------------------------------------------
def _cpu_inputs(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_states=torch.randn((n, 3072)).to(torch.bfloat16),
        topk_ids=torch.arange(n * 8, dtype=torch.int64).view(n, 8) % 512,
        topk_weights=torch.rand((n, 8), dtype=torch.float32),
        num_tokens=n,
    )


class _RecordingEntry:
    """Stand-in for a tvm-ffi stage entry: records its name (and arguments)
    when launched."""

    def __init__(self, name: str, log: list, calls: dict | None = None) -> None:
        self.name = name
        self.log = log
        self.calls = calls

    def __call__(self, *args) -> None:
        self.log.append(self.name)
        if self.calls is not None:
            self.calls.setdefault(self.name, []).append(args)


def _recording_entries(log: list, calls: dict | None = None) -> dict:
    return {stage: _RecordingEntry(stage, log, calls) for stage in cake_plan.STAGES}


def test_compute_launches_the_five_stages_and_nothing_else(
    sm103_host, plan_device, monkeypatch
) -> None:
    """compute() launches exactly publish_push, local_pack, fc1, fc2, combine;
    the host clears nothing (publish_push zeroes the combine read window and
    resets the routing state on the device; the inbox flags, the fc2
    completion counter and the generation words are never touched by the
    host), and staging itself launches nothing."""

    from flashinfer.experimental.cake_w8a8_megamoe_ep16 import cake_backend

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        cake_backend, "validate_transformed_mega_weights", lambda tw, device=None: None
    )
    ws = _cpu_workspace(rank=3)
    log: list = []
    calls: dict = {}
    ws.entries = _recording_entries(log, calls)
    terms = ws.local_terms.local
    terms.fill_(1.0)
    ws.expert_counts.fill_(5)
    ws.route_map.fill_(9)
    ws.overflow.fill_(2)
    flags = ws.push_flags.local.view(torch.int32)
    flags.fill_(4)
    generation = ws.push_generation.view(torch.int32)
    generation.fill_(6)
    completion = ws.fc2_completion.local.view(torch.int32)
    completion.fill_(96)
    n, bound = 5, 16
    t = _cpu_inputs(n)
    sm103_host.set_compile_tokens_per_rank(ws, bound)
    sm103_host.stage_inputs(t, ws, quantize_input=True)
    assert ws.staged_bound == bound and ws.staged_tokens == n
    assert log == []  # staging launches nothing
    assert torch.equal(ws.hidden[:n], t.hidden_states)
    assert torch.all(ws.ids[n:bound] == MASKED_ROUTE)
    out = sm103_host.compute(ws, _fake_weights(), output=None)
    assert out.shape == (n, 3072)
    assert log == [
        "publish_push",
        "local_pack",
        "fc1",
        "fc2",
        "combine",
    ]
    assert log == list(cake_plan.STAGES)
    # No host-side memset: the term buffer, the routing state, the inbox flags,
    # the completion counter and the generation words are exactly as the
    # (recording) programs left them (the counter is monotonic and never reset).
    assert torch.all(terms == 1.0)
    assert torch.all(ws.expert_counts == 5) and torch.all(ws.route_map == 9)
    assert int(ws.overflow.item()) == 2
    assert torch.all(flags == 4) and torch.all(generation == 6)
    assert torch.all(completion == 96)
    # The publish launch carries the buffers it resets, the generation words,
    # the bound, the process group and the push buffers; local_pack shares them.
    (publish_args,) = calls["publish_push"]
    assert publish_args[3] is terms
    assert publish_args[4] is ws.expert_counts
    assert publish_args[5] is ws.route_map
    assert publish_args[6] is ws.overflow
    assert publish_args[7] is ws.push_generation
    assert publish_args[8:11] == (bound, 16, 3) and publish_args[-1] == bound
    assert publish_args[12] is ws.push_inbox.local
    assert publish_args[14] is ws.push_flags.local
    (pack_args,) = calls["local_pack"]
    assert pack_args[7] is ws.push_generation and pack_args[-1] == 16 * bound
    assert pack_args[8:13] == (bound, 32, 1, 16, 3)
    assert pack_args[14] is ws.push_inbox.local
    assert pack_args[16] is ws.push_flags.local
    # fc2 signals into the counter, combine waits on the same counter and reads
    # the generation word local_pack advances.
    (fc2_args,) = calls["fc2"]
    assert fc2_args[16] is terms
    assert fc2_args[18] is ws.fc2_completion.local and fc2_args[-1] == 128
    (combine_args,) = calls["combine"]
    assert combine_args[0] is terms
    assert combine_args[2] is ws.push_generation
    assert combine_args[3:7] == (bound, 3, 16, 3)
    assert combine_args[8] is ws.fc2_completion.local
    assert combine_args[-1] == 3 * bound
    # Without a compile bound the bound is the capacity of the workspace.
    log.clear()
    calls.clear()
    sm103_host.set_compile_tokens_per_rank(ws, None)
    sm103_host.stage_inputs(t, ws, quantize_input=True)
    assert ws.staged_bound == 64
    sm103_host.compute(ws, _fake_weights(), output=None)
    assert log == list(cake_plan.STAGES)
    assert calls["publish_push"][0][8:11] == (64, 16, 3)
    assert calls["publish_push"][0][-1] == 64
    assert calls["local_pack"][0][-1] == 16 * 64
    assert calls["fc2"][0][9] == 64 * 8  # routes_per_rank
    assert calls["combine"][0][3:5] == (64, 3) and calls["combine"][0][-1] == 192
    assert torch.all(terms == 1.0)
    assert torch.all(flags == 4) and torch.all(generation == 6)
    assert torch.all(completion == 96)
    with pytest.raises(MoEEpConfigError, match="prequantized"):
        sm103_host.stage_inputs(t, ws, quantize_input=False)


def test_compute_orders_staging_before_the_first_launch(
    sm103_host, plan_device, monkeypatch
) -> None:
    """Host-side staging runs before publish_push; between the five launches
    there is no host step."""

    from flashinfer.experimental.cake_w8a8_megamoe_ep16 import cake_backend

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        cake_backend, "validate_transformed_mega_weights", lambda tw, device=None: None
    )
    order: list = []
    monkeypatch.setattr(
        cake_backend,
        "_stage_inputs",
        lambda *args, **kwargs: order.append(("stage", kwargs["bound"])) or 5,
    )
    ws = _cpu_workspace(rank=7)
    ws.entries = _recording_entries(order)
    sm103_host.set_compile_tokens_per_rank(ws, 32)
    sm103_host.stage_inputs(_cpu_inputs(5), ws, quantize_input=True)
    assert order == [("stage", 32)]
    assert ws.staged_tokens == 5 and ws.staged_bound == 32
    sm103_host.compute(ws, _fake_weights(), output=None)
    assert order == [
        ("stage", 32),
        "publish_push",
        "local_pack",
        "fc1",
        "fc2",
        "combine",
    ]
    with pytest.raises(MoEEpConfigError, match="stage_inputs"):
        sm103_host.compute(_cpu_workspace(rank=7), None, output=None)


def test_destroying_a_layer_releases_its_launch_plans(
    sm103_host, plan_device, monkeypatch
) -> None:
    from flashinfer.experimental.cake_w8a8_megamoe_ep16 import cake_backend

    # The CPU stand-in weights fail the CUDA-tensor check of the real validator
    # (covered by test_weight_pack_validation); the plan cache is under test here.
    monkeypatch.setattr(
        cake_backend, "validate_transformed_mega_weights", lambda tw, device=None: None
    )
    ws = _cpu_workspace(rank=3)
    other = Sm103Mxfp8Mxfp8CakeMegaKernelBackend(
        Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig()
    )
    tw_a, tw_b = _fake_weights(), _fake_weights()
    calls = sm103_host._launch_plan(ws, tw_a, 16)
    assert [entry.name for entry, _ in calls] == list(cake_plan.STAGES)
    assert sm103_host._launch_plan(ws, tw_a, 16) is calls  # cached per (weights, bound)
    sm103_host._launch_plan(ws, tw_a, 64)
    other._launch_plan(ws, tw_b, 16)
    assert set(ws.launch_plans) == {(id(tw_a), 16), (id(tw_a), 64), (id(tw_b), 16)}
    # Releasing one layer's workspace reference drops only its entries.
    sm103_host._forget_local_workspace_state(ws)
    assert set(ws.launch_plans) == {(id(tw_b), 16)}
    assert ws.launch_plans[(id(tw_b), 16)][1] is tw_b
    sm103_host._forget_local_workspace_state(ws)  # idempotent
    sm103_host._forget_local_workspace_state(object())  # unknown workspace: no-op
    other._forget_local_workspace_state(ws)
    assert not ws.launch_plans
    with pytest.raises(MoEEpConfigError, match="TransformedCakeW8a8MegaWeights"):
        sm103_host._launch_plan(ws, ("w13", "w2"), 16)


def test_routing_overflow_reads_the_staged_workspace(sm103_host) -> None:
    from flashinfer.experimental import cake_w8a8_megamoe_ep16 as package

    ws = _cpu_workspace()
    ws.overflow.fill_(3)
    layer = SimpleNamespace(_staged_kernel=sm103_host, _staged_workspace=ws)
    assert routing_overflow(layer) is ws.overflow
    assert int(routing_overflow(layer).item()) == 3
    assert package.routing_overflow is routing_overflow
    assert "routing_overflow" in package.__all__
    with pytest.raises(RuntimeError, match="staged forward"):
        routing_overflow(SimpleNamespace(_staged_kernel=None, _staged_workspace=None))
    with pytest.raises(TypeError, match=KERNEL_NAME):
        routing_overflow(SimpleNamespace(_staged_kernel=object(), _staged_workspace=ws))
    ws.destroyed = True
    with pytest.raises(RuntimeError, match="destroyed"):
        routing_overflow(layer)


# ---------------------------------------------------------------------------
# 16-rank GPU tests (skip unless torchrun WORLD_SIZE == 16 on sm_103).
# ---------------------------------------------------------------------------
WORLD = 16
LOCAL_EXPERTS = 32
EXPERTS = 512
HIDDEN = 3072
INTERMEDIATE = 5120
TOP_K = 8


def _ep16_device():
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD:
        pytest.skip("requires torchrun with exactly 16 ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = torch.cuda.device_count()
    device_index = 0 if visible == 1 else local_rank
    if device_index >= visible:
        pytest.skip("requires one visible CUDA device per local rank")
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (10, 3):
        pytest.skip("requires SM103")
    return device


@pytest.fixture(scope="module")
def ep16():
    device = _ep16_device()
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=20))
    if dist.get_world_size() != WORLD:
        pytest.skip("requires an EP16 process group")
    yield SimpleNamespace(device=device, rank=dist.get_rank())
    if initialized_here:
        dist.barrier()
        dist.destroy_process_group()


def _contract_weights(rank: int, device: torch.device, seed: int = 13007):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    w13 = torch.randn(
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w13.mul_(HIDDEN**-0.5)
    w2 = torch.randn(
        (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(INTERMEDIATE**-0.5)
    return w13.contiguous(), w2.contiguous()


def _make_layer(ep16, w13, w2, max_tokens_per_rank: int = 64) -> MoEEpMegaLayer:
    return MoEEpMegaLayer(
        bootstrap=BootstrapConfig(
            world_size=WORLD, rank=ep16.rank, device=ep16.device.index
        ),
        fleet_params=FleetParams(
            num_experts=EXPERTS,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=HIDDEN,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(megakernel=Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig()),
    )


def _inputs(ep16, tokens: int, topk_ids: torch.Tensor, seed: int = 7) -> MoEEpTensors:
    generator = torch.Generator(device=ep16.device).manual_seed(seed + ep16.rank)
    hidden = torch.randn(
        (tokens, HIDDEN), dtype=torch.bfloat16, device=ep16.device, generator=generator
    )
    logits = torch.randn(
        (tokens, TOP_K), dtype=torch.float32, device=ep16.device, generator=generator
    )
    return MoEEpTensors(
        hidden_states=hidden.contiguous(),
        topk_ids=topk_ids.to(torch.int64).contiguous(),
        topk_weights=torch.softmax(logits, dim=-1).contiguous(),
    )


def _spread_routing(ep16, tokens: int) -> torch.Tensor:
    token = torch.arange(tokens, dtype=torch.int64, device=ep16.device).unsqueeze(1)
    slot = torch.arange(TOP_K, dtype=torch.int64, device=ep16.device).unsqueeze(0)
    return (token * TOP_K + slot + ep16.rank * TOP_K) % EXPERTS


def _random_routing_within_capacity(ep16, tokens: int, seed: int) -> torch.Tensor:
    """Distinct top-8 per token on every rank; every expert's global load <= 32."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    load = torch.zeros(EXPERTS, dtype=torch.int64)
    rows = []
    for _rank in range(WORLD):
        rank_rows = []
        for _token in range(tokens):
            chosen = []
            order = torch.randperm(EXPERTS, generator=generator).tolist()
            for expert in order:
                if load[expert] < 32:
                    chosen.append(expert)
                    load[expert] += 1
                if len(chosen) == TOP_K:
                    break
            rank_rows.append(chosen)
        rows.append(rank_rows)
    assert int(load.max()) <= 32
    return torch.tensor(rows[ep16.rank], dtype=torch.int64, device=ep16.device)


def _hotset_routing(ep16, tokens: int, hot_expert: int, hot_load: int) -> torch.Tensor:
    """Spread routing with exactly ``hot_load`` global routes moved onto ``hot_expert``."""

    ids = _spread_routing(ep16, tokens).clone()
    # Move the hot expert's spread routes eight experts up (outside this row's
    # eight consecutive ids) so no row carries a duplicate expert.
    ids = torch.where(ids == hot_expert, (ids + 8) % EXPERTS, ids)
    route_index = torch.arange(tokens, device=ep16.device) * WORLD + ep16.rank
    hot_rows = route_index < hot_load
    slot_free = torch.arange(TOP_K, device=ep16.device).unsqueeze(0) == 0
    ids = torch.where(
        hot_rows.unsqueeze(1) & slot_free, torch.full_like(ids, hot_expert), ids
    )
    return ids


def _all_gather_rows(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(WORLD)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.stack(gathered)


def _oracle(
    ep16, t: MoEEpTensors, w13: torch.Tensor, w2: torch.Tensor, bound: int
) -> torch.Tensor:
    """Pure-Torch block-32 MXFP8 reference (same construction as the FlashInfer sm100 oracle)."""

    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl import (
        weights as wh,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        compute_megamoe_reference_mxfp8,
        mxfp8_quantize_per_block_32,
    )

    tokens = t.num_tokens
    hidden = torch.zeros((bound, HIDDEN), dtype=torch.bfloat16, device=ep16.device)
    ids = torch.full((bound, TOP_K), -1, dtype=torch.int64, device=ep16.device)
    weights = torch.zeros((bound, TOP_K), dtype=torch.float32, device=ep16.device)
    hidden[:tokens] = t.hidden_states
    ids[:tokens] = t.topk_ids
    weights[:tokens] = t.topk_weights
    gathered_hidden = _all_gather_rows(hidden).reshape(WORLD * bound, HIDDEN)
    gathered_ids = _all_gather_rows(ids).reshape(WORLD * bound, TOP_K)
    gathered_weights = _all_gather_rows(weights).reshape(WORLD * bound, TOP_K)
    activation_q, activation_sf = mxfp8_quantize_per_block_32(
        gathered_hidden.float(), torch.float8_e4m3fn
    )
    fc1_logical = wh._fc1_weight_from_w13(w13, intermediate_size=INTERMEDIATE)
    fc1_q, fc1_sf, fc2_q, fc2_sf = [], [], [], []
    for expert in range(LOCAL_EXPERTS):
        q, sf = wh._quantize_mxfp8_weight_k_major(
            fc1_logical[expert].transpose(0, 1), kind="mxfp8_e4m3"
        )
        fc1_q.append(q.transpose(0, 1))
        fc1_sf.append(sf)
        q, sf = wh._quantize_mxfp8_weight_k_major(w2[expert], kind="mxfp8_e4m3")
        fc2_q.append(q.transpose(0, 1))
        fc2_sf.append(sf)
    offset = ep16.rank * LOCAL_EXPERTS
    owned = (gathered_ids >= offset) & (gathered_ids < offset + LOCAL_EXPERTS)
    local_ids = torch.where(
        owned, gathered_ids - offset, torch.full_like(gathered_ids, -1)
    )
    local_terms = compute_megamoe_reference_mxfp8(
        input_activation=activation_q.unsqueeze(0),
        input_activation_sf=activation_sf.unsqueeze(0),
        input_topk_idx=local_ids.unsqueeze(0),
        input_topk_weights=gathered_weights.unsqueeze(0),
        fc1_weight=torch.stack(fc1_q).unsqueeze(0),
        fc1_weight_sf=torch.stack(fc1_sf).unsqueeze(0),
        fc2_weight=torch.stack(fc2_q).unsqueeze(0),
        fc2_weight_sf=torch.stack(fc2_sf).unsqueeze(0),
        ab_dtype=torch.float8_e4m3fn,
        gate_up_clamp=None,
        apply_topk_in_fc1=True,
    )[0]
    received = torch.empty_like(local_terms)
    dist.all_to_all_single(received, local_terms.contiguous())
    received = received.reshape(WORLD, bound, TOP_K, HIDDEN)
    combined = torch.zeros((bound, HIDDEN), dtype=torch.float32, device=ep16.device)
    for route in range(TOP_K):
        for owner in range(WORLD):
            combined.add_(received[owner, :, route, :].float())
    return combined.to(torch.bfloat16)[:tokens]


def _check(output: torch.Tensor, expected: torch.Tensor) -> None:
    """``output`` matches the oracle: atol=rtol=0.1 elementwise, rel-L2 <= 0.02.

    The failure message carries the mismatch statistics and rows, so one
    rank's report localizes the error without the other ranks' logs.
    """

    where = f"test line {inspect.currentframe().f_back.f_lineno}"  # type: ignore[union-attr]
    assert tuple(output.shape) == tuple(expected.shape), (where, output.shape)
    assert output.dtype == torch.bfloat16, where
    out = output.float()
    ref = expected.float()
    diff = (out - ref).abs()
    finite = bool(torch.isfinite(out).all().item())
    rel_l2 = float((diff.norm() / ref.norm().clamp_min(1e-6)).item())
    bad = ~(diff <= 0.1 + 0.1 * ref.abs())
    if not finite or bool(bad.any().item()) or rel_l2 > 0.02:
        rows = torch.nonzero(bad.any(dim=1)).flatten().tolist()
        raise AssertionError(
            f"{where}: finite={finite} max|diff|={float(diff.max().item()):.4f} "
            f"rel_l2={rel_l2:.4f} mismatched rows {len(rows)}/{output.shape[0]}"
            f" {rows[:16]}"
        )


def _all_ranks(flag: bool, device: torch.device) -> bool:
    t = torch.tensor(1 if flag else 0, dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(int(t.item()))


def _counter_words(tensor: torch.Tensor) -> list[int]:
    """Host copy of a device u32 [2] word pair after a stream synchronize."""

    torch.cuda.synchronize(tensor.device)
    # Read the words as int32 (the values stay far below 2^31 here).
    return tensor.view(torch.int32).to(torch.int64).tolist()


def _check_completion(ep16, ws, forwards: int) -> None:
    """After ``forwards`` executed forwards, every rank's fc2 completion
    counter reads ``[16 * forwards, 0]`` (word 0 took one add from every rank
    per forward and is never reset; word 1 returned to 0) and the generation
    word local_pack advances reads ``forwards``. The check is collective: the
    maximum over the ranks must equal this rank's value."""

    where = f"test line {inspect.currentframe().f_back.f_lineno}"  # type: ignore[union-attr]
    counter = _counter_words(ws.fc2_completion.local)
    generation = _counter_words(ws.push_generation)
    assert counter == [16 * forwards, 0], (where, counter, forwards)
    assert generation == [forwards, 0], (where, generation, forwards)
    assert _any_rank_max(counter[0], ep16.device) == 16 * forwards, where
    assert _all_ranks(True, ep16.device)


def _check_counted_combine_matches_terms(
    ep16, ws, output: torch.Tensor, bound: int
) -> None:
    """The counted combine's output is bit-identical to the reference
    combine of this rank's term window: the eight route rows of each token
    added in route order in FP32 (starting from zero), rounded to bf16 once.
    Read after a synchronize and before the next forward (whose publish_push
    zeroes the window)."""

    where = f"test line {inspect.currentframe().f_back.f_lineno}"  # type: ignore[union-attr]
    torch.cuda.synchronize(ep16.device)
    window = ws.local_terms.local[ep16.rank * bound : (ep16.rank + 1) * bound]
    assert tuple(window.shape) == (bound, TOP_K, HIDDEN), where
    accum = torch.zeros((bound, HIDDEN), dtype=torch.float32, device=ep16.device)
    for route_slot in range(TOP_K):
        accum += window[:, route_slot, :].float()
    reference = accum.to(torch.bfloat16)[: output.shape[0]]
    assert torch.equal(output, reference), (
        where,
        int((output != reference).sum().item()),
    )


def _any_rank_max(value: int, device: torch.device) -> int:
    t = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def _say(ep16, message: str) -> None:
    """Rank-tagged progress line (visible with ``pytest -s``)."""

    print(f"[ep16 rank {ep16.rank}] {message}", file=sys.stderr, flush=True)


def _fail_fast(ep16, exc: BaseException) -> None:
    """Report this rank's failure now and leave the job.

    The EP16 tests are collective. A rank that stops early desynchronizes its
    peers' next collective, and a collective teardown would then hang, so the
    failure would surface only as a timeout without a traceback. Print the
    traceback and exit this rank with a non-zero status; the launcher
    (``srun --kill-on-bad-exit=1``, ``torchrun``) stops the remaining ranks.
    """

    _say(ep16, f"FAILED: {type(exc).__name__}: {exc}")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(1)


def test_ep16_correctness_against_the_mxfp8_oracle(ep16) -> None:
    w13, w2 = _contract_weights(ep16.rank, ep16.device)
    layer = _make_layer(ep16, w13, w2)
    try:
        layer.warmup()
        _say(ep16, "warmup ok")
        ws = layer._workspace
        # Forwards executed on the pooled workspace so far (the warmup is one);
        # the fc2 completion counter must read 16 * forwards on every rank.
        _check_completion(ep16, ws, forwards=1)
        _say(ep16, "fc2 completion counter after the warmup: [16, 0]")

        def overflow() -> int:
            return _any_rank_max(int(routing_overflow(layer).item()), ep16.device)

        # 1. spread routing at 16/32/64 tokens (warmup pattern).
        for tokens in (16, 32, 64):
            t = _inputs(ep16, tokens, _spread_routing(ep16, tokens))
            out = layer.forward(t).clone()
            _check_counted_combine_matches_terms(ep16, ws, out, bound=64)
            _check(out, _oracle(ep16, t, w13, w2, bound=64))
            assert overflow() == 0
            _say(ep16, f"stage 1 ok: spread routing, {tokens} tokens")
        _check_completion(ep16, ws, forwards=4)
        # 2. random top-8 within capacity by construction.
        t = _inputs(
            ep16, 64, _random_routing_within_capacity(ep16, 64, seed=11), seed=19
        )
        _check(layer.forward(t), _oracle(ep16, t, w13, w2, bound=64))
        _say(ep16, "stage 2 ok: random routing within capacity")
        # 3. hotset skew: one expert at exactly 32 routes.
        t = _inputs(
            ep16, 32, _hotset_routing(ep16, 32, hot_expert=77, hot_load=32), seed=23
        )
        _check(layer.forward(t), _oracle(ep16, t, w13, w2, bound=64))
        assert overflow() == 0
        _say(ep16, "stage 3 ok: one expert at 32 routes")
        # 3b. masked slots inside live tokens: a slot that carried a route in the
        # previous forward and is -1 now must contribute zero, not the stale term.
        t = _inputs(ep16, 32, _spread_routing(ep16, 32), seed=24)
        _check(layer.forward(t), _oracle(ep16, t, w13, w2, bound=64))
        ids = t.topk_ids.clone()
        ids[:, 3] = -1
        ids[::4, 5] = -1
        masked = MoEEpTensors(
            hidden_states=t.hidden_states,
            topk_ids=ids.contiguous(),
            topk_weights=t.topk_weights,
        )
        _say(ep16, "stage 3b: unmasked forward ok")
        for _ in range(2):
            _check(layer.forward(masked), _oracle(ep16, masked, w13, w2, bound=64))
            assert overflow() == 0
        _say(ep16, "stage 3b ok: masked slots inside live tokens")
        # 4. zero-route experts: token t of rank r sends slot s to expert s of
        # owner (t + r) % 16, so the first 8 experts of every rank carry 16 routes
        # each and the other 24 carry none (within the 32-routes-per-expert bound).
        token = torch.arange(16, dtype=torch.int64, device=ep16.device).unsqueeze(1)
        slot = torch.arange(TOP_K, dtype=torch.int64, device=ep16.device).unsqueeze(0)
        ids = ((token + ep16.rank) % WORLD) * LOCAL_EXPERTS + slot
        t = _inputs(ep16, 16, ids, seed=29)
        _check(layer.forward(t), _oracle(ep16, t, w13, w2, bound=64))
        assert overflow() == 0
        _say(ep16, "stage 4 ok: zero-route experts")
        # 5. ragged per-rank token counts (1..64) through the padding path.
        tokens = 1 + (ep16.rank * 9) % 64
        t = _inputs(ep16, tokens, _spread_routing(ep16, tokens), seed=31)
        out = layer.forward(t).clone()
        _check_counted_combine_matches_terms(ep16, ws, out, bound=64)
        _check(out, _oracle(ep16, t, w13, w2, bound=64))
        _say(ep16, f"stage 5 ok: ragged counts ({tokens} tokens)")
        # 6. repeated forwards are bit-identical, and the counted combine
        # equals the reference combine of the term window every time.
        t = _inputs(ep16, 64, _spread_routing(ep16, 64), seed=37)
        first = layer.forward(t).clone()
        _check_counted_combine_matches_terms(ep16, ws, first, bound=64)
        for _ in range(4):
            out = layer.forward(t).clone()
            assert torch.equal(out, first)
            _check_counted_combine_matches_terms(ep16, ws, out, bound=64)
        _say(ep16, "stage 6 ok: repeated forwards are bit-identical")
        # Every forward so far (warmup, 3 + 1 + 1 + 1 + 2 + 1 + 1 + 5) added
        # exactly 16 to word 0 on every rank; word 1 is back to 0.
        _check_completion(ep16, ws, forwards=ws.launches)
        assert ws.launches == 16
        _say(ep16, f"fc2 completion counter after {ws.launches} forwards ok")
        # 7. 33 routes to one expert set the device overflow flag; the layer stays usable.
        t = _inputs(
            ep16, 32, _hotset_routing(ep16, 32, hot_expert=77, hot_load=33), seed=41
        )
        layer.forward(t)
        torch.cuda.synchronize(ep16.device)
        assert overflow() >= 1
        t = _inputs(ep16, 16, _spread_routing(ep16, 16), seed=43)
        _check(layer.forward(t), _oracle(ep16, t, w13, w2, bound=64))
        assert overflow() == 0
        _say(ep16, "stage 7 ok: overflow flag and recovery")
        # 8. CUDA graph replay of the bucketed call equals the eager result.
        t = _inputs(ep16, 16, _spread_routing(ep16, 16), seed=47)
        layer.stage_inputs(t, compile_tokens_per_rank=16)
        eager = layer.compute_staged(output=None).clone()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(ep16.device)
        dist.barrier()
        with torch.cuda.graph(graph):
            layer.stage_inputs(t, compile_tokens_per_rank=16)
            view = layer.compute_staged(output=None)
        torch.cuda.synchronize(ep16.device)
        dist.barrier()
        # The capture itself executes nothing (ws.launches counts the recorded
        # compute), each replay is one forward: the counter follows the
        # executed forwards, so the three programmatic graph edges and the
        # counted combine work under replay without any host reset.
        captured, replays = 1, 3
        for _ in range(replays):
            graph.replay()
            torch.cuda.synchronize(ep16.device)
            assert torch.equal(view, eager)
        _check_counted_combine_matches_terms(ep16, ws, view, bound=16)
        _check(view, _oracle(ep16, t, w13, w2, bound=16))
        _check_completion(ep16, ws, forwards=ws.launches - captured + replays)
        del graph
        capture = torch.cuda.CUDAGraph()
        with (
            pytest.raises(MoEEpConfigError, match="explicit rank-uniform bound"),
            torch.cuda.graph(capture),
        ):
            layer.stage_inputs(t)  # no bound: refused before any launch
        del capture
        dist.barrier()
        _say(ep16, "stage 8 ok: CUDA graph replay")
        # 9. a second layer with other weights shares the pooled workspace.
        w13_b, w2_b = _contract_weights(ep16.rank, ep16.device, seed=17000)
        other = _make_layer(ep16, w13_b, w2_b)
        other.warmup()
        assert other._workspace is ws
        t = _inputs(ep16, 32, _spread_routing(ep16, 32), seed=53)
        _check(other.forward(t), _oracle(ep16, t, w13_b, w2_b, bound=64))
        _check(layer.forward(t), _oracle(ep16, t, w13, w2, bound=64))
        other.destroy()
        assert not ws.destroyed
        # The pooled workspace's counter counted both layers' forwards.
        _check_completion(ep16, ws, forwards=ws.launches - captured + replays)
        _say(ep16, "stage 9 ok: second layer shares the pooled workspace")
    except BaseException as exc:  # noqa: BLE001 - reported, then this rank exits
        _fail_fast(ep16, exc)
    layer.destroy()
    dist.barrier()


# Skew stress: forwards executed while the ranks are deliberately out of
# step. Every rank enqueues an independent random stream delay in front of
# each forward's compute sequence (``torch.cuda._sleep``; up to
# SKEW_MAX_DELAY_CYCLES ~ 50 us at ~2 GHz), so the ranks reach publish_push,
# local_pack's flag waits, fc2's remote pushes and the combine's counter poll
# of a forward at different times, and the cross-forward reuse rule (a rank's
# publish_push(g + 1) against its peers' forward g) is exercised under real
# inter-rank skew rather than lockstep.
SKEW_EAGER_FORWARDS = 48
SKEW_GRAPH_REPLAYS = 16
SKEW_MAX_DELAY_CYCLES = 100_000


def _skew_delay_cycles(rank: int, index: int) -> int:
    """Per-rank, per-forward stream delay in [0, SKEW_MAX_DELAY_CYCLES] from
    an RNG seeded by (rank, forward index): the ranks' delays are independent
    of each other and the sequence is reproducible."""

    generator = torch.Generator(device="cpu").manual_seed(
        1_000_003 * (rank + 1) + index
    )
    return int(
        torch.randint(0, SKEW_MAX_DELAY_CYCLES + 1, (1,), generator=generator).item()
    )


def _zero_route_routing(ep16, tokens: int) -> torch.Tensor:
    """Token t of rank r sends slot s to expert s of owner (t + r) % 16: with
    16 tokens the first 8 experts of every rank carry 16 routes each and the
    other 24 carry none."""

    token = torch.arange(tokens, dtype=torch.int64, device=ep16.device).unsqueeze(1)
    slot = torch.arange(TOP_K, dtype=torch.int64, device=ep16.device).unsqueeze(0)
    return ((token + ep16.rank) % WORLD) * LOCAL_EXPERTS + slot


def test_ep16_cross_forward_reuse_under_rank_skew(ep16) -> None:
    """Cross-forward reuse under real inter-rank skew.

    Before every forward's compute sequence each rank enqueues an independent
    random stream delay (``torch.cuda._sleep``, 0 .. SKEW_MAX_DELAY_CYCLES
    cycles, RNG seeded by rank and forward index), so a rank enters
    ``publish_push(g + 1)`` while its peers are still at other points of
    forwards ``g`` / ``g + 1`` (waiting on inbox flags in ``local_pack``,
    pushing routes in ``fc2``, polling the completion counter in ``combine``).
    The per-rank token counts vary across the forwards within the staged
    capacity (spread, zero-route, ragged, random, single-token and hotset
    routings). After every forward, stream-synced, every rank's
    ``fc2_completion`` reads ``[16 * N, 0]`` and ``push_generation`` ``[N, 0]``
    (``N`` = forwards executed since allocation, the warmup included), the
    counted combine's output is bit-identical to the reference combine of this
    rank's term window, and the output is bit-identical to the first forward of
    the same inputs (which the pure-Torch MXFP8 oracle validated). Then one
    CUDA graph with a fixed rank-dependent delay in front of the compute
    sequence is captured (no host write occurs under capture) and replayed with
    the same checks after every replay.
    """

    w13, w2 = _contract_weights(ep16.rank, ep16.device)
    layer = _make_layer(ep16, w13, w2)
    try:
        layer.warmup()
        ws = layer._workspace
        executed = ws.launches  # forwards executed so far: the warmup is one
        assert executed == 1
        _check_completion(ep16, ws, forwards=executed)
        ragged = 1 + (ep16.rank * 9) % 64
        cases = (
            ("spread-64", _inputs(ep16, 64, _spread_routing(ep16, 64), seed=101)),
            (
                "zero-route-16",
                _inputs(ep16, 16, _zero_route_routing(ep16, 16), seed=103),
            ),
            (
                "ragged",
                _inputs(ep16, ragged, _spread_routing(ep16, ragged), seed=107),
            ),
            (
                "random-32",
                _inputs(
                    ep16,
                    32,
                    _random_routing_within_capacity(ep16, 32, seed=109),
                    seed=113,
                ),
            ),
            ("spread-1", _inputs(ep16, 1, _spread_routing(ep16, 1), seed=127)),
            (
                "hotset-48",
                _inputs(
                    ep16,
                    48,
                    _hotset_routing(ep16, 48, hot_expert=77, hot_load=32),
                    seed=131,
                ),
            ),
        )

        def skewed_forward(
            t: MoEEpTensors, cycles: int, bound: int | None
        ) -> torch.Tensor:
            # Staging is device copies only; the delay is stream-ordered
            # between them and this rank's publish_push.
            layer.stage_inputs(t, compile_tokens_per_rank=bound)
            torch.cuda._sleep(cycles)
            return layer.compute_staged(output=None)

        def overflow() -> int:
            torch.cuda.synchronize(ep16.device)
            return int(routing_overflow(layer).item())

        # 1. eager forwards under random per-rank skew (bound = capacity 64).
        reference: dict[str, torch.Tensor] = {}
        for index in range(SKEW_EAGER_FORWARDS):
            name, t = cases[index % len(cases)]
            cycles = _skew_delay_cycles(ep16.rank, index)
            out = skewed_forward(t, cycles, bound=None).clone()
            executed += 1
            _check_completion(ep16, ws, forwards=executed)
            _check_counted_combine_matches_terms(ep16, ws, out, bound=64)
            assert overflow() == 0, (index, name)
            if name in reference:
                assert torch.equal(out, reference[name]), (index, name, cycles)
            else:
                _check(out, _oracle(ep16, t, w13, w2, bound=64))
                reference[name] = out
        assert len(reference) == len(cases)
        assert executed == 1 + SKEW_EAGER_FORWARDS
        _say(
            ep16,
            f"stage 1 ok: {SKEW_EAGER_FORWARDS} eager forwards under random skew "
            f"(counter [{16 * executed}, 0])",
        )
        # 2. one CUDA graph with a fixed rank-dependent delay in front of the
        # compute sequence, replayed with the same checks after every replay.
        name, t = cases[3]  # random-32: bound 32
        bound = t.num_tokens
        fixed_cycles = (ep16.rank * 7919 % 16) * 6_000
        eager = skewed_forward(t, fixed_cycles, bound=bound).clone()
        executed += 1
        _check_completion(ep16, ws, forwards=executed)
        _check_counted_combine_matches_terms(ep16, ws, eager, bound=bound)
        _check(eager, _oracle(ep16, t, w13, w2, bound=bound))
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(ep16.device)
        dist.barrier()
        with torch.cuda.graph(graph):
            view = skewed_forward(t, fixed_cycles, bound=bound)
        torch.cuda.synchronize(ep16.device)
        dist.barrier()
        # The capture executed nothing: one compute was recorded, the
        # counters still read the executed forwards.
        assert ws.launches == executed + 1
        _check_completion(ep16, ws, forwards=executed)
        for replay in range(SKEW_GRAPH_REPLAYS):
            graph.replay()
            executed += 1
            torch.cuda.synchronize(ep16.device)
            _check_completion(ep16, ws, forwards=executed)
            assert torch.equal(view, eager), (replay, fixed_cycles)
            _check_counted_combine_matches_terms(ep16, ws, view, bound=bound)
            assert overflow() == 0, replay
        assert executed == 2 + SKEW_EAGER_FORWARDS + SKEW_GRAPH_REPLAYS
        del graph
        dist.barrier()
        _say(
            ep16,
            f"stage 2 ok: {SKEW_GRAPH_REPLAYS} graph replays with a fixed "
            f"{fixed_cycles}-cycle delay (counter [{16 * executed}, 0])",
        )
    except BaseException as exc:  # noqa: BLE001 - reported, then this rank exits
        _fail_fast(ep16, exc)
    layer.destroy()
    dist.barrier()


def test_ep16_ktile_major_payload_matches_the_row_major_permutation(ep16) -> None:
    """The quantization program with ``ktile_major_rows = R`` writes exactly the
    pure-Torch permutation of its row-major (``R = 0``) output; the scale
    carriers are identical. Rank-local (no collectives); runs the real program."""

    import tvm_ffi

    from flashinfer.experimental.cake_w8a8_megamoe_ep16.cake_jit import load_module

    module = load_module(ep16.device)
    grid_x = cake_plan.quantize_grid(ep16.device.index)
    generator = torch.Generator(device=ep16.device).manual_seed(61 + ep16.rank)
    try:
        for mode, rows, columns, gate_width, interleave, box_rows in (
            (SCALE_MODE_W13, 512, 256, 256, 1, W13_KTILE_ROWS),
            (SCALE_MODE_W2_SWAP, 256, 384, 0, 0, W2_KTILE_ROWS),
        ):
            source = torch.randn(
                (LOCAL_EXPERTS, rows, columns),
                dtype=torch.bfloat16,
                device=ep16.device,
                generator=generator,
            )
            outputs = {}
            for ktile_major_rows in (0, box_rows):
                quantized = torch.empty(
                    source.shape, dtype=torch.float8_e4m3fn, device=ep16.device
                )
                scales = torch.empty(
                    (source.numel() // 32,), dtype=torch.uint8, device=ep16.device
                )
                with torch.cuda.device(ep16.device), tvm_ffi.use_torch_stream():
                    launch_quantize_weights(
                        module,
                        source=source,
                        quantized=quantized,
                        scales_cp=scales,
                        rows=rows,
                        columns=columns,
                        gate_width=gate_width,
                        interleave_gate_up=interleave,
                        scale_tile_n256_k128=mode,
                        ktile_major_rows=ktile_major_rows,
                        grid_x=grid_x,
                    )
                torch.cuda.synchronize(ep16.device)
                outputs[ktile_major_rows] = (quantized.view(torch.uint8), scales)
            row_major, row_major_scales = outputs[0]
            ktile_major, ktile_major_scales = outputs[box_rows]
            assert torch.equal(ktile_major_scales, row_major_scales)
            assert torch.equal(
                ktile_major, reference_ktile_major_payload(row_major, box_rows=box_rows)
            )
            assert torch.equal(
                reference_row_major_payload(ktile_major, box_rows=box_rows), row_major
            )
            assert not torch.equal(ktile_major, row_major)  # a real permutation
        _say(ep16, "k-tile-major payload ok: matches the row-major permutation")
    except BaseException as exc:  # noqa: BLE001 - reported, then this rank exits
        _fail_fast(ep16, exc)


def test_ep16_collective_config_mismatch_fails_on_every_rank(ep16) -> None:
    w13, w2 = _contract_weights(ep16.rank, ep16.device)
    config = Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig()
    fleet = FleetParams(
        num_experts=EXPERTS,
        max_tokens_per_rank=32 if ep16.rank % 2 else 64,
        token_hidden_size=HIDDEN,
    )
    try:
        with pytest.raises(MoEEpConfigError, match="same sm103 Cake backend"):
            MoEEpMegaLayer(
                bootstrap=BootstrapConfig(
                    world_size=WORLD, rank=ep16.rank, device=ep16.device.index
                ),
                fleet_params=fleet,
                weights=MoEWeightPack(w13=w13, w2=w2),
                backend=MegaConfig(megakernel=config),
            )
    except BaseException as exc:  # noqa: BLE001 - reported, then this rank exits
        _fail_fast(ep16, exc)
    dist.barrier()
    assert _all_ranks(True, ep16.device)
    assert dataclasses.asdict(config)["kernel_name"] == KERNEL_NAME
