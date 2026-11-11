# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side (no GPU) contracts of the Cake SM103 BF16 x NVFP4 MegaMoE backend.

The device programs are replaced by a recording program table, the workspace
by CPU buffers of the backend's own buffer table, so the staging rules, the
positional calls the backend assembles (argument names, scalars, views, grids)
and the pool key are checked without a GPU.
"""

from __future__ import annotations

import contextlib
import dataclasses
import random
import types
from typing import Any

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MoEEpTensors,
    PrequantizedMoEWeights,
    Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake import (
    cake_backend,
    cake_plan,
    cake_programs,
    cake_symmetric,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake import (
    preprocess_mega_weights as preprocess_sm103_bf16_nvfp4_cake_mega_weights,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake.cake_backend import (
    CakeWorkspace,
    Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend,
    resolve_bound,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake.cake_plan import (
    CELLS,
    COMBINE,
    COMBINE_ARGS,
    FUSED,
    FUSED_ARGS,
    GRID_ARGS,
    PEER_TABLE_CAPACITY,
    PROGRAM_IDENTIFIERS,
    SUPPLIED_ARGS,
    TOP_K,
    Cell,
    Geometry,
    combine_grid,
    local_buffers,
    max_tiles,
    pool_rows,
    program_identifier,
    select_cell,
    symmetric_buffers,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake.cake_programs import (
    ProgramHandle,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake.cake_weights import (
    TransformedCakeW4a16MegaWeights,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

KERNEL_NAME = "sm103_bf16_nvfp4_bf16_cake"
HIDDEN, INTERMEDIATE, EXPERTS = 6144, 256, 32


# --------------------------------------------------------------------- config


def test_config_defaults_and_kernel_name() -> None:
    config = Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=2048)
    assert config.kernel_name == KERNEL_NAME
    assert config.top_k == 8
    assert config.knobs is None
    assert config.gate_up_clamp is None and config.resolved_gate_up_clamp is None
    assert config.activation == "swiglu"
    assert (
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(
            intermediate_size=2048, gate_up_clamp=7.0
        ).resolved_gate_up_clamp
        == 7.0
    )


def test_config_field_names_mirror_the_cutedsl_sibling() -> None:
    from flashinfer.moe_ep import Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig

    mine = {
        f.name for f in dataclasses.fields(Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig)
    }
    theirs = {
        f.name for f in dataclasses.fields(Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig)
    }
    assert theirs <= mine, sorted(theirs - mine)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_name": "sm100_bf16_nvfp4_bf16_cutedsl"},
        {"top_k": 6},
        {"intermediate_size": 0},
        {"apply_routing_weights_before_fc2": True},
        {"apply_topk_in_fc1": True},
        {"enable_in_kernel_fc2_reduce": True},
        {"fc1_norm_const": torch.ones(4)},
        {"swiglu_alpha": 1.702},
        {"swiglu_beta": 1.0},
        {"activation": "situ"},
        {"activation": "relu2"},
        {"situ_beta": 0.5},
        {"situ_linear_beta": 0.5},
        {"activation_clamp": 7.0},
        {"gate_up_clamp": -1.0},
        {"gate_up_clamp": 7.0, "activation_clamp": 7.0},
        {"knobs": "fast"},
        {"knobs": {"n": 16}},
        {"knobs": {"n": 128, "cta_group": 1}},
        {"knobs": {"n": 32, "cta_group": 2}},
        {"knobs": {"n": 64, "stages": 4}},
    ],
)
def test_config_rejects_out_of_scope_forms(kwargs: dict) -> None:
    args = {"intermediate_size": 2048, **kwargs}
    with pytest.raises(ValueError):
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(**args)


def test_config_accepts_auto_knobs_as_the_cell_rule() -> None:
    config = Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(
        intermediate_size=2048, knobs="auto"
    )
    assert cake_plan.cell_from_knobs(None) is None
    assert Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(config)._pinned_cell is None


def test_config_accepts_every_shipped_cell_as_knobs() -> None:
    for cell in CELLS:
        config = Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(
            intermediate_size=2048, knobs=cell.as_knobs()
        )
        assert cake_plan.cell_from_knobs(config.knobs) == cell


# ------------------------------------------------------------------ registry


def test_backend_is_registered_under_the_kernel_name() -> None:
    config = Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=2048)
    kernel = create_mega_kernel(config)
    assert isinstance(kernel, Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend)
    assert kernel.kernel_name() == KERNEL_NAME
    assert kernel.supports_output_view
    assert not kernel.supports_live_token_tuning
    assert not kernel.supports_auto_workspaces
    with pytest.raises(RuntimeError):
        kernel.get_tuned_knobs()


def test_runtime_requirements_include_nvshmem_at_every_world_size() -> None:
    from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST

    kernel = create_mega_kernel(
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=2048)
    )
    for world, rank in ((1, 0), (8, 3)):
        assert kernel.runtime_requirements(
            BootstrapConfig(world_size=world, rank=rank)
        ) == {TORCH_DIST, NVSHMEM}


@pytest.mark.parametrize(
    "fleet, intermediate, world",
    [
        (
            dict(num_experts=256, max_tokens_per_rank=64, token_hidden_size=6144),
            2048,
            8,
        ),
        (
            dict(num_experts=256, max_tokens_per_rank=4096, token_hidden_size=7168),
            2048,
            1,
        ),
    ],
)
def test_validate_init_accepts_the_target_geometries(
    monkeypatch: pytest.MonkeyPatch, fleet: dict, intermediate: int, world: int
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kernel = create_mega_kernel(
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=intermediate)
    )
    kernel.validate_init(
        BootstrapConfig(world_size=world, rank=0), FleetParams(**fleet)
    )


@pytest.mark.parametrize(
    "fleet, intermediate, world, message",
    [
        (
            dict(num_experts=256, max_tokens_per_rank=64, token_hidden_size=6144),
            2048,
            4,
            "world",
        ),
        (
            dict(num_experts=256, max_tokens_per_rank=64, token_hidden_size=6144),
            2048,
            2,
            "world",
        ),
        (
            dict(num_experts=256, max_tokens_per_rank=64, token_hidden_size=6016),
            2048,
            8,
            "256",
        ),
        (
            dict(num_experts=256, max_tokens_per_rank=64, token_hidden_size=6144),
            1600,
            8,
            "256",
        ),
        (
            dict(num_experts=255, max_tokens_per_rank=64, token_hidden_size=6144),
            2048,
            8,
            "divisible",
        ),
    ],
)
def test_validate_init_rejects_unsupported_geometries(
    monkeypatch: pytest.MonkeyPatch,
    fleet: dict,
    intermediate: int,
    world: int,
    message: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kernel = create_mega_kernel(
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=intermediate)
    )
    with pytest.raises(MoEEpConfigError, match=message):
        kernel.validate_init(
            BootstrapConfig(world_size=world, rank=0), FleetParams(**fleet)
        )


def test_validate_init_rejects_a_non_default_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kernel = create_mega_kernel(
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=2048)
    )
    with pytest.raises(MoEEpConfigError, match="stream"):
        kernel.validate_init(
            BootstrapConfig(world_size=8, rank=0, stream=7),
            FleetParams(
                num_experts=256, max_tokens_per_rank=64, token_hidden_size=6144
            ),
        )


# ------------------------------------------------------------------- weights


def _canonical(experts: int, hidden: int, intermediate: int, device="cpu"):
    g = torch.Generator().manual_seed(1)
    w13 = torch.randint(
        256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8, generator=g
    )
    w2 = torch.randint(
        256, (experts, hidden, intermediate // 2), dtype=torch.uint8, generator=g
    )
    s13 = torch.randint(
        0x28,
        0x40,
        (experts, 2 * intermediate, hidden // 16),
        dtype=torch.uint8,
        generator=g,
    )
    s2 = torch.randint(
        0x28,
        0x40,
        (experts, hidden, intermediate // 16),
        dtype=torch.uint8,
        generator=g,
    )
    return PrequantizedMoEWeights(
        w13=w13.to(device),
        w13_scale=s13.view(torch.float8_e4m3fn).to(device),
        w2=w2.to(device),
        w2_scale=s2.view(torch.float8_e4m3fn).to(device),
    )


def test_preprocess_weights_keeps_the_canonical_pack() -> None:
    weights = _canonical(4, HIDDEN, INTERMEDIATE)
    transformed = preprocess_sm103_bf16_nvfp4_cake_mega_weights(
        weights, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
    )
    assert isinstance(transformed, TransformedCakeW4a16MegaWeights)
    assert transformed.w13 is weights.w13 and transformed.w2 is weights.w2
    assert transformed.w13_scale is weights.w13_scale
    assert transformed.w2_scale is weights.w2_scale
    assert transformed.w13.dtype == torch.uint8 and transformed.w2.dtype == torch.uint8
    assert transformed.w13_scale.dtype == torch.float8_e4m3fn
    assert (
        transformed.local_experts,
        transformed.hidden_size,
        transformed.intermediate_size,
    ) == (
        4,
        HIDDEN,
        INTERMEDIATE,
    )


def test_preprocess_weights_accepts_the_fp4_dtype_view() -> None:
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        pytest.skip("torch has no float4_e2m1fn_x2")
    weights = _canonical(2, HIDDEN, INTERMEDIATE)
    viewed = PrequantizedMoEWeights(
        w13=weights.w13.view(fp4),
        w13_scale=weights.w13_scale,
        w2=weights.w2.view(fp4),
        w2_scale=weights.w2_scale,
    )
    transformed = preprocess_sm103_bf16_nvfp4_cake_mega_weights(
        viewed, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
    )
    assert transformed.w13.dtype == torch.uint8
    assert torch.equal(transformed.w13, weights.w13)


def test_preprocess_weights_refuses_unquantized_and_misshaped_packs() -> None:
    from flashinfer.moe_ep import MoEWeightPack

    weights = _canonical(2, HIDDEN, INTERMEDIATE)
    with pytest.raises(ValueError, match="PrequantizedMoEWeights"):
        preprocess_sm103_bf16_nvfp4_cake_mega_weights(
            MoEWeightPack(
                w13=torch.zeros(2, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
                w2=torch.zeros(2, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16),
            ),
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
        )
    with pytest.raises(ValueError, match="shape"):
        preprocess_sm103_bf16_nvfp4_cake_mega_weights(
            weights, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN * 2
        )
    bad_scale = PrequantizedMoEWeights(
        w13=weights.w13,
        w13_scale=weights.w13_scale.view(torch.uint8),
        w2=weights.w2,
        w2_scale=weights.w2_scale,
    )
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        preprocess_sm103_bf16_nvfp4_cake_mega_weights(
            bad_scale, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
        )


# ---------------------------------------------------------------------- plan


def test_program_identifiers_cover_every_cell_and_h_max() -> None:
    # 4 cells x 2 known hidden sizes (default h_max 6144, the 7168 variant) + 1 combine
    # unless the combine is fused into the tail (the shipped switch decides).
    separate = cake_plan.program_identifiers(fused_combine=False)
    assert len(separate) == 2 * len(CELLS) + 1 and separate[-1] == COMBINE
    assert cake_plan.program_identifiers() == PROGRAM_IDENTIFIERS
    assert len(PROGRAM_IDENTIFIERS) == 2 * len(CELLS) + (
        0 if cake_plan.FUSED_COMBINE else 1
    )
    # The fused combine retires the combine program: eight fused programs remain.
    fused_only = cake_plan.program_identifiers(fused_combine=True)
    assert len(fused_only) == 2 * len(CELLS) and COMBINE not in fused_only
    assert fused_only == separate[:-1]
    assert program_identifier(FUSED, Cell(32)) == "FUSED_N32"
    assert program_identifier(FUSED, Cell(32), 6144) == "FUSED_N32"
    assert program_identifier(FUSED, Cell(32), 1024) == "FUSED_N32"
    assert program_identifier(FUSED, Cell(128, 2), 7168) == "FUSED_N128_CG2_H7168"
    assert program_identifier(FUSED, Cell(64, 2)) == "FUSED_N64_CG2"
    assert (
        program_identifier(COMBINE)
        == program_identifier(COMBINE, None, 7168)
        == "COMBINE"
    )
    assert cake_plan.fused_h_max(6144) == 6144 and cake_plan.fused_h_max(7168) == 7168
    with pytest.raises(ValueError):
        program_identifier(FUSED, Cell(32), 8448)
    with pytest.raises(ValueError):
        program_identifier(FUSED)
    with pytest.raises(ValueError):
        program_identifier("OTHER", Cell(32))


@pytest.mark.parametrize(
    "tokens, world, experts, expected",
    [
        # m = world x tokens x 8 / experts expected rows per expert;
        # smallest n in (32, 64) with n >= m + 3.1 sqrt(m), else the 2-CTA n = 128
        (0, 8, 256, Cell(32)),
        (8, 8, 256, Cell(32)),  # m = 2
        (32, 8, 256, Cell(32)),  # m = 8 -> 16.8
        (64, 8, 256, Cell(32)),  # m = 16 -> 28.4
        (128, 8, 256, Cell(64)),  # m = 32 -> 49.5
        (512, 8, 256, Cell(128, 2)),  # m = 128 -> 163
        (4096, 8, 256, Cell(128, 2)),
        (1, 1, 256, Cell(32)),
        (512, 1, 256, Cell(32)),  # m = 16 -> 28.4
        (4096, 1, 256, Cell(128, 2)),  # m = 128
        (64, 8, 32, Cell(128, 2)),  # the host-test geometry: m = 128
    ],
)
def test_select_cell_follows_the_token_rule(tokens, world, experts, expected) -> None:
    assert select_cell(tokens, world, TOP_K, experts) == expected
    with pytest.raises(ValueError):
        select_cell(-1, world, TOP_K, experts)


def test_select_cell_never_picks_the_pinned_only_cell() -> None:
    for tokens in range(0, 5000, 7):
        assert select_cell(tokens, 8, TOP_K, 256) != Cell(64, 2)


def test_max_tiles_bounds_every_routing() -> None:
    rng = random.Random(3)
    for _ in range(200):
        world = rng.choice((1, 8))
        local = rng.choice((1, 3, 4, 32))
        cap = rng.randint(1, 96)
        n = rng.choice((32, 64, 128))
        routes = world * cap * TOP_K
        counts = [0] * local
        for _ in range(routes):
            counts[rng.randrange(local)] += 1
        tiles = sum((c + n - 1) // n for c in counts)
        assert tiles <= max_tiles(world, cap, local, n)
    assert pool_rows(8, 512, 32, 128) == max_tiles(8, 512, 32, 128) * 128
    assert max_tiles(8, 512, 32, 128) == (8 * 512 * 8 + 127 * 32) // 128


def test_combine_grid_covers_the_capacity() -> None:
    assert combine_grid(0, 6144) == (6, 1, 1)
    assert combine_grid(1, 6144) == (6, 1, 1)
    assert combine_grid(64, 7168) == (7, 64, 1)
    assert combine_grid(512, 6144) == (6, 512, 1)
    # One token per CTA up to COMBINE_MAX_ROWS rows; above it every CTA of row y
    # walks the tokens y, y + rows, ... (4 per CTA at a capacity of 4096).
    assert cake_plan.COMBINE_MAX_ROWS == 1024
    assert combine_grid(1024, 6144) == (6, 1024, 1)
    assert combine_grid(1025, 6144) == (6, 1024, 1)
    assert combine_grid(4096, 7168) == (7, 1024, 1)


def test_cta_hist_scratch_covers_the_programs_rounds_of_32_ctas() -> None:
    # The fused program reads the per-CTA histogram in rounds of 32 CTAs, so the
    # scratch holds at least round_up(grid, 32) rows of E_local words; the grid is
    # at most one CTA per SM.
    for sm_count in (32, 72, 100, 132, 148, 160, 192):
        g = Geometry(8, 0, 256, 6144, 2048, 8, 64, sm_count=sm_count)
        rows = local_buffers(g)["cta_hist"][0][0]
        assert rows == g.max_grid_ctas >= -(-sm_count // 32) * 32 >= sm_count


def test_buffer_tables_follow_the_geometry() -> None:
    g = Geometry(8, 3, 256, 6144, 2048, 8, 512, sm_count=148)
    sym = symmetric_buffers(g)
    assert sym["published_hidden"] == ((512, 6144), torch.bfloat16)
    # The published routing retires once the route tables and the fused combine
    # both ship; the route tables come with the first.
    published = not (cake_plan.ROUTE_TABLES and cake_plan.FUSED_COMBINE)
    assert ("published_ids" in sym) is published
    assert ("published_weights" in sym) is published
    if published:
        assert sym["published_ids"] == ((512, 8), torch.int32)
    assert ("route_offs" in sym) is cake_plan.ROUTE_TABLES
    assert sym["route_terms"] == ((512, 8, 6144), torch.bfloat16)
    assert sym["completion"][1] == torch.uint32
    loc = local_buffers(g)
    assert loc["pool"][0] == (g.max_pool_rows, 6144)
    assert loc["fc1_out"][0] == (g.max_pool_rows, 2048)
    assert loc["cta_hist"][0] == (296, 32)
    assert loc["expert_seg"][0] == (33,)
    assert loc["fc1_done"] == loc["fc2_done"] == loc["fc1_ready"]
    assert loc["fc1_done"] == ((g.max_route_tiles,), torch.uint32)
    assert loc["epoch"][1] == torch.uint32 and loc["sync_words"][1] == torch.uint32
    assert g.max_pool_rows >= max(g.pool_rows(cell) for cell in CELLS)
    assert g.max_route_tiles == g.max_tiles(Cell(32))
    assert (g.fc1_m_tiles, g.fc2_m_tiles) == (32, 48)
    for name in FUSED_ARGS:
        assert name.isidentifier()
    # The combine program's own arguments: the live count, and the output unless
    # the fused tail writes it.
    # ... less what the both-switches form retires from the fused list (the
    # publications; the combine program, which took them too, no longer ships).
    retired = set(cake_plan.fused_args(route_tables=False, fused_combine=False))
    retired -= set(FUSED_ARGS)
    assert set(COMBINE_ARGS) - set(FUSED_ARGS) - retired == (
        {"tokens_per_rank"}
        if cake_plan.FUSED_COMBINE
        else {"output", "tokens_per_rank"}
    )
    assert tuple(FUSED_ARGS[-3:]) == GRID_ARGS == tuple(COMBINE_ARGS[-3:])
    assert cake_plan.fused_args() == FUSED_ARGS  # the shipped switches
    assert (
        cake_plan.fused_args(route_tables=False, fused_combine=False)
        == cake_plan.FUSED_BASE_ARGS
    )


def test_program_form_switches_shape_the_tables() -> None:
    # The shipped head: both switches off, the tables as the programs take them.
    assert (cake_plan.ROUTE_TABLES, cake_plan.FUSED_COMBINE) == (False, False)
    assert cake_plan.fused_per_forward_args(fused_combine=False) == {
        "hidden",
        "topk_ids",
        "topk_weights",
    }
    g = Geometry(8, 3, 256, 6144, 2048, 8, 512, sm_count=148)
    base = symmetric_buffers(g, route_tables=False, fused_combine=False)
    assert "route_offs" not in base and "published_ids" in base
    # Route tables: two banks of the global-expert prefix and of the route entries.
    tables = symmetric_buffers(g, route_tables=True, fused_combine=False)
    assert tables["route_offs"] == ((2, 257), torch.int32)
    assert tables["route_list"] == ((2, 512 * 8), torch.int32)
    small = Geometry(2, 1, 64, 6144, 2048, 8, 16, sm_count=148)
    assert symmetric_buffers(small, route_tables=True)["route_offs"] == (
        (2, 257),
        torch.int32,
    )
    assert "published_ids" in tables and "published_weights" in tables
    args = cake_plan.fused_args(route_tables=True, fused_combine=False)
    assert args[-7:] == (
        "route_offs",
        "route_offs_peers",
        "route_list",
        "route_list_peers",
        *GRID_ARGS,
    )
    assert "output" not in args
    # Fused combine: the fused program writes the output; the published routing
    # retires only once the route tables carry it too.
    assert "output" in cake_plan.fused_args(route_tables=False, fused_combine=True)
    assert "output" in cake_plan.fused_per_forward_args(fused_combine=True)
    assert "published_ids" in symmetric_buffers(
        g, route_tables=False, fused_combine=True
    )
    both = symmetric_buffers(g, route_tables=True, fused_combine=True)
    assert "published_ids" not in both and "published_weights" not in both
    assert {"route_offs", "route_list", "published_hidden", "published_meta"} < set(
        both
    )
    assert cake_plan.fused_args(route_tables=True, fused_combine=True)[-4:] == (
        "output",
        *GRID_ARGS,
    )
    # The both-switches form's argument list agrees with its buffer table: the
    # publications are gone from both, every remaining buffer crosses as
    # (local, peers); the fused combine alone keeps the publications.
    both_args = cake_plan.fused_args(route_tables=True, fused_combine=True)
    assert (
        "published_ids" not in both_args and "published_weights_peers" not in both_args
    )
    assert "published_ids" in cake_plan.fused_args(
        route_tables=False, fused_combine=True
    )
    assert set(both) < set(both_args) and {f"{name}_peers" for name in both} < set(
        both_args
    )
    # Geometry rules behind the switches: the global expert bound at a world
    # size above one with route tables; the combine slice rule drops with the
    # fused combine.
    common = dict(num_experts=512, hidden=6144, intermediate=2048, top_k=8, capacity=64)
    cake_plan.validate_geometry(world_size=8, **common, route_tables=False)
    with pytest.raises(ValueError, match="in total"):
        cake_plan.validate_geometry(world_size=8, **common, route_tables=True)
    cake_plan.validate_geometry(
        world_size=1,
        num_experts=256,
        hidden=6144,
        intermediate=2048,
        top_k=8,
        capacity=64,
        route_tables=True,
    )
    narrow = dict(num_experts=256, intermediate=2048, top_k=8, capacity=64)
    with pytest.raises(ValueError, match="slice"):
        cake_plan.validate_geometry(
            world_size=8, hidden=6144 + 256, **narrow, fused_combine=False
        )
    cake_plan.validate_geometry(
        world_size=8, hidden=6144 + 256, **narrow, fused_combine=True
    )


def test_peer_tables_are_capacity_padded_cpu_int64() -> None:
    table = cake_symmetric.peer_table([0x10, 2**63 + 8], world_size=2)
    tensor = cake_symmetric.peer_table_tensor(table, rank=1)
    assert tensor.device.type == "cpu" and tensor.dtype == torch.int64
    assert tensor.shape == (PEER_TABLE_CAPACITY,)
    assert tensor[0].item() == 0x10 and tensor[1].item() == 8 - 2**63
    # pad slots repeat the self entry (never a null pointer)
    assert tensor[2:].eq(8 - 2**63).all()
    assert cake_symmetric.peer_table_tensor(table, rank=0)[2:].eq(0x10).all()
    with pytest.raises(ValueError):
        cake_symmetric.peer_table_tensor(table, rank=2)
    with pytest.raises(ValueError):
        cake_symmetric.peer_table([0x10, 0], world_size=2)
    with pytest.raises(ValueError):
        cake_symmetric.peer_table([0x10] * 9, world_size=9)


def _fake_loader(identifiers) -> types.SimpleNamespace:
    """A loader module publishing one program constant per identifier, shipped
    for both exact targets (only the attributes ``available`` / ``targets`` read)."""
    return types.SimpleNamespace(
        **{
            identifier: types.SimpleNamespace(targets=("sm_100a", "sm_103a"))
            for identifier in identifiers
        }
    )


def test_shipped_programs_require_every_program_of_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every program the plan enumerates (the fused program of every cell at every
    # known pull-slot row, and the combine program) must ship for the route to be
    # available; a missing one is named when it is loaded.
    complete = _fake_loader(PROGRAM_IDENTIFIERS)
    monkeypatch.setattr(
        cake_programs.ShippedPrograms, "loader", staticmethod(lambda: complete)
    )
    assert cake_programs.ShippedPrograms.available()
    assert cake_programs.ShippedPrograms().targets() == ((10, 0), (10, 3))
    missing = program_identifier(FUSED, Cell(32), 7168)
    partial = _fake_loader(
        [identifier for identifier in PROGRAM_IDENTIFIERS if identifier != missing]
    )
    monkeypatch.setattr(
        cake_programs.ShippedPrograms, "loader", staticmethod(lambda: partial)
    )
    assert not cake_programs.ShippedPrograms.available()
    assert cake_programs.ShippedPrograms().targets() == ()
    with pytest.raises(RuntimeError, match="FUSED_N32_H7168"):
        cake_programs.ShippedPrograms().handle(
            FUSED, Cell(32), torch.device("cpu"), 7168
        )
    # The default-row program of the same cell is still served by name.
    assert getattr(partial, program_identifier(FUSED, Cell(32), 6144)) is not None


def test_program_handle_refuses_unknown_argument_names() -> None:
    ProgramHandle(FUSED, "FUSED_N32", SUPPLIED_ARGS[FUSED], lambda *a: None)
    with pytest.raises(RuntimeError, match="does not supply"):
        ProgramHandle(
            FUSED,
            "FUSED_N32",
            ("hidden", "launch_epoch", *GRID_ARGS),
            lambda *a: None,
        )
    with pytest.raises(RuntimeError, match="grid"):
        ProgramHandle(COMBINE, "COMBINE", ("route_terms", "output"), lambda *a: None)


def test_resolve_bound_rules() -> None:
    assert resolve_bound(5, None, 64) == 5
    assert resolve_bound(0, None, 64) == 0
    assert resolve_bound(5, 32, 64) == 32
    with pytest.raises(MoEEpConfigError):
        resolve_bound(40, 32, 64)
    with pytest.raises(MoEEpConfigError):
        resolve_bound(5, 65, 64)
    with pytest.raises(MoEEpConfigError):
        resolve_bound(5, True, 64)


# ------------------------------------------------- host path with a fake table


class RecordingTable:
    """A program table whose programs record their positional calls."""

    def __init__(self, probe=None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.handles: dict[str, ProgramHandle] = {}
        self.probe = probe or (lambda cta_group: 148 // cta_group)

    def targets(self):
        return ((10, 3),)

    def handle(self, kernel, cell, device, hidden=None):
        identifier = program_identifier(kernel, cell, hidden)
        handle = self.handles.get(identifier)
        if handle is None:
            args = SUPPLIED_ARGS[kernel]

            def launch(*values, _identifier=identifier, _args=args):
                self.calls.append((_identifier, dict(zip(_args, values, strict=True))))

            handle = ProgramHandle(
                kernel,
                identifier,
                args,
                launch,
                cluster=(cell.cta_group, 1, 1) if cell else (1, 1, 1),
                cooperative=kernel == FUSED,
                max_active_clusters=(
                    lambda index, _cg=(cell.cta_group if cell else 1): self.probe(_cg)
                ),
            )
            self.handles[identifier] = handle
        return handle


def _cpu_workspace(backend, *, world: int, rank: int, capacity: int, experts: int):
    geometry = Geometry(
        world, rank, experts, HIDDEN, INTERMEDIATE, TOP_K, capacity, sm_count=148
    )
    device = torch.device("cpu")
    symmetric = {}
    for name, (shape, dtype) in symmetric_buffers(geometry).items():
        tensor = torch.zeros(shape, dtype=dtype)
        table = cake_symmetric.peer_table([tensor.data_ptr()] * world, world_size=world)
        symmetric[name] = cake_symmetric.SymmetricBuffer(
            tensor,
            table,
            cake_symmetric.peer_table_tensor(table, rank=rank),
            nvshmem=False,
        )
    pad = torch.zeros((cake_plan.PG_FLAG_WORDS,), dtype=torch.uint32)
    table = cake_symmetric.peer_table([pad.data_ptr()] * world, world_size=world)
    local = {
        name: torch.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in local_buffers(geometry).items()
    }
    local["staged_ids"].fill_(-1)
    local["fc1_alpha"].fill_(1.0)
    local["fc2_alpha"].fill_(1.0)
    ws = CakeWorkspace(
        device=device,
        group=None,
        rank=rank,
        world_size=world,
        geometry=geometry,
        swiglu_limit=7.0,
        use_torch_stream=contextlib.nullcontext,
        symmetric=symmetric,
        pg_flags=cake_symmetric.SymmetricBuffer(
            pad,
            table,
            cake_symmetric.peer_table_tensor(table, rank=rank),
            nvshmem=False,
        ),
        local=local,
    )
    # what _allocate_workspace agrees collectively: the fused grid per cell
    ws.fused_grids.update(backend._local_fused_grids(geometry, device, CELLS))
    return ws


@pytest.fixture
def host_path(monkeypatch: pytest.MonkeyPatch):
    table = RecordingTable()
    monkeypatch.setattr(cake_programs, "_programs", table)
    config = Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(
        intermediate_size=INTERMEDIATE, gate_up_clamp=7.0
    )
    backend = Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(config)
    backend.bind_ep_bootstrap(BootstrapConfig(world_size=8, rank=3))
    ws = _cpu_workspace(backend, world=8, rank=3, capacity=64, experts=EXPERTS)
    weights = preprocess_sm103_bf16_nvfp4_cake_mega_weights(
        _canonical(EXPERTS // 8, HIDDEN, INTERMEDIATE),
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
    )
    yield backend, ws, weights, table


def _inputs(tokens: int, *, ids_dtype=torch.int32, seed: int = 0) -> MoEEpTensors:
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, HIDDEN, generator=g).bfloat16()
    ids = (
        torch.stack(
            [torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)]
        )
        if tokens
        else torch.empty(0, TOP_K, dtype=torch.int64)
    )
    weights = torch.softmax(torch.randn(tokens, TOP_K, generator=g), dim=-1)
    return MoEEpTensors(
        hidden_states=hidden, topk_ids=ids.to(ids_dtype), topk_weights=weights
    )


def _launch_names(cell: Cell) -> list[str]:
    """The programs one forward launches, in order: the fused program of the
    cell, then the combine program unless the combine is fused into the tail."""
    return [f"FUSED_{cell.suffix}", *([] if cake_plan.FUSED_COMBINE else ["COMBINE"])]


def test_host_path_binds_direct_inputs_and_launches_the_route(host_path) -> None:
    backend, ws, weights, table = host_path
    t = _inputs(40)
    backend.stage_inputs(t, ws, quantize_input=True)
    assert ws.staged.hidden is t.hidden_states and ws.staged.ids is t.topk_ids
    assert ws.staged.weights is t.topk_weights
    out = backend.compute(ws, weights, output=None)
    assert out.shape == (40, HIDDEN) and out.data_ptr() == ws.output.data_ptr()
    cell = select_cell(ws.capacity, 8, TOP_K, EXPERTS)
    assert [name for name, _ in table.calls] == _launch_names(cell)
    fused = table.calls[0][1]
    assert fused["hidden"] is t.hidden_states
    assert fused["topk_ids"] is t.topk_ids and fused["topk_weights"] is t.topk_weights
    assert fused["T_r"] == 40 and fused["cap"] == 64
    assert (fused["H"], fused["I"], fused["E_local"]) == (HIDDEN, INTERMEDIATE, 4)
    assert fused["fc1_m_tiles"] == fused["fc1_m_div"] == INTERMEDIATE // 64
    assert fused["fc2_m_tiles"] == fused["fc2_m_div"] == HIDDEN // 128
    assert fused["fc1_out_st"].data_ptr() == fused["fc1_out_ld"].data_ptr()
    assert fused["fc1_out_st"].shape == (ws.geometry.pool_rows(cell), INTERMEDIATE)
    assert fused["swiglu_limit"] == 7.0
    assert fused["mega_pg_world"] == 8 and fused["mega_pg_rank"] == 3
    assert fused["mega_pg_flags"] is ws.pg_flags.peers
    assert fused["W13"] is weights.w13 and fused["S13"].dtype == torch.uint8
    assert fused["S13"].data_ptr() == weights.w13_scale.data_ptr()
    assert fused["pool"].shape == (ws.geometry.pool_rows(cell), HIDDEN)
    assert (
        fused["pool"].data_ptr()
        == fused["pool_ptr"].data_ptr()
        == ws.local["pool"].data_ptr()
    )
    assert fused["fc1_ready"].shape == (ws.geometry.max_tiles(cell),)
    assert fused["route_terms"] is ws.symmetric["route_terms"].local
    assert fused["route_terms_peers"] is ws.symmetric["route_terms"].peers
    assert (
        fused["epoch"] is ws.local["epoch"]
        and fused["sync_words"] is ws.local["sync_words"]
    )
    tiles = ws.geometry.max_tiles(cell)
    for name in ("fc1_ready", "fc1_done", "fc2_done"):
        assert fused[name].data_ptr() == ws.local[name].data_ptr()
        assert fused[name].shape == (tiles,) and fused[name].dtype == torch.uint32
    assert (fused["grid_x"], fused["grid_y"], fused["grid_z"]) == (148, 1, 1)
    if cake_plan.FUSED_COMBINE:
        # The fused tail writes the caller's output; no combine program runs.
        assert fused["output"] is ws.output and len(table.calls) == 1
    else:
        combine = table.calls[1][1]
        assert combine["output"] is ws.output and combine["T_r"] == 40
        assert combine["tokens_per_rank"] is ws.symmetric["published_meta"].local
        assert combine["epoch"] is ws.local["epoch"]
        assert combine["route_terms"] is ws.symmetric["route_terms"].local
        assert (
            combine["grid_x"],
            combine["grid_y"],
            combine["grid_z"],
        ) == combine_grid(ws.capacity, HIDDEN)
        assert combine["hidden"] == HIDDEN  # the i32 size; the fused one is the rows
    assert ws.launches == 1


def test_host_path_stages_int64_ids_and_buckets(host_path) -> None:
    backend, ws, weights, table = host_path
    t = _inputs(10, ids_dtype=torch.int64, seed=2)
    backend.set_compile_tokens_per_rank(ws, 32)
    backend.stage_inputs(t, ws, quantize_input=True)
    staged = ws.staged
    assert staged.bound == 32 and staged.tokens == 10
    assert staged.ids.dtype == torch.int32 and staged.ids.shape == (32, TOP_K)
    assert torch.equal(staged.ids[:10], t.topk_ids.to(torch.int32))
    assert staged.ids[10:].eq(-1).all()
    assert staged.hidden.shape == (32, HIDDEN)
    assert torch.equal(staged.hidden[:10], t.hidden_states)
    assert staged.weights.shape == (32, TOP_K)
    output = torch.empty(10, HIDDEN, dtype=torch.bfloat16)
    ws.output[:10].fill_(2.0)
    out = backend.compute(ws, weights, output=output)
    assert out.data_ptr() == output.data_ptr() and out.shape == (10, HIDDEN)
    assert out.eq(2.0).all()
    fused = table.calls[0][1]
    assert fused["T_r"] == 32 and fused["hidden"] is staged.hidden
    assert table.calls[-1][1]["output"] is ws.output
    if not cake_plan.FUSED_COMBINE:
        assert table.calls[1][1]["grid_y"] == ws.capacity


def test_host_path_writes_a_bound_sized_caller_output_in_place(host_path) -> None:
    backend, ws, weights, table = host_path
    t = _inputs(16, seed=5)
    backend.stage_inputs(t, ws, quantize_input=True)
    output = torch.empty(16, HIDDEN, dtype=torch.bfloat16)
    out = backend.compute(ws, weights, output=output)
    assert out.data_ptr() == output.data_ptr()
    assert table.calls[-1][1]["output"] is output


def test_host_path_zero_tokens_still_launches_the_route(host_path) -> None:
    backend, ws, weights, table = host_path
    backend.stage_inputs(_inputs(0), ws, quantize_input=True)
    out = backend.compute(ws, weights, output=None)
    assert out.shape == (0, HIDDEN)
    assert table.calls[0][1]["T_r"] == 0
    assert len(table.calls) == (1 if cake_plan.FUSED_COMBINE else 2)
    if not cake_plan.FUSED_COMBINE:
        assert len(table.calls) == (1 if cake_plan.FUSED_COMBINE else 2)
    if not cake_plan.FUSED_COMBINE:
        assert table.calls[1][1]["grid_y"] == ws.capacity > 0


def test_host_path_one_launch_under_the_fused_combine_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fused-combine form: the fused program takes ``output`` as a per-forward
    # argument, no combine program is loaded or launched, and the caller's
    # bound-sized output is written in place.
    monkeypatch.setattr(cake_backend, "FUSED_COMBINE", True)
    monkeypatch.setattr(
        cake_backend,
        "FUSED_PER_FORWARD_ARGS",
        cake_plan.fused_per_forward_args(fused_combine=True),
    )
    fused_args = cake_plan.fused_args(route_tables=True, fused_combine=True)
    monkeypatch.setitem(SUPPLIED_ARGS, FUSED, fused_args)
    table = RecordingTable()
    monkeypatch.setattr(cake_programs, "_programs", table)
    config = Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(
        intermediate_size=INTERMEDIATE, gate_up_clamp=7.0
    )
    backend = Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(config)
    backend.bind_ep_bootstrap(BootstrapConfig(world_size=8, rank=3))
    ws = _cpu_workspace(backend, world=8, rank=3, capacity=64, experts=EXPERTS)
    # The route tables of the switched form, allocated as the buffer table spells them.
    geometry = ws.geometry
    for name, (shape, dtype) in symmetric_buffers(
        geometry, route_tables=True, fused_combine=True
    ).items():
        if name in ws.symmetric:
            continue
        tensor = torch.zeros(shape, dtype=dtype)
        peers = cake_symmetric.peer_table([tensor.data_ptr()] * 8, world_size=8)
        ws.symmetric[name] = cake_symmetric.SymmetricBuffer(
            tensor,
            peers,
            cake_symmetric.peer_table_tensor(peers, rank=3),
            nvshmem=False,
        )
    weights = preprocess_sm103_bf16_nvfp4_cake_mega_weights(
        _canonical(EXPERTS // 8, HIDDEN, INTERMEDIATE),
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
    )
    t = _inputs(16, seed=9)
    backend.stage_inputs(t, ws, quantize_input=True)
    output = torch.empty(16, HIDDEN, dtype=torch.bfloat16)
    out = backend.compute(ws, weights, output=output)
    assert out.data_ptr() == output.data_ptr()
    cell = select_cell(ws.capacity, 8, TOP_K, EXPERTS)
    assert [name for name, _ in table.calls] == [f"FUSED_{cell.suffix}"]
    fused = table.calls[0][1]
    assert fused["output"] is output and fused["T_r"] == 16
    assert fused["route_offs"] is ws.symmetric["route_offs"].local
    assert fused["route_list_peers"] is ws.symmetric["route_list"].peers
    assert ws.plans and next(iter(ws.plans.values())).combine is None
    assert ws.launches == 1


def test_host_path_rejects_aliasing_and_requires_staging(host_path) -> None:
    backend, ws, weights, _ = host_path
    with pytest.raises(MoEEpConfigError, match="stage_inputs"):
        backend.compute(ws, weights, output=None)
    t = _inputs(4)
    aliased = MoEEpTensors(
        hidden_states=ws.local["staged_hidden"][:4],
        topk_ids=t.topk_ids,
        topk_weights=t.topk_weights,
    )
    with pytest.raises(MoEEpConfigError, match="alias"):
        backend.stage_inputs(aliased, ws, quantize_input=True)
    with pytest.raises(MoEEpConfigError):
        backend.set_compile_tokens_per_rank(ws, 65)
    with pytest.raises(MoEEpConfigError, match="warmed"):
        backend.validate_capture_ready(ws, weights)


def test_host_path_pins_a_cell_and_caches_plans_per_bound(host_path) -> None:
    backend, ws, weights, table = host_path
    backend._pinned_cell = Cell(128, 2)
    for tokens in (8, 8, 12):
        backend.stage_inputs(_inputs(tokens, seed=tokens), ws, quantize_input=True)
        backend.compute(ws, weights, output=None)
    names = [name for name, _ in table.calls]
    assert names == _launch_names(Cell(128, 2)) * 3
    assert table.calls[0][1]["grid_x"] == 148 // 2 * 2
    assert set(ws.plans) == {
        (id(weights), 8, "N128_CG2"),
        (id(weights), 12, "N128_CG2"),
    }
    backend._forget_local_workspace_state(ws)
    assert not ws.plans


def test_host_path_reports_the_selected_cell_as_tuned_knobs(host_path) -> None:
    backend, ws, weights, _ = host_path
    with pytest.raises(RuntimeError):
        backend.get_tuned_knobs()
    backend.stage_inputs(_inputs(5), ws, quantize_input=True)
    backend.compute(ws, weights, output=None)
    assert (
        backend.get_tuned_knobs()
        == select_cell(ws.capacity, 8, TOP_K, EXPERTS).as_knobs()
    )


def test_host_path_per_forward_alphas_are_copied(host_path) -> None:
    backend, ws, weights, table = host_path
    t = _inputs(6)
    t.fc1_alpha = torch.full((4,), 0.5)
    t.fc2_alpha = torch.full((4,), 0.25)
    backend.stage_inputs(t, ws, quantize_input=True)
    assert ws.local["fc1_alpha"].eq(0.5).all() and ws.local["fc2_alpha"].eq(0.25).all()
    backend.compute(ws, weights, output=None)
    assert table.calls[0][1]["fc1_alpha"] is ws.local["fc1_alpha"]


def test_host_path_refuses_foreign_weights_and_shapes(host_path) -> None:
    backend, ws, weights, _ = host_path
    backend.stage_inputs(_inputs(4), ws, quantize_input=True)
    with pytest.raises(MoEEpConfigError, match="TransformedCakeW4a16MegaWeights"):
        backend.compute(ws, (weights.w13, weights.w2), output=None)
    wrong = TransformedCakeW4a16MegaWeights(
        w13=weights.w13[:2],
        w13_scale=weights.w13_scale[:2],
        w2=weights.w2[:2],
        w2_scale=weights.w2_scale[:2],
    )
    with pytest.raises(MoEEpConfigError, match="experts"):
        backend.compute(ws, wrong, output=None)


def test_validate_forward_checks_dtypes_and_fields(host_path) -> None:
    backend, _, _, _ = host_path
    fleet = FleetParams(
        num_experts=EXPERTS, max_tokens_per_rank=64, token_hidden_size=HIDDEN
    )
    cpu_inputs = _inputs(4)
    with pytest.raises(MoEEpConfigError):
        backend.validate_forward(
            MoEEpTensors(
                cpu_inputs.hidden_states.float(),
                cpu_inputs.topk_ids,
                cpu_inputs.topk_weights,
            ),
            fleet,
            quantize_input=True,
        )
    with pytest.raises(MoEEpConfigError, match="fc1_norm_const"):
        bad = _inputs(4)
        bad.fc1_norm_const = torch.ones(4)
        backend.validate_forward(bad, fleet, quantize_input=True)


def test_workspace_pool_key_separates_clamp_knobs_and_alphas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    fleet = FleetParams(num_experts=256, max_tokens_per_rank=64, token_hidden_size=6144)

    def key(**kwargs):
        backend = Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(
            Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=2048, **kwargs)
        )
        backend.bind_ep_bootstrap(BootstrapConfig(world_size=8, rank=1))
        return backend._workspace_pool_key(fleet)

    base = key()
    assert base == key()
    assert base[0] == KERNEL_NAME
    assert base != key(gate_up_clamp=7.0)
    assert base != key(knobs={"n": 64})
    assert key(knobs={"n": 64, "cta_group": 1}) == key(knobs={"cta_group": 1, "n": 64})
    alpha = torch.ones(32)
    assert key(fc1_alpha=alpha) == key(fc1_alpha=alpha)
    assert key(fc1_alpha=alpha) != key(fc1_alpha=torch.ones(32))


def test_cake_backend_exports_match_the_package() -> None:
    assert cake_backend.KERNEL_NAME == KERNEL_NAME
    from flashinfer.moe_ep.backends.mega.kernel import sm103

    assert hasattr(sm103, "bf16_nvfp4_bf16_cake")


# ----------------------------------------------------- design-review rulings


def test_backend_duck_types_the_config() -> None:
    class Foreign:  # the fields of the config, not the class
        kernel_name = KERNEL_NAME
        intermediate_size = 2048
        top_k = 8
        gate_up_clamp = 7.0
        knobs = {"n": 64}

    backend = Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(Foreign())
    adopted = backend._kernel_config
    assert isinstance(adopted, Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig)
    assert adopted.gate_up_clamp == 7.0 and backend._pinned_cell == Cell(64)

    class Wrong(Foreign):
        kernel_name = "sm100_bf16_nvfp4_bf16_cutedsl"

    with pytest.raises(MoEEpConfigError, match="kernel_name"):
        Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(Wrong())

    class OutOfScope(Foreign):
        swiglu_alpha = 1.702

    with pytest.raises(MoEEpConfigError, match="swiglu_alpha"):
        Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(OutOfScope())


def test_host_path_rejects_strided_or_offset_inputs(host_path) -> None:
    backend, ws, _, _ = host_path
    fleet = FleetParams(
        num_experts=EXPERTS, max_tokens_per_rank=64, token_hidden_size=HIDDEN
    )
    t = _inputs(8)
    strided = MoEEpTensors(
        hidden_states=torch.randn(8, 2 * HIDDEN).bfloat16()[:, :HIDDEN],
        topk_ids=t.topk_ids,
        topk_weights=t.topk_weights,
    )
    with pytest.raises(MoEEpConfigError, match="contiguous"):
        backend.validate_forward(strided, fleet, quantize_input=True)
    with pytest.raises(MoEEpConfigError, match="contiguous"):
        backend.stage_inputs(strided, ws, quantize_input=True)
    offset = MoEEpTensors(
        hidden_states=t.hidden_states,
        topk_ids=torch.zeros(9 * TOP_K + 1, dtype=torch.int32)[1:].view(9, TOP_K)[:8],
        topk_weights=t.topk_weights,
    )
    assert offset.topk_ids.is_contiguous() and offset.topk_ids.data_ptr() % 16 == 4
    with pytest.raises(MoEEpConfigError, match="aligned"):
        backend.stage_inputs(offset, ws, quantize_input=True)


def test_local_fused_grids_follow_the_sm_rule_capped_by_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = RecordingTable(probe=lambda cta_group: 120)
    monkeypatch.setattr(cake_programs, "_programs", table)
    backend = Sm103Bf16Nvfp4Bf16CakeMegaKernelBackend(
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=INTERMEDIATE)
    )
    geometry = Geometry(8, 0, EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, 64, sm_count=148)
    grids = backend._local_fused_grids(geometry, torch.device("cpu"), CELLS)
    # cg1 cells: min(148, 120) clusters; cg2 cells: min(74, 120) clusters x 2 CTAs
    assert grids == {"N32": 120, "N64": 120, "N64_CG2": 148, "N128_CG2": 148}
    assert backend._candidate_cells() == cake_plan.SELECTABLE_CELLS
    backend._pinned_cell = Cell(64, 2)
    assert backend._candidate_cells() == (Cell(64, 2),)


def test_compute_refuses_a_cell_the_workspace_was_not_allocated_for(
    host_path,
) -> None:
    backend, ws, weights, _ = host_path
    backend._pinned_cell = Cell(64, 2)
    ws.fused_grids.pop("N64_CG2")
    backend.stage_inputs(_inputs(4), ws, quantize_input=True)
    with pytest.raises(MoEEpConfigError, match="without cell"):
        backend.compute(ws, weights, output=None)
