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

# Host-only checks of the ``sm107_nvfp4_nvfp4_bf16_cake`` mega backend: its
# configuration, tiers, buffers and grids, the loader's programs against the
# arguments the host supplies, and the launches a forward performs, recorded
# through a fake program table on CPU tensors. No GPU.

from __future__ import annotations

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cake import (
    cake_backend,
    cake_plan,
    cake_programs,
)
from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cake.cake_config import (
    Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig,
)
from flashinfer.moe_ep.core.kernel.registry import _MEGA_KERNEL_REGISTRY

META = torch.device("meta")


def test_registered_under_its_kernel_name() -> None:
    cls = _MEGA_KERNEL_REGISTRY["sm107_nvfp4_nvfp4_bf16_cake"]
    assert cls is cake_backend.Sm107Nvfp4CakeMegaKernelBackend
    assert cls.kernel_name() == "sm107_nvfp4_nvfp4_bf16_cake"


def test_config_refuses_what_the_programs_do_not_build() -> None:
    config = Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=3072)
    assert config.top_k == 6 and config.gate_up_clamp == 10.0
    for kwargs in (
        {"top_k": 8},
        {"gate_up_clamp": None},
        {"gate_up_clamp": 7.0},
        {"activation": "situ"},
        {"apply_routing_weights_before_fc2": False},
        {"combine_dtype": "nvfp4"},
        {"kernel_name": "sm107_nvfp4_nvfp4_bf16_cutedsl"},
    ):
        with pytest.raises(ValueError):
            Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=3072, **kwargs)


def test_tiers_and_geometry() -> None:
    assert [tier.tokens for tier in cake_plan.TIERS] == [1024, 4096, 8192]
    assert [cake_plan.select_tier(n).tokens for n in (1, 1024, 1025, 4096, 8192)] == [
        1024,
        1024,
        4096,
        4096,
        8192,
    ]
    with pytest.raises(ValueError, match="exceeds the largest"):
        cake_plan.select_tier(8193)
    assert [tier.dedup for tier in cake_plan.TIERS] == [False, True, True]
    assert [tier.reduce for tier in cake_plan.TIERS] == [
        "REDUCE_ROUTE",
        "REDUCE_RANK",
        "REDUCE_RANK",
    ]
    cake_plan.validate_geometry(
        hidden=7168, intermediate=3072, num_experts=48, top_k=6, world_size=4
    )
    with pytest.raises(ValueError, match="world_size"):
        cake_plan.validate_geometry(
            hidden=7168, intermediate=3072, num_experts=48, top_k=6, world_size=8
        )
    route, rank = cake_plan.TIERS[0], cake_plan.TIERS[1]
    assert [tier.mega_grid_multiple for tier in cake_plan.TIERS] == [2, 4, 4]
    assert cake_plan.mega_grid(212, route) == 212
    assert cake_plan.mega_grid(211, route) == 210
    assert cake_plan.mega_grid(212, rank) == 212
    assert cake_plan.mega_grid(210, rank) == 208
    assert cake_plan.scale_multiplier(1.0) == float(
        torch.tensor(1.0 / 6.0, dtype=torch.float32)
    )


def test_buffers() -> None:
    for tier in cake_plan.TIERS:
        symmetric = cake_plan.symmetric_buffers(tier)
        assert symmetric["pre_reduced"][0] == (tier.tokens, tier.slots, 7168)
        assert symmetric["activation"][0] == (tier.tokens, 3584)
        assert symmetric["activation_sf"][0] == (tier.tokens, 448)
        assert symmetric["token_src_metadata"][0] == (tier.pool_rows, 2)
        assert symmetric["fc1_topk_scores"][0] == (
            (tier.pool_rows, 2) if tier.dedup else (tier.pool_rows,)
        )
        assert symmetric["sizes_by_rank"][0] == (16, 48)
        assert symmetric["metadata_ready"][0] == (3,)
        assert symmetric["x_ready"][0] == (8,)
        fills = cake_plan.symmetric_fills(tier)
        assert set(fills) <= set(symmetric)
        assert ("fc1_topk_scores" in fills) is tier.dedup
        local = cake_plan.local_buffers(tier)
        assert ("dedup_rows" in local) is tier.dedup and (
            "fc2_done" in local
        ) is tier.dedup
        assert local["pool_sf"][0] == (tier.scale_rows // 128, 112, 512)
        assert local["fc1_out_sf"][0] == (tier.scale_rows // 128, 48, 512)


def _loader_programs():
    loader = pytest.importorskip(cake_programs.LOADER_MODULE)
    return {name: getattr(loader, name) for name in cake_plan.PROGRAM_IDENTIFIERS}


def _workspace(tier: cake_plan.Tier) -> cake_backend.CakeNvfp4Workspace:
    """A workspace of meta tensors from the plan's own tables (no memory, no
    NVSHMEM): the recorded launches read shapes and views only."""

    class _Buffer:
        def __init__(self, shape, dtype):
            self.local = torch.empty(shape, dtype=dtype, device=META)
            self.peers = torch.zeros(cake_plan.PEER_TABLE_CAPACITY, dtype=torch.int64)

        def free(self):
            pass

    return cake_backend.CakeNvfp4Workspace(
        tier=tier,
        device=torch.device("cpu"),
        rank=0,
        world_size=4,
        symmetric={
            name: _Buffer(shape, dtype)
            for name, (shape, dtype) in cake_plan.symmetric_buffers(tier).items()
        },
        flags=_Buffer((cake_plan.PG_FLAG_WORDS,), torch.uint32),
        local={
            name: torch.empty(shape, dtype=dtype, device=META)
            for name, (shape, dtype) in cake_plan.local_buffers(tier).items()
        },
        output=torch.empty((tier.tokens, 7168), dtype=torch.bfloat16, device=META),
        mega_ctas=212,
        ones=torch.empty(12, dtype=torch.float32, device=META),
    )


def test_the_host_supplies_every_argument_of_every_program() -> None:
    programs = _loader_programs()
    weights = {"W1", "S1", "W2", "S2", "fc1_alpha", "fc2_alpha", "norm_const"}
    per_call = {
        "STAGE": {
            "x",
            "data",
            "sf",
            "route_ids",
            "route_scores",
            "table_ids",
            "table_scores",
            "norm",
            "mult",
        },
        "REDUCE_ROUTE": {"slot_rows", "topk_idx", "out", "world"},
        "REDUCE_RANK": {"slot_rows", "topk_idx", "out", "world"},
    }
    for tier in cake_plan.TIERS:
        supplied = set(cake_backend._views(_workspace(tier))) | weights
        for identifier in (tier.router, tier.mega):
            args = tuple(programs[identifier].args)
            assert args[-3:] == cake_plan.GRID_ARGS
            missing = set(args[:-3]) - supplied
            assert not missing, (identifier, sorted(missing))
    for identifier, names in per_call.items():
        assert set(programs[identifier].args[:-3]) == names, identifier


class _Recording:
    """A program table that records each launch's identifier, values and grid."""

    def __init__(self, programs) -> None:
        self.programs = programs
        self.calls = []

    def available(self) -> bool:
        return True

    def handle(self, identifier, device):
        program = self.programs[identifier]

        def launch(*values, identifier=identifier):
            self.calls.append((identifier, values))

        return cake_programs.ProgramHandle(
            identifier=identifier,
            args=tuple(program.args),
            launch=launch,
            cluster=tuple(program.cluster),
            cooperative=bool(program.launch_record()["cooperative"]),
            use_pdl=bool(program.launch_record()["use_pdl"]),
        )


@pytest.mark.parametrize("tokens", [1000, 4096, 5000])
def test_a_forward_launches_router_mega_and_reduce(monkeypatch, tokens) -> None:
    programs = _loader_programs()
    table = _Recording(programs)
    monkeypatch.setattr(cake_programs, "_programs", table)
    import tvm_ffi

    monkeypatch.setattr(tvm_ffi, "use_torch_stream", lambda: _NullContext())
    tier = cake_plan.select_tier(tokens)
    workspace = _workspace(tier)
    workspace.staged_tokens = tokens
    workspace.scalars = {
        name: workspace.ones for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
    }
    # The transformed weights' shapes: K-major codes as permuted views, flat
    # scale planes per expert.
    weights = (
        (
            torch.empty((12, 6144, 3584), dtype=torch.uint8, device=META).permute(
                0, 2, 1
            ),
            torch.empty((12, 6144 * 448), dtype=torch.uint8, device=META),
        ),
        (
            torch.empty((12, 7168, 1536), dtype=torch.uint8, device=META).permute(
                0, 2, 1
            ),
            torch.empty((12, 7168 * 192), dtype=torch.uint8, device=META),
        ),
    )
    backend = object.__new__(cake_backend.Sm107Nvfp4CakeMegaKernelBackend)
    out = backend.compute(workspace, weights, output=None)
    assert out.shape == (tokens, 7168)
    assert [identifier for identifier, _ in table.calls] == [
        tier.router,
        tier.mega,
        tier.reduce,
    ]
    grids = [values[-3:] for _, values in table.calls]
    assert grids == [(tier.router_ctas, 1, 1), (212, 1, 1), (tokens, 1, 1)]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
