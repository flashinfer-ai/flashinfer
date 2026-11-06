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

"""Integration tests of the Cake SM103 BF16 x NVFP4 MegaMoE backend.

Launch on 8 Blackwell GPUs of one host (EP world 8) or as plain pytest on one GPU
(EP world 1; the backend ships both worlds):

    torchrun --nproc_per_node=8 -m pytest tests/moe_ep/test_moe_ep_sm103_w4a16_cake_mega.py \
        -m "gpu_8 and arch_blackwell" -v
    python -m pytest tests/moe_ep/test_moe_ep_sm103_w4a16_cake_mega.py -m arch_blackwell -v

The expert weights are random packed E2M1 codes with E4M3 block scales
(``tests.test_helpers.w4a16_reference.make_w4a16_weights``), the activations
Gaussian BF16. Every output is compared against the independent BF16 / FP32
route of ``w4a16_reference`` on the band ``|y - ref| <= 2e-3 max|ref| + 0.05 |ref|``, and
against an optional second oracle named by ``CAKE_W4A16_ORACLE_MODULE``
(``module:function(problem, local_weights) -> bf16 [T, H]``). Repeated forwards
on the same inputs are bitwise identical; a captured graph replays with new
inputs and without host state. The layer fixture runs the tests at both hidden
sizes of the route (6144 and 7168 columns); one EP world 1 case runs 600 tokens,
past the kernel's redundant id scan, on a layer whose capacity holds them.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import pytest
import torch

HIDDEN = 6144
HIDDEN_SIZES = (6144, 7168)
INTERMEDIATE = 512
EXPERTS = 32
TOP_K = 8
CAPACITY = 96
WIDE_TOKENS = 600
WIDE_CAPACITY = 640
GATE_UP_CLAMP = 7.0
ATOL_FRACTION = 2e-3
RTOL = 0.05
SUPPORTED_WORLDS = (1, 8)

pytestmark = [pytest.mark.arch_blackwell] + (
    [pytest.mark.gpu_8] if int(os.environ.get("WORLD_SIZE", "1")) > 1 else []
)


@dataclass
class Problem:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    def tensors(self):
        from flashinfer.moe_ep import MoEEpTensors

        return MoEEpTensors(
            hidden_states=self.hidden_states,
            topk_ids=self.topk_ids,
            topk_weights=self.topk_weights,
        )

    def reference_problem(self) -> dict:
        return {
            "hidden_states": self.hidden_states,
            "topk_ids": self.topk_ids,
            "topk_weights": self.topk_weights,
            "gate_up_clamp": GATE_UP_CLAMP,
            "activation": "swiglu",
        }


# --------------------------------------------------------------------- launch


def _world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def skip_unless_supported() -> tuple[int, int]:
    rank, world_size = _world()
    if world_size not in SUPPORTED_WORLDS:
        pytest.skip(
            f"the route ships EP worlds {SUPPORTED_WORLDS}, launched {world_size}"
        )
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"needs {local_world_size} visible GPUs, one per local rank")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("needs an sm_100 / sm_103 device")
    from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake import (
        programs_available,
    )

    if not programs_available():
        pytest.skip(
            "no Cake W4A16 MegaMoE programs (ship the loader or install a table "
            "with set_programs)"
        )
    return rank, world_size


def init_distributed(tmp_path_factory) -> tuple[int, int]:
    import torch.distributed as dist

    rank, world_size = skip_unless_supported()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
        else:
            rendezvous = tmp_path_factory.mktemp("cake-w4a16-ep1") / "process-group"
            dist.init_process_group(
                backend="nccl",
                init_method=rendezvous.as_uri(),
                rank=0,
                world_size=1,
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
    return rank, world_size


# -------------------------------------------------------------------- problem


def make_problem(
    tokens: int,
    *,
    rank: int,
    seed: int,
    masked_slots: float = 0.0,
    all_masked: bool = False,
    ids_dtype: torch.dtype = torch.int64,
    hidden: int | None = None,
) -> Problem:
    device = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=device).manual_seed(seed * 1000 + rank)
    states = torch.randn(
        tokens,
        HIDDEN if hidden is None else hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    )
    scores = torch.rand(tokens, EXPERTS, device=device, generator=g)
    ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int64)
    weights = torch.softmax(
        torch.randn(tokens, TOP_K, device=device, generator=g), dim=-1
    ).float()
    if all_masked:
        ids.fill_(-1)
    elif masked_slots > 0:
        mask = torch.rand(tokens, TOP_K, device=device, generator=g) < masked_slots
        ids[mask] = -1
    return Problem(states, ids.to(ids_dtype).contiguous(), weights.contiguous())


def reference(problem: Problem, weights) -> torch.Tensor:
    from tests.test_helpers.w4a16_reference import w4a16_reference

    return w4a16_reference(problem.reference_problem(), weights)


def second_oracle(problem: Problem, weights) -> torch.Tensor | None:
    spec = os.environ.get("CAKE_W4A16_ORACLE_MODULE", "").strip()
    if not spec:
        return None
    module_name, _, attribute = spec.partition(":")
    oracle = getattr(importlib.import_module(module_name), attribute or "oracle")
    return oracle(problem.reference_problem(), weights)


def check_band(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == torch.bfloat16
    actual_f, expected_f = actual.float(), expected.float()
    assert torch.isfinite(actual_f).all(), f"{what}: non-finite output"
    if expected.numel() == 0:
        return
    scale = expected_f.abs().max().item()
    tolerance = ATOL_FRACTION * scale + RTOL * expected_f.abs()
    error = (actual_f - expected_f).abs()
    worst = (error - tolerance).max().item()
    bad = int((error > tolerance).sum().item())
    assert bad == 0, (
        f"{what}: {bad} of {expected.numel()} elements outside the band "
        f"(max excess {worst:.4g}, max|ref| {scale:.4g})"
    )


# ---------------------------------------------------------------------- layer


def make_layer(
    rank: int,
    world_size: int,
    weights,
    *,
    hidden: int | None = None,
    capacity: int | None = None,
    knobs=None,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, device=torch.cuda.current_device()
        ),
        fleet_params=FleetParams(
            num_experts=EXPERTS,
            max_tokens_per_rank=CAPACITY if capacity is None else capacity,
            token_hidden_size=HIDDEN if hidden is None else hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOP_K,
                gate_up_clamp=GATE_UP_CLAMP,
                knobs=knobs,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


@pytest.fixture(scope="module", params=HIDDEN_SIZES, ids=lambda hidden: f"h{hidden}")
def ep_layer(request, tmp_path_factory):
    """``(rank, world_size, layer, weights, hidden)``: one layer per hidden size of
    the route, shared by the tests at that size (a layer pins its hidden size)."""

    import torch.distributed as dist

    from tests.test_helpers.w4a16_reference import make_w4a16_weights

    hidden = request.param
    rank, world_size = init_distributed(tmp_path_factory)
    weights = make_w4a16_weights(hidden, INTERMEDIATE, EXPERTS // world_size, rank)
    layer = make_layer(rank, world_size, weights, hidden=hidden)
    layer.warmup(make_problem(16, rank=rank, seed=1, hidden=hidden).tensors())
    torch.cuda.synchronize()
    yield rank, world_size, layer, weights, hidden
    layer.destroy()
    dist.barrier()


def _forward(layer, problem: Problem) -> torch.Tensor:
    out = layer(problem.tensors())
    torch.cuda.synchronize()
    return out


def _check(layer, problem: Problem, weights, what: str) -> torch.Tensor:
    out = _forward(layer, problem)
    assert tuple(out.shape) == tuple(problem.hidden_states.shape)
    check_band(out, reference(problem, weights), f"{what} vs w4a16_reference")
    other = second_oracle(problem, weights)
    if other is not None:
        check_band(out, other, f"{what} vs {os.environ['CAKE_W4A16_ORACLE_MODULE']}")
    return out


# ---------------------------------------------------------------------- tests


def test_uniform_rows_match_the_reference(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    _check(
        layer,
        make_problem(64, rank=rank, seed=100, hidden=hidden),
        weights,
        "uniform 64",
    )


def test_capacity_rows_match_the_reference(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    _check(
        layer,
        make_problem(CAPACITY, rank=rank, seed=110, hidden=hidden),
        weights,
        "capacity",
    )


def test_int32_ids_bind_directly_and_match(ep_layer) -> None:
    """int32 routing ids are bound in place (no staging copy). The layer pins the
    input dtypes per layer instance at its first real forward, so the int32
    path gets its own layer (the shared fixture's forwards use int64 ids)."""

    import torch.distributed as dist

    rank, world_size, _, weights, hidden = ep_layer
    layer = make_layer(rank, world_size, weights, hidden=hidden)
    try:
        layer.warmup(
            make_problem(
                16, rank=rank, seed=1, hidden=hidden, ids_dtype=torch.int32
            ).tensors()
        )
        problem = make_problem(
            48, rank=rank, seed=120, hidden=hidden, ids_dtype=torch.int32
        )
        _check(layer, problem, weights, "int32 ids")
        # the same rows through int64 ids on the shared layer agree bitwise
        shared = ep_layer[2]
        other = Problem(
            problem.hidden_states,
            problem.topk_ids.to(torch.int64),
            problem.topk_weights,
        )
        a = _forward(layer, problem).clone()
        b = _forward(shared, other).clone()
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    finally:
        layer.destroy()
        dist.barrier()


def test_heterogeneous_rows_including_an_empty_rank(ep_layer) -> None:
    """Per-rank row counts differ in one call; a rank may hold no token."""

    rank, world_size, layer, weights, hidden = ep_layer
    pattern = (17, 0, 96, 1, 33, 64, 2, 80)
    rows = pattern[rank % len(pattern)] if world_size > 1 else 0
    out = _check(
        layer,
        make_problem(rows, rank=rank, seed=130, hidden=hidden),
        weights,
        "heterogeneous",
    )
    assert out.shape[0] == rows


def test_masked_slots_contribute_zero(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    _check(
        layer,
        make_problem(40, rank=rank, seed=140, hidden=hidden, masked_slots=0.3),
        weights,
        "masked slots",
    )


def test_fully_masked_rank_returns_zeros(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    masked = rank == world_size - 1
    problem = make_problem(24, rank=rank, seed=150, hidden=hidden, all_masked=masked)
    out = _check(layer, problem, weights, "fully masked")
    if masked:
        assert torch.equal(out.float(), torch.zeros_like(out.float()))


def test_repeated_forwards_are_bitwise_identical(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    problem = make_problem(50, rank=rank, seed=160, hidden=hidden)
    first = _forward(layer, problem).clone()
    other = make_problem(31, rank=rank, seed=161, hidden=hidden)
    _forward(layer, other)
    again = _forward(layer, problem)
    assert torch.equal(first.view(torch.int16), again.view(torch.int16))
    check_band(first, reference(problem, weights), "repeat vs w4a16_reference")


def test_stale_rows_beyond_the_live_count_are_ignored(ep_layer) -> None:
    """After a full-capacity forward, a shorter forward must not read stale rows."""

    rank, world_size, layer, weights, hidden = ep_layer
    _forward(layer, make_problem(CAPACITY, rank=rank, seed=170, hidden=hidden))
    _check(
        layer,
        make_problem(CAPACITY - 3, rank=rank, seed=171, hidden=hidden),
        weights,
        "stale rows",
    )


def test_world1_rows_past_the_scan_bound_match_the_reference(ep_layer) -> None:
    """An EP world 1 rank with more than 512 tokens (4096 routing ids at top-8) is
    past the kernel's redundant id scan: that forward exchanges the histogram, on
    a layer whose capacity holds the rows."""

    import torch.distributed as dist

    rank, world_size, _, weights, hidden = ep_layer
    if world_size != 1:
        pytest.skip("the id-scan bound is an EP world 1 path")
    layer = make_layer(rank, world_size, weights, hidden=hidden, capacity=WIDE_CAPACITY)
    try:
        layer.warmup(make_problem(16, rank=rank, seed=1, hidden=hidden).tensors())
        _check(
            layer,
            make_problem(WIDE_TOKENS, rank=rank, seed=220, hidden=hidden),
            weights,
            f"{WIDE_TOKENS} rows at world 1",
        )
    finally:
        layer.destroy()
        dist.barrier()


def test_output_view_is_the_workspace_output(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    if not layer.supports_output_view:
        pytest.skip("backend without output views")
    problem = make_problem(20, rank=rank, seed=180, hidden=hidden)
    view = layer.forward(problem.tensors(), return_workspace_view=True)
    torch.cuda.synchronize()
    assert view.data_ptr() == layer.output_buffer.data_ptr()
    check_band(
        view.clone(), reference(problem, weights), "output view vs w4a16_reference"
    )


def test_compile_bucket_bounds_the_rows(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    problem = make_problem(21, rank=rank, seed=190, hidden=hidden)
    layer.stage_inputs(problem.tensors(), compile_tokens_per_rank=64)
    out = layer.compute_staged(output=None)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (21, hidden)
    check_band(out.clone(), reference(problem, weights), "bucket 64 vs w4a16_reference")


def test_cuda_graph_replays_without_host_state(ep_layer) -> None:
    rank, world_size, layer, weights, hidden = ep_layer
    problem = make_problem(40, rank=rank, seed=200, hidden=hidden)
    tensors = problem.tensors()
    # Warm the bound and the launch plan eagerly, then capture one forward.
    eager = layer(tensors).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(tensors)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))
    # New inputs in place: the replay computes them (no host-side state).
    fresh = make_problem(40, rank=rank, seed=201, hidden=hidden)
    problem.hidden_states.copy_(fresh.hidden_states)
    problem.topk_ids.copy_(fresh.topk_ids)
    problem.topk_weights.copy_(fresh.topk_weights)
    graph.replay()
    torch.cuda.synchronize()
    check_band(
        captured.clone(), reference(fresh, weights), "graph replay vs w4a16_reference"
    )
    first = captured.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first.view(torch.int16), captured.view(torch.int16))


def test_pinned_cells_match_the_reference(ep_layer) -> None:
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.sm103.bf16_nvfp4_bf16_cake import CELLS

    rank, world_size, layer, weights, hidden = ep_layer
    problem = make_problem(32, rank=rank, seed=210, hidden=hidden)
    expected = reference(problem, weights)
    for cell in CELLS:
        pinned = make_layer(
            rank, world_size, weights, hidden=hidden, knobs=cell.as_knobs()
        )
        try:
            pinned.warmup(make_problem(8, rank=rank, seed=2, hidden=hidden).tensors())
            out = _forward(pinned, problem)
            check_band(out, expected, f"cell {cell.suffix}")
        finally:
            pinned.destroy()
            dist.barrier()
