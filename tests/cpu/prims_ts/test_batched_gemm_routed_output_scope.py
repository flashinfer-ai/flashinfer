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

"""Repeat-check scope for routed plain output with expert padding."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
    _active_route_output,
    _same_launch_output,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("is_swap_ab", [False, True])
def test_routed_repeat_scope_ignores_padding_but_detects_active_changes(is_swap_ab):
    active = torch.tensor([True, False, True, False])
    first = torch.tensor([[1.0, 2.0], [float("nan"), 0.0], [3.0, 4.0], [7.0, 8.0]])
    second = first.clone()
    second[~active] = torch.tensor([[9.0, float("nan")], [11.0, 12.0]])
    # The last row is declared capacity beyond the routed padded extent.
    second = torch.cat([second, torch.tensor([[13.0, 14.0]])])
    if is_swap_ab:
        first, second = first.T, second.T

    def observable(output):
        return _active_route_output(output, active, is_swap_ab=is_swap_ab)

    assert _same_launch_output(observable(first), observable(second))
    second[0, 0] += 1
    assert not _same_launch_output(observable(first), observable(second))
    second[0, 0] = float("nan")
    assert not _same_launch_output(observable(first), observable(second))
    # Selection must retain poison on real routes for the coverage check.
    assert torch.isnan(observable(second)[0, 0])


@pytest.mark.parametrize("is_swap_ab", [False, True])
def test_routed_repeat_scope_rejects_missing_token_storage(is_swap_ab):
    output = torch.zeros((3, 2))
    if is_swap_ab:
        output = output.T
    with pytest.raises(AssertionError, match="token extent"):
        _active_route_output(
            output, torch.tensor([True, False, True, False]), is_swap_ab=is_swap_ab
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("is_swap_ab", [False, True])
def test_routed_repeat_scope_compares_signed_zero_bits(dtype, is_swap_ab):
    first = torch.zeros((2, 2), dtype=dtype)
    second = first.clone()
    active = torch.tensor([True, False])
    if is_swap_ab:
        first, second = first.T, second.T

    def observable(output):
        return _active_route_output(output, active, is_swap_ab=is_swap_ab)

    second[1, 1] = -0.0
    assert _same_launch_output(observable(first), observable(second))
    second[0, 0] = -0.0
    assert not _same_launch_output(observable(first), observable(second))


@pytest.mark.parametrize("path", ["preloaded", "cooperative", "steps"])
def test_finalize_padded_n_skips_reductions_without_changing_scratch_protocol(
    monkeypatch, path
):
    from types import SimpleNamespace as NS
    from tests.prims_ts.python_body import python_body
    from flashinfer.prims_ts.batched_gemm import gmem_c_resources as module
    from flashinfer.prims_ts.batched_gemm.epilogue_schedule import (
        EpilogueAction,
        EpilogueOp,
    )

    monkeypatch.setattr(module, "Int32", int)
    monkeypatch.setattr(module.cutlass, "Int64", int)
    monkeypatch.setattr(module.cutlass, "range_constexpr", range)
    monkeypatch.setattr(module.cute.arch, "warp_idx", lambda: 0)
    monkeypatch.setattr(module.cute.arch, "lane_idx", lambda: 0)
    monkeypatch.setattr(module, "nonnegative_div", lambda a, b: a // b)
    monkeypatch.setattr(module, "nonnegative_mod", lambda a, b: a % b)
    events = []
    for name in (
        "cp_reduce_async_bulk_global_shared_cta",
        "cp_async_bulk_commit_group",
        "cp_async_bulk_wait_group",
        "barrier_cta_sync",
    ):
        monkeypatch.setattr(
            module.prims, name, lambda *a, name=name, **kw: events.append(name)
        )
    monkeypatch.setattr(module.cute.arch, "fence_view_async_shared", lambda: None)
    memory = NS(subview=lambda *a: memory, store=lambda *a, **kw: None)
    cfg = NS(
        tile_m=256,
        tile_n=256,
        num_bytes_c_per_stage=128 * 256 * 2,
        num_stages_c_smem=2,
        dtype_c_bits=16,
        finalize_smem_row_stride=256,
        warp_layout=NS(epilogue_warp_idx=0),
        preload_epilogue_plan=NS(gated=False, finalize=True, barrier_threads=256),
    )
    resource = NS(
        cfg=cfg,
        tile_token_limit=256,
        problem_n=1280,
        num_tokens=1,
        route_map_view=NS(load=lambda **kw: [0]),
        gC=memory,
        sC=memory,
    )
    protocols = []
    for tile_n in (4, 5):
        events.clear()
        stage = NS(work_tile=NS(tile_idx=(0, tile_n, 0)))
        if path == "preloaded":
            python_body(module.GmemCResource._store_preloaded_finalize)(resource, stage)
            expected_reductions = 2
        elif path == "cooperative":
            python_body(module.GmemCResource._reduce_cooperative_fragment)(
                resource, None, 0, tile_n, 3
            )
            expected_reductions = 1
        else:
            action = python_body(module.GmemCResource._step_output_action)
            for op in (
                EpilogueOp.EMIT,
                EpilogueOp.COMMIT,
                EpilogueOp.SOURCE_WAIT,
                EpilogueOp.BARRIER,
            ):
                action(resource, stage, EpilogueAction(op, fragment=0, slot=0))
            expected_reductions = 1
        reduce = "cp_reduce_async_bulk_global_shared_cta"
        assert events.count(reduce) == (expected_reductions if tile_n == 4 else 0)
        protocols.append([event for event in events if event != reduce])
    assert protocols[0] == protocols[1]
    assert "cp_async_bulk_commit_group" in protocols[1]
    assert "cp_async_bulk_wait_group" in protocols[1]
