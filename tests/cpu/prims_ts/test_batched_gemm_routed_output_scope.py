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
    _active_output_scale_offsets,
    _active_quantized_output,
    _active_route_output,
    _poison_checked_output,
    _same_launch_output,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8, torch.float32])
def test_repeat_poison_detects_missing_c_or_scale_store_and_preserves_guards(dtype):
    storage = torch.full((32,), 19, dtype=dtype)
    output = storage[4:-4]
    touched = torch.zeros_like(output, dtype=torch.bool)
    touched[4:12] = True
    first = torch.arange(8).to(dtype)
    output[touched] = first
    for repeat in (1, 2, 3):
        _poison_checked_output(output, repeat, touched)
        assert not _same_launch_output(output[touched], first)
        assert torch.all(output[~touched] == 19)
        assert torch.all(storage[:4] == 19) and torch.all(storage[-4:] == 19)
        output[touched] = first


@pytest.mark.parametrize("written_bits", [0x0F, 0xF0], ids=["low-only", "high-only"])
def test_fp4_repeat_poison_detects_partial_byte_store(written_bits):
    expected = torch.tensor([0xA5], dtype=torch.uint8)
    masks = set()
    for repeat in range(256):
        output = expected.clone()
        _poison_checked_output(output, repeat)
        masks.add((output ^ expected).item())
        # Simulate a kernel restoring only one of the two packed outputs.
        output = (expected & written_bits) | (output & (0xFF ^ written_bits))
        assert not _same_launch_output(expected, output), repeat
    assert len(masks) > 1


@pytest.mark.parametrize("values_per_byte", [1, 2])
@pytest.mark.parametrize("layout", [1, 2])
def test_quantized_repeat_excludes_padded_tokens_but_keeps_guard_checks(
    values_per_byte, layout
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _sf_c_index_grid
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import SfLayout

    layout = int(SfLayout.R8c4 if layout == 1 else SfLayout.R128c4)
    active = torch.tensor([True, False, True, False])
    width, pitch, offset = 64, 128, 32
    storage = torch.full((4 * pitch // values_per_byte + 16,), 19, dtype=torch.uint8)
    scales = torch.full((4096,), 23, dtype=torch.uint8)
    offsets = _active_output_scale_offsets(
        active, width=width, pitch=pitch, offset=offset, block_size=16, layout=layout
    )
    full_grid = _sf_c_index_grid(
        output_m=width,
        n=4,
        sf_layout_c=layout,
        sf_block_size=16,
        is_swap_ab=True,
        device="cpu",
        pitch=pitch,
        m_offset=offset,
    )
    assert torch.equal(offsets, full_grid[::16, active].T)

    def observable():
        return _active_quantized_output(
            storage,
            active,
            width=width,
            pitch=pitch,
            offset=offset,
            values_per_byte=values_per_byte,
        )

    first, first_scales = observable().clone(), scales[offsets].clone()
    for row in (1, 3):
        storage[
            row * pitch // values_per_byte : (row + 1) * pitch // values_per_byte
        ] = 99
    padded_offsets = _active_output_scale_offsets(
        ~active, width=width, pitch=pitch, offset=offset, block_size=16, layout=layout
    )
    scales[padded_offsets] = 99
    assert _same_launch_output(first, observable())
    assert _same_launch_output(first_scales, scales[offsets])
    storage[-1] = 0  # Numeric scope does not replace the independent guard audit.
    assert _same_launch_output(first, observable())
    assert storage[-1] != 19
    storage[offset // values_per_byte] ^= 1
    scales[offsets[0, 0]] ^= 1
    assert not _same_launch_output(first, observable())
    assert not _same_launch_output(first_scales, scales[offsets])


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
