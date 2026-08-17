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

from pathlib import Path

import torch

from flashinfer.fused_moe.sm90_nvfp4_repack import repack_nvfp4_sm90_v3
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_w4a8_gemm import (
    create_sm90_push_nvfp4_w4a8_gemm,
    get_sm90_push_nvfp4_w4a8_gemm_uri,
)
from tests.gemm.test_sm90_w4a8_gemm import (
    _checkpoint,
    _grouped_reference,
    _nonuniform_activation_scales,
    requires_sm90,
)


def _source() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe/src/"
        "nvfp4_w4a8_gemm/binding.cu"
    ).read_text(encoding="utf-8")


def _cache_stats(runner) -> tuple[int, ...]:
    return tuple(int(value) for value in runner.ffi_runner.tma_cache_stats())


def _view(device: torch.device, seed: int):
    torch.manual_seed(seed)
    return repack_nvfp4_sm90_v3(
        _checkpoint(device, rows=64),
        group_size=128,
        residual_scheme="generic",
    )


def _runner(max_m: int, view, **kwargs):
    return create_sm90_push_nvfp4_w4a8_gemm(
        max_m,
        view,
        payload_layout=3,
        allow_legacy_layout=True,
        **kwargs,
    )


def test_sm90_w4a8_tma_cache_and_tactic_contracts_are_source_visible():
    source = "".join(_source().split())
    assert "constexprsize_tkMaxWeightTmaCacheCapacity=128" in source
    assert "tma_cache_capacity_=static_cast<size_t>(tma_cache_capacity)" in source
    assert "std::list<W4A8WeightTmaCacheEntry>weight_tma_cache_" in source
    assert "weight_tma_cache_.splice(weight_tma_cache_.begin()" in source
    assert "weight_tma_cache_.pop_back()" in source
    assert "W4A8ResolvedTmaMapsresolve_tma_maps" in source
    assert (
        "resolved.group_scales=static_cast<constfloat*>(group_scales.data_ptr())"
        in source
    )
    assert "std::lock_guard<std::mutex>cache_lock(tma_cache_mutex_)" in source
    assert "dispatch_launch<DebugFp32>" in source
    assert (
        "find_w4a8_kernel_variant(BlockM,BlockN,GroupSize,Scheme,pipeline_stages)"
        in source
    )
    assert "tvm::ffi::Moduleinit(){returnmake_runner(3,4,false," in source
    assert "tvm::ffi::Moduleinit_with_tactics" in source


def test_sm90_w4a8_pipeline_stages_share_the_compiled_module_uri():
    default_uri = get_sm90_push_nvfp4_w4a8_gemm_uri(m64_stages=3, m128_stages=4)
    alternate_uri = get_sm90_push_nvfp4_w4a8_gemm_uri(m64_stages=2, m128_stages=3)
    assert alternate_uri == default_uri


@requires_sm90
def test_sm90_w4a8_tma_cache_reuses_weight_a_after_b_and_tracks_activation():
    device = torch.device("cuda")
    view_a = _view(device, 11)
    view_b = _view(device, 13)
    activation_a = torch.randn(17, 128, device=device).to(torch.float8_e4m3fn)
    activation_b = activation_a.clone()
    scales = _nonuniform_activation_scales(1, 32, device)
    offsets = torch.tensor([0, 17], dtype=torch.int64, device=device)
    runner = _runner(17, view_a)

    output_a = runner.run(activation_a, scales, offsets)
    runner.weight_view = view_b
    output_b = runner.run(activation_a, scales, offsets)
    runner.weight_view = view_a
    output_a_again = runner.run(activation_a, scales, offsets)

    assert _cache_stats(runner) == (2, 1, 1, 2, 0, 2, 128)
    torch.testing.assert_close(
        output_a.float(),
        _grouped_reference(activation_a, scales, view_a, offsets),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        output_b.float(),
        _grouped_reference(activation_a, scales, view_b, offsets),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(output_a_again, output_a, rtol=0, atol=0)
    assert not torch.equal(output_b, output_a)

    runner.run(activation_b, scales, offsets)
    assert _cache_stats(runner) == (2, 2, 2, 2, 0, 2, 128)

    runner.ffi_runner.configure_workspace(runner.workspace)
    assert _cache_stats(runner) == (0, 0, 0, 0, 0, 0, 128)


@requires_sm90
def test_sm90_w4a8_tma_cache_keys_activation_rows_separately_from_address():
    device = torch.device("cuda")
    view = _view(device, 19)
    backing = torch.randn(17, 128, device=device).to(torch.float8_e4m3fn)
    activation_17 = backing[:17]
    activation_9 = backing[:9]
    assert activation_17.data_ptr() == activation_9.data_ptr()
    scales = _nonuniform_activation_scales(1, 32, device)
    runner = _runner(17, view)

    runner.run(
        activation_17,
        scales,
        torch.tensor([0, 17], dtype=torch.int64, device=device),
    )
    runner.run(
        activation_9,
        scales,
        torch.tensor([0, 9], dtype=torch.int64, device=device),
    )

    assert _cache_stats(runner) == (0, 2, 1, 1, 0, 1, 128)


@requires_sm90
def test_sm90_w4a8_tma_cache_evicts_the_least_recently_used_weight():
    device = torch.device("cuda")
    views = [_view(device, seed) for seed in range(129)]
    activation = torch.randn(1, 128, device=device).to(torch.float8_e4m3fn)
    scales = torch.ones(1, 32, dtype=torch.float32, device=device)
    offsets = torch.tensor([0, 1], dtype=torch.int64, device=device)
    runner = _runner(1, views[0])

    for view in views:
        runner.weight_view = view
        runner.run(activation, scales, offsets)

    assert _cache_stats(runner) == (128, 1, 0, 129, 1, 128, 128)

    runner.weight_view = views[0]
    runner.run(activation, scales, offsets)
    assert _cache_stats(runner) == (129, 1, 0, 130, 2, 128, 128)


@requires_sm90
def test_sm90_w4a8_tma_cache_capacity_one_matches_single_slot_semantics():
    device = torch.device("cuda")
    views = [_view(device, seed) for seed in (29, 31)]
    activation = torch.randn(3, 128, device=device).to(torch.float8_e4m3fn)
    scales = torch.ones(1, 32, dtype=torch.float32, device=device)
    offsets = torch.tensor([0, 3], dtype=torch.int64, device=device)
    runner = _runner(
        3,
        views[0],
        tma_cache_capacity=1,
    )

    first = runner.run(activation, scales, offsets)
    runner.weight_view = views[1]
    runner.run(activation, scales, offsets)
    runner.weight_view = views[0]
    replay = runner.run(activation, scales, offsets)

    assert torch.equal(first, replay)
    assert _cache_stats(runner) == (2, 1, 0, 3, 2, 1, 1)


@requires_sm90
def test_sm90_w4a8_pipeline_stage_selection_contract():
    device = torch.device("cuda")
    view = _view(device, 17)

    default_runner = _runner(1, view)
    assert int(default_runner.ffi_runner.selected_pipeline_stage(64)) == 3
    assert int(default_runner.ffi_runner.selected_pipeline_stage(128)) == 4

    alternate_runner = _runner(
        1,
        view,
        m64_stages=2,
        m128_stages=3,
    )
    assert int(alternate_runner.ffi_runner.selected_pipeline_stage(64)) == 2
    assert int(alternate_runner.ffi_runner.selected_pipeline_stage(128)) == 3
