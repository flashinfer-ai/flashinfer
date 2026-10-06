"""Phase-2a integration test for the NVFP4 split-materialized dynamic back-end.

Drives ``MoEDynamicKernelBackend(quant_recipe="nvfp4",
materialize_intermediate=True, mma_tiler_mn=(128, 128),
share_input_across_experts=True)`` through the production ``@cute.jit`` host
adapter ``_DynamicMoELaunch`` so that ONE traced call launches the
cooperative route/pack front-end, then the external
``Nvfp4MaterializedPhase1Kernel`` / ``Nvfp4MaterializedPhase2Kernel`` on the
same stream.

Gates:
  * correctness vs ``moe_reference_nvfp4`` (the same direct-division, gs=1
    domain builder used by tests/moe/test_nvfp4_phase_kernels.py) on M=64 and
    M=256, with the ``_bf16_output_bound`` absolute ceiling;
  * split vs non-split (monolithic) back-end agreement on identical inputs
    (cos > 0.9999 against each other and against the oracle);
  * CUDA-graph capture of the three-kernel launch with bitwise-identical
    replays (top-k=1 so every output byte has a single atomic producer) and
    zero new device allocations during replay;
  * construction-time fail-closed gates for invalid split combinations.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import make_ptr

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe._shared.kernels.reference import compare_to_reference
from b12x.moe.fused_moe._impl import _DynamicMoELaunch

from b12x.testing.reference.helpers import require_b12x
from tests.experimental.b12x.moe.test_nvfp4_phase_kernels import (
    _bf16_output_bound,
    _build_domain,
    _gptr,
)

_TILE_M = 128
_TILE_N = 128


def _fake_i32(shape):
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, shape, assumed_align=4)


def _fake_f32(shape):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, shape, assumed_align=16
    )


def _split_workspace(domain):
    """Host-side launch buffers mirroring the _impl materialized sizing."""
    E, K, n = domain["E"], domain["K"], domain["n"]
    m, top_k = domain["m"], domain["top_k"]
    gate_tile_cnt = n // _TILE_N
    # Worst case: every expert keeps one partial tile alive.
    phys_tiles = E + (m * top_k + _TILE_M - 1) // _TILE_M
    rows_padded = phys_tiles * _TILE_M
    max_tasks = phys_tiles * max(gate_tile_cnt, 1)
    intermediate_tiles = n // _TILE_N
    device = torch.device("cuda")

    packed_a = torch.zeros(rows_padded * (K // 2), dtype=torch.uint8, device=device)
    # Adapter view: rows_padded * align_up(K/16, 4) bytes; the swizzled atom
    # plane itself needs rows_padded * (K/64) * 512 / rows_padded rows, which
    # the same buffer covers.
    scale_flat = torch.zeros(
        rows_padded * ((K + 63) // 64) * 4, dtype=torch.uint8, device=device
    )
    words_per_row = intermediate_tiles * 16
    intermediate_u32 = torch.zeros(
        rows_padded * words_per_row + intermediate_tiles * rows_padded * 2,
        dtype=torch.int32,
        device=device,
    )

    def z1():
        return torch.zeros(1, dtype=torch.int32, device=device)

    def zt():
        return torch.zeros(max_tasks, dtype=torch.int32, device=device)

    ws = {
        "phys_tiles": phys_tiles,
        "rows_padded": rows_padded,
        "max_tasks": max_tasks,
        "gate_tile_cnt": gate_tile_cnt,
        "packed_a": packed_a,
        "scale_flat": scale_flat,
        "intermediate_u32": intermediate_u32,
        "barrier_count": z1(),
        "barrier_epoch": z1(),
        "pair_head": z1(),
        "producers_done": z1(),
        "all_pub": z1(),
        "task_head": z1(),
        "task_tail": z1(),
        "task_ready": zt(),
        "task_expert": zt(),
        "task_m_tile": zt(),
        "task_slice_begin": zt(),
        "task_slice_count": zt(),
        "task_valid_rows": zt(),
        "tile_write_count": torch.zeros(phys_tiles, dtype=torch.int32, device=device),
        "row_counts": torch.zeros(E, dtype=torch.int32, device=device),
        "expert_write_rows": torch.zeros(E, dtype=torch.int32, device=device),
        "expert_tile_base": torch.zeros(E + 1, dtype=torch.int32, device=device),
        "token_map": torch.zeros(rows_padded, dtype=torch.int32, device=device),
        "token_weights": torch.zeros(rows_padded, dtype=torch.float32, device=device),
        "scatter_output": torch.zeros(m, K, dtype=torch.bfloat16, device=device),
        "ones_f32": torch.ones(E, dtype=torch.float32, device=device),
    }
    return ws


def _compile_launch(domain, *, materialize: bool, spec_name: str):
    E, K, n = domain["E"], domain["K"], domain["n"]
    top_k = domain["top_k"]
    w1_n = domain["w1_n"]
    kernel = MoEDynamicKernelBackend(
        16,
        (_TILE_M, _TILE_N),
        quant_recipe="nvfp4",
        activation="silu",
        share_input_across_experts=True,
        num_topk=top_k,
        materialize_intermediate=materialize,
        deterministic_output=False,
    )
    if materialize:
        assert kernel.nvfp4_split_materialized
        assert kernel.external_materialized_fc1
        assert kernel.external_materialized_fc2
    else:
        assert not kernel.external_materialized_fc1

    launch = _DynamicMoELaunch(kernel, k=K, n=n, w1_n=w1_n, num_topk=top_k)
    weight_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (w1_n, K, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (K, n, E), stride_order=(1, 0, 2), assumed_align=16
    )

    def fake_ptr_u8():
        return make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)

    def fake_ptr_i32():
        return make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)

    def fake_ptr_u32():
        return make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)

    def fake_ptr_sf():
        return make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)

    compiled = b12x_compile(
        launch,
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(weight_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_sf(),
        fake_ptr_u8(),
        fake_ptr_u8(),
        fake_ptr_u32(),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        _fake_i32((1,)),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        fake_ptr_i32(),
        b_w13_fake,
        fake_ptr_sf(),
        fake_ptr_sf(),  # sfb_w13_gate_ptr (unused placeholder under packed w13)
        b_down_fake,
        fake_ptr_sf(),
        _fake_i32((E,)),
        _fake_i32((E,)),
        _fake_i32((E + 1,)),
        _fake_f32((E,)),
        _fake_f32((E,)),
        _fake_f32((E,)),
        _fake_f32((E,)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_fields(
            spec_name,
            1,
            ("experts", E),
            ("hidden", K),
            ("intermediate", n),
            ("top_k", top_k),
            ("materialized", materialize),
            ("tile_m", _TILE_M),
        ),
    )
    return compiled


def _make_launcher(compiled, domain, ws, scales=None):
    E, K, n = domain["E"], domain["K"], domain["n"]
    m, top_k = domain["m"], domain["top_k"]
    w1_n = domain["w1_n"]
    weight_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    flat_ids = domain["topk_ids"].reshape(-1).contiguous()
    flat_weights = domain["topk_weights"].reshape(-1).contiguous()
    mac = 4
    # input_global_scale(a1), alpha(FC1 dequant), down_alpha(FC2 dequant),
    # global_scale(a2 requant).  Default to ones so the layout/routing gates
    # stay scale-invariant; the non-unity cross-check overrides them.
    scales = scales or {}
    ones = ws["ones_f32"]
    input_gs = scales.get("input_global_scale", ones)
    alpha = scales.get("alpha", ones)
    down_alpha = scales.get("down_alpha", ones)
    global_scale = scales.get("global_scale", ones)

    def _launch():
        compiled(
            _gptr(cutlass.BFloat16, domain["x"]),
            _gptr(cutlass.Int32, flat_ids, 4),
            _gptr(cutlass.Float32, flat_weights, 4),
            _gptr(weight_dtype, ws["packed_a"]),
            _gptr(sf_dtype, ws["scale_flat"]),
            _gptr(cutlass.Uint8, ws["packed_a"]),
            _gptr(cutlass.Uint8, ws["scale_flat"]),
            _gptr(cutlass.Uint32, ws["intermediate_u32"]),
            ws["barrier_count"],
            ws["barrier_epoch"],
            ws["pair_head"],
            ws["producers_done"],
            ws["all_pub"],
            ws["task_head"],
            ws["task_tail"],
            _gptr(cutlass.Int32, ws["task_ready"], 4),
            _gptr(cutlass.Int32, ws["task_expert"], 4),
            _gptr(cutlass.Int32, ws["task_m_tile"], 4),
            _gptr(cutlass.Int32, ws["task_slice_begin"], 4),
            _gptr(cutlass.Int32, ws["task_slice_count"], 4),
            _gptr(cutlass.Int32, ws["task_valid_rows"], 4),
            _gptr(cutlass.Int32, ws["tile_write_count"], 4),
            domain["w13_packed"],
            _gptr(sf_dtype, domain["w13_sfb"]),
            _gptr(sf_dtype, domain["w13_sfb"]),
            domain["w2_packed"],
            _gptr(sf_dtype, domain["w2_sfb"]),
            ws["row_counts"],
            ws["expert_write_rows"],
            ws["expert_tile_base"],
            input_gs,
            alpha,
            down_alpha,
            global_scale,
            _gptr(cutlass.BFloat16, ws["scatter_output"]),
            _gptr(cutlass.Int32, ws["token_map"], 4),
            _gptr(cutlass.Float32, ws["token_weights"], 4),
            m,
            m * top_k,
            m,
            ws["rows_padded"],
            ws["max_tasks"],
            ws["phys_tiles"],
            mac,
            current_cuda_stream(),
        )

    return _launch


def _run_split_backend(domain):
    ws = _split_workspace(domain)
    compiled = _compile_launch(
        domain,
        materialize=True,
        spec_name="tests.nvfp4_split_backend.split",
    )
    _make_launcher(compiled, domain, ws)()
    torch.cuda.synchronize()
    return ws["scatter_output"], ws, compiled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_split_backend_matches_oracle_m64() -> None:
    """Route/pack front-end + external phase kernels vs the NVFP4 oracle."""
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=64, top_k=2, seed=21)
    out, _ws, _compiled = _run_split_backend(domain)
    assert out.abs().sum().item() > 0, "split launch produced all zeros"
    metrics = compare_to_reference(out.float(), domain["oracle"])
    assert metrics.cos > 0.9999, metrics
    bound = _bf16_output_bound(domain["oracle"])
    assert metrics.max_abs <= bound, (metrics, bound)
    assert metrics.rmse <= bound, (metrics, bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_split_backend_matches_oracle_m256() -> None:
    """Multi-tile experts (several 128-row tiles per expert) at M=256."""
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=256, top_k=2, seed=22)
    out, _ws, _compiled = _run_split_backend(domain)
    assert out.abs().sum().item() > 0
    metrics = compare_to_reference(out.float(), domain["oracle"])
    assert metrics.cos > 0.9999, metrics
    bound = _bf16_output_bound(domain["oracle"])
    assert metrics.max_abs <= bound, (metrics, bound)
    assert metrics.rmse <= bound, (metrics, bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_split_matches_monolithic_backend() -> None:
    """Same inputs through the split back-end and the monolithic back-end."""
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=128, top_k=2, seed=23)
    split_out, _ws, _compiled = _run_split_backend(domain)

    ws_mono = _split_workspace(domain)
    compiled_mono = _compile_launch(
        domain,
        materialize=False,
        spec_name="tests.nvfp4_split_backend.monolithic",
    )
    _make_launcher(compiled_mono, domain, ws_mono)()
    torch.cuda.synchronize()
    mono_out = ws_mono["scatter_output"]

    assert mono_out.abs().sum().item() > 0
    metrics_split = compare_to_reference(split_out.float(), domain["oracle"])
    metrics_mono = compare_to_reference(mono_out.float(), domain["oracle"])
    assert metrics_split.cos > 0.9999, metrics_split
    assert metrics_mono.cos > 0.9999, metrics_mono
    cross = compare_to_reference(split_out.float(), mono_out.float())
    assert cross.cos > 0.9999, cross


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_split_cuda_graph_replay_bitwise_and_allocations() -> None:
    """Capture cooperative+phase1+phase2 in one graph; replays are bitwise
    identical (top-k=1: one atomic producer per output byte) and allocate
    nothing."""
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=64, top_k=1, seed=24)
    ws = _split_workspace(domain)
    compiled = _compile_launch(
        domain,
        materialize=True,
        spec_name="tests.nvfp4_split_backend.split",
    )
    launch = _make_launcher(compiled, domain, ws)

    # Warm-up (also the reference run for bitwise comparison).
    launch()
    torch.cuda.synchronize()
    reference = ws["scatter_output"].clone()
    metrics = compare_to_reference(reference.float(), domain["oracle"])
    assert metrics.cos > 0.9999, metrics

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated()
    for replay_idx in range(3):
        ws["scatter_output"].fill_(float("nan"))  # poison: replay must rewrite
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            ws["scatter_output"].view(torch.uint8), reference.view(torch.uint8)
        ), f"replay {replay_idx} is not bitwise identical"
    after = torch.cuda.memory_allocated()
    assert before == after, (
        f"CUDA-graph replay allocated new device memory ({before} -> {after} bytes)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_split_matches_monolithic_under_real_scales() -> None:
    """Fold-argument wiring check: distinct per-expert alpha / down_alpha /
    global_scale (a2) must produce the SAME result on the split and monolithic
    back-ends.  The layout/oracle gates run at gs=1 and cannot catch a swapped
    or mis-folded scale operand (e.g. alpha applied once vs to both gate and
    up, or down_alpha vs global_scale order), so this drives the real
    non-unity dequant/requant path.  The a1 input scale is folded by the
    shared route/pack front-end, so it is exercised identically on both arms.
    """
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=128, top_k=2, seed=25)
    E = domain["E"]
    idx = torch.arange(E, dtype=torch.float32, device="cuda")
    scales = {
        "input_global_scale": (0.75 + 0.05 * idx),
        "alpha": (1.5 - 0.08 * idx),
        "down_alpha": (0.85 + 0.06 * idx),
        "global_scale": (1.25 + 0.04 * idx),
    }
    ws_split = _split_workspace(domain)
    compiled_split = _compile_launch(
        domain, materialize=True, spec_name="tests.nvfp4_split_backend.split_real"
    )
    _make_launcher(compiled_split, domain, ws_split, scales)()
    torch.cuda.synchronize()
    split_out = ws_split["scatter_output"]

    ws_mono = _split_workspace(domain)
    compiled_mono = _compile_launch(
        domain, materialize=False, spec_name="tests.nvfp4_split_backend.mono_real"
    )
    _make_launcher(compiled_mono, domain, ws_mono, scales)()
    torch.cuda.synchronize()
    mono_out = ws_mono["scatter_output"]

    assert split_out.abs().sum().item() > 0
    cross = compare_to_reference(split_out.float(), mono_out.float())
    assert cross.cos > 0.9999, cross
    bound = _bf16_output_bound(mono_out.float())
    assert cross.max_abs <= bound, (cross, bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_split_backend_rejects_invalid_materialized_combos() -> None:
    """Fail-closed construction gates (no launch, no GPU state required)."""
    require_b12x()
    base = dict(
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        quant_recipe="nvfp4",
        activation="silu",
        num_topk=2,
        materialize_intermediate=True,
    )
    # Accepted: shared-input SiLU grouped route/pack front-end.
    MoEDynamicKernelBackend(**{**base, "share_input_across_experts": True})
    for name, bad in [
        ("grouped-quantized per-route input", {"share_input_across_experts": False}),
        ("relu2 activation", {"activation": "relu2"}),
        ("M32 tile", {"mma_tiler_mn": (32, 128)}),
        ("direct routing", {"direct_routing": True}),
        ("deterministic output", {"deterministic_output": True}),
        ("dynamic down scale", {"dynamic_down_scale": True}),
    ]:
        try:
            MoEDynamicKernelBackend(
                **{**base, "share_input_across_experts": True, **bad}
            )
            pytest.fail(f"{name}: expected ValueError but construction succeeded")
        except ValueError:
            pass


if __name__ == "__main__":
    raise SystemExit("run via pytest")
