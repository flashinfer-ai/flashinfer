"""Trellis weight-source parity for the dynamic W4A8 MoE kernel.

Drives MoEDynamicKernelBackend(quant_recipe="w4a8_trellis") with a uniform
K2 payload and compares against the route-major trellis W4A8 scaffold with
identity Hadamard transforms. Both paths quantize activations to MXFP8 and
decode the identical payload through the lut_e4m3 value table, so outputs agree to
accumulation-order tolerance.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import make_ptr

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x.moe.fused_moe._impl import _DynamicMoEW4A8Launch, current_cuda_stream
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x._lib.quant.lut_e4m3 import (
    lut_e4m3_direct_table_cpu,
    lut_e4m3_value_table_cpu,
)
from b12x.testing.reference.trellis_moe import (
    build_trellis_weight,
    trellis_moe_reference,
)
from b12x.testing.reference.helpers import require_b12x
from tests.experimental.b12x.moe.test_w4a8_dynamic_kernel import (
    _OPT_LEVEL_2,
    _fake_f32,
    _fake_i32,
    _gptr,
)

require_b12x(allow_module_level=True)

_TILE_M = 16
_TILE_N = 128
_BITS = 2


def _h128(device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return (h / (128.0 ** 0.5)).to(device=device, dtype=torch.float32)


def _run_trellis_dynamic(
    *,
    activation: str,
    E: int,
    m: int,
    K: int,
    n: int,
    top_k: int,
    seed: int,
    keepalive: list | None = None,
    mac: int = 4,
    recipe: str = "w4a8_trellis",
    tile_m: int = _TILE_M,
    split_materialized: bool = False,
    intermediate_hadamard: bool = False,
    direct_lut: bool = False,
    direct: bool = False,
    topk_ids_override: torch.Tensor | None = None,
):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    w1_n = 2 * n

    x = (torch.randn(m, K, device=device) * 1.0e-2).to(torch.bfloat16)
    gen = torch.Generator(device="cpu").manual_seed(seed * 7 + 5)
    gate_payload, gate_w = build_trellis_weight(gen, E, n, K, _BITS, device)
    up_payload, up_w = build_trellis_weight(gen, E, n, K, _BITS, device)
    down_payload, down_w = build_trellis_weight(gen, E, K, n, _BITS, device)
    w13_i16 = torch.stack((gate_payload, up_payload), dim=0).contiguous()
    w2_i16 = down_payload.contiguous()
    w13_ref = torch.stack((gate_w, up_w), dim=0)
    if topk_ids_override is None:
        topk_ids = torch.stack(
            [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
        ).to(torch.int32)
    else:
        if tuple(topk_ids_override.shape) != (m, top_k):
            raise ValueError(
                "topk_ids_override must have shape "
                f"{(m, top_k)}, got {tuple(topk_ids_override.shape)}"
            )
        topk_ids = topk_ids_override.to(
            device=device, dtype=torch.int32
        ).contiguous()
    topk_weights = torch.softmax(
        torch.randn(m, top_k, device=device), dim=-1
    ).float()

    active_routes = (topk_ids >= 0) & (topk_ids < E)
    reference_ids = torch.where(
        active_routes, topk_ids, torch.zeros_like(topk_ids)
    )
    reference_weights = torch.where(
        active_routes, topk_weights, torch.zeros_like(topk_weights)
    )

    # ---- torch oracle in the kernel bases ----
    rot_cols = (6 if intermediate_hadamard else 3) * n
    rotations = (
        torch.rand((E, rot_cols), generator=gen, dtype=torch.float32)
        .mul_(1.0)
        .add_(0.5)
    ).to(device=device, dtype=torch.float16)
    if intermediate_hadamard:
        signs = (
            torch.randint(0, 2, (E, 3 * n), generator=gen)
            .mul_(2)
            .sub_(1)
            .to(torch.float16)
        ).to(device)
        rotations[:, 3 * n :] = signs
    oracle = trellis_moe_reference(
        x.float(),
        w13_ref,
        down_w,
        rotations,
        reference_ids,
        reference_weights,
        intermediate_hadamard=intermediate_hadamard,
    )
    torch.cuda.synchronize()

    # ---- dynamic kernel, trellis source ----
    # The reference mirrors the kernel transforms exactly, so both consume
    # the same input and compare in the raw scatter basis.
    x_dyn = x
    # Projection-major flat payloads: w13 [proj][E][K16][N16], down
    # [E][K16][N16] (the prepared trellis layout, shared with the micro path).
    w13_flat = w13_i16.contiguous().view(torch.int32).reshape(-1)
    down_flat = w2_i16.contiguous().view(torch.int32).reshape(-1)
    if direct_lut:
        # Rate-indexed 192 KiB state table; the phase kernels gather from it
        # in global memory (bit-identical to the value-table decode).
        value_table = lut_e4m3_direct_table_cpu().to(device)
    else:
        value_table = lut_e4m3_value_table_cpu().to(device)
    rot_flat = rotations.reshape(-1).contiguous()
    sentinel_u32 = torch.zeros(1, dtype=torch.uint32, device=device)
    # Dummy packed-FP4 weight tensors: shape-only (TMA descriptors and tile
    # counts); never dereferenced by the trellis staging.
    w13_packed = torch.zeros(E, w1_n, K // 2, dtype=torch.uint8, device=device)
    w2_packed = torch.zeros(E, K, n // 2, dtype=torch.uint8, device=device)
    scale_flat = torch.zeros(
        (E + m * top_k + 1) * tile_m * (K // 8), dtype=torch.uint8, device=device
    )

    phys_tiles = E + (m * top_k + tile_m - 1) // tile_m
    rows_padded = phys_tiles * tile_m
    gate_tile_cnt = (w1_n // _TILE_N) // 2
    max_tasks = phys_tiles * max(gate_tile_cnt, 1)

    packed_a = torch.zeros(rows_padded * K, dtype=torch.uint8, device=device)
    intermediate_u32 = torch.zeros(
        rows_padded * (n + n // 32) // 4, dtype=torch.int32, device=device
    )

    def z1():
        return torch.zeros(1, dtype=torch.int32, device=device)

    barrier_count, barrier_epoch = z1(), z1()
    pair_head, producers_done, all_pub = z1(), z1(), z1()
    task_head, task_tail = z1(), z1()

    def zt():
        return torch.zeros(max_tasks, dtype=torch.int32, device=device)

    task_ready, task_expert, task_m_tile = zt(), zt(), zt()
    task_slice_begin, task_slice_count, task_valid_rows = zt(), zt(), zt()
    tile_write_count = torch.zeros(phys_tiles, dtype=torch.int32, device=device)
    row_counts = torch.zeros(E, dtype=torch.int32, device=device)
    expert_write_rows = torch.zeros(E, dtype=torch.int32, device=device)
    expert_tile_base = torch.zeros(E + 1, dtype=torch.int32, device=device)
    token_map = torch.zeros(rows_padded, dtype=torch.int32, device=device)
    token_weights = torch.zeros(rows_padded, dtype=torch.float32, device=device)
    scatter_output = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    ones = torch.ones(E, device=device)
    flat_ids = topk_ids.reshape(-1).contiguous()
    flat_weights = topk_weights.reshape(-1).contiguous()

    import os as _os
    _direct = direct or _os.environ.get("BENCH_DIRECT", "0") == "1"
    _share = (
        split_materialized
        or _direct
        or _os.environ.get("BENCH_SHARE", "0") == "1"
    )
    if recipe == "w4a8_trellis":
        kernel = MoEDynamicKernelBackend(
            16,
            (tile_m, _TILE_N),
            activation=activation,
            quant_recipe="w4a8_trellis",
            w4a8_repacked=True,
            num_topk=top_k,
            trellis_bits=_BITS,
            trellis_intermediate_hadamard=intermediate_hadamard,
            trellis_direct_lut=direct_lut,
            direct_routing=_direct,
            materialize_intermediate=split_materialized or _direct,
            share_input_across_experts=_share,
        )
    else:
        kernel = MoEDynamicKernelBackend(
            16,
            (tile_m, _TILE_N),
            activation=activation,
            quant_recipe=recipe,
            w4a8_repacked=True,
            num_topk=top_k,
            direct_routing=_direct,
            materialize_intermediate=split_materialized or _direct,
            share_input_across_experts=_share,
        )
        w13_flat = torch.zeros(
            E * (w1_n // 256) * (K // 128) * 4096,
            dtype=torch.int32, device=device,
        )
        down_flat = torch.zeros(
            E * (K // 256) * (n // 128) * 4096,
            dtype=torch.int32, device=device,
        )
        sentinel_u32 = torch.zeros(
            max(E * (w1_n // 256) * (K // 128) * 256,
                E * (K // 256) * (n // 128) * 256),
            dtype=torch.uint32, device=device,
        )
    launch = _DynamicMoEW4A8Launch(kernel, k=K, n=n, w1_n=w1_n, num_topk=top_k)

    weight_dtype = cutlass.Float4E2M1FN
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (w1_n, K, E), stride_order=(1, 0, 2), assumed_align=16
    )
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (K, n, E), stride_order=(1, 0, 2), assumed_align=16
    )

    def fake_ptr_u8():
        return make_ptr(
            cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
        )

    def fake_ptr_i32():
        return make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)

    def fake_ptr_u32():
        return make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)

    compiled = b12x_compile(
        launch,
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(weight_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(),
        fake_ptr_u8(),
        fake_ptr_u32(),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        b_w13_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        b_down_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(),
        fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(),
        _fake_i32((E,)), _fake_i32((E,)), _fake_i32((E + 1,)),
        _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1, 1, 1, 1, 1, 1, 1,
        current_cuda_stream(),
        fake_ptr_u8(),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        compile_spec=KernelCompileSpec.from_fields(
            "tests.w4a8_dynamic.trellis",
            1,
            ("recipe", recipe),
            ("tile_m", tile_m),
            ("split", int(split_materialized)),
            ("intermediate_hadamard", int(intermediate_hadamard)),
            ("direct", int(_direct)),
            ("direct_lut", int(direct_lut)),
            ("share", int(_share)),
            ("activation", activation),
            ("bits", _BITS),
            ("experts", E),
            ("hidden", K),
            ("intermediate", n),
            ("top_k", top_k),
        ),
        dsl_compile_options=_OPT_LEVEL_2,
    )

    compiled(
        _gptr(cutlass.BFloat16, x_dyn),
        _gptr(cutlass.Int32, flat_ids, 4),
        _gptr(cutlass.Float32, flat_weights, 4),
        _gptr(weight_dtype, packed_a),
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        _gptr(cutlass.Uint8, packed_a),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint32, intermediate_u32),
        barrier_count, barrier_epoch, pair_head, producers_done, all_pub,
        task_head, task_tail,
        _gptr(cutlass.Int32, task_ready, 4),
        _gptr(cutlass.Int32, task_expert, 4),
        _gptr(cutlass.Int32, task_m_tile, 4),
        _gptr(cutlass.Int32, task_slice_begin, 4),
        _gptr(cutlass.Int32, task_slice_count, 4),
        _gptr(cutlass.Int32, task_valid_rows, 4),
        _gptr(cutlass.Int32, tile_write_count, 4),
        w13_packed,
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        w2_packed,
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint32, w13_flat),
        _gptr(cutlass.Uint32, sentinel_u32),
        _gptr(cutlass.Uint32, down_flat),
        _gptr(cutlass.Uint32, sentinel_u32),
        row_counts, expert_write_rows, expert_tile_base,
        ones, ones, ones, ones,
        _gptr(cutlass.BFloat16, scatter_output),
        _gptr(cutlass.Int32, token_map, 4),
        _gptr(cutlass.Float32, token_weights, 4),
        m,
        m * top_k,
        m,
        rows_padded,
        max_tasks,
        phys_tiles,
        mac,
        current_cuda_stream(),
        _gptr(cutlass.Uint8, value_table),
        _gptr(cutlass.Float16, rot_flat),
    )
    torch.cuda.synchronize()
    if keepalive is not None:
        keepalive.append(dict(locals()))
    return scatter_output.float(), oracle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["situ"])
@pytest.mark.parametrize("m", [1, 3])
def test_dynamic_trellis_matches_scaffold(activation: str, m: int) -> None:
    got, want = _run_trellis_dynamic(
        activation=activation, E=8, m=m, K=512, n=256, top_k=4, seed=20260811
    )
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got) > 0
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), want.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
    rel_l2 = ((got - want).norm() / want.norm().clamp_min(1e-9)).item()
    assert rel_l2 < 0.12, rel_l2
    for row in range(got.shape[0]):
        row_cos = torch.nn.functional.cosine_similarity(
            got[row : row + 1], want[row : row + 1]
        ).item()
        assert row_cos > 0.995, (row, row_cos)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dynamic_w4a8_m1_direct_ignores_inactive_routes() -> None:
    """M=1 direct scheduling must skip every inactive expert route."""

    keepalive: list[dict[str, object]] = []
    got, want = _run_trellis_dynamic(
        activation="silu",
        E=4,
        m=1,
        K=256,
        n=128,
        top_k=2,
        seed=20260815,
        recipe="w4a8_mx",
        direct=True,
        topk_ids_override=torch.full((1, 2), -1, dtype=torch.int32),
        keepalive=keepalive,
    )
    torch.testing.assert_close(want, torch.zeros_like(want), rtol=0, atol=0)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    packed_a = keepalive[0]["packed_a"]
    assert isinstance(packed_a, torch.Tensor)
    assert torch.count_nonzero(packed_a).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dynamic_w4a8_shared_input_ignores_inactive_routes() -> None:
    """Shared-input scheduling must not materialize an inactive token."""

    keepalive: list[dict[str, object]] = []
    got, want = _run_trellis_dynamic(
        activation="silu",
        E=4,
        m=3,
        K=256,
        n=128,
        top_k=2,
        seed=20260815,
        recipe="w4a8_mx",
        split_materialized=True,
        topk_ids_override=torch.full((3, 2), -1, dtype=torch.int32),
        keepalive=keepalive,
    )
    torch.testing.assert_close(want, torch.zeros_like(want), rtol=0, atol=0)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    packed_a = keepalive[0]["packed_a"]
    assert isinstance(packed_a, torch.Tensor)
    assert torch.count_nonzero(packed_a).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [33, 64])
def test_dynamic_trellis_split_materialized_matches_scaffold(m: int) -> None:
    got, want = _run_trellis_dynamic(
        activation="situ",
        E=8,
        m=m,
        K=512,
        n=256,
        top_k=4,
        seed=20260812,
        tile_m=64,
        split_materialized=True,
        mac=64,
    )
    assert torch.isfinite(got).all()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), want.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
    rel_l2 = ((got - want).norm() / want.norm().clamp_min(1e-9)).item()
    assert rel_l2 < 0.12, rel_l2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [33])
def test_dynamic_trellis_intermediate_hadamard_split_matches_scaffold(m: int) -> None:
    got, want = _run_trellis_dynamic(
        activation="situ",
        E=8,
        m=m,
        K=512,
        n=256,
        top_k=4,
        seed=20260814,
        tile_m=64,
        split_materialized=True,
        intermediate_hadamard=True,
        mac=64,
    )
    assert torch.isfinite(got).all()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), want.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
    rel_l2 = ((got - want).norm() / want.norm().clamp_min(1e-9)).item()
    assert rel_l2 < 0.12, rel_l2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 3])
def test_dynamic_trellis_intermediate_hadamard_matches_scaffold(m: int) -> None:
    got, want = _run_trellis_dynamic(
        activation="situ",
        E=8,
        m=m,
        K=512,
        n=256,
        top_k=4,
        seed=20260813,
        intermediate_hadamard=True,
    )
    assert torch.isfinite(got).all()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), want.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
    rel_l2 = ((got - want).norm() / want.norm().clamp_min(1e-9)).item()
    assert rel_l2 < 0.12, rel_l2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("intermediate_hadamard", [False, True])
def test_dynamic_trellis_direct_table_matches_value_table(intermediate_hadamard: bool) -> None:
    """The direct table precomposes the lut_e4m3 permutation with the value
    table, so the split phase kernels decode identical weight
    bytes under either mode (pinned bit-level by
    test_trellis_direct_lut_decode). The split path's scattered top-k
    accumulation is not run-to-run deterministic, so outputs are compared
    within that noise floor and against the reference."""

    kwargs = dict(
        activation="situ", E=8, m=96, K=512, n=256, top_k=4,
        seed=20260812, tile_m=64, split_materialized=True, intermediate_hadamard=intermediate_hadamard,
    )
    got_value_table, want = _run_trellis_dynamic(**kwargs, direct_lut=False)
    got_direct, _ = _run_trellis_dynamic(**kwargs, direct_lut=True)
    assert torch.allclose(got_value_table, got_direct, rtol=0.0, atol=0.06)
    cosine = torch.nn.functional.cosine_similarity(
        got_direct.reshape(1, -1), want.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
