"""Direct-drive integration test for the w4a8 dynamic MoE kernel recipe.

Drives MoEDynamicKernelBackend(quant_recipe="w4a8_mx"/"w4a8_nvfp4") through
the _DynamicMoEW4A8Launch adapter with synthetic experts and gates the output
against the pure-Torch oracle (moe_reference_w4a8_mx).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import make_ptr

from b12x._lib.intrinsics import _fp4_encode_nibbles, fp4_quantize_values_torch
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from cutlass.base_dsl.compiler import OptLevel as _DSLOptLevel

_OPT_LEVEL_2 = _DSLOptLevel(2)
from b12x.moe.fused_moe._impl import (
    _DynamicMoEW4A8Launch,
    _pad_w4a8_grid_columns,
    _pad_w4a8_grid_rows,
    current_cuda_stream,
)
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe._shared.kernels.reference import (
    compare_to_reference,
    decompose_nvfp4_scales_to_mx_residual,
    moe_reference_w4a8_mx,
)

from b12x.testing.reference.helpers import require_b12x

_TILE_M = 128
_TILE_N = 128


def _pack_fp4_rows(values: torch.Tensor) -> torch.Tensor:
    nib = _fp4_encode_nibbles(values)
    pair = nib.view(*values.shape[:-1], values.shape[-1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).contiguous()


def _quantize_weight_mxfp4(w: torch.Tensor):
    rows, cols = w.shape
    blocked = w.view(rows, cols // 32, 32)
    bmax = blocked.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(bmax > 0, bmax / 6.0, torch.ones_like(bmax))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    byte = torch.where(bmax > 0, exponent + 127, torch.zeros_like(exponent)).to(torch.uint8)
    scale = torch.where(bmax > 0, torch.exp2(exponent), torch.zeros_like(exponent))
    q = fp4_quantize_values_torch(
        torch.where(scale > 0, blocked / scale.clamp(min=1e-30), torch.zeros_like(blocked)).view(rows, cols)
    )
    return _pack_fp4_rows(q), byte.squeeze(-1)


def _quantize_weight_nvfp4(w: torch.Tensor):
    rows, cols = w.shape
    blocked = w.view(rows, cols // 16, 16)
    bmax = blocked.abs().amax(dim=-1, keepdim=True)
    scale = (bmax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
    q = fp4_quantize_values_torch((blocked / scale.clamp(min=1e-30)).view(rows, cols))
    return _pack_fp4_rows(q), scale.squeeze(-1)


def _gptr(dtype, t: torch.Tensor, align: int = 16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _fake_i32(shape):
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, shape, assumed_align=4)


def _fake_f32(shape):
    return cute.runtime.make_fake_compact_tensor(cutlass.Float32, shape, assumed_align=16)


def _active_route_reference_inputs(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent inactive routes with safe indices and zero oracle weights."""

    active = (topk_ids >= 0) & (topk_ids < num_experts)
    safe_ids = torch.where(active, topk_ids, torch.zeros_like(topk_ids))
    active_weights = torch.where(active, topk_weights, torch.zeros_like(topk_weights))
    return safe_ids, active_weights


def _run_w4a8_dynamic(
    *,
    recipe: str,
    activation: str,
    E: int,
    m: int,
    K: int,
    n: int,
    top_k: int,
    seed: int,
    tile_m: int = _TILE_M,
    return_launcher: bool = False,
    return_state: bool = False,
    return_debug: bool = False,
    route_weight_slot: int | None = None,
    capture_intermediate: bool = False,
    capture_fc1_raw: bool = False,
    capture_fc1_staged: bool = False,
    topk_ids_override: torch.Tensor | None = None,
    compiled_override=None,
):
    if return_state and not return_launcher:
        raise ValueError("return_state requires return_launcher=True")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    is_gated = activation in {"silu", "situ"}
    w1_n = 2 * n if is_gated else n

    x = (torch.randn(m, K, device=device) * 2.0).to(torch.bfloat16)
    w13_full = torch.randn(E, w1_n, K, device=device) * 0.05
    w2_full = torch.randn(E, K, n, device=device) * 0.05
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
        topk_ids = topk_ids_override.to(device=device, dtype=torch.int32).contiguous()
    topk_weights = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()
    if route_weight_slot is not None:
        if not 0 <= route_weight_slot < top_k:
            raise ValueError(
                f"route_weight_slot must be in [0, {top_k}), got "
                f"{route_weight_slot}"
            )
        topk_weights.zero_()
        topk_weights[:, route_weight_slot] = 1.0

    if recipe == "w4a8_mx":
        w13_q = [_quantize_weight_mxfp4(w13_full[e]) for e in range(E)]
        w2_q = [_quantize_weight_mxfp4(w2_full[e]) for e in range(E)]
        w13_mx = torch.stack([q[1] for q in w13_q]).contiguous()
        w2_mx = torch.stack([q[1] for q in w2_q]).contiguous()
        w13_res = torch.zeros(E, w1_n, K // 16, dtype=torch.uint8, device=device)
        w2_res = torch.zeros(E, K, n // 16, dtype=torch.uint8, device=device)
        ref_res_w13 = None
        ref_res_w2 = None
        ref_w13_mx = w13_mx
        ref_w2_mx = w2_mx
    else:
        w13_q = [_quantize_weight_nvfp4(w13_full[e]) for e in range(E)]
        w2_q = [_quantize_weight_nvfp4(w2_full[e]) for e in range(E)]
        w13_scales = torch.stack([q[1] for q in w13_q])
        w2_scales = torch.stack([q[1] for q in w2_q])
        w13_mx, w13_res_e4m3 = decompose_nvfp4_scales_to_mx_residual(w13_scales)
        w2_mx, w2_res_e4m3 = decompose_nvfp4_scales_to_mx_residual(w2_scales)
        w13_mx = w13_mx.contiguous()
        w2_mx = w2_mx.contiguous()
        w13_res = w13_res_e4m3.view(torch.uint8).contiguous()
        w2_res = w2_res_e4m3.view(torch.uint8).contiguous()
        ref_w13_mx = w13_mx
        ref_w2_mx = w2_mx
        ref_res_w13 = w13_res_e4m3
        ref_res_w2 = w2_res_e4m3
        w13_mx = _pad_w4a8_grid_rows(
            w13_mx,
            logical_n=n,
            is_gated=is_gated,
        )
        w13_res = _pad_w4a8_grid_rows(
            w13_res,
            logical_n=n,
            is_gated=is_gated,
        )
        w2_mx = _pad_w4a8_grid_columns(
            w2_mx,
            logical_n=n,
            elements_per_scale=32,
        )
        w2_res = _pad_w4a8_grid_columns(
            w2_res,
            logical_n=n,
            elements_per_scale=16,
        )

    w13_packed = torch.stack([q[0] for q in w13_q]).contiguous()
    w2_packed = torch.stack([q[0] for q in w2_q]).contiguous()
    ref_w13_packed = w13_packed
    if recipe == "w4a8_nvfp4":
        w13_packed = _pad_w4a8_grid_rows(
            w13_packed,
            logical_n=n,
            is_gated=is_gated,
        )
    ones = torch.ones(E, device=device)

    reference_ids, reference_weights = _active_route_reference_inputs(
        topk_ids, topk_weights, E
    )
    reference = moe_reference_w4a8_mx(
        x.float(),
        ref_w13_packed, ref_w13_mx, ref_res_w13, ones,
        w2_packed, ref_w2_mx, ref_res_w2, ones,
        reference_ids, reference_weights, E, K, n,
        activation=activation,
    )

    # ---- workspace ----
    # Worst-case physical tiles: every expert pays one partial tile plus the
    # full tiles its routed rows occupy.
    phys_tiles = E + (m * top_k + tile_m - 1) // tile_m
    rows_padded = phys_tiles * tile_m
    gate_tile_cnt = (
        (n + _TILE_N - 1) // _TILE_N
        if is_gated
        else (w1_n + _TILE_N - 1) // _TILE_N
    )
    max_tasks = phys_tiles * max(gate_tile_cnt, 1)
    mac = 4

    packed_a = torch.zeros(rows_padded * K, dtype=torch.uint8, device=device)
    # Sized for the (unused) vec16 SF TMA descriptor view: rows * K/8 bytes.
    scale_flat = torch.zeros(rows_padded * (K // 8), dtype=torch.uint8, device=device)
    intermediate_words = rows_padded * (n + n // 32) // 4
    if capture_fc1_staged:
        intermediate_words = max(
            intermediate_words,
            rows_padded * (K + K // 32) // 4,
        )
    intermediate_u32 = torch.zeros(
        intermediate_words,
        dtype=torch.int32,
        device=device,
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
    # The repacked-weight arguments are part of the launch ABI even when this
    # test exercises the ordinary (non-repacked) W4A8 path.  The kernel does
    # not dereference them in that specialization, so one aligned sentinel is
    # sufficient while keeping the compile and runtime argument lists exact.
    repacked_sentinel = torch.zeros(1, dtype=torch.uint32, device=device)
    scatter_output = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    flat_ids = topk_ids.reshape(-1).contiguous()
    flat_weights = topk_weights.reshape(-1).contiguous()

    kernel = MoEDynamicKernelBackend(
        16,
        (tile_m, _TILE_N),
        activation=activation,
        quant_recipe=recipe,
    )
    launch = _DynamicMoEW4A8Launch(kernel, k=K, n=n, w1_n=w1_n, num_topk=top_k)

    weight_dtype = cutlass.Float4E2M1FN
    dynamic_w1_n = (
        (2 if is_gated else 1) * ((n + _TILE_N - 1) // _TILE_N) * _TILE_N
        if recipe == "w4a8_nvfp4"
        else w1_n
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (dynamic_w1_n, K, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
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

    def _compile_or_override(*args, **kwargs):
        if compiled_override is not None:
            return compiled_override
        return b12x_compile(*args, **kwargs)

    compiled = _compile_or_override(
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
        compile_spec=KernelCompileSpec.from_fields(
            "tests.w4a8_dynamic.direct",
            1,
            ("recipe", recipe),
            ("activation", activation),
            ("tile_m", tile_m),
            ("experts", E),
            ("hidden", K),
            ("intermediate", n),
            ("top_k", top_k),
        ),
        dsl_compile_options=_OPT_LEVEL_2,
    )

    compiled(
        _gptr(cutlass.BFloat16, x),
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
        _gptr(cutlass.Uint8, w13_mx),
        _gptr(cutlass.Uint8, w2_mx),
        _gptr(cutlass.Uint8, w13_res),
        _gptr(cutlass.Uint8, w2_res),
        _gptr(cutlass.Uint32, repacked_sentinel),
        _gptr(cutlass.Uint32, repacked_sentinel),
        _gptr(cutlass.Uint32, repacked_sentinel),
        _gptr(cutlass.Uint32, repacked_sentinel),
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
    )
    torch.cuda.synchronize()
    if sum(
        int(enabled)
        for enabled in (
            capture_intermediate,
            capture_fc1_raw,
            capture_fc1_staged,
        )
    ) > 1:
        raise ValueError(
            "intermediate, raw-FC1, and staged-FC1 captures are mutually exclusive"
        )
    intermediate_debug = {}
    if capture_intermediate:
        if not return_debug:
            raise ValueError("capture_intermediate requires return_debug=True")
        from b12x._lib.intrinsics import (
            _ue8m0_output_scale_torch,
            pow2_ceil_ue8m0_torch,
        )
        from b12x.moe._shared.kernels.reference import (
            _make_fp4_lut,
            _quant_dequant_mxfp8_rows,
            _w4a8_effective_weight,
        )

        words_per_row = n // 4
        payload_words = rows_padded * words_per_row
        scale_words = rows_padded * (n // 128)
        raw_bytes = intermediate_u32.view(torch.uint8)
        device_payload = raw_bytes[: payload_words * 4].view(rows_padded, n)
        device_scale = (
            raw_bytes[payload_words * 4 : (payload_words + scale_words) * 4]
            .view(n // 128, rows_padded, 4)
            .permute(1, 0, 2)
            .reshape(rows_padded, n // 32)
        )

        reference_activated = torch.zeros(
            rows_padded, n, dtype=torch.float32, device=device
        )
        physical_expert = torch.full(
            (rows_padded,), -1, dtype=torch.int32, device=device
        )
        physical_expert_local = torch.full_like(physical_expert, -1)
        x_qd = _quant_dequant_mxfp8_rows(x.float())
        fp4_lut = _make_fp4_lut(device)
        expert_bases = expert_tile_base.tolist()
        expert_counts = row_counts.tolist()
        for expert in range(E):
            count = int(expert_counts[expert])
            if count == 0:
                continue
            physical_begin = int(expert_bases[expert]) * tile_m
            physical_rows = torch.arange(
                physical_begin,
                physical_begin + count,
                dtype=torch.long,
                device=device,
            )
            token_rows = token_map[physical_rows].long()
            w13_eff = _w4a8_effective_weight(
                w13_packed[expert],
                w13_mx[expert],
                None if ref_res_w13 is None else ref_res_w13[expert],
                w1_n,
                K,
                fp4_lut,
            ).view(w1_n, K)
            fc1 = x_qd[token_rows] @ w13_eff.T
            activated = torch.square(torch.relu(fc1)).to(torch.bfloat16).float()
            reference_activated[physical_rows] = activated
            physical_expert[physical_rows] = expert
            physical_expert_local[physical_rows] = torch.arange(
                count, dtype=torch.int32, device=device
            )

        ref_blocks = reference_activated.view(rows_padded, n // 32, 32)
        ref_block_max = ref_blocks.abs().amax(dim=-1, keepdim=True)
        _, reference_scale = pow2_ceil_ue8m0_torch(ref_block_max / 448.0)
        ref_inv_scale = _ue8m0_output_scale_torch(reference_scale)
        reference_payload = (
            (ref_blocks * ref_inv_scale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .view(rows_padded, n)
        )
        reference_scale = reference_scale.squeeze(-1)
        device_dequant = (
            device_payload.view(torch.float8_e4m3fn)
            .float()
            .view(rows_padded, n // 32, 32)
            * _ue8m0_output_scale_torch(device_scale).reciprocal().unsqueeze(-1)
        ).view(rows_padded, n)
        reference_dequant = (
            reference_payload.view(torch.float8_e4m3fn)
            .float()
            .view(rows_padded, n // 32, 32)
            * _ue8m0_output_scale_torch(reference_scale).reciprocal().unsqueeze(-1)
        ).view(rows_padded, n)
        # reciprocal(0) is inf for byte-zero blocks, but the payload is also
        # zero. Replace those products explicitly so zero blocks remain finite.
        device_dequant = torch.where(
            device_scale.repeat_interleave(32, dim=1) == 0,
            torch.zeros_like(device_dequant),
            device_dequant,
        )
        reference_dequant = torch.where(
            reference_scale.repeat_interleave(32, dim=1) == 0,
            torch.zeros_like(reference_dequant),
            reference_dequant,
        )
        intermediate_debug = {
            "intermediate_device_payload": device_payload.detach().cpu(),
            "intermediate_device_scale": device_scale.detach().cpu(),
            "intermediate_device_dequant": device_dequant.detach().cpu(),
            "intermediate_reference_activated": reference_activated.detach().cpu(),
            "intermediate_reference_payload": reference_payload.detach().cpu(),
            "intermediate_reference_scale": reference_scale.detach().cpu(),
            "intermediate_reference_dequant": reference_dequant.detach().cpu(),
            "intermediate_physical_expert": physical_expert.detach().cpu(),
            "intermediate_physical_expert_local": physical_expert_local.detach().cpu(),
        }
    fc1_raw_debug = {}
    if capture_fc1_raw:
        if not return_debug:
            raise ValueError("capture_fc1_raw requires return_debug=True")
        from b12x.moe._shared.kernels.reference import (
            _make_fp4_lut,
            _quant_dequant_mxfp8_rows,
            _w4a8_effective_weight,
        )

        device_fc1_raw = (
            intermediate_u32[: rows_padded * 32]
            .view(torch.float32)
            .view(rows_padded, 32)
        )
        reference_fc1_raw = torch.zeros_like(device_fc1_raw)
        physical_expert = torch.full(
            (rows_padded,), -1, dtype=torch.int32, device=device
        )
        physical_expert_local = torch.full_like(physical_expert, -1)
        x_qd = _quant_dequant_mxfp8_rows(x.float())
        fp4_lut = _make_fp4_lut(device)
        expert_bases = expert_tile_base.tolist()
        expert_counts = row_counts.tolist()
        for expert in range(E):
            count = int(expert_counts[expert])
            if count == 0:
                continue
            physical_begin = int(expert_bases[expert]) * tile_m
            physical_rows = torch.arange(
                physical_begin,
                physical_begin + count,
                dtype=torch.long,
                device=device,
            )
            token_rows = token_map[physical_rows].long()
            w13_eff = _w4a8_effective_weight(
                w13_packed[expert],
                w13_mx[expert],
                None if ref_res_w13 is None else ref_res_w13[expert],
                w1_n,
                K,
                fp4_lut,
            ).view(w1_n, K)
            reference_fc1_raw[physical_rows] = (
                x_qd[token_rows] @ w13_eff[:32].T
            )
            physical_expert[physical_rows] = expert
            physical_expert_local[physical_rows] = torch.arange(
                count, dtype=torch.int32, device=device
            )
        fc1_raw_debug = {
            "fc1_raw_device": device_fc1_raw.detach().cpu(),
            "fc1_raw_reference": reference_fc1_raw.detach().cpu(),
            "fc1_raw_physical_expert": physical_expert.detach().cpu(),
            "fc1_raw_physical_expert_local": physical_expert_local.detach().cpu(),
        }
    fc1_staged_debug = {}
    if capture_fc1_staged:
        if not return_debug:
            raise ValueError("capture_fc1_staged requires return_debug=True")
        staged_tiles = K // 128
        payload_words_per_row = staged_tiles * 32
        payload_words = rows_padded * payload_words_per_row
        raw_bytes = intermediate_u32.view(torch.uint8)
        device_payload = raw_bytes[: payload_words * 4].view(rows_padded, K)
        device_scale = (
            raw_bytes[
                payload_words * 4 :
                (payload_words + staged_tiles * rows_padded) * 4
            ]
            .view(staged_tiles, rows_padded, 4)
            .permute(1, 0, 2)
            .reshape(rows_padded, K // 32)
        )
        source_payload = packed_a.view(rows_padded, K)
        source_scale = scale_flat[: rows_padded * (K // 32)].view(
            rows_padded, K // 32
        )
        physical_expert = torch.full(
            (rows_padded,), -1, dtype=torch.int32, device=device
        )
        physical_expert_local = torch.full_like(physical_expert, -1)
        expert_bases = expert_tile_base.tolist()
        expert_counts = row_counts.tolist()
        for expert in range(E):
            count = int(expert_counts[expert])
            if count == 0:
                continue
            physical_begin = int(expert_bases[expert]) * tile_m
            physical_rows = torch.arange(
                physical_begin,
                physical_begin + count,
                dtype=torch.long,
                device=device,
            )
            physical_expert[physical_rows] = expert
            physical_expert_local[physical_rows] = torch.arange(
                count, dtype=torch.int32, device=device
            )
        fc1_staged_debug = {
            "fc1_staged_device_payload": device_payload.detach().cpu(),
            "fc1_staged_source_payload": source_payload.detach().cpu(),
            "fc1_staged_device_scale": device_scale.detach().cpu(),
            "fc1_staged_source_scale": source_scale.detach().cpu(),
            "fc1_staged_down_residual": w2_res.detach().cpu(),
            "fc1_staged_physical_expert": physical_expert.detach().cpu(),
            "fc1_staged_physical_expert_local": physical_expert_local.detach().cpu(),
        }
    if return_debug:
        debug = {
            "topk_ids": topk_ids.detach().cpu(),
            "topk_weights": topk_weights.detach().cpu(),
            "row_counts": row_counts.detach().cpu(),
            "expert_write_rows": expert_write_rows.detach().cpu(),
            "expert_tile_base": expert_tile_base.detach().cpu(),
            "token_map": token_map.detach().cpu(),
            "token_weights": token_weights.detach().cpu(),
            "task_ready": task_ready.detach().cpu(),
            "task_expert": task_expert.detach().cpu(),
            "task_m_tile": task_m_tile.detach().cpu(),
            "task_slice_begin": task_slice_begin.detach().cpu(),
            "task_slice_count": task_slice_count.detach().cpu(),
            "task_valid_rows": task_valid_rows.detach().cpu(),
            "tile_write_count": tile_write_count.detach().cpu(),
            **intermediate_debug,
            **fc1_raw_debug,
            **fc1_staged_debug,
        }
        if ref_res_w13 is not None and ref_res_w2 is not None:
            for variant, fc1_residual, fc2_residual in (
                ("no_fc1_residual", None, ref_res_w2),
                ("no_fc2_residual", ref_res_w13, None),
                ("no_residual", None, None),
            ):
                debug[f"reference_{variant}"] = moe_reference_w4a8_mx(
                    x.float(),
                    w13_packed,
                    w13_mx,
                    fc1_residual,
                    ones,
                    w2_packed,
                    w2_mx,
                    fc2_residual,
                    ones,
                    reference_ids,
                    reference_weights,
                    E,
                    K,
                    n,
                    activation=activation,
                )
        return scatter_output, reference, debug
    if return_launcher:
        def _relaunch_with(compiled_kernel):
            compiled_kernel(
                _gptr(cutlass.BFloat16, x),
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
                _gptr(cutlass.Uint8, w13_mx),
                _gptr(cutlass.Uint8, w2_mx),
                _gptr(cutlass.Uint8, w13_res),
                _gptr(cutlass.Uint8, w2_res),
                _gptr(cutlass.Uint32, repacked_sentinel),
                _gptr(cutlass.Uint32, repacked_sentinel),
                _gptr(cutlass.Uint32, repacked_sentinel),
                _gptr(cutlass.Uint32, repacked_sentinel),
                row_counts, expert_write_rows, expert_tile_base,
                ones, ones, ones, ones,
                _gptr(cutlass.BFloat16, scatter_output),
                _gptr(cutlass.Int32, token_map, 4),
                _gptr(cutlass.Float32, token_weights, 4),
                m, m * top_k, m, rows_padded, max_tasks, phys_tiles, mac,
                current_cuda_stream(),
            )

        def _relaunch():
            _relaunch_with(compiled)

        if return_state:
            def _current_reference():
                live_ids, live_weights = _active_route_reference_inputs(
                    flat_ids.view(m, top_k), flat_weights.view(m, top_k), E
                )
                return moe_reference_w4a8_mx(
                    x.float(),
                    w13_packed,
                    w13_mx,
                    ref_res_w13,
                    ones,
                    w2_packed,
                    w2_mx,
                    ref_res_w2,
                    ones,
                    live_ids,
                    live_weights,
                    E,
                    K,
                    n,
                    activation=activation,
                )

            state = {
                "live_inputs": {
                    "x": x,
                    "topk_ids": flat_ids.view(m, top_k),
                    "topk_weights": flat_weights.view(m, top_k),
                },
                "read_only_inputs": {
                    "w13_packed": w13_packed,
                    "w2_packed": w2_packed,
                    "w13_mx": w13_mx,
                    "w2_mx": w2_mx,
                    "w13_res": w13_res,
                    "w2_res": w2_res,
                    "ones": ones,
                    "repacked_sentinel": repacked_sentinel,
                },
                "mutable_allocations": {
                    "packed_a": packed_a,
                    "scale_flat": scale_flat,
                    "intermediate_u32": intermediate_u32,
                    "barrier_count": barrier_count,
                    "barrier_epoch": barrier_epoch,
                    "pair_head": pair_head,
                    "producers_done": producers_done,
                    "all_pub": all_pub,
                    "task_head": task_head,
                    "task_tail": task_tail,
                    "task_ready": task_ready,
                    "task_expert": task_expert,
                    "task_m_tile": task_m_tile,
                    "task_slice_begin": task_slice_begin,
                    "task_slice_count": task_slice_count,
                    "task_valid_rows": task_valid_rows,
                    "tile_write_count": tile_write_count,
                    "row_counts": row_counts,
                    "expert_write_rows": expert_write_rows,
                    "expert_tile_base": expert_tile_base,
                    "token_map": token_map,
                    "token_weights": token_weights,
                    "scatter_output": scatter_output,
                },
                "current_reference": _current_reference,
                "relaunch_with": _relaunch_with,
            }
            return scatter_output, reference, _relaunch, state
        return scatter_output, reference, _relaunch
    return scatter_output, reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("recipe", ["w4a8_mx", "w4a8_nvfp4"])
@pytest.mark.parametrize("activation", ["silu", "situ", "relu2"])
def test_w4a8_dynamic_matches_oracle(recipe: str, activation: str) -> None:
    require_b12x()
    out, ref = _run_w4a8_dynamic(
        recipe=recipe, activation=activation,
        E=4, m=8, K=256, n=128, top_k=2, seed=11,
    )
    assert out.abs().sum().item() > 0, "kernel produced all zeros"
    metrics = compare_to_reference(out.float(), ref)
    assert metrics.cos > 0.999, metrics
    ref_rms = ref.float().square().mean().sqrt().item()
    assert metrics.rmse <= max(0.03 * ref_rms, 5e-3), (metrics, ref_rms)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_nvfp4_dynamic_n160_tail_matches_oracle() -> None:
    require_b12x()
    out, ref = _run_w4a8_dynamic(
        recipe="w4a8_nvfp4",
        activation="silu",
        E=4,
        m=32,
        K=256,
        n=160,
        top_k=2,
        seed=160,
        tile_m=32,
    )

    metrics = compare_to_reference(out.float(), ref)
    assert metrics.cos > 0.999, metrics
    ref_rms = ref.float().square().mean().sqrt().item()
    assert metrics.rmse <= max(0.03 * ref_rms, 5e-3), (metrics, ref_rms)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("inactive_expert", [-1, 4])
def test_w4a8_dynamic_ignores_inactive_routes(inactive_expert: int) -> None:
    """Inactive routes must not enter expert storage or contribute output."""

    require_b12x()
    topk_ids = torch.tensor(
        [
            [inactive_expert, inactive_expert],
            [0, inactive_expert],
            [1, 2],
            [inactive_expert, 3],
            [2, inactive_expert],
            [3, 0],
            [inactive_expert, 1],
            [0, 2],
        ],
        dtype=torch.int32,
    )
    out, ref, debug = _run_w4a8_dynamic(
        recipe="w4a8_mx",
        activation="silu",
        E=4,
        m=8,
        K=256,
        n=128,
        top_k=2,
        seed=211,
        topk_ids_override=topk_ids,
        return_debug=True,
    )

    active_routes = int(((topk_ids >= 0) & (topk_ids < 4)).sum().item())
    assert int(debug["row_counts"].sum().item()) == active_routes
    assert int(debug["expert_write_rows"].sum().item()) == active_routes
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), rtol=0, atol=0)
    metrics = compare_to_reference(out.float(), ref)
    assert metrics.cos > 0.999, metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_experts", (31, 32, 33))
def test_grouped_expert_prefix_handles_warp_tail_and_multiple_tiles(num_experts):
    """Empty experts and inactive routes must leave compact tile offsets exact."""
    require_b12x()
    routes = torch.tensor([[0, num_experts - 1]] * 33, dtype=torch.int32)
    routes[::4, 0] = -1
    out, ref, debug = _run_w4a8_dynamic(
        recipe="w4a8_mx", activation="silu", E=num_experts,
        m=33, K=256, n=128, top_k=2, seed=329, tile_m=16,
        topk_ids_override=routes, return_debug=True,
    )
    counts = torch.bincount(routes[routes >= 0].long(), minlength=num_experts)
    expected = torch.cat((
        torch.zeros(1, dtype=torch.int64), ((counts + 15) // 16).cumsum(0),
    ))
    torch.testing.assert_close(
        debug["expert_tile_base"].long(), expected, rtol=0, atol=0,
    )
    assert torch.isfinite(out).all() and torch.count_nonzero(out)
    assert compare_to_reference(out.float(), ref).cos > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ["silu", "situ", "relu2"])
def test_w4a8_dynamic_small_tile_parallel_regime_matches_oracle(
    activation: str,
) -> None:
    """Exercise the production M16/four-MMA/two-DMA regime directly."""
    require_b12x()
    kernel = MoEDynamicKernelBackend(
        16,
        (16, _TILE_N),
        activation=activation,
        quant_recipe="w4a8_mx",
    )
    assert kernel.atom_shape == (1, 4, 1)
    assert kernel.num_mma_warps == 4
    assert kernel.num_dma_warps == 2
    out, ref = _run_w4a8_dynamic(
        recipe="w4a8_mx",
        activation=activation,
        E=4,
        m=17,
        K=256,
        n=128,
        top_k=2,
        seed=29,
        tile_m=16,
    )
    metrics = compare_to_reference(out.float(), ref)
    assert metrics.cos > 0.999, metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_nvfp4_relu2_m32_rows_16_17_match_oracle() -> None:
    """Guard the FC1-A/FC2-residual alias handoff at the first M16 boundary."""
    require_b12x()
    m = 18
    topk_ids = torch.tensor(
        [[0, 1 + token % 3] for token in range(m)],
        dtype=torch.int32,
    )
    out, ref, debug = _run_w4a8_dynamic(
        recipe="w4a8_nvfp4",
        activation="relu2",
        E=4,
        m=m,
        K=256,
        n=128,
        top_k=2,
        seed=1032,
        tile_m=32,
        route_weight_slot=0,
        return_debug=True,
        topk_ids_override=topk_ids,
    )

    expert_zero_base = int(debug["expert_tile_base"][0].item()) * 32
    assert int(debug["row_counts"][0].item()) == m
    target_rows = torch.tensor(
        [expert_zero_base + 16, expert_zero_base + 17], dtype=torch.long
    )
    target_tokens = debug["token_map"][target_rows].long()
    assert len(set(target_tokens.tolist())) == 2
    assert all(0 <= token < m for token in target_tokens.tolist())
    assert debug["token_weights"][target_rows].tolist() == [1.0, 1.0]

    for local, token in zip((16, 17), target_tokens.tolist(), strict=True):
        metrics = compare_to_reference(
            out[token : token + 1].float(), ref[token : token + 1]
        )
        ref_rms = ref[token].float().square().mean().sqrt().item()
        assert metrics.cos > 0.999, (local, metrics)
        assert metrics.rmse <= max(0.03 * ref_rms, 5e-3), (
            local,
            metrics,
            ref_rms,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_dynamic_boundary_m_sizes() -> None:
    require_b12x()
    for m in (1, 3, 127, 129):
        out, ref = _run_w4a8_dynamic(
            recipe="w4a8_mx", activation="silu",
            E=4, m=m, K=256, n=128, top_k=2, seed=100 + m,
        )
        assert out.abs().sum().item() > 0, m
        metrics = compare_to_reference(out.float(), ref)
        assert metrics.cos > 0.999, (m, metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_dynamic_graph_replay_tracks_routing_updates() -> None:
    """Capture the w4a8 launch in a CUDA graph; replay must track routing."""
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(7)
    E, m, K, n, top_k = 4, 8, 256, 128, 2
    w1_n = 2 * n

    x = (torch.randn(m, K, device=device) * 2.0).to(torch.bfloat16)
    w13_full = torch.randn(E, w1_n, K, device=device) * 0.05
    w2_full = torch.randn(E, K, n, device=device) * 0.05
    w13_q = [_quantize_weight_mxfp4(w13_full[e]) for e in range(E)]
    w2_q = [_quantize_weight_mxfp4(w2_full[e]) for e in range(E)]
    w13_packed = torch.stack([q[0] for q in w13_q]).contiguous()
    w2_packed = torch.stack([q[0] for q in w2_q]).contiguous()
    w13_mx = torch.stack([q[1] for q in w13_q]).contiguous()
    w2_mx = torch.stack([q[1] for q in w2_q]).contiguous()
    w13_res = torch.zeros(E, w1_n, K // 16, dtype=torch.uint8, device=device)
    w2_res = torch.zeros(E, K, n // 16, dtype=torch.uint8, device=device)
    ones = torch.ones(E, device=device)

    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()
    flat_ids = topk_ids.reshape(-1).contiguous()
    flat_weights = topk_weights.reshape(-1).contiguous()

    phys_tiles = E + 1
    rows_padded = phys_tiles * _TILE_M
    max_tasks = phys_tiles
    packed_a = torch.zeros(rows_padded * K, dtype=torch.uint8, device=device)
    scale_flat = torch.zeros(rows_padded * (K // 8), dtype=torch.uint8, device=device)
    intermediate_u32 = torch.zeros(
        rows_padded * (n + n // 32) // 4,
        dtype=torch.int32,
        device=device,
    )
    repacked_sentinel = torch.zeros(1, dtype=torch.uint32, device=device)
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

    kernel = MoEDynamicKernelBackend(
        16, (_TILE_M, _TILE_N), activation="silu", quant_recipe="w4a8_mx"
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
        fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u32(),
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
        dsl_compile_options=_OPT_LEVEL_2,
    )

    def _launch():
        compiled(
            _gptr(cutlass.BFloat16, x),
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
            _gptr(cutlass.Uint8, w13_mx),
            _gptr(cutlass.Uint8, w2_mx),
            _gptr(cutlass.Uint8, w13_res),
            _gptr(cutlass.Uint8, w2_res),
            _gptr(cutlass.Uint32, repacked_sentinel),
            _gptr(cutlass.Uint32, repacked_sentinel),
            _gptr(cutlass.Uint32, repacked_sentinel),
            _gptr(cutlass.Uint32, repacked_sentinel),
            row_counts, expert_write_rows, expert_tile_base,
            ones, ones, ones, ones,
            _gptr(cutlass.BFloat16, scatter_output),
            _gptr(cutlass.Int32, token_map, 4),
            _gptr(cutlass.Float32, token_weights, 4),
            m, m * top_k, m, rows_padded, max_tasks, phys_tiles, 4,
            current_cuda_stream(),
        )

    def _oracle():
        live_ids, live_weights = _active_route_reference_inputs(
            flat_ids.view(m, top_k), flat_weights.view(m, top_k), E
        )
        return moe_reference_w4a8_mx(
            x.float(), w13_packed, w13_mx, None, ones,
            w2_packed, w2_mx, None, ones,
            live_ids, live_weights, E, K, n,
            activation="silu",
        )

    # Warm up (also pre-zeros barrier state path), then capture.
    _launch()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        _launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for round_idx in range(3):
        torch.manual_seed(100 + round_idx)
        new_ids = torch.stack(
            [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
        ).to(torch.int32)
        if round_idx == 1:
            new_ids[-3:, 1] = -1
        elif round_idx == 2:
            new_ids.fill_(-1)
        new_w = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()
        flat_ids.copy_(new_ids.reshape(-1))
        flat_weights.copy_(new_w.reshape(-1))
        x.copy_((torch.randn(m, K, device=device) * 2.0).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        ref = _oracle()
        if round_idx == 2:
            torch.testing.assert_close(
                scatter_output, torch.zeros_like(scatter_output), rtol=0, atol=0
            )
        else:
            metrics = compare_to_reference(scatter_output.float(), ref)
            assert scatter_output.abs().sum().item() > 0, round_idx
            assert metrics.cos > 0.999, (round_idx, metrics)
