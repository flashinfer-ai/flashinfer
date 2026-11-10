"""Direct-drive integration test for the NVFP4 (W4A4) split-materialized
phase kernels.

Drives ``Nvfp4MaterializedPhase1Kernel``/``Nvfp4MaterializedPhase2Kernel``
DIRECTLY over a synthetic expert-major packed A + SFA domain built with the
same quantization the oracle (``moe_reference_nvfp4``) applies to routed
activations, and gates FC1/intermediate/FC2 outputs against per-16-block
NVFP4 torch reference math.  Also proves the pool-scaled Int64 addressing
contract (rows parked past 2^31/stride) and frozen kernel resolution under
multiple live task counts.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import Int32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.nvfp4_phase1 import Nvfp4MaterializedPhase1Kernel
from b12x.moe._shared.kernels.nvfp4_phase2 import Nvfp4MaterializedPhase2Kernel
from b12x.moe._shared.kernels.reference import compare_to_reference

from b12x.testing.reference.helpers import require_b12x

_TILE_M = 128
_TILE_N = 128
# Physical tile base that drives live row-scaled offsets past 2^31.  With
# n=128 the intermediate payload stride is 16 u32 per row, so a live row id
# above 2^31/16 is required; the extra tiles give a safety margin.
_INT32_BOUNDARY_TILE_BASE = (2**31 // 16) // _TILE_M + 1024


def _fake_i32(shape):
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, shape, assumed_align=4)


def _fake_f32(shape):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, shape, assumed_align=16
    )


def _fake_u8(shape):
    return cute.runtime.make_fake_compact_tensor(cutlass.Uint8, shape, assumed_align=16)


def _gptr(dtype, t: torch.Tensor, align: int = 16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _pack_fp4_rows(values: torch.Tensor) -> torch.Tensor:
    """Pack quantized FP4 values (float codes) into uint8 byte rows."""
    from b12x._lib.intrinsics import _fp4_encode_nibbles

    nib = _fp4_encode_nibbles(values)
    pair = nib.view(*values.shape[:-1], values.shape[-1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).contiguous()


def _quantize_nvfp4_rows(x: torch.Tensor, global_scale: float):
    """Per-16-block NVFP4 quantize-dequantize + packed bytes + scale plane.

    Matches ``quantize_block_fp4`` (direct-division oracle semantics):
    scale = e4m3(amax * gs / 6); payload = fp4(x / (scale/gs)).
    Returns (packed_bytes [R, C/2], dequant_values [R, C], scale_f32 [R, C/16]).
    """
    from b12x._lib.intrinsics import FLOAT8_E4M3_MAX, fp4_quantize_values_torch

    x2 = x.float() if x.dim() == 2 else x.float().unsqueeze(0)
    rows, cols = x2.shape
    blocked = x2.view(rows, cols // 16, 16)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    raw_scale = (block_max * global_scale / 6.0).clamp(max=FLOAT8_E4M3_MAX)
    scale = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)
    eff_scale = (scale / global_scale).clamp(min=1e-30)
    scaled = blocked / eff_scale
    quant = fp4_quantize_values_torch(scaled)
    dequant = quant * scale
    packed = _pack_fp4_rows(quant.reshape(rows, cols))
    if x.dim() == 1:
        return (
            packed.view(cols // 2),
            dequant.view(cols),
            scale.view(cols // 16),
        )
    return packed, dequant.reshape(rows, cols), scale.reshape(rows, cols // 16)


def _oracle_swizzled_scale(scale_f32: torch.Tensor, rows: int) -> torch.Tensor:
    """Swizzled E4M3 scale plane in the layout ``unswizzle_block_scale``
    inverts (the publisher's canonical F8_128x4 atom).

    Delegates to ``_swizzle_scale_plane`` so the oracle and the kernels
    consume the byte-identical scale layout; a partial (row-permutation
    only) swizzle would make the oracle decode wrong scales for planes
    with more than one K/4 atom column.
    """
    return _swizzle_scale_plane(scale_f32, rows)


def _swizzle_scale_plane(scale_f32: torch.Tensor, rows: int) -> torch.Tensor:
    """Vectorized F8_128x4 swizzle of one [rows, k_blocks] E4M3 scale plane.

    Matches the publisher's offset math:
    offset(row, kb) = (row//128)*(K4*512) + (kb//4)*512 + (row%128%32)*16 +
    ((row%128)//32)*4 + (kb%4).
    """
    k_blocks = scale_f32.shape[1]
    k4 = (k_blocks + 3) // 4
    rows_p = ((rows + 127) // 128) * 128
    e4m3 = scale_f32.to(torch.float8_e4m3fn).view(torch.uint8)
    padded = torch.zeros(rows_p, k4 * 4, dtype=torch.uint8, device=e4m3.device)
    padded[:rows, :k_blocks] = e4m3
    r = torch.arange(rows_p, device=e4m3.device).view(rows_p, 1)
    kb = torch.arange(k4 * 4, device=e4m3.device).view(1, k4 * 4)
    off = (
        (r // 128) * (k4 * 512)
        + (kb // 4) * 512
        + ((r % 128) % 32) * 16
        + ((r % 128) // 32) * 4
        + (kb % 4)
    )
    # Compact per-plane size: one padded 128-row atom contributes k4*512
    # bytes, matching tile_atom_to_shape_SF's contiguous atom packing so the
    # per-expert stride is (rows/128)*k4*512.
    flat = torch.zeros(
        (rows_p // 128) * k4 * 512, dtype=torch.uint8, device=e4m3.device
    )
    flat[off.view(-1)] = padded.view(-1)
    return flat


def _route_domain(m, E, top_k, topk_ids, tile_base=0):
    """Expert-major physical-row assignment mirroring the front-end contract.

    Experts occupy consecutive physical 128-row tiles; each routed pair
    claims the next free row of its expert; tasks are published as
    (source tile, intermediate tile) slots with per-slot valid-row counts.

    ``tile_base`` shifts every expert's first physical tile to a high index.
    The live rows then carry large physical row ids while only their tail of
    the pool is written, which is how the Int64 offset contract is exercised
    without allocating the untouched prefix.

    Returns only the O(m*top_k) live pair rows.  The caller allocates the full
    device ``token_map``/``token_weights`` and scatters these indices, so a
    high-``tile_base`` domain never materializes dense host lists.
    """
    row_counts = [0] * E
    for pair in range(m * top_k):
        row_counts[int(topk_ids.reshape(-1)[pair])] += 1
    expert_tile_base = [tile_base]
    for e in range(E):
        tiles = (row_counts[e] + _TILE_M - 1) // _TILE_M
        expert_tile_base.append(expert_tile_base[-1] + tiles)
    phys_tiles = expert_tile_base[-1]
    rows_capacity = phys_tiles * _TILE_M
    cursor = [0] * E
    pair_phys = [0] * (m * top_k)
    for t in range(m):
        for k_i in range(top_k):
            eid = int(topk_ids[t, k_i])
            local = cursor[eid]
            tile = expert_tile_base[eid] + local // _TILE_M
            pair_phys[t * top_k + k_i] = tile * _TILE_M + (local % _TILE_M)
            cursor[eid] += 1
    return expert_tile_base, phys_tiles, rows_capacity, pair_phys


def _build_domain(
    *, E: int, K: int, n: int, m: int, top_k: int, seed: int, tile_base: int = 0
):
    """Build synthetic weights + routed inputs + the expert-major domain."""
    from b12x.moe._shared.kernels.reference import moe_reference_nvfp4

    device = torch.device("cuda")
    torch.manual_seed(seed)
    w1_n = 2 * n
    x = (torch.randn(m, K, device=device) * 2.0).to(torch.bfloat16)
    w13_full = torch.randn(E, w1_n, K, device=device) * 0.05
    w2_full = torch.randn(E, K, n, device=device) * 0.05
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, top_k, device=device), dim=-1).float()

    ones = torch.ones(E, device=device)
    a1_gscale = torch.ones(E, device=device)
    a2_gscale = torch.ones(E, device=device)

    w13_q = [_quantize_nvfp4_rows(w13_full[e], 1.0) for e in range(E)]
    w2_q = [_quantize_nvfp4_rows(w2_full[e], 1.0) for e in range(E)]
    w13_packed = torch.stack([q[0] for q in w13_q]).contiguous()
    w2_packed = torch.stack([q[0] for q in w2_q]).contiguous()
    w13_dequant = torch.stack([q[1] for q in w13_q]).contiguous()
    w2_dequant = torch.stack([q[1] for q in w2_q]).contiguous()
    w13_scales = torch.stack([q[2] for q in w13_q]).contiguous()
    w2_scales = torch.stack([q[2] for q in w2_q]).contiguous()

    w13_oracle_sf = (
        torch.stack([_oracle_swizzled_scale(w13_scales[e], w1_n) for e in range(E)])
        .contiguous()
        .view(E, -1)
    )
    w2_oracle_sf = (
        torch.stack([_oracle_swizzled_scale(w2_scales[e], K) for e in range(E)])
        .contiguous()
        .view(E, -1)
    )
    oracle = moe_reference_nvfp4(
        x.float(),
        w13_packed,
        w13_oracle_sf,
        ones,
        w2_packed,
        w2_oracle_sf,
        ones,
        a1_gscale,
        a2_gscale,
        topk_ids,
        topk_weights,
        E,
        K,
        n,
        activation="silu",
        quant_scale_math="direct_division",
    )

    (
        expert_tile_base,
        phys_tiles,
        rows_capacity,
        pair_phys,
    ) = _route_domain(m, E, top_k, topk_ids, tile_base=tile_base)

    # Device tensors sized to the full pool; only the m*top_k live pair rows
    # are written, so a high-tile_base domain does not build host-side lists
    # proportional to rows_capacity.
    token_map = torch.zeros(rows_capacity, dtype=torch.int32, device=device)
    token_weights = torch.zeros(rows_capacity, dtype=torch.float32, device=device)
    phys_of_pair = torch.tensor(pair_phys, dtype=torch.int64, device=device)
    token_of_pair = torch.arange(m, device=device, dtype=torch.int32).repeat_interleave(
        top_k
    )
    token_map[phys_of_pair] = token_of_pair
    token_weights[phys_of_pair] = topk_weights.reshape(-1)

    intermediate_tiles = n // _TILE_N
    task_expert = torch.zeros(
        phys_tiles * intermediate_tiles, dtype=torch.int32, device=device
    )
    task_valid_rows = torch.zeros_like(task_expert)
    row_counts = [0] * E
    for pair in range(m * top_k):
        row_counts[int(topk_ids.reshape(-1)[pair])] += 1
    for e in range(E):
        count = row_counts[e]
        for tile in range(expert_tile_base[e], expert_tile_base[e + 1]):
            base_rows = tile - expert_tile_base[e]
            valid = count - base_rows * _TILE_M
            valid = min(max(valid, 0), _TILE_M)
            for it in range(intermediate_tiles):
                task_expert[tile * intermediate_tiles + it] = e
                task_valid_rows[tile * intermediate_tiles + it] = valid
    expert_tile_base_t = torch.tensor(
        expert_tile_base, dtype=torch.int32, device=device
    )

    # The route/pack front-end materializes one quantized activation row per
    # ROUTE (expert-major physical row), fanning a shared-token quantization
    # out to each routed expert's physical row.  SFA is the same row's scale
    # in the F8_128x4 swizzled plane, indexed by physical row.  phase1 reads
    # both directly by physical row; token_map is only used for FC2 scatter.
    packed_a = torch.zeros(rows_capacity * K // 2, dtype=torch.uint8, device=device)
    k4 = K // 64
    # One F8_128x4 atom per 128 physical rows carries k4*512 bytes, so the
    # single activation SFA plane is (rows_capacity//128)*k4*512 bytes.  The
    # swizzle writes at (phys//128)*(k4*512) + ... and phase1 reads at
    # sf_atom*(k4*512) + ...; both stay inside this exact extent.
    scale_flat = torch.zeros(
        ((rows_capacity + 127) // 128) * k4 * 512,
        dtype=torch.uint8,
        device=device,
    )
    x_quantized = [_quantize_nvfp4_rows(x[t].float(), 1.0) for t in range(m)]
    for pair in range(m * top_k):
        t = pair // top_k
        phys = pair_phys[pair]
        p, _, s = x_quantized[t]
        packed_a.view(-1)[phys * (K // 2) : (phys + 1) * (K // 2)] = p
        for kb in range(K // 16):
            off = (
                (phys // 128) * (k4 * 512)
                + (kb // 4) * 512
                + (phys % 128 % 32) * 16
                + ((phys % 128) // 32) * 4
                + (kb % 4)
            )
            scale_flat.view(-1)[off] = s[kb].to(torch.float8_e4m3fn).view(torch.uint8)

    w13_sfb = torch.cat(
        [_swizzle_scale_plane(w13_scales[e], w1_n) for e in range(E)]
    ).contiguous()
    w2_sfb = torch.cat(
        [_swizzle_scale_plane(w2_scales[e], K) for e in range(E)]
    ).contiguous()

    return {
        "x": x,
        "w13_packed": w13_packed,
        "w2_packed": w2_packed,
        "w13_dequant": w13_dequant,
        "w2_dequant": w2_dequant,
        "w13_scales": w13_scales,
        "w2_scales": w2_scales,
        "w13_sfb": w13_sfb,
        "w2_sfb": w2_sfb,
        "oracle": oracle,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "token_map": token_map,
        "token_weights": token_weights,
        "task_expert": task_expert,
        "task_valid_rows": task_valid_rows,
        "expert_tile_base": expert_tile_base_t,
        "packed_a": packed_a,
        "scale_flat": scale_flat,
        "phys_tiles": phys_tiles,
        "rows_capacity": rows_capacity,
        "intermediate_tiles": intermediate_tiles,
        "w1_n": w1_n,
        "m": m,
        "K": K,
        "n": n,
        "E": E,
        "top_k": top_k,
        "x_quantized": x_quantized,
        "phys_of_pair": phys_of_pair,
    }


def _compile_phase1(domain, *, spec_name="tests.nvfp4_phase_kernels.p1"):
    kernel = Nvfp4MaterializedPhase1Kernel(source_tile_m=_TILE_M)
    E, K, n = domain["E"], domain["K"], domain["n"]

    def _fake_u32(shape):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Uint32, shape, assumed_align=16
        )

    return b12x_compile(
        kernel,
        _fake_u32((domain["rows_capacity"] * K // 8,)),
        _fake_u8((domain["scale_flat"].numel(),)),
        _fake_u32((E * domain["w1_n"] * K // 8,)),
        _fake_u8((domain["w13_sfb"].numel(),)),
        _fake_u32((domain["rows_capacity"] * 16,)),
        _fake_i32((domain["rows_capacity"],)),
        _fake_i32((domain["task_expert"].numel(),)),
        _fake_i32((domain["task_expert"].numel(),)),
        _fake_i32((E + 1,)),
        _fake_f32((E,)),
        _fake_f32((E,)),
        Int32(K // 128),
        Int32(domain["intermediate_tiles"]),
        Int32(domain["w1_n"] // 128),
        Int32(4),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_fields(
            spec_name,
            1,
            ("experts", E),
            ("hidden", K),
            ("intermediate", n),
            ("source_tile_m", _TILE_M),
        ),
    )


def _compile_phase2(domain, *, spec_name="tests.nvfp4_phase_kernels.p2"):
    kernel = Nvfp4MaterializedPhase2Kernel(source_tile_m=_TILE_M)
    E, K, n = domain["E"], domain["K"], domain["n"]

    def _fake_u32(shape):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Uint32, shape, assumed_align=16
        )

    def _fake_bf16(shape):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, shape, assumed_align=16
        )

    return b12x_compile(
        kernel,
        _fake_u32((domain["rows_capacity"] * 16,)),
        _fake_u32((E * K * domain["n"] // 8,)),
        _fake_u8((domain["w2_sfb"].numel(),)),
        _fake_bf16((domain["m"], K)),
        _fake_i32((domain["rows_capacity"],)),
        _fake_f32((domain["rows_capacity"],)),
        _fake_i32((domain["task_expert"].numel(),)),
        _fake_i32((domain["task_expert"].numel(),)),
        _fake_i32((E + 1,)),
        _fake_f32((E,)),
        Int32(domain["intermediate_tiles"]),
        Int32(K // 128),
        Int32(4),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_fields(
            spec_name,
            1,
            ("experts", E),
            ("hidden", K),
            ("intermediate", n),
            ("source_tile_m", _TILE_M),
        ),
    )


def _allocate_intermediate(domain):
    """Allocate the intermediate workspace for the domain's full capacity.

    The phase1 kernel writes both payload and scale planes into one contiguous
    buffer: payload [rows_capacity * (n//128) * 16] u32 followed by scale
    [(n//128) * rows_capacity * 2] u32.  One pool covers both.  A high-``tile_base``
    domain keeps its live rows in this pool's tail; the untouched prefix is
    still allocated so the kernel's own row-scaled offsets stay in bounds.
    """
    rows_capacity = domain["rows_capacity"]
    intermediate_tiles = domain["intermediate_tiles"]
    words_per_row = intermediate_tiles * 16
    total_elements = (
        rows_capacity * words_per_row + intermediate_tiles * rows_capacity * 2
    )
    return torch.zeros(total_elements, dtype=torch.int32, device="cuda")


def _launch_phase1(compiled, domain, intermediate_u32, alpha_t, gs_t):
    E, K = domain["E"], domain["K"]
    return compiled(
        _gptr(cutlass.Uint32, domain["packed_a"].view(torch.int32)),
        _gptr(cutlass.Uint8, domain["scale_flat"], 16),
        _gptr(cutlass.Uint32, domain["w13_packed"].view(torch.int32)),
        _gptr(cutlass.Uint8, domain["w13_sfb"], 16),
        _gptr(cutlass.Uint32, intermediate_u32),
        _gptr(cutlass.Int32, domain["token_map"], 4),
        _gptr(cutlass.Int32, domain["task_expert"], 4),
        _gptr(cutlass.Int32, domain["task_valid_rows"], 4),
        _gptr(cutlass.Int32, domain["expert_tile_base"], 4),
        _gptr(cutlass.Float32, alpha_t, 4),
        _gptr(cutlass.Float32, gs_t, 4),
        Int32(K // 128),
        Int32(domain["intermediate_tiles"]),
        Int32(domain["w1_n"] // 128),
        Int32(4),
        current_cuda_stream(),
    )


def _launch_phase2(compiled, domain, intermediate_u32, down_alpha_t, scatter_output):
    E, K = domain["E"], domain["K"]
    return compiled(
        _gptr(cutlass.Uint32, intermediate_u32),
        _gptr(cutlass.Uint32, domain["w2_packed"].view(torch.int32)),
        _gptr(cutlass.Uint8, domain["w2_sfb"], 16),
        _gptr(cutlass.BFloat16, scatter_output),
        _gptr(cutlass.Int32, domain["token_map"], 4),
        _gptr(cutlass.Float32, domain["token_weights"], 4),
        _gptr(cutlass.Int32, domain["task_expert"], 4),
        _gptr(cutlass.Int32, domain["task_valid_rows"], 4),
        _gptr(cutlass.Int32, domain["expert_tile_base"], 4),
        _gptr(cutlass.Float32, down_alpha_t, 4),
        Int32(domain["intermediate_tiles"]),
        Int32(K // 128),
        Int32(4),
        current_cuda_stream(),
    )


def _run_phases(domain, *, compiled_p1=None, compiled_p2=None, scatter_output=None):
    E = domain["E"]
    ones = torch.ones(E, device="cuda")
    intermediate_u32 = _allocate_intermediate(domain)
    if compiled_p1 is None:
        compiled_p1 = _compile_phase1(domain)
    if compiled_p2 is None:
        compiled_p2 = _compile_phase2(domain)
    if scatter_output is None:
        scatter_output = torch.zeros(
            domain["m"], domain["K"], dtype=torch.bfloat16, device="cuda"
        )
    _launch_phase1(compiled_p1, domain, intermediate_u32, ones, ones)
    torch.cuda.synchronize()
    _launch_phase2(compiled_p2, domain, intermediate_u32, ones, scatter_output)
    torch.cuda.synchronize()
    return scatter_output, intermediate_u32


def _torch_chain_reference(domain):
    """Torch per-token FC1->SiLU->requant->FC2 chain (direct-division).

    The FC1 intermediate is rounded to BF16 before requantization to mirror
    the shared ``sC`` epilogue staging dtype both the monolithic and the split
    NVFP4 kernels use, so the gate measures the kernel against its real
    numerical contract rather than an unattainable FP32 intermediate.
    """
    K, n, E = domain["K"], domain["n"], domain["E"]
    out = torch.zeros(domain["m"], K, dtype=torch.float32, device="cuda")
    w13_d = domain["w13_dequant"]
    w2_d = domain["w2_dequant"]
    topk_ids = domain["topk_ids"]
    topk_weights = domain["topk_weights"]
    for t in range(domain["m"]):
        for k_i in range(domain["top_k"]):
            eid = int(topk_ids[t, k_i])
            x_deq = domain["x_quantized"][t][1]
            gate = w13_d[eid][n:] @ x_deq
            up = w13_d[eid][:n] @ x_deq
            inter = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
            _, inter_q, _ = _quantize_nvfp4_rows(inter, 1.0)
            down = w2_d[eid] @ inter_q
            out[t] += float(topk_weights[t, k_i]) * down
    return out


def _bf16_output_bound(reference: torch.Tensor) -> float:
    """Absolute-error ceiling for a BF16-stored kernel output.

    ``scatter_output`` is BF16, so a kernel cannot be closer than one BF16 ULP
    of the FP32 oracle it is checked against.  Allow three ULPs of the
    reference amplitude (accumulation-order slack) and keep the original
    8e-4 floor for small-magnitude domains.
    """
    ulp = reference.abs().max().item() * 2.0**-8
    return max(8e-4, 3.0 * ulp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_phase_chain_small() -> None:
    """Full phase1->phase2 chain vs the NVFP4 oracle (E=8, K=256, I=128)."""
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=64, top_k=2, seed=11)
    out, _ = _run_phases(domain)
    assert out.abs().sum().item() > 0, "kernel produced all zeros"
    metrics = compare_to_reference(out.float(), domain["oracle"])
    assert metrics.cos > 0.9999, metrics
    bound = _bf16_output_bound(domain["oracle"])
    assert metrics.max_abs <= bound, (metrics, bound)
    assert metrics.rmse <= bound, (metrics, bound)
    # direct torch chain must match the oracle too (sanity of the harness)
    chain = _torch_chain_reference(domain)
    chain_metrics = compare_to_reference(chain, domain["oracle"])
    assert chain_metrics.cos > 0.99999, chain_metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_phase_chain_m256() -> None:
    """Full chain on M=256 with multi-tile experts."""
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=256, top_k=2, seed=12)
    out, _ = _run_phases(domain)
    assert out.abs().sum().item() > 0
    metrics = compare_to_reference(out.float(), domain["oracle"])
    assert metrics.cos > 0.9999, metrics
    bound = _bf16_output_bound(domain["oracle"])
    assert metrics.max_abs <= bound, (metrics, bound)
    assert metrics.rmse <= bound, (metrics, bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_phase_intermediate_matches_torch() -> None:
    """Unit-level phase1 check: materialized intermediate vs torch math."""
    require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=64, top_k=2, seed=13)
    E = domain["E"]
    ones = torch.ones(E, device="cuda")
    intermediate_u32 = _allocate_intermediate(domain)
    compiled_p1 = _compile_phase1(domain)
    _launch_phase1(compiled_p1, domain, intermediate_u32, ones, ones)
    torch.cuda.synchronize()

    rows_capacity = domain["rows_capacity"]
    words_per_row = domain["intermediate_tiles"] * 16
    raw = intermediate_u32.view(torch.uint8)
    payload = raw[: rows_capacity * words_per_row * 4].view(
        rows_capacity, domain["n"] // 2
    )
    sf_plane = raw[rows_capacity * words_per_row * 4 :]

    from b12x._lib.intrinsics import FLOAT8_E4M3_MAX, fp4_quantize_values_torch

    worst = 0.0
    checked = 0
    for pair in range(domain["m"] * domain["top_k"]):
        t = pair // domain["top_k"]
        eid = int(domain["topk_ids"][t, pair % domain["top_k"]])
        # reference intermediate row
        x_deq = domain["x_quantized"][t][1]
        gate = domain["w13_dequant"][eid][domain["n"] :] @ x_deq
        up = domain["w13_dequant"][eid][: domain["n"]] @ x_deq
        inter = torch.nn.functional.silu(gate) * up
        # Match the kernel's BF16 shared sC staging before the FP4 requant.
        inter = inter.to(torch.bfloat16).float()
        _, inter_q, _ = _quantize_nvfp4_rows(inter, 1.0)
        phys = int(domain["phys_of_pair"][pair])
        # decode device payload+scale for this row
        row_bytes = payload[phys]  # [n//2]
        # sf_plane is already sliced past the payload plane, so the byte
        # offset of (it, phys) is relative to the scale-plane base.
        scale_bytes = []
        for it in range(domain["intermediate_tiles"]):
            base = (it * rows_capacity + phys) * 8
            w0 = sf_plane[base : base + 8]
            scale_bytes.append(w0)
        deq_row = torch.zeros(domain["n"], dtype=torch.float32, device="cuda")
        lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            + [-v for v in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)],
            dtype=torch.float32,
            device="cuda",
        )
        for it in range(domain["intermediate_tiles"]):
            w0, w1 = scale_bytes[it][:4], scale_bytes[it][4:]
            for blk in range(8):
                kb = it * 8 + blk
                if blk < 4:
                    sbyte = int(w0[blk])
                else:
                    sbyte = int(w1[blk - 4])
                sf = (
                    torch.tensor([sbyte], dtype=torch.uint8, device="cuda")
                    .view(torch.float8_e4m3fn)
                    .float()
                    .item()
                )
                lo = (kb * 16) // 2
                # decode 16 values from 8 bytes
                vals = []
                for bidx in range(8):
                    b = int(row_bytes[lo + bidx])
                    vals.append(lut[b & 0xF].item())
                    vals.append(lut[(b >> 4) & 0xF].item())
                for vi, v in enumerate(vals):
                    deq_row[kb * 16 + vi] = v * sf
        if checked < 200:
            diff = (deq_row - inter_q).abs().max().item()
            worst = max(worst, diff)
            checked += 1
    assert worst <= 1e-6, f"intermediate mismatch: max abs {worst}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_phase_live_row_beyond_int32_offset() -> None:
    """A live physical row whose scaled pool offset exceeds 2^31 stays correct.

    The phase kernels scale a physical row id into pool offsets
    (``physical_row * words_per_row`` for the payload, ``sf_atom * k4 * 512``
    for the scale plane).  Parking the live domain at a high ``tile_base`` makes
    those kernel-computed products cross the Int32 boundary while only the tail
    of each pool is written, so Int32 truncation in either phase kernel would
    corrupt the output.  Shifting merely the base pointer of a small pool does
    not exercise the kernels' own offset math, so this test does not do that.
    """
    require_b12x()
    K = n = 128
    m, top_k, E = 8, 1, 2
    # Upper bound on the pools this domain allocates, so a smaller GPU skips
    # instead of OOM-ing.  Live rows sit in the tail; the prefix is allocated.
    rows_capacity = (_INT32_BOUNDARY_TILE_BASE + E) * _TILE_M
    words_per_row = (n // _TILE_N) * 16
    required_bytes = (
        rows_capacity * (K // 2)  # packed_a
        + (rows_capacity // _TILE_M) * (K // 64) * 512  # scale_flat
        + rows_capacity * words_per_row * 4  # intermediate payload
        + (n // _TILE_N) * rows_capacity * 2 * 4  # intermediate scale plane
        + rows_capacity * 4 * 2  # token_map + token_weights
    )
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes + 2 * 1024**3:
        pytest.skip(
            "Int32-boundary live-row test requires "
            f"{required_bytes + 2 * 1024**3} bytes free, found {free_bytes}"
        )
    domain = _build_domain(
        E=E, K=K, n=n, m=m, top_k=top_k, seed=17, tile_base=_INT32_BOUNDARY_TILE_BASE
    )
    assert domain["intermediate_tiles"] == 1, "premise: one intermediate tile"
    max_live_phys = int(domain["phys_of_pair"].max())
    # Prove the domain actually crosses the boundary the kernels must survive.
    assert max_live_phys * words_per_row > 2**31, (
        f"live intermediate offset does not exceed 2^31: "
        f"{max_live_phys} * {words_per_row} = {max_live_phys * words_per_row}"
    )
    assert max_live_phys * (K // 2) > 2**31, (
        "live activation byte offset must also exceed 2^31"
    )
    out, _ = _run_phases(domain)
    assert out.abs().sum().item() > 0, "kernel produced all zeros"
    metrics = compare_to_reference(out.float(), domain["oracle"])
    assert metrics.cos > 0.9999, metrics
    bound = _bf16_output_bound(domain["oracle"])
    assert metrics.rmse <= bound, (metrics, bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_phase_frozen_resolution_two_counts() -> None:
    """One compiled callable, two different live task counts, both correct."""
    require_b12x()
    domain_a = _build_domain(E=8, K=256, n=128, m=64, top_k=2, seed=15)
    domain_b = _build_domain(E=8, K=256, n=128, m=256, top_k=2, seed=16)
    # The compiled callable is capacity-specialized: sharing one callable
    # requires the same physical-tile capacity.  The shared-callable claim is
    # about LIVE task counts, which differ (128 vs 512 routed rows), so the
    # capacities must match while the written work must not.
    assert domain_a["phys_tiles"] == domain_b["phys_tiles"], (
        f"shared-callable test requires equal capacity: "
        f"{domain_a['phys_tiles']} vs {domain_b['phys_tiles']}"
    )
    assert domain_a["rows_capacity"] == domain_b["rows_capacity"]
    compiled_p1 = _compile_phase1(domain_a)
    compiled_p2 = _compile_phase2(domain_a)

    # launch A (small live counts)
    out_a, intermediate_a = _run_phases(
        domain_a, compiled_p1=compiled_p1, compiled_p2=compiled_p2
    )
    metrics_a = compare_to_reference(out_a.float(), domain_a["oracle"])
    assert metrics_a.cos > 0.9999, metrics_a

    # same compiled callables, larger live counts
    out_b, intermediate_b = _run_phases(
        domain_b, compiled_p1=compiled_p1, compiled_p2=compiled_p2
    )
    metrics_b = compare_to_reference(out_b.float(), domain_b["oracle"])
    assert metrics_b.cos > 0.9999, metrics_b
    assert out_b.abs().sum().item() > 0
    # The live routing materialized strictly more rows for domain_b, so it must
    # leave strictly more nonzero intermediate words (payload + scale planes)
    # while sharing the same compiled callable.
    written_a = int((intermediate_a != 0).sum())
    written_b = int((intermediate_b != 0).sum())
    assert written_b > written_a, (
        f"larger live count must write more intermediate words: "
        f"b={written_b} vs a={written_a}"
    )


if __name__ == "__main__":
    raise SystemExit("run via pytest")
