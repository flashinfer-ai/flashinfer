"""W4A8 throughput-tier MoE pipeline driver (v1).

End-to-end forward for one MoE layer, all custom kernels:

    pack_routes_w4a8 (w4a16 semantics, expert runs padded to whole 48-row
    GEMM groups) -> mxfp8_quantize_rows(x) [m rows only]
    -> FC1 grouped GEMM (gather_a: A rows through packed_route_indices;
       experts_per_group keeps every CTA group on the uniform fast path)
       -> cache13 [cap_rows, 2n] bf16
    -> fused silu(gate)*up + MXFP8 quant -> act_q/act_sf
    -> FC2 grouped GEMM (dense over the packed rows) -> cache2 [cap_rows, K]
    -> weighted top-k gather-sum -> out [m, K] bf16

Determinism / route semantics: ``packed_route_indices[s]`` encodes
``token*topk + j``; an inverse map (packed position of every (token, j)
route, built by a tiny scatter kernel) lets the final sum read the FC2 rows
in fixed j-order per token -- same determinism approach as the w4a16
scatter + W4A16TopKSumKernel pair, with the topk-weight multiply riding in
the sum (this GEMM's epilogue does not multiply weights).

Serving discipline (hot path = :func:`w4a8_tier_forward`):
- All buffers live in a workspace built once by
  :func:`build_w4a8_tier_workspace` (capacity-sized for the worst-case route
  packing); the hot path performs no device allocation, no ``.item()``, and
  no host sync. ``packed_route_count`` stays on device; grid sizing uses
  worst-case capacity blocks with -1 expert ids masking dead CTAs and the
  act kernel early-exiting on the device count.
- Cute kernels are compiled on first call (do the warmup call before
  capture/timing); subsequent calls reuse cached compiled objects and cached
  cute views (rebuilt only if a tensor's data_ptr changes, so CUDA-graph
  capture sees stable pointers).

CONTRACT (v1): every ``topk_ids`` entry must be a valid expert in
``[0, num_experts)`` (no expert_map / dropped routes) -- the inverse map is
fully overwritten each call only when all m*topk routes are present.

Group padding note: route packing pads each expert run to whole 48-row
groups (measured at DS4 m=4096: 16-row padding + mixed-expert slow-path
sweeps cost FC1 1.94ms / FC2 0.91ms vs 1.45ms / 0.69ms with group padding
+ experts_per_group, at ~+14% padded rows -- net -0.71ms).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import torch
import triton
import triton.language as tl

from .act import silu_mul_mxfp8_quantize_rows
from .gemm import (
    W4A8GemmKernel,
    repack_w4a8_weights,
    to_cute_u32,
)
from .quant import mxfp8_quantize_rows
from .route import pack_routes_w4a8, w4a8_route_capacity

_BLOCK_M = 16
_GROUP_BLOCKS = 3  # = gemm._BLOCKS_PER_CTA
_GROUP_ROWS = _GROUP_BLOCKS * _BLOCK_M  # route-pack padding granularity
_INVALID_ROUTE = torch.iinfo(torch.int32).max
_GEMM_COMPILE_CACHE: dict[tuple, object] = {}


# --------------------------------------------------------------------------
# Triton glue kernels
# --------------------------------------------------------------------------


@triton.jit
def _route_inverse_kernel(
    packed_route_indices,
    inverse,
    num_slots,
    total_routes,
    BLOCK_T: tl.constexpr,
):
    """inverse[idx] = packed slot of route idx, for every valid slot."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offs < num_slots
    idx = tl.load(packed_route_indices + offs, mask=mask, other=0x7FFFFFFF)
    valid = mask & (idx < total_routes)
    tl.store(inverse + idx, offs, mask=valid)


@triton.jit
def _topk_weighted_sum_kernel(
    fc2,  # bf16 [cap_rows, hidden] (packed rows)
    inverse,  # i32 [m*topk]: packed row of route (t, j)
    weights,  # f32 [m, topk]
    out,  # bf16 [m, hidden]
    hidden,
    TOPK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        src = tl.load(inverse + token * TOPK + j)
        w = tl.load(weights + token * TOPK + j).to(tl.float32)
        acc += w * tl.load(fc2 + src * hidden + cols).to(tl.float32)
    tl.store(out + token * hidden + cols, acc.to(out.dtype.element_ty))


# --------------------------------------------------------------------------
# Prepare (host, one-time)
# --------------------------------------------------------------------------


def prepare_w4a8_tier_weights(
    w13_fp4: torch.Tensor,  # [E, 2n, K/2] u8, kernel order [up; gate]
    w13_mx: torch.Tensor,  # [E, 2n, K/32] u8
    w2_fp4: torch.Tensor,  # [E, K, n/2] u8
    w2_mx: torch.Tensor,  # [E, K, n/32] u8
) -> dict:
    """Repack both expert weight sets for the W4A8 GEMM (one-time, host)."""
    w13_rp, w13_sfb = repack_w4a8_weights(w13_fp4, w13_mx)
    w2_rp, w2_sfb = repack_w4a8_weights(w2_fp4, w2_mx)
    return {
        "w13_rp": w13_rp,
        "w13_sfb": w13_sfb,
        "w2_rp": w2_rp,
        "w2_sfb": w2_sfb,
    }


def build_w4a8_tier_workspace(
    *,
    m: int,
    hidden_size: int,  # K
    intermediate_size: int,  # n (per-partition I_tp)
    num_experts: int,
    topk: int,
    device: torch.device,
) -> dict:
    """Allocate every pipeline buffer at worst-case route-pack capacity."""
    k = int(hidden_size)
    n = int(intermediate_size)
    e = int(num_experts)
    topk = int(topk)
    m = int(m)
    total_routes = m * topk
    # Route packing pads expert runs to whole 3-block GEMM groups (48 rows):
    # every group is expert-uniform (the GEMM's experts_per_group fast path).
    cap_rows, cap_groups = w4a8_route_capacity(total_routes, e, _GROUP_ROWS)
    cap_blocks = cap_rows // _BLOCK_M

    pri = torch.full((cap_rows,), _INVALID_ROUTE, dtype=torch.int32, device=device)
    ws = {
        "m": m,
        "k": k,
        "n": n,
        "num_experts": e,
        "topk": topk,
        "total_routes": total_routes,
        "cap_blocks": cap_blocks,
        "cap_groups": cap_groups,
        "cap_rows": cap_rows,
        # route packing (pri tail beyond route_pack's written capacity stays
        # at the one-time invalid fill -- the GEMM gather contract).
        "pri": pri,
        "beids": torch.empty(cap_groups, dtype=torch.int32, device=device),
        "count": torch.zeros(1, dtype=torch.int32, device=device),
        "eoff": torch.empty(e + 1, dtype=torch.int32, device=device),
        "ecnt": torch.zeros(e, dtype=torch.int32, device=device),
        "inv": torch.zeros(max(total_routes, 1), dtype=torch.int32, device=device),
        # activations
        "xq": torch.empty(m, k, dtype=torch.float8_e4m3fn, device=device),
        "xsf": torch.empty(m, k // 32, dtype=torch.uint8, device=device),
        "cache13": torch.zeros(cap_rows, 2 * n, dtype=torch.bfloat16, device=device),
        "act_q": torch.zeros(cap_rows, n, dtype=torch.float8_e4m3fn, device=device),
        "act_sf": torch.zeros(cap_rows, n // 32, dtype=torch.uint8, device=device),
        "cache2": torch.zeros(cap_rows, k, dtype=torch.bfloat16, device=device),
        "out": torch.empty(m, k, dtype=torch.bfloat16, device=device),
        "_views": {},
        "_compiled": {},
    }
    return ws


def _cached_u32_view(ws: dict, key: str, t: torch.Tensor):
    entry = ws["_views"].get(key)
    if entry is None or entry[0] is not t:
        entry = (t, to_cute_u32(t))
        ws["_views"][key] = entry
    return entry[1]


def _cached_i32_view(ws: dict, key: str, t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    entry = ws["_views"].get(key)
    if entry is None or entry[0] is not t:
        cute_t = from_dlpack(t, assumed_align=4)
        cute_t.element_type = cutlass.Int32
        entry = (t, cute_t)
        ws["_views"][key] = entry
    return entry[1]


def _gemm_compiled(ws: dict, key: str, kernel: W4A8GemmKernel, args: tuple):
    import cutlass.cute as cute

    cache_key = (kernel.__cache_key__, ws["m"], ws["cap_rows"])
    compiled = _GEMM_COMPILE_CACHE.get(cache_key)
    if compiled is None:
        compiled = cute.compile(kernel, *args)
        _GEMM_COMPILE_CACHE[cache_key] = compiled
    ws["_compiled"][key] = compiled
    return compiled


# --------------------------------------------------------------------------
# Hot path
# --------------------------------------------------------------------------


def w4a8_tier_forward(
    x: torch.Tensor,  # [m, K] bf16
    w13_rp: torch.Tensor,
    w13_sfb: torch.Tensor,
    w2_rp: torch.Tensor,
    w2_sfb: torch.Tensor,
    topk_ids: torch.Tensor,  # [m, topk] i32, all valid experts
    topk_weights: torch.Tensor,  # [m, topk] f32
    workspace: dict,
    *,
    out: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    stage_events: list | None = None,
) -> torch.Tensor:
    """One MoE layer forward; see the module docstring for the contract.

    ``stage_events``: debug-only -- when a list is supplied, (label, event)
    pairs are appended around every stage for per-stage timing breakdowns.
    """
    ws = workspace
    m, k, n = ws["m"], ws["k"], ws["n"]
    e, topk = ws["num_experts"], ws["topk"]
    cap_blocks, cap_rows = ws["cap_blocks"], ws["cap_rows"]
    total_routes = ws["total_routes"]
    if out is None:
        out = ws["out"]

    def _mark(label: str) -> None:
        if stage_events is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            stage_events.append((label, ev))

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _mark("start")

    # 1) Route pack (expert runs padded to whole 48-row GEMM groups).
    pack_routes_w4a8(
        topk_ids,
        e,
        _GROUP_ROWS,
        packed_route_indices=ws["pri"],
        block_expert_ids=ws["beids"],
        packed_route_count=ws["count"],
        expert_offsets=ws["eoff"],
        expert_counts=ws["ecnt"],
    )
    # Inverse map: packed position of every (token, j) route.
    _route_inverse_kernel[(triton.cdiv(cap_rows, 256),)](
        ws["pri"], ws["inv"], cap_rows, total_routes, BLOCK_T=256, num_warps=4
    )
    _mark("route_pack")

    # 2) MXFP8 quant of x (m rows only; FC1 gathers through the routes).
    mxfp8_quantize_rows(x, out_values=ws["xq"], out_scales=ws["xsf"])
    _mark("quant_x")

    # 3) FC1 grouped GEMM, A-row gather -> cache13 [cap_rows, 2n].
    fc1_kernel = W4A8GemmKernel(
        size_n=2 * n,
        size_k=k,
        num_experts=e,
        gather_a=True,
        topk=topk,
        experts_per_group=True,
    )
    fc1_args = (
        _cached_u32_view(ws, "xq", ws["xq"]),
        _cached_u32_view(ws, "xsf", ws["xsf"]),
        _cached_u32_view(ws, "w13_rp", w13_rp),
        _cached_u32_view(ws, "w13_sfb", w13_sfb),
        _cached_u32_view(ws, "cache13", ws["cache13"]),
        _cached_i32_view(ws, "beids", ws["beids"]),
        _cached_i32_view(ws, "pri", ws["pri"]),
        cutlass.Int32(cap_blocks),
        cutlass.Int32(cap_rows),
        cutlass.Int32(total_routes),
        cutlass.Int32(fc1_kernel.grid_x(cap_blocks)),
        stream,
    )
    fc1 = ws["_compiled"].get("fc1")
    if fc1 is None:
        fc1 = _gemm_compiled(ws, "fc1", fc1_kernel, fc1_args)
    fc1(*fc1_args)
    _mark("fc1")

    # 4) Fused silu(gate)*up + MXFP8 quant over the live packed rows.
    silu_mul_mxfp8_quantize_rows(
        ws["cache13"],
        out_values=ws["act_q"],
        out_scales=ws["act_sf"],
        valid_rows=ws["count"],
        swiglu_limit=swiglu_limit,
    )
    _mark("act_quant")

    # 5) FC2 grouped GEMM, dense over the packed rows -> cache2 [cap_rows, K].
    fc2_kernel = W4A8GemmKernel(
        size_n=k, size_k=n, num_experts=e, experts_per_group=True
    )
    fc2_args = (
        _cached_u32_view(ws, "act_q", ws["act_q"]),
        _cached_u32_view(ws, "act_sf", ws["act_sf"]),
        _cached_u32_view(ws, "w2_rp", w2_rp),
        _cached_u32_view(ws, "w2_sfb", w2_sfb),
        _cached_u32_view(ws, "cache2", ws["cache2"]),
        _cached_i32_view(ws, "beids", ws["beids"]),
        _cached_i32_view(ws, "pri", ws["pri"]),
        cutlass.Int32(cap_blocks),
        cutlass.Int32(cap_rows),
        cutlass.Int32(0),
        cutlass.Int32(fc2_kernel.grid_x(cap_blocks)),
        stream,
    )
    fc2 = ws["_compiled"].get("fc2")
    if fc2 is None:
        fc2 = _gemm_compiled(ws, "fc2", fc2_kernel, fc2_args)
    fc2(*fc2_args)
    _mark("fc2")

    # 6) Deterministic weighted top-k sum (fixed j order per token).
    block_c = 512 if k % 512 == 0 else 256
    if k % block_c != 0:
        raise ValueError(f"hidden_size {k} must be divisible by {block_c}")
    _topk_weighted_sum_kernel[(m, k // block_c)](
        ws["cache2"],
        ws["inv"],
        topk_weights,
        out,
        k,
        TOPK=topk,
        BLOCK_C=block_c,
        num_warps=4,
    )
    _mark("topk_sum")
    return out


def w4a8_tier_stage_timings(
    *forward_args,
    iters: int = 20,
    **forward_kwargs,
) -> dict[str, float]:
    """Debug helper: average per-stage ms via CUDA events (not the hot path)."""
    timings: dict[str, float] = {}
    for _ in range(iters):
        events: list = []
        w4a8_tier_forward(*forward_args, stage_events=events, **forward_kwargs)
        torch.cuda.synchronize()
        for (_l0, e0), (l1, e1) in zip(events, events[1:], strict=False):
            timings[l1] = timings.get(l1, 0.0) + e0.elapsed_time(e1)
    return {k: v / iters for k, v in timings.items()}
