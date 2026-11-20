"""One-launch staging of pre-quantized MXFP8 rows into the NVLinkOneSided workspace.

One program per workspace row ``t``. Below the token count it copies the row's
E4M3 data and UE8M0 scales with the sentinel rewrite of ``staging.py`` (data
``0xFF`` -> ``0x7F``, scale ``0xFF`` -> ``0xFE``), the staged scales also into
the shared expert's scale plane at the swizzled slot row of ``t`` when the
workspace has one, the Top K ids narrowed to the workspace's int32 and the FP32
routing weights. From the token count on it masks the row's routes (``-1``),
so rows a larger earlier batch left behind dispatch nothing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Data bytes each loop step of a program moves.
_CHUNK = 2048


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "sf_stride",
        "x_sf_out_stride",
        "tile_rows",
        "slot_rows",
    ]
)
def _stage_kernel(
    x,
    sf,
    ids,
    weights,
    x_out,
    sf_out,
    ids_out,
    weights_out,
    shared_sf_out,
    num_tokens,
    sf_stride,
    x_sf_out_stride,
    tile_rows,
    slot_rows,
    HIDDEN: tl.constexpr,
    SF_COLS: tl.constexpr,
    SF_BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    SHARED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, TOP_K_BLOCK)
    routes = k < TOP_K
    if row < num_tokens:
        for start in tl.static_range(0, HIDDEN, CHUNK):
            cols = start + tl.arange(0, CHUNK)
            live = cols < HIDDEN
            data = tl.load(x + row * HIDDEN + cols, mask=live)
            data = tl.where(data == 0xFF, 0x7F, data).to(tl.uint8)
            tl.store(x_out + row * HIDDEN + cols, data, mask=live)
        cols = tl.arange(0, SF_BLOCK)
        live = cols < SF_COLS
        scales = tl.load(sf + row * sf_stride + cols, mask=live)
        scales = tl.where(scales == 0xFF, 0xFE, scales).to(tl.uint8)
        tl.store(sf_out + row * x_sf_out_stride + cols, scales, mask=live)
        if SHARED:
            # The 32x4x4 atom swizzle of the shared expert's scale plane
            # (Sm100GenphaseSymmBuffer.shared_expert_scale_layout).
            slot = row // tile_rows * slot_rows + row % tile_rows
            offset = (
                ((slot // 128) * (SF_COLS // 4) + cols // 4) * 512
                + (slot % 32) * 16
                + (slot // 32 % 4) * 4
                + cols % 4
            )
            tl.store(shared_sf_out + offset, scales, mask=live)
        route_ids = tl.load(ids + row * TOP_K + k, mask=routes)
        tl.store(ids_out + row * TOP_K + k, route_ids.to(tl.int32), mask=routes)
        route_weights = tl.load(weights + row * TOP_K + k, mask=routes)
        tl.store(weights_out + row * TOP_K + k, route_weights, mask=routes)
    else:
        unrouted = tl.full((TOP_K_BLOCK,), -1, tl.int32)
        tl.store(ids_out + row * TOP_K + k, unrouted, mask=routes)


def stage_prequantized(
    hidden_states: torch.Tensor,
    scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    workspace,
    sf_cols: int,
) -> None:
    """Enqueue the staging of every workspace row on the current stream.

    Contiguous E4M3 ``hidden_states``, int32/int64 ``topk_ids`` and FP32
    ``topk_weights``, UE8M0 ``scales`` rows with unit column stride; the
    workspace's ``x``, ``topk_idx`` and ``topk_weights`` contiguous and its
    ``x_sf`` rows with unit column stride.
    """
    capacity, top_k = workspace.topk_idx.shape
    shared = workspace.config.shared_expert is not None
    sf_out = workspace.x_sf.view(torch.uint8)
    if shared:
        shared_sf, tile_rows, slot_rows = workspace.shared_expert_scale_layout()
    else:
        shared_sf, tile_rows, slot_rows = sf_out, 1, 1  # not written
    _stage_kernel[(capacity,)](
        hidden_states.view(torch.uint8),
        scales.view(torch.uint8),
        topk_ids,
        topk_weights,
        workspace.x.view(torch.uint8),
        sf_out,
        workspace.topk_idx,
        workspace.topk_weights,
        shared_sf.view(torch.uint8),
        hidden_states.shape[0],
        scales.stride(0),
        workspace.x_sf.stride(0),
        tile_rows,
        slot_rows,
        HIDDEN=workspace.config.hidden,
        SF_COLS=sf_cols,
        SF_BLOCK=triton.next_power_of_2(sf_cols),
        TOP_K=top_k,
        TOP_K_BLOCK=triton.next_power_of_2(top_k),
        CHUNK=min(_CHUNK, triton.next_power_of_2(workspace.config.hidden)),
        SHARED=shared,
        num_warps=4,
    )
