"""Verified inverse mappings for the W4A8-MX N256/K128 repacked layout.

The w4a8_mx weight prep repacks logical FP4 weights and e8m0 grids in place
(`_logical_weight_to_w4a8_rp_inplace`, `_e8m0_scale_to_w4a8_sfb_inplace`).
Both transforms are pure power-of-two permutations, so a logical
(row, int32-word) or (row, scale-col) coordinate maps to the repacked buffer
with a handful of bitfield ops. These are the formulas a tiny-decode kernel
(micro-class) needs to consume the repacked buffers directly, with zero extra
weight memory.

Verified bit-exact against the real prep kernels on:
  - w13 orientation (rows=2n, k_dim=k, row_rotation=n for the w31 gated layout)
  - w2 orientation  (rows=k,  k_dim=n, no rotation)
  - both e8m0 grids
Note: the sfb prep clamps e8m0 bytes >= 249 (2^122+; unreachable for real
weights), so synthetic inputs must stay below that.

Coalescing caveat for kernel authors: within one logical row, consecutive
int32 words sit 4 words (16 B) apart in the repacked layout; the 16 B
rp-contiguous unit covers logical rows {r, r+8, r+16, r+24} at one k-window
(the n8i mode). A warp that wants full sectors must therefore cover four
8-apart rows per instruction instead of one row per warp.
"""

import pytest
import torch

from b12x.moe.fused_moe._impl import (
    _e8m0_scale_to_w4a8_sfb_inplace,
    _logical_weight_to_w4a8_rp_inplace,
)


def rp_word_offset(
    r: int,
    w: int,
    *,
    rows: int,
    k_tiles: int,
    rot: int,
    half: int = 0,
) -> int:
    """Logical (row, int32-word) -> flat int32-word offset in the rp buffer."""
    if half:
        half_pad = -(-half // 128) * 128
        rel = (r - rot) % rows
        p = rel if rel < half else half_pad + (rel - half)
    else:
        p = (r - rot) % rows
    nt, row = p >> 8, p & 255
    n8c, n8i, r8 = row >> 5, (row >> 3) & 3, row & 7
    kt, k32, cgrp = w >> 4, (w & 15) >> 2, w & 3
    idx = n8i | (cgrp << 2) | (r8 << 4) | (n8c << 7) | (k32 << 10)
    return (nt * k_tiles + kt) * 4096 + idx


def sfb_byte_offset(r: int, c: int, *, rows: int, k_tiles: int, rot: int) -> int:
    """Logical (row, per-32 scale col) -> flat byte offset in the sfb buffer."""
    p = (r - rot) % rows
    nt, q = p >> 8, p & 255
    n32, row8 = q >> 3, q & 7
    kt, kb = c >> 2, c & 3
    return kb | (row8 << 2) | (n32 << 5) | ((nt * k_tiles + kt) << 10)


@pytest.mark.parametrize(
    "rows,kdim,rot,half",
    [
        (2048, 4096, 1024, 0),
        (4096, 1024, 0, 0),
        # ceil-tiled tails (2048/TP6 = 352): w13 = 704 rows rotated by 352,
        # w2 = K tail (352 = 2x128 + 96)
        (704, 4096, 352, 0),
        (4096, 352, 0, 0),
        (352, 352, 0, 0),
        # gated half-aligned tail layout (the serving w13 path)
        (704, 4096, 352, 352),
    ],
    ids=["w13_rotated", "w2", "w13_n_tail", "w2_k_tail", "both_tails",
         "w13_half_tail"],
)
def test_rp_weight_inverse(rows: int, kdim: int, rot: int, half: int) -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    logical = torch.randint(0, 256, (1, rows, kdim // 2), dtype=torch.uint8, device=dev)
    qwords = logical.clone().view(torch.int32).reshape(1, rows, kdim // 8)
    rp = _logical_weight_to_w4a8_rp_inplace(
        logical.clone(),
        size_k=kdim,
        size_n=rows,
        row_rotation=(rot or None),
        gated_half_rows=(half or None),
    )
    rp_flat = rp.reshape(-1).view(torch.int32)
    k_tiles = -(-kdim // 128)
    rs = torch.randint(0, rows, (4096,))
    ws = torch.randint(0, kdim // 8, (4096,))
    offs = torch.tensor(
        [
            rp_word_offset(
                int(r), int(w), rows=rows, k_tiles=k_tiles, rot=rot, half=half
            )
            for r, w in zip(rs, ws)
        ],
        device=dev,
    )
    assert torch.equal(rp_flat[offs], qwords[0][rs.to(dev), ws.to(dev)])
    n_tiles = -(-rows // 256)
    if n_tiles * 256 != rows or k_tiles * 128 != kdim:
        # every rp word not addressed by a logical (row, word) must be zero
        seen = torch.zeros(n_tiles * k_tiles * 4096, dtype=torch.bool, device=dev)
        all_r = torch.arange(rows).repeat_interleave(kdim // 8)
        all_w = torch.arange(kdim // 8).repeat(rows)
        all_offs = torch.tensor(
            [
                rp_word_offset(
                    int(r), int(w), rows=rows, k_tiles=k_tiles, rot=rot, half=half
                )
                for r, w in zip(all_r, all_w)
            ],
            device=dev,
        )
        seen[all_offs] = True
        assert (rp_flat[~seen] == 0).all()


@pytest.mark.parametrize(
    "rows,kdim,rot",
    [
        (2048, 4096, 1024),
        (4096, 1024, 0),
        (704, 4096, 352),
        (4096, 352, 0),
        (352, 352, 0),
    ],
    ids=["w13_rotated", "w2", "w13_n_tail", "w2_k_tail", "both_tails"],
)
def test_sfb_grid_inverse(rows: int, kdim: int, rot: int) -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    scale_cols = kdim // 32
    scale = (
        torch.arange(rows * scale_cols, dtype=torch.int32, device=dev).reshape(
            1, rows, scale_cols
        )
        % 200
        + 1
    ).to(torch.uint8)
    sfb = _e8m0_scale_to_w4a8_sfb_inplace(
        scale.clone(), weight_E=1, rows=rows, k_dim=kdim, row_rotation=(rot or None)
    )
    sfb_flat = sfb.reshape(-1).view(torch.uint8)
    k_tiles = -(-kdim // 128)
    rs = torch.randint(0, rows, (4096,))
    cs = torch.randint(0, scale_cols, (4096,))
    offs = torch.tensor(
        [
            sfb_byte_offset(int(r), int(c), rows=rows, k_tiles=k_tiles, rot=rot)
            for r, c in zip(rs, cs)
        ],
        device=dev,
    )
    assert torch.equal(sfb_flat[offs], scale[0][rs.to(dev), cs.to(dev)])
