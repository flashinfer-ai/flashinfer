#!/usr/bin/env python3
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
"""Correctness checks for :mod:`vsa_attention`.

Four independent checks, all on a small grid so the dense references fit:

1. **Layout** — tile/untile is an exact round-trip and the cube permutation
   really groups each ``(4, 4, 4)`` neighbourhood into one 64-token block.
2. **Kernel** — the full VSA path matches an eager PyTorch reference that
   reimplements the same math (masked pooling, top-k, masked dense attention
   over the selected blocks, gated combine).
3. **Head independence** — splitting the heads across ranks (what Ulysses
   does) must not change the result, since VSA runs per rank on a head slice.
4. **Approximation quality** — VSA at a few sparsities versus full dense
   attention, reported as cosine similarity so the sparse branch is visibly
   doing something sensible rather than silently returning garbage.

Run: ``python examples/pytorch/wan/vsa_sanity.py``
"""

import math

import torch

from vsa_attention import (
    VSA_BLOCK_SIZE,
    VSA_TILE_SIZE,
    build_vsa_metadata,
    vsa_attention,
)


def _reference_vsa(q, k, v, gate, meta, sparsity):
    """Eager reimplementation of the VSA math, independent of the kernel."""
    b, _, h, d = q.shape
    scale = 1.0 / math.sqrt(d)
    nb, blk = meta.num_blocks, VSA_BLOCK_SIZE

    def tile(x):
        buf = x.new_zeros((b, meta.padded_seq_len, h, d))
        buf[:, meta.non_pad_index] = x[:, meta.tile_partition_indices]
        return buf

    q_t, k_t, v_t, g_t = (tile(x) for x in (q, k, v, gate))
    sizes = meta.variable_block_sizes.view(1, -1, 1, 1).float()

    def pool(x):
        return (x.view(b, nb, blk, h, d).float().sum(2) / sizes).permute(0, 2, 1, 3)

    q_c, k_c, v_c = pool(q_t), pool(k_t), pool(v_t)
    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
    out_c = torch.matmul(torch.softmax(scores, -1), v_c)  # [B,H,NB,D]

    topk = meta.topk_for_sparsity(sparsity)
    sel = torch.topk(scores, topk, dim=-1).indices  # [B,H,NB,topk]
    block_mask = torch.zeros(b, h, nb, nb, dtype=torch.bool, device=q.device)
    block_mask.scatter_(-1, sel, True)

    # Token-level mask: block selection AND per-block valid-token count.
    token_valid = (
        torch.arange(blk, device=q.device)[None, :] < meta.variable_block_sizes[:, None]
    ).reshape(-1)  # [S_padded]
    token_mask = block_mask.repeat_interleave(blk, dim=2).repeat_interleave(blk, dim=3)
    token_mask &= token_valid.view(1, 1, 1, -1)

    qq = q_t.permute(0, 2, 1, 3).float()
    kk = k_t.permute(0, 2, 1, 3).float()
    vv = v_t.permute(0, 2, 1, 3).float()
    logits = (qq @ kk.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~token_mask, float("-inf"))
    out_s = (torch.softmax(logits, -1) @ vv).permute(0, 2, 1, 3).to(q.dtype)

    out = out_s.view(b, nb, blk, h, d) + out_c.permute(0, 2, 1, 3).unsqueeze(2).to(
        q.dtype
    ) * g_t.view(b, nb, blk, h, d)
    return out.view(b, meta.padded_seq_len, h, d)[:, meta.untile_combined_index]


def _dense(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    qq, kk, vv = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    return (torch.softmax(qq @ kk.transpose(-1, -2) * scale, -1) @ vv).permute(
        0, 2, 1, 3
    )


def _cos(a, b):
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Deliberately not divisible by the tile size in T and H, so partial cubes
    # (and therefore block_sizes masking) are exercised.
    grid = (5, 9, 12)
    b, h, d = 1, 4, 128
    meta = build_vsa_metadata(grid, device)
    s = meta.seq_len

    print(f"grid={grid}  tokens={s}  tiles={meta.num_tiles}  blocks={meta.num_blocks}")
    print(f"padded={meta.padded_seq_len}  (+{meta.padded_seq_len / s - 1:.1%})")
    print(
        "block sizes: "
        f"min={meta.variable_block_sizes.min().item()} "
        f"max={meta.variable_block_sizes.max().item()} "
        f"sum={meta.variable_block_sizes.sum().item()} (== tokens: "
        f"{meta.variable_block_sizes.sum().item() == s})"
    )

    # --- 1. layout ------------------------------------------------------------
    x = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16)
    buf = x.new_zeros((b, meta.padded_seq_len, h, d))
    buf[:, meta.non_pad_index] = x[:, meta.tile_partition_indices]
    assert torch.equal(buf[:, meta.untile_combined_index], x), "tile/untile mismatch"

    # First block must hold exactly the (4,4,4) corner cube of the grid.
    t, hh, ww = grid
    ids = torch.arange(t * hh * ww, device=device).reshape(t, hh, ww)
    ts, hs, ws = VSA_TILE_SIZE
    corner = ids[:ts, :hs, :ws].flatten()
    got = meta.tile_partition_indices[: corner.numel()]
    assert torch.equal(got, corner), "first cube is not the (4,4,4) corner"
    print("[1] layout: tile/untile round-trip exact, cube grouping correct  OK")

    # --- 2. kernel vs eager reference ----------------------------------------
    q, k, v, g = (
        torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16) for _ in range(4)
    )
    for sparsity in (0.0, 0.5, 0.9):
        out = vsa_attention(q, k, v, g, meta, sparsity)
        ref = _reference_vsa(q, k, v, g, meta, sparsity)
        err = (out.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        topk = meta.topk_for_sparsity(sparsity)
        print(
            f"[2] sparsity={sparsity:<4} topk={topk:>4}/{meta.num_blocks}  "
            f"max_abs_err={err:.4f}  ref_max={scale:.3f}  rel={err / scale:.2e}"
        )
        assert err / scale < 2e-2, f"kernel disagrees with reference at {sparsity}"
    print("[2] kernel matches the eager VSA reference  OK")

    # --- 3. head independence (the invariant Ulysses relies on) --------------
    # Under Ulysses each rank runs VSA over H/world_size heads of the full
    # sequence, so per-slice results must concatenate back to the all-heads
    # result exactly, or sequence-parallel output would not be single-GPU
    # output.
    heads_mp = 8
    qm, km, vm, gm = (
        torch.randn(b, s, heads_mp, d, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    )
    full = vsa_attention(qm, km, vm, gm, meta, 0.5)
    ref_scale = full.float().abs().max().item()
    for world in (2, 4, 8):
        h_local = heads_mp // world
        split = torch.cat(
            [
                vsa_attention(
                    *(
                        x[:, :, r * h_local : (r + 1) * h_local].contiguous()
                        for x in (qm, km, vm, gm)
                    ),
                    meta,
                    0.5,
                )
                for r in range(world)
            ],
            dim=2,
        )
        err = (full.float() - split.float()).abs().max().item()
        print(
            f"[3] world={world} (h_local={h_local}): max_abs_diff={err:.5f} "
            f"rel={err / ref_scale:.2e}"
        )
        # The kernel picks its tiling from the head count, so a differently
        # shaped launch can reassociate the softmax accumulation. Anything at
        # bf16 rounding level is fine; a selection change would be far larger.
        assert err / ref_scale < 5e-3, f"head split changes the result at {world=}"
    print("[3] head independence holds up to bf16 rounding  OK")

    # --- 4. approximation quality vs dense attention -------------------------
    # gate=0 isolates the sparse branch; gate=1 is the full VSA output.
    zero = torch.zeros_like(g)
    dense = _dense(q, k, v)
    for sparsity in (0.0, 0.5, 0.75, 0.9):
        sparse_only = vsa_attention(q, k, v, zero, meta, sparsity)
        print(
            f"[4] sparsity={sparsity:<5} cos(sparse-branch, dense)={_cos(sparse_only, dense):.4f}"
        )
    print("\nall checks passed")


if __name__ == "__main__":
    main()
