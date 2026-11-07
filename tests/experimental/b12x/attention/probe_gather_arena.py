"""Gathered-arena MG KVARN_DUAL A/B probe (server gather-bug hunt).

Builds kvarn records in the GATHERED-ARENA geometry exactly as
vllm `_gather_full_ckv` lays them out (tile-stride 143,616 B = 256x536 +
6400 tail, page_size = group = 256), launches `run_unified_prefill_mg`
with dense-causal selection (what the full-CKV gather route feeds MG),
and compares against a torch reference unpacked from the same arena
bytes. Also runs the pool geometry (page 4352) as a control.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/js/.cache/b12x-kvarn-wt")
sys.path.insert(0, "/home/js/lil/vllm")

from b12x.attention.sparse_mla._preparation import (  # noqa: E402
    plan_cache_writer,
    writer_state,
)
from b12x.attention._shared.mla.prefill_mg import (  # noqa: E402
    run_unified_prefill_mg,
)
from b12x.attention._shared.mla.traits import (  # noqa: E402
    ComputeMode,
    ModelType,
    ScaleFormat,
)
from vllm.v1.attention.ops.kvarn_mla import (  # noqa: E402
    kvarn_mla_layout_from_dtype,
    unpack_kvarn_mla_records,
)

DEV = torch.device("cuda", 0)
GROUP = 256
PAGE = 4352
RECORD = 536
HEADS = 64
TILE_STRIDE = GROUP * RECORD + 6400  # 143616
L = 512
SM_SCALE = 0.088388


def write_records(n_tokens: int):
    """Write n_tokens synthetic records to a fake pool, return flat bytes."""
    cache = torch.zeros((4, PAGE, RECORD), dtype=torch.uint8, device=DEV)
    latents = torch.randn(n_tokens, L, device=DEV, dtype=torch.float32) * 0.5
    slots = torch.arange(n_tokens, device=DEV, dtype=torch.int64)
    wplan = plan_cache_writer(latents.to(torch.bfloat16), cache, slots)
    writer_state(wplan, device=DEV).run(latents.to(torch.bfloat16), cache, slots)
    torch.cuda.synchronize()
    return cache.reshape(-1, RECORD)[:n_tokens].contiguous(), latents


def build_arena(recs: torch.Tensor, used_rows: int, n_tokens: int):
    """Scatter token records at their gathered staging positions."""
    tiles = (used_rows + GROUP - 1) // GROUP
    buf = torch.zeros(tiles * TILE_STRIDE, dtype=torch.uint8, device=DEV)
    arena = buf.as_strided((tiles, GROUP, RECORD), (TILE_STRIDE, RECORD, 1))
    tok = torch.arange(n_tokens, device=DEV)
    rank = (tok // 4) % 2
    padded = used_rows // 2
    local = (tok // 8) * 4 + tok % 4
    gath = (rank * padded + local).to(torch.int64)
    flat = torch.zeros(used_rows * RECORD, dtype=torch.uint8, device=DEV)
    idx = (gath[:, None] * RECORD + torch.arange(RECORD, device=DEV)).reshape(-1)
    flat[idx] = recs.reshape(-1)
    arena.view(tiles, GROUP * RECORD).copy_(flat.view(tiles, GROUP * RECORD))
    return buf, arena


def run_arena(n_tokens: int) -> float:
    rows = ((n_tokens + 1) // 2) * 2  # even split across 2 ranks
    padded = ((rows + 1) // 2 + GROUP - 1) // GROUP * GROUP  # per-rank pad
    used_rows = 2 * padded
    recs, latents = write_records(n_tokens)
    buf, arena = build_arena(recs, used_rows, n_tokens)

    q = torch.randn((n_tokens, HEADS, L), device=DEV, dtype=torch.bfloat16)
    width = used_rows
    sel = torch.full((n_tokens, width), -1, dtype=torch.int32, device=DEV)
    idx = torch.arange(used_rows, device=DEV, dtype=torch.int32)
    # Dense-causal: row r attends gathered rows [0, r]; gathered row of
    # token r = rank(r)*padded + local(r) with interleave 4, ws 2.
    tok = torch.arange(n_tokens, device=DEV)
    rank = (tok // 4) % 2
    local = (tok // 8) * 4 + tok % 4
    gath = rank * padded + local
    for r in range(n_tokens):
        sel[r, : r + 1] = torch.sort(gath[: r + 1]).values
    topk_length = (torch.arange(1, n_tokens + 1, device=DEV)).to(torch.int32)

    output = torch.zeros((n_tokens, HEADS, L), dtype=torch.bfloat16, device=DEV)
    run_unified_prefill_mg(
        q=q,
        kv_cache=arena,
        topk_indices=sel,
        sm_scale=SM_SCALE,
        page_block_size=GROUP,
        topk_length=topk_length,
        output=output,
        stride_kv_block=arena.stride(0),
        compute_mode=int(ComputeMode.BF16),
        mg_n_hg=1,
        model_type=int(ModelType.KVARN_MLA),
        scale_format=int(ScaleFormat.KVARN_DUAL),
        latent_scale=1.0,
        fp8_rope=False,
        latent_scale_per_token=False,
    )
    torch.cuda.synchronize()

    flat = np.frombuffer(bytes(buf.detach().cpu().untyped_storage()), dtype=np.uint8)
    worst = (0.0, -1)
    for r in range(n_tokens):
        sel_r = sel[r, : topk_length[r]].to(torch.int64)
        rec_off = (sel_r // GROUP) * TILE_STRIDE + (sel_r % GROUP) * RECORD
        recs_r = torch.from_numpy(
            np.stack([flat[o : o + RECORD] for o in rec_off.tolist()])
        )
        k_hat, v_hat = unpack_kvarn_mla_records(
            recs_r, kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
        )
        k_hat, v_hat = k_hat.to(DEV), v_hat.to(DEV)
        score = (q[r].float() @ k_hat.T) * SM_SCALE
        ref = torch.softmax(score, dim=-1) @ v_hat
        err = (output[r].float() - ref).abs().mean()
        rel = float(err / ref.abs().mean().clamp_min(1e-6))
        if rel > worst[0]:
            worst = (rel, r)
    return worst


def diagnose(n_tokens: int, row: int):
    """Which prefix of the causal candidate list does the kernel match?"""
    import torch

    rows_pad = ((n_tokens + 1) // 2 + GROUP - 1) // GROUP * GROUP
    used_rows = 2 * rows_pad
    recs, _ = write_records(n_tokens)
    buf, arena = build_arena(recs, used_rows, n_tokens)
    q = torch.randn((n_tokens, HEADS, L), device=DEV, dtype=torch.bfloat16)
    sel = torch.full((n_tokens, used_rows), -1, dtype=torch.int32, device=DEV)
    tok = torch.arange(n_tokens, device=DEV)
    rank = (tok // 4) % 2
    local = (tok // 8) * 4 + tok % 4
    gath = (rank * rows_pad + local).to(torch.int64)
    for r in range(n_tokens):
        sel[r, : r + 1] = torch.sort(gath[: r + 1]).values
    topk_length = torch.arange(1, n_tokens + 1, device=DEV).to(torch.int32)
    output = torch.zeros((n_tokens, HEADS, L), dtype=torch.bfloat16, device=DEV)
    run_unified_prefill_mg(
        q=q,
        kv_cache=arena,
        topk_indices=sel,
        sm_scale=SM_SCALE,
        page_block_size=GROUP,
        topk_length=topk_length,
        output=output,
        stride_kv_block=arena.stride(0),
        compute_mode=int(ComputeMode.BF16),
        mg_n_hg=1,
        model_type=int(ModelType.KVARN_MLA),
        scale_format=int(ScaleFormat.KVARN_DUAL),
        latent_scale=1.0,
        fp8_rope=False,
        latent_scale_per_token=False,
    )
    torch.cuda.synchronize()
    flat = np.frombuffer(bytes(buf.detach().cpu().untyped_storage()), dtype=np.uint8)

    def ref_for(slots):
        rec_off = (slots // GROUP) * TILE_STRIDE + (slots % GROUP) * RECORD
        recs_r = torch.from_numpy(
            np.stack([flat[o : o + RECORD] for o in rec_off.tolist()])
        )
        k_hat, v_hat = unpack_kvarn_mla_records(
            recs_r, kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
        )
        score = (q[row].float() @ k_hat.T.to(DEV)) * SM_SCALE
        return (torch.softmax(score, dim=-1) @ v_hat.to(DEV)).cpu()

    sel_row = sel[row, : topk_length[row]].to(torch.int64)
    out = output[row].float().cpu()
    n = topk_length[row].item()
    print(f"row {row}: {n} candidates")
    for k in list(range(1, min(n, 300) + 1, max(1, n // 24))) + [n]:
        ref = ref_for(sel_row[:k])
        rel = float((out - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6))
        print(f"  vs first {k:3d} cands: rel={rel:.4f}")
    for label, slots in [
        ("dense 0..n-1", torch.arange(n, device=DEV)),
        ("sorted my sel", torch.sort(sel_row).values),
        ("raw sel order", sel_row),
    ]:
        ref = ref_for(slots)
        rel = float((out - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6))
        print(f"  vs {label:14s}: rel={rel:.4f}")


def live_slots(n_tokens: int, row: int):
    """Which candidate slots does the kernel actually read at `row`?

    Runs the launch twice per probe slot with that slot's record replaced
    by a distinct value; a nonzero output diff at `row` means the slot is
    live in the kernel's attention.
    """
    rows_pad = ((n_tokens + 1) // 2 + GROUP - 1) // GROUP * GROUP
    used_rows = 2 * rows_pad
    recs, _ = write_records(n_tokens)
    buf, arena = build_arena(recs, used_rows, n_tokens)
    q = torch.randn((n_tokens, HEADS, L), device=DEV, dtype=torch.bfloat16)
    sel = torch.full((n_tokens, used_rows), -1, dtype=torch.int32, device=DEV)
    tok = torch.arange(n_tokens, device=DEV)
    rank = (tok // 4) % 2
    local = (tok // 8) * 4 + tok % 4
    gath = (rank * rows_pad + local).to(torch.int64)
    for r in range(n_tokens):
        sel[r, : r + 1] = torch.sort(gath[: r + 1]).values
    topk_length = torch.arange(1, n_tokens + 1, device=DEV).to(torch.int32)
    output = torch.zeros((n_tokens, HEADS, L), dtype=torch.bfloat16, device=DEV)
    run_unified_prefill_mg(
        q=q,
        kv_cache=arena,
        topk_indices=sel,
        sm_scale=SM_SCALE,
        page_block_size=GROUP,
        topk_length=topk_length,
        output=output,
        stride_kv_block=arena.stride(0),
        compute_mode=int(ComputeMode.BF16),
        mg_n_hg=1,
        model_type=int(ModelType.KVARN_MLA),
        scale_format=int(ScaleFormat.KVARN_DUAL),
        latent_scale=1.0,
        fp8_rope=False,
        latent_scale_per_token=False,
    )
    torch.cuda.synchronize()
    base = output.clone()
    sel_row = sel[row, : topk_length[row]].to(torch.int64)
    live, dead = [], []
    for s in sel_row.tolist():
        buf2 = buf.clone()
        off = (s // GROUP) * TILE_STRIDE + (s % GROUP) * RECORD
        buf2[off : off + RECORD] = 0x30
        arena2 = buf2.as_strided(
            (buf2.numel() // TILE_STRIDE, GROUP, RECORD),
            (TILE_STRIDE, RECORD, 1),
        )
        out2 = torch.zeros_like(output)
        run_unified_prefill_mg(
            q=q,
            kv_cache=arena2,
            topk_indices=sel,
            sm_scale=SM_SCALE,
            page_block_size=GROUP,
            topk_length=topk_length,
            output=out2,
            stride_kv_block=arena2.stride(0),
            compute_mode=int(ComputeMode.BF16),
            mg_n_hg=1,
            model_type=int(ModelType.KVARN_MLA),
            scale_format=int(ScaleFormat.KVARN_DUAL),
            latent_scale=1.0,
            fp8_rope=False,
            latent_scale_per_token=False,
        )
        torch.cuda.synchronize()
        d = float((out2[row].float() - base[row].float()).abs().max())
        (live if d > 1e-3 else dead).append(s)
    print(f"row {row}: {len(sel_row)} candidates, live={len(live)} dead={len(dead)}")
    if dead:
        print(f"  dead slots: {dead[:40]}")
    return live, dead


def recover_weights(n_tokens: int, row: int):
    """LSQ-recover the kernel's effective softmax weights at `row`."""
    torch.manual_seed(3)
    rows_pad = ((n_tokens + 1) // 2 + GROUP - 1) // GROUP * GROUP
    used_rows = 2 * rows_pad
    recs, _ = write_records(n_tokens)
    buf, arena = build_arena(recs, used_rows, n_tokens)
    q = torch.randn((n_tokens, HEADS, L), device=DEV, dtype=torch.bfloat16)
    sel = torch.full((n_tokens, used_rows), -1, dtype=torch.int32, device=DEV)
    tok = torch.arange(n_tokens, device=DEV)
    rank = (tok // 4) % 2
    local = (tok // 8) * 4 + tok % 4
    gath = (rank * rows_pad + local).to(torch.int64)
    for r in range(n_tokens):
        sel[r, : r + 1] = torch.sort(gath[: r + 1]).values
    topk_length = torch.arange(1, n_tokens + 1, device=DEV).to(torch.int32)
    output = torch.zeros((n_tokens, HEADS, L), dtype=torch.bfloat16, device=DEV)
    run_unified_prefill_mg(
        q=q,
        kv_cache=arena,
        topk_indices=sel,
        sm_scale=SM_SCALE,
        page_block_size=GROUP,
        topk_length=topk_length,
        output=output,
        stride_kv_block=arena.stride(0),
        compute_mode=int(ComputeMode.BF16),
        mg_n_hg=1,
        model_type=int(ModelType.KVARN_MLA),
        scale_format=int(ScaleFormat.KVARN_DUAL),
        latent_scale=1.0,
        fp8_rope=False,
        latent_scale_per_token=False,
    )
    torch.cuda.synchronize()
    flat = np.frombuffer(bytes(buf.detach().cpu().untyped_storage()), dtype=np.uint8)
    sel_row = sel[row, : topk_length[row]].to(torch.int64)
    rec_off = (sel_row // GROUP) * TILE_STRIDE + (sel_row % GROUP) * RECORD
    recs_r = torch.from_numpy(
        np.stack([flat[o : o + RECORD] for o in rec_off.tolist()])
    )
    k_hat, v_hat = unpack_kvarn_mla_records(
        recs_r, kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
    )
    k_hat, v_hat = k_hat.float(), v_hat.float()
    n = sel_row.shape[0]
    out = output[row].float()  # [HEADS, L]
    # per head: out[h] = V @ p_h -> stack heads: [HEADS*L, n]
    A = torch.cat([v_hat.T] * HEADS, dim=0).cpu()
    b = out.reshape(-1).cpu()
    p_est, *_ = torch.linalg.lstsq(A, b).solution
    p_est = p_est.clamp_min(0)
    p_ref = (
        torch.softmax((q[row].float().cpu() @ k_hat.T.cpu()) * SM_SCALE, dim=-1)
        .mean(0)
        .cpu()
    )
    # per-head reference for comparison of shape
    print(f"row {row}: n={n}")
    print(f"  sum p_est={p_est.sum():.3f}  sum p_ref={p_ref.sum():.3f}")
    d = (p_est - p_ref).abs()
    top = d.argsort(descending=True)[:12]
    print("  largest weight diffs (idx, est, ref):")
    for i in top.tolist():
        print(f"    {i:4d}  {p_est[i]:+.4f}  {p_ref[i]:.4f}")
    # correlation between est and ref, and est vs shifted ref
    for shift in (-8, -4, -1, 0, 1, 4, 8):
        if shift < 0:
            a, b2 = p_est[-shift:], p_ref[:shift]
        elif shift > 0:
            a, b2 = p_est[:-shift], p_ref[shift:]
        else:
            a, b2 = p_est, p_ref
        c = float(torch.corrcoef(torch.stack([a, b2]))[0, 1])
        print(f"  corr(est, ref shift {shift:+d}) = {c:.4f}")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 2:
        (recover_weights if _sys.argv[1] == "w" else live_slots)(
            int(_sys.argv[2]), int(_sys.argv[3])
        )
    else:
        for n in (64, 258, 300, 511):
            rel, row = run_arena(n)
            verdict = "OK" if rel < 0.05 else "MISMATCH"
            print(f"S={n:4d} worst_row={row:4d} rel={rel:.4f}  {verdict}")
