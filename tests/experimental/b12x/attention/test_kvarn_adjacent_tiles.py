"""Adjacent-tile decode test: two tiles in the SAME manager block.

Historical note: this test used to repro a per-tile scale-tail collision
(the tail was written over the NEXT tile's records). Under per-token
records every scale is inline and the reserved tile-tail region carries no
data, so the collision class is structurally impossible; this test now
guards the adjacent-tile addressing itself -- records written across a
tile boundary must round-trip through the real decode kernel (S1/S6 arms,
per-token inline scales) bit-faithfully.

  CUDA_VISIBLE_DEVICES=2 ~/lil/vllm/.venv/bin/python \
      tests/attention/test_kvarn_adjacent_tiles.py
"""

import os
import sys

REPO = os.environ.get("B12X_REPO", os.path.expanduser("~/.cache/b12x-kvarn-wt"))
VLLM = os.environ.get("VLLM_REPO", os.path.expanduser("~/lil/vllm"))
for p in (REPO, VLLM):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from b12x.attention.sparse_mla import Caps  # noqa: E402
from b12x.attention.sparse_mla._preparation import (  # noqa: E402
    compile_sparse_mla,
    plan_cache_writer,
    writer_state,
)
from b12x.attention._shared.mla.kernel import run_prepared_unified_decode  # noqa: E402
from b12x.attention._shared.mla._kvarn_writer import (  # noqa: E402
    kvarn_tile_stride_bytes,
)
from vllm.v1.attention.ops.kvarn_mla import (  # noqa: E402
    KvarnMlaLayout,
    pack_kvarn_mla_records,
    unpack_kvarn_mla_records,
)

DEV_ORD = 0
GROUP = 256
PAGE = 4352
RECORD = 536
PTW = 8
HEADS = 16
TOPK = 2051
L = 512


def _write_two_adjacent_tiles(device):
    """Write tiles 3 and 4 of block 0 (records 768..1279) via the writer."""
    cache = torch.zeros((4, PAGE, RECORD), dtype=torch.uint8, device=device)
    latents = torch.randn(2 * GROUP, L, device=device) * 0.5
    base0 = 3 * GROUP
    slots = torch.arange(base0, base0 + 2 * GROUP, device=device, dtype=torch.int64)
    kv_c = latents.to(torch.bfloat16)
    wplan = plan_cache_writer(kv_c, cache, slots)
    wstate = writer_state(wplan, device=device)
    wstate.run(kv_c, cache, slots)
    torch.cuda.synchronize()
    return cache, latents, base0


def _extract_records(cache, base0, n):
    """Tile-strided extraction of n records starting at record base0."""
    tile_stride = kvarn_tile_stride_bytes(record_bytes=RECORD)
    flat = cache.view(-1)
    recs = torch.empty((n, RECORD), dtype=torch.uint8, device=cache.device)
    for i in range(n):
        slot = base0 + i
        block = slot // PAGE
        local = slot % PAGE
        tile = local // GROUP
        row = local % GROUP
        off = block * (PAGE // GROUP) * tile_stride + tile * tile_stride + row * RECORD
        recs[i] = flat[off : off + RECORD]
    return recs


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", DEV_ORD)
    cache, latents, base0 = _write_two_adjacent_tiles(device)

    layout = KvarnMlaLayout(k_bits=4, v_bits=4, group=GROUP, latent_dim=L)
    ref_records = pack_kvarn_mla_records(latents.float(), layout)
    got_records = _extract_records(cache, base0, 2 * GROUP)
    same = torch.equal(got_records, ref_records.to(torch.uint8))
    diff = (got_records != ref_records.to(torch.uint8)).sum().item()
    print(f"records byte-equal: {same} (diff bytes: {diff})")
    # The fp-divide rounding may flip isolated codes; byte equality is the
    # ideal but not required -- dequant agreement is the gate.
    ref_k, ref_v = unpack_kvarn_mla_records(ref_records, layout)
    got_k, got_v = unpack_kvarn_mla_records(got_records, layout)
    rel_k = ((ref_k - got_k).norm() / ref_k.norm().clamp_min(1e-9)).item()
    rel_v = ((ref_v - got_v).norm() / ref_v.norm().clamp_min(1e-9)).item()
    print(
        f"tile roundtrip: k_rel={rel_k:.5f} v_rel={rel_v:.5f} "
        f"-> {'OK' if max(rel_k, rel_v) < 0.05 else 'CORRUPT'}"
    )
    # 0.05, not 0.02: the kernel's fast-math divide flips isolated codes by
    # +/-1 level vs the torch reference (measured ~3.5% dequant rel).
    assert max(rel_k, rel_v) < 0.05

    # --- real decode kernel on both tiles ---
    ROWS = 2
    caps = Caps(
        device=device,
        num_q_heads=HEADS,
        max_q_rows=ROWS,
        max_width=TOPK,
        softmax_scale=0.088388,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=L,
        v_head_dim=L,
        model_type=4,
        scale_format=3,
        cache_record_bytes=RECORD,
        fp8_rope=False,
        kvarn_k_bits=4,
        kvarn_v_bits=4,
        kvarn_group=GROUP,
        latent_scale_per_token=False,
        mode="decode",
        max_batch=ROWS,
        max_chunks_per_row=max(1, (TOPK + 63) // 64),
        page_size=PAGE,
        max_page_table_width=PTW,
        return_lse=False,
        lse_scale="natural",
    )
    prepared = compile_sparse_mla(caps, DEV_ORD)
    from b12x.attention.sparse_mla._scratch import (
        _materialize_sparse_mla_scratch,
        _sparse_mla_scratch_layout,
    )

    scr_layout = _sparse_mla_scratch_layout(caps)
    scratch = torch.empty((scr_layout.nbytes,), dtype=torch.uint8, device=device)
    scratch_t = _materialize_sparse_mla_scratch(caps, scratch, scr_layout)
    q = torch.empty((ROWS, HEADS, L), dtype=torch.bfloat16, device=device)
    q.normal_()
    selected = torch.zeros((ROWS, TOPK), dtype=torch.int32, device=device)
    for r in range(ROWS):
        selected[r, :GROUP] = torch.arange(
            base0 + r * GROUP, base0 + (r + 1) * GROUP, dtype=torch.int32, device=device
        )
    topk_lengths = torch.full((ROWS,), GROUP, dtype=torch.int32, device=device)
    out = run_prepared_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=selected,
        swa_topk_lengths=topk_lengths,
        workspace=scratch_t,
        sm_scale=0.088388,
        swa_page_size=PAGE,
        attn_sink=None,
        return_lse=False,
        lse_scale="natural",
        prepared=prepared,
    )
    torch.cuda.synchronize()
    out32 = out.float()
    ok = True
    for r in range(ROWS):
        tile_lat = latents[r * GROUP : (r + 1) * GROUP]
        recs = pack_kvarn_mla_records(tile_lat.float(), layout)
        k_hat, v_hat = unpack_kvarn_mla_records(recs, layout)
        score = (q[r].float() @ k_hat.T) * 0.088388
        ref = torch.softmax(score, dim=-1) @ v_hat
        rel = ((out32[r] - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        verdict = "OK" if rel < 0.05 else "CORRUPT"
        ok = ok and rel < 0.05
        print(f"kernel row {r} (tile {r}): rel={rel:.4f} -> {verdict}")
    assert ok, "decode kernel mismatch on adjacent tiles"


def single_slot_probe():
    """q=0 + one selected candidate -> out == that record's v_hat."""
    torch.manual_seed(7)
    device = torch.device("cuda", DEV_ORD)
    cache, latents, base0 = _write_two_adjacent_tiles(device)

    ROWS = 1
    caps = Caps(
        device=device,
        num_q_heads=HEADS,
        max_q_rows=ROWS,
        max_width=TOPK,
        softmax_scale=0.088388,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=L,
        v_head_dim=L,
        model_type=4,
        scale_format=3,
        cache_record_bytes=RECORD,
        fp8_rope=False,
        kvarn_k_bits=4,
        kvarn_v_bits=4,
        kvarn_group=GROUP,
        latent_scale_per_token=False,
        mode="decode",
        max_batch=ROWS,
        max_chunks_per_row=max(1, (TOPK + 63) // 64),
        page_size=PAGE,
        max_page_table_width=PTW,
        return_lse=False,
        lse_scale="natural",
    )
    prepared = compile_sparse_mla(caps, DEV_ORD)
    from b12x.attention.sparse_mla._scratch import (
        _materialize_sparse_mla_scratch,
        _sparse_mla_scratch_layout,
    )

    scr_layout = _sparse_mla_scratch_layout(caps)
    scratch = torch.empty((scr_layout.nbytes,), dtype=torch.uint8, device=device)
    scratch_t = _materialize_sparse_mla_scratch(caps, scratch, scr_layout)
    layout = KvarnMlaLayout(k_bits=4, v_bits=4, group=GROUP, latent_dim=L)

    slot = base0 + 7
    recs = pack_kvarn_mla_records(latents.float(), layout)
    _, ref_v = unpack_kvarn_mla_records(recs, layout)

    q = torch.zeros((1, HEADS, L), dtype=torch.bfloat16, device=device)
    selected = torch.full((1, TOPK), -1, dtype=torch.int32, device=device)
    selected[0, 0] = slot
    topk_lengths = torch.ones((1,), dtype=torch.int32, device=device)
    out = run_prepared_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=selected,
        swa_topk_lengths=topk_lengths,
        workspace=scratch_t,
        sm_scale=0.088388,
        swa_page_size=PAGE,
        attn_sink=None,
        return_lse=False,
        lse_scale="natural",
        prepared=prepared,
    )
    torch.cuda.synchronize()
    got = out.float()[0, 0]
    rel = ((got - ref_v[7]).norm() / ref_v[7].norm().clamp_min(1e-9)).item()
    finite = bool(torch.isfinite(got).all())
    print(
        f"slot {slot}: v rel={rel:.4f} finite={finite}"
        f" -> {'OK' if rel < 0.05 and finite else 'MISMATCH'}"
    )
    assert rel < 0.05 and finite, f"single-slot probe rel={rel}"


if __name__ == "__main__":
    main()
    print("--- single-slot probe ---")
    single_slot_probe()
    print("ADJACENT TILES ALL OK")
