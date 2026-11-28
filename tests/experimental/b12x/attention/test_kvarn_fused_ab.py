"""Fused-kernel A/B: b12x kvarn writer + decode vs the torch codec.

Writes random rotated latents at multi-block/multi-tile slots through the
REAL writer kernel, runs the REAL decode kernel, and compares its per-head
latent output (P @ l_hat) against a reference computed with
unpack_kvarn_mla_records on the same cache bytes.

  CUDA_VISIBLE_DEVICES=2 ~/lil/vllm/.venv/bin/python \
      tests/attention/test_kvarn_fused_ab.py
"""

import math
import os
import sys

import pytest

REPO = os.environ.get("B12X_REPO", os.path.expanduser("~/.cache/b12x-kvarn-wt"))
VLLM = os.environ.get("VLLM_REPO", os.path.expanduser("~/lil/vllm"))
for p in (REPO, VLLM):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("kvarn fused A/B test needs CUDA", allow_module_level=True)

from b12x.attention.sparse_mla import Caps  # noqa: E402
from b12x.attention.sparse_mla._preparation import (  # noqa: E402
    compile_sparse_mla,
    plan_cache_writer,
    writer_state,
)
from b12x.attention._shared.mla.kernel import run_prepared_unified_decode  # noqa: E402

# Fork-only reference codec: skip collection cleanly when unavailable.
kvarn_mla = pytest.importorskip("vllm.v1.attention.ops.kvarn_mla")
unpack_kvarn_mla_records = kvarn_mla.unpack_kvarn_mla_records
kvarn_mla_layout_from_dtype = kvarn_mla.kvarn_mla_layout_from_dtype

DEV_ORD = 0
GROUP = 256
PAGE = 4352
RECORD = 536
PTW = 8  # blocks
HEADS = 16
TOPK = 2051
L = 512


def test_main() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda", DEV_ORD)
    layout = (34, PAGE, RECORD)
    cache = torch.zeros(layout, dtype=torch.uint8, device=device)

    # Two tiles in different blocks: block 1 tile 1, block 3 tile 5.
    tile_specs = [(1 * PAGE + 1 * GROUP, 1), (3 * PAGE + 5 * GROUP, 1)]
    n_tokens = GROUP * len(tile_specs)
    latents = torch.randn(n_tokens, L, device=device, dtype=torch.float32) * 0.5
    slots = torch.cat(
        [
            torch.arange(base, base + GROUP, device=device, dtype=torch.int64)
            for base, _ in tile_specs
        ]
    )

    kv_c_bf16 = latents.to(torch.bfloat16)
    wplan = plan_cache_writer(kv_c_bf16, cache, slots)
    wstate = writer_state(wplan, device=device)
    wstate.run(kv_c_bf16, cache, slots)
    torch.cuda.synchronize()
    print("writer ran; non-zero cache bytes:", int((cache != 0).sum()))

    # --- decode kernel on the same cache ---
    ROWS = 4
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
    # Row r attends to the r-th tile's slots (cyclic over the two tiles).
    for r in range(ROWS):
        base, _ = tile_specs[r % len(tile_specs)]
        selected[r, :GROUP] = torch.arange(
            base, base + GROUP, dtype=torch.int32, device=device
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
    # out: [ROWS, HEADS, L] = P @ l_hat per head (pre-W_UV latents).
    out32 = out.float()

    # --- reference from the codec ---
    max_err = 0.0
    for r in range(ROWS):
        base, _ = tile_specs[r % len(tile_specs)]
        # Dense contiguous cache: block stride = cache.stride(0), rows are
        # packed at RECORD within the page (the writer and decode io use the
        # same dense addressing, not the gathered-arena tile stride).
        blk = base // PAGE
        local = base % PAGE
        flat = cache.view(-1)
        rec_off = blk * cache.stride(0) + local * RECORD
        recs = torch.stack(
            [
                flat[rec_off + i * RECORD : rec_off + (i + 1) * RECORD]
                for i in range(GROUP)
            ]
        )
        k_hat, v_hat = unpack_kvarn_mla_records(recs, _caps_layout())
        # reference: P = softmax(q_abs . k_hat * scale); out = P @ v_hat
        qabs = q[r].float()  # [HEADS, L]
        score = (qabs @ k_hat.T) * 0.088388
        Pr = torch.softmax(score, dim=-1)  # [HEADS, GROUP]
        ref = Pr @ v_hat  # [HEADS, L]
        err = (out32[r] - ref).abs()
        rel = err.mean() / ref.abs().mean().clamp_min(1e-6)
        max_err = max(max_err, float(rel))
        print(
            f"row {r}: mean|ref|={ref.abs().mean():.4f} "
            f"mean|err|={err.mean():.6f} rel={float(rel):.4f}"
        )
    assert max_err < 0.05, f"FUSED AB mismatch: max_err {max_err:.4f}"


def _caps_layout():
    return kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
