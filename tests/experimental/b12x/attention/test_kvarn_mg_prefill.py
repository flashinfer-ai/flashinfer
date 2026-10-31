"""MG prefill KVARN_DUAL arm A/B: writer -> MG extend vs the torch codec.

Writes random latents through the REAL writer kernel at multi-block /
multi-tile slots, runs the MG prefill (extend) kernel over a >256 selection
(multi-tile IO, one tile partial), and compares against a reference that
dequantizes the same cache bytes with ``unpack_kvarn_mla_records``.

  CUDA_VISIBLE_DEVICES=0 ~/lil/vllm/.venv/bin/python \
      tests/attention/test_kvarn_mg_prefill.py
"""

import os
import sys

REPO = os.environ.get("B12X_REPO", os.path.expanduser("~/.cache/b12x-kvarn-wt"))
VLLM = os.environ.get("VLLM_REPO", os.path.expanduser("~/lil/vllm"))
for p in (REPO, VLLM):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from b12x.attention.sparse_mla._preparation import (  # noqa: E402
    plan_cache_writer,
    writer_state,
)
from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg  # noqa: E402
from b12x.attention._shared.mla.traits import (  # noqa: E402
    ComputeMode,
    ModelType,
    ScaleFormat,
)
from vllm.v1.attention.ops.kvarn_mla import (  # noqa: E402
    kvarn_mla_layout_from_dtype,
    unpack_kvarn_mla_records,
)

DEV_ORD = 0
GROUP = 256
PAGE = 4352
RECORD = 536
HEADS = 16
TOPK_WIDTH = 2051
TOPK_LENGTH = 520  # 8 full tiles + a partial ninth tile (streamed tiles 1..9)
L = 512
SM_SCALE = 0.088388


def _records_at(cache: torch.Tensor, slots: torch.Tensor):
    """Decode K/V hats for slot ids: the kernel's linear row addressing
    (block*stride0 + local*stride1), matching the IO arm."""
    flat = np.frombuffer(
        bytes(cache.detach().cpu().contiguous().untyped_storage()),
        dtype=np.uint8,
    )
    stride0, stride1 = cache.stride(0), cache.stride(1)
    blk = slots // PAGE
    local = slots % PAGE
    rec_off = blk * stride0 + local * stride1
    recs = torch.from_numpy(
        np.stack(
            [flat[rec_off[i] : rec_off[i] + RECORD] for i in range(slots.shape[0])]
        )
    )
    k_hat, v_hat = unpack_kvarn_mla_records(
        recs, kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
    )
    return k_hat.to(cache.device), v_hat.to(cache.device)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", DEV_ORD)
    cache_full = torch.zeros((34, PAGE, 602), dtype=torch.uint8, device=device)
    cache = cache_full[..., :RECORD]

    # Two tiles in different blocks: block 1 tile 1, block 3 tile 5.
    tile_bases = [1 * PAGE + 1 * GROUP, 3 * PAGE + 5 * GROUP]
    n_tokens = GROUP * len(tile_bases)
    latents = torch.randn(n_tokens, L, device=device, dtype=torch.float32) * 0.5
    slots = torch.cat(
        [
            torch.arange(base, base + GROUP, device=device, dtype=torch.int64)
            for base in tile_bases
        ]
    )
    wplan = plan_cache_writer(latents.to(torch.bfloat16), cache, slots)
    writer_state(wplan, device=device).run(latents.to(torch.bfloat16), cache, slots)
    torch.cuda.synchronize()
    assert int((cache != 0).sum()) > 0, "writer wrote nothing"

    # Selection per row: tile A (256), tile B (256), then tile B's first 8
    # (partial tile), then -1 padding to the container width.
    sel = torch.full((2, TOPK_WIDTH), -1, dtype=torch.int32, device=device)
    for r, base in enumerate(tile_bases):
        other = tile_bases[1 - r]
        sel[r, 0:GROUP] = torch.arange(
            base, base + GROUP, dtype=torch.int32, device=device
        )
        sel[r, GROUP : 2 * GROUP] = torch.arange(
            other, other + GROUP, dtype=torch.int32, device=device
        )
        sel[r, 2 * GROUP : TOPK_LENGTH] = torch.arange(
            other, other + (TOPK_LENGTH - 2 * GROUP), dtype=torch.int32, device=device
        )
    topk_length = torch.full((2,), TOPK_LENGTH, dtype=torch.int32, device=device)

    q = torch.randn((2, HEADS, L), device=device, dtype=torch.bfloat16)
    output = torch.zeros((2, HEADS, L), dtype=torch.bfloat16, device=device)
    run_unified_prefill_mg(
        q=q,
        kv_cache=cache,
        topk_indices=sel,
        sm_scale=SM_SCALE,
        page_block_size=PAGE,
        topk_length=topk_length,
        output=output,
        stride_kv_block=cache.stride(0),
        compute_mode=int(ComputeMode.BF16),
        mg_n_hg=1,
        model_type=int(ModelType.KVARN_MLA),
        scale_format=int(ScaleFormat.KVARN_DUAL),
        latent_scale=1.0,
        fp8_rope=False,
        latent_scale_per_token=False,
    )
    torch.cuda.synchronize()

    max_rel = 0.0
    for r in range(2):
        k_hat, v_hat = _records_at(cache_full, sel[r, :TOPK_LENGTH].to(torch.int64))
        score = (q[r].float() @ k_hat.T) * SM_SCALE
        prob = torch.softmax(score, dim=-1)
        ref = prob @ v_hat
        err = (output[r].float() - ref).abs()
        rel = float(err.mean() / ref.abs().mean().clamp_min(1e-6))
        max_rel = max(max_rel, rel)
        print(f"row {r}: mean|ref|={ref.abs().mean():.4f} rel={rel:.4f}")
    print("MG KVARN PREFILL:", "OK" if max_rel < 0.05 else "MISMATCH")


if __name__ == "__main__":
    main()
