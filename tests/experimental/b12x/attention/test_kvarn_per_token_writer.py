"""Per-token KVarN writer kernel vs vllm reference codec (offline check).

Run: CUDA_VISIBLE_DEVICES=2 ~/.cache/b12x-kvarn-wt/../lil/vllm/.venv/bin/python \
    tests/attention/test_kvarn_per_token_writer.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VLLM = os.environ.get("VLLM_REPO", os.path.expanduser("~/lil/vllm"))
for p in (REPO, VLLM):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from b12x.attention._shared.mla._kvarn_writer import (  # noqa: E402
    concat_and_cache_glm_next_mla_kvarn,
    dequant_kvarn_mla_records,
    kvarn_record_bytes,
    kvarn_tile_stride_bytes,
)
from vllm.v1.attention.ops.kvarn_mla import (  # noqa: E402
    hadamard_matrix,
    kvarn_mla_layout_from_dtype,
    pack_kvarn_mla_records,
    unpack_kvarn_mla_records,
)

DEVICE = "cuda"
GROUP = 256
LATENT = 512


def _rotate(latent, device):
    h = hadamard_matrix(LATENT, device)
    return (latent @ h.T).to(torch.bfloat16)


def _run_variant(k_bits, v_bits, seed=0):
    torch.manual_seed(seed)
    layout = kvarn_mla_layout_from_dtype(f"kvarn_mla_k{k_bits}v{v_bits}_g256")
    record_bytes = kvarn_record_bytes(k_bits, v_bits)
    assert record_bytes == layout.record_bytes

    # Two tiles per manager page -> exercises the tile-strided addressing.
    tiles_per_page = 2
    page_block_size = GROUP * tiles_per_page
    tile_stride = kvarn_tile_stride_bytes(record_bytes=record_bytes)
    num_pages = 3
    block_stride = tile_stride * tiles_per_page
    capacity = num_pages * page_block_size
    cache = torch.zeros(num_pages * block_stride, dtype=torch.uint8, device=DEVICE)
    kv_cache = cache.as_strided(
        (num_pages, page_block_size, record_bytes), (block_stride, record_bytes, 1)
    )

    # Tokens scattered across tiles and pages; a couple of negative slots.
    tokens = 700
    slots = torch.randperm(capacity, device=DEVICE)[:tokens].to(torch.int64)
    slots[:5] = -1  # padded slots must be skipped
    latent = torch.randn(tokens, LATENT, device=DEVICE, dtype=torch.float32) * 0.5
    l_rot = _rotate(latent, DEVICE)

    concat_and_cache_glm_next_mla_kvarn(
        l_rot, kv_cache, slots, k_bits=k_bits, v_bits=v_bits, group=GROUP
    )
    torch.cuda.synchronize()

    valid = slots >= 0
    s = slots.clamp_min(0).long()
    offs = (
        (s // page_block_size) * block_stride
        + ((s % page_block_size) // GROUP) * tile_stride
        + (s % GROUP) * record_bytes
    )
    rec_bytes = cache[offs.unsqueeze(1) + torch.arange(record_bytes, device=DEVICE)]

    k_hat, v_hat = dequant_kvarn_mla_records(rec_bytes, k_bits, v_bits)
    ref_records = pack_kvarn_mla_records(l_rot.to(torch.float32), layout)
    ref_k, ref_v = unpack_kvarn_mla_records(ref_records, layout)

    for name, got, want in (("k", k_hat, ref_k), ("v", v_hat, ref_v)):
        err = (got[valid] - want[valid]).float()
        snr = 10 * torch.log10((want[valid].float() ** 2).mean() / (err**2).mean())
        # Kernel and reference quantizers agree to fp-division rounding (the
        # DSL emits fast-math divide; rare codes round +/-1 level).
        assert snr > 30, (
            f"{k_bits}v{v_bits} {name}: kernel-vs-ref SNR {snr:.1f} dB"
        )  # v2: 3 levels
        print(
            f"k{k_bits}v{v_bits} {name}: kernel-vs-ref SNR={snr:.1f} dB "
            f"max|diff|={err.abs().max():.5f} mean|ref|={want[valid].abs().mean():.4f}"
        )

    # Scale footer spot check: inline fp16 scale/zp must match the reference
    # record bytes at the new offsets.
    footer_a = rec_bytes[valid, k_bits * 64 + v_bits * 64 :][:, :8]
    footer_b = ref_records[valid, k_bits * 64 + v_bits * 64 :][:, :8]
    assert torch.equal(footer_a.view(torch.uint8), footer_b.view(torch.uint8)), (
        "inline scale footer mismatch"
    )
    # Padded slots: records untouched (zeros).
    assert int(rec_bytes[~valid].abs().sum()) == 0
    print(f"k{k_bits}v{v_bits}: OK")


if __name__ == "__main__":
    for kb, vb in ((4, 4), (5, 5), (5, 4), (4, 2)):
        _run_variant(kb, vb)
    print("ALL OK")
