"""Standalone probe: b12x tail-decode arm (GLM fp8 math over a 536B pool).

Writes stock GLM fp8 records (512 E4M3 latent + 4 inline fp32 group scales,
528B content) into a 536B-stride kvarn pool via the real GLM_NEXT fp8 writer
kernel, runs the REAL tail-decode arm (ModelType.KVARN_TAIL_FP8: GLM fp8 math,
scalar staging of the 536B row stride into the 528B smem row), and compares
the per-head latent output against a torch reference that dequants the same
record bytes and computes softmax(QK)V in the H-rotated latent frame (Q used
raw; no Hadamard -- the kvarn folds live in vLLM weights, not in this probe).

  CUDA_VISIBLE_DEVICES=4 B12X_COMPILE_DISK_CACHE=0 \
      ~/lil/vllm/.venv/bin/python tests/attention/test_kvarn_tail_fp8_probe.py
"""

import os
import sys

REPO = os.environ.get("B12X_REPO", os.path.expanduser("~/.cache/b12x-kvarn-wt"))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from b12x.attention.sparse_mla import Caps  # noqa: E402
from b12x.attention.sparse_mla._preparation import compile_sparse_mla  # noqa: E402
from b12x.attention._shared.mla.kernel import run_prepared_unified_decode  # noqa: E402
from b12x.attention.sparse_mla._scratch import (  # noqa: E402
    _materialize_sparse_mla_scratch,
    _sparse_mla_scratch_layout,
)
from b12x.attention._shared.mla.kv_cache import (  # noqa: E402
    _compile_glm_next_mla_cache_writer,
    _glm_next_cache_writer_launch,
)
from b12x.attention._shared.mla.prefill import run_unified_prefill  # noqa: E402
from b12x._lib.compiler import run_compiled  # noqa: E402

DEV_ORD = 0  # first device visible under CUDA_VISIBLE_DEVICES
L = 512
HEADS = 16
PAGE = 256  # tokens per paged block (probe geometry; 536B record stride)
RECORD = 536  # kvarn pool row stride (528B fp8 content + 8B pad)
SM_SCALE = 256**-0.5
POOL_SLOTS = PAGE  # one page of slots


def _write_fp8_records(latent_bf16: torch.Tensor, cache: torch.Tensor) -> None:
    """Populate the 536B pool with stock GLM fp8 records via the real writer.

    The writer addresses records by kv_cache.stride(1) (=536 here), so the
    528B semantic record lands at [0,528) of each 536B row and [528,536) is
    left as pad -- exactly the tail-pool layout the decode arm expects.
    """
    slots = torch.arange(latent_bf16.shape[0], dtype=torch.int64, device=cache.device)
    compiled = _compile_glm_next_mla_cache_writer(latent_bf16, cache, slots)
    _, args, _ = _glm_next_cache_writer_launch(latent_bf16, cache, slots)
    run_compiled(compiled, args)


def _dequant_fp8_records(rec: torch.Tensor) -> torch.Tensor:
    """Dequant [T, 536] uint8 pool rows -> [T, 512] fp32 latent.

    Record layout (kv_cache.py writer doc): [0,512) E4M3 latent (four 128-dim
    groups), [512,528) 4 x fp32 group scales (group amax / 448.0).
    """
    t = rec.shape[0]
    lat = rec[:, :512].contiguous().view(torch.float8_e4m3fn).float().reshape(t, L)
    scales = rec[:, 512:528].contiguous().view(torch.float32).reshape(t, 4)
    return lat * scales.repeat_interleave(128, dim=1)


def _quantize_q_s0(q: torch.Tensor) -> torch.Tensor:
    """Mirror the kernel's S0 Q quantization (decode_math.py S0 docstring).

    Per 128-dim quant tile: scale = pow2ceil(max(amax, 1e-4) / 448),
    q_fp8 = E4M3(q / scale). Returns the dequantized fp32 Q the MMA
    effectively consumes, so the reference isolates KERNEL correctness
    from the format's Q quantization noise.
    """
    q32 = q.float()
    tiles = q32.reshape(*q32.shape[:-1], 4, 128)
    amax = tiles.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q_fp8 = (tiles / scale).to(torch.float8_e4m3fn).float()
    return (q_fp8 * scale).reshape(q32.shape)


def _reference(
    q0: torch.Tensor, k_hat: torch.Tensor, v_hat: torch.Tensor
) -> torch.Tensor:
    score = (q0.float() @ k_hat.transpose(0, 1)) * SM_SCALE
    prob = torch.softmax(score, dim=-1)
    return prob @ v_hat


def _run_tail_decode(
    cache: torch.Tensor,
    q: torch.Tensor,
    selected: torch.Tensor,
    topk_lengths: torch.Tensor,
    prepared,
    scratch_t,
    cache_seqlens: torch.Tensor | None = None,
):
    return run_prepared_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=selected,
        swa_topk_lengths=topk_lengths,
        workspace=scratch_t,
        sm_scale=SM_SCALE,
        swa_page_size=PAGE,
        attn_sink=None,
        return_lse=False,
        lse_scale="natural",
        prepared=prepared,
    )


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", DEV_ORD)
    print(f"device={device} ({torch.cuda.get_device_name(DEV_ORD)})")

    # One page of 536B pool slots, populated with stock GLM fp8 records.
    cache = torch.zeros((1, PAGE, RECORD), dtype=torch.uint8, device=device)
    latent = (torch.randn((POOL_SLOTS, L), device=device) / 4).to(torch.bfloat16)
    _write_fp8_records(latent, cache)
    torch.cuda.synchronize()
    nonzero = int((cache[:, :, :528] != 0).sum())
    print(f"writer populated {nonzero} nonzero content bytes (of {POOL_SLOTS * 528})")

    rows = 1
    max_width = POOL_SLOTS
    caps = Caps(
        device=device,
        num_q_heads=HEADS,
        max_q_rows=rows,
        max_width=max_width,
        softmax_scale=SM_SCALE,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=L,
        v_head_dim=L,
        model_type=5,  # ModelType.KVARN_TAIL_FP8
        scale_format=1,  # ScaleFormat.ARBITRARY_FP32 (GLM fp8 math)
        kvarn_k_bits=4,  # k4v4 pool: slot width 536 = RECORD below
        kvarn_v_bits=4,
        kvarn_group=256,
        cache_record_bytes=RECORD,
        fp8_rope=False,
        latent_scale_per_token=False,
        mode="decode",
        max_batch=rows,
        max_chunks_per_row=max(1, (max_width + 63) // 64),
        page_size=PAGE,
        max_page_table_width=1,
        return_lse=False,
        lse_scale="natural",
    )
    prepared = compile_sparse_mla(caps, DEV_ORD)
    scr_layout = _sparse_mla_scratch_layout(caps)
    scratch = torch.empty((scr_layout.nbytes,), dtype=torch.uint8, device=device)
    scratch_t = _materialize_sparse_mla_scratch(caps, scratch, scr_layout)

    q = torch.empty((rows, HEADS, L), dtype=torch.bfloat16, device=device)
    q.normal_()
    selected = torch.arange(max_width, dtype=torch.int32, device=device).unsqueeze(0)

    k_hat_all = _dequant_fp8_records(cache.reshape(PAGE, RECORD))

    worst_rel = 0.0
    worst_abs = 0.0
    for tail_len in (64, 129, 256):
        topk_lengths = torch.full((rows,), tail_len, dtype=torch.int32, device=device)
        out = _run_tail_decode(cache, q, selected, topk_lengths, prepared, scratch_t)
        torch.cuda.synchronize()
        out32 = out[0].float()  # [HEADS, L]

        k_hat = k_hat_all[:tail_len]
        v_hat = k_hat  # MLA: V == K latent (same record bytes)
        ref = _reference(_quantize_q_s0(q[0]), k_hat, v_hat)  # [HEADS, L]

        err = (out32 - ref).abs()
        rel = err.mean() / ref.abs().mean().clamp_min(1e-6)
        max_abs = float(err.max())
        worst_rel = max(worst_rel, float(rel))
        worst_abs = max(worst_abs, max_abs)
        print(
            f"tail_len={tail_len:3d}: mean|ref|={ref.abs().mean():.4f} "
            f"mean|err|={err.mean():.6f} max|err|={max_abs:.6f} "
            f"rel={float(rel):.5f} ({float(rel) * 100:.3f}%)"
        )

    print(
        f"worst: max|err|={worst_abs:.6f} rel={worst_rel:.5f} ({worst_rel * 100:.3f}%)"
    )
    ok = worst_rel < 0.01 and worst_abs < 0.05
    print("TAIL FP8 PROBE:", "OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


def extend_probe() -> None:
    """EXTEND mode: fp8 arm via run_unified_prefill vs torch reference."""
    torch.manual_seed(11)
    device = torch.device("cuda", DEV_ORD)
    cache = torch.zeros((1, PAGE, RECORD), dtype=torch.uint8, device=device)
    latent = (torch.randn((POOL_SLOTS, L), device=device) / 4).to(torch.bfloat16)
    _write_fp8_records(latent, cache)
    torch.cuda.synchronize()
    k_hat_all = _dequant_fp8_records(cache.reshape(PAGE, RECORD))
    v_hat_all = k_hat_all  # latent-only attention: v == k latent

    rows = 8
    q = torch.empty((rows, HEADS, L), dtype=torch.bfloat16, device=device)
    q.normal_()
    # causal windows: row r attends records [0, s_r), s_r = 64*(r+1)
    s_per_row = torch.tensor(
        [64 * (r + 1) for r in range(rows)], dtype=torch.int32, device=device
    )
    selected = (
        torch.arange(POOL_SLOTS, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(rows, POOL_SLOTS)
        .contiguous()
    )
    out = run_unified_prefill(
        q=q,
        kv_cache=cache,
        topk_indices=selected,
        sm_scale=SM_SCALE,
        page_block_size=PAGE,
        topk_length=s_per_row,
        scale_format=1,
        model_type=5,
        kvarn_k_bits=4,
        kvarn_v_bits=4,
        kvarn_group=256,
        fp8_rope=False,
        latent_scale_per_token=False,
    )
    torch.cuda.synchronize()
    out = out[0] if isinstance(out, tuple) else out
    worst = 0.0
    for r in range(rows):
        s = int(s_per_row[r])
        ref = _reference(q[r], k_hat_all[:s], v_hat_all[:s])
        err = (out[r].float() - ref).abs().max().item()
        worst = max(worst, err)
    print(
        f"EXTEND FP8 PROBE: worst|err|={worst:.6f} -> "
        f"{'OK' if worst < 5e-2 else 'MISMATCH'}"
    )


if __name__ == "__main__":
    main()
    extend_probe()
