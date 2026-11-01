"""MG extend determinism + correctness regression gate.

Guards the IO->math handoff race fixed in ``io_mg.py`` (the KVarN scalar
gather published tile completion via a single leader arrive without a
rendezvous between the other IO lanes' stores and that arrive, letting
math read not-yet-stored record tails; nondeterministic corruption that
grew with candidate count).

Fails (bit-exactness) if any of repeated identical MG KVARN_DUAL launches
over a multi-chunk gathered-arena layout diverges, and fails if the
output drifts from the torch codec reference beyond codec noise.

  CUDA_VISIBLE_DEVICES=<free gpu> ~/lil/vllm/.venv/bin/python \
      tests/attention/test_mg_extend_determinism.py
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

from b12x.attention.sparse_mla import ModelType, plan  # noqa: E402
from vllm.v1.attention.ops.kvarn_mla import (  # noqa: E402
    kvarn_mla_layout_from_dtype,
    unpack_kvarn_mla_records,
)

DEV = torch.device("cuda", 0)
GROUP = 256
RECORD = 536
TILE_STRIDE = GROUP * RECORD + 6400  # gathered-arena tile stride (143616)
HEADS = 64
WIDTH = 512
REPS = 4


def _write_records(n_tokens: int):
    from b12x.attention.sparse_mla._preparation import (
        plan_cache_writer,
        writer_state,
    )

    pool = torch.zeros((4, 4352, RECORD), dtype=torch.uint8, device=DEV)
    latents = torch.randn(n_tokens, 512, device=DEV, dtype=torch.float32) * 0.5
    slots = torch.arange(n_tokens, device=DEV, dtype=torch.int64)
    wplan = plan_cache_writer(latents.to(torch.bfloat16), pool, slots)
    writer_state(wplan, device=DEV).run(latents.to(torch.bfloat16), pool, slots)
    torch.cuda.synchronize()
    return pool.reshape(-1, RECORD)[:n_tokens].contiguous()


def _arena(recs: torch.Tensor, used_rows: int, n_tokens: int):
    tiles = (used_rows + GROUP - 1) // GROUP
    buf = torch.zeros(tiles * TILE_STRIDE, dtype=torch.uint8, device=DEV)
    arena = buf.as_strided((tiles, GROUP, RECORD), (TILE_STRIDE, RECORD, 1))
    tok = torch.arange(n_tokens, device=DEV)
    rank = (tok // 4) % 2
    local = (tok // 8) * 4 + tok % 4
    gath = (rank * (used_rows // 2) + local).to(torch.int64)
    flat = torch.zeros(used_rows * RECORD, dtype=torch.uint8, device=DEV)
    idx = gath[:, None] * RECORD + torch.arange(RECORD, device=DEV)
    flat[idx.reshape(-1)] = recs.reshape(-1)
    arena.view(tiles, GROUP * RECORD).copy_(flat.view(tiles, GROUP * RECORD))
    return buf, arena, gath


def _launch(plan, scratch, q, arena, sel, lengths):
    from b12x.attention.sparse_mla import bind, run

    b = bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=arena,
        selected_indices=sel,
        cache_lengths=lengths.clone(),
        selected_lengths=lengths,
    )
    out = run(b)
    return (out[0] if isinstance(out, tuple) else out).clone()


def test_mg_extend_bit_exact_and_correct(n_tokens=258, used_rows=512):
    torch.manual_seed(3)
    recs = _write_records(n_tokens)
    buf, arena, gath = _arena(recs, used_rows, n_tokens)
    g = torch.Generator(device="cpu").manual_seed(3)
    q = torch.randn((n_tokens, HEADS, 512), generator=g).to(DEV, torch.bfloat16)
    sel = torch.full((n_tokens, WIDTH), -1, dtype=torch.int32, device=DEV)
    for r in range(n_tokens):
        sel[r, : r + 1] = torch.sort(gath[: r + 1]).values
    lengths = torch.arange(1, n_tokens + 1, device=DEV).to(torch.int32)

    from b12x.attention.sparse_mla import Caps

    caps = Caps(
        device=DEV,
        num_q_heads=HEADS,
        max_q_rows=n_tokens,
        max_width=WIDTH,
        softmax_scale=0.088388,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=512,
        v_head_dim=512,
        mode="extend",
        max_batch=n_tokens,
        max_chunks_per_row=(WIDTH + 63) // 64,
        page_size=GROUP,
        return_lse=False,
        lse_scale="natural",
        model_type=int(ModelType.KVARN_MLA),
        scale_format=3,
        kvarn_k_bits=4,
        kvarn_v_bits=4,
        kvarn_group=256,
        cache_record_bytes=536,
    )
    p = plan(caps)
    from b12x.attention.sparse_mla._scratch import plan_sparse_mla_scratch

    sp = plan_sparse_mla_scratch(caps)
    spec = sp.scratch_specs()[0]
    scratch = torch.zeros(spec.shape, dtype=spec.dtype, device=DEV)

    outs = [_launch(p, scratch, q, arena, sel, lengths) for _ in range(REPS)]
    torch.cuda.synchronize()
    for i in range(1, REPS):
        d = (outs[i].float() - outs[0].float()).abs().max().item()
        assert d == 0.0, f"nondeterministic MG extend: rep {i} vs 0 diff {d}"

    flat = np.frombuffer(bytes(buf.detach().cpu().untyped_storage()), dtype=np.uint8)
    worst = 0.0
    for r in range(n_tokens):
        sr = sel[r, : lengths[r]].to(torch.int64)
        off = (sr // GROUP) * TILE_STRIDE + (sr % GROUP) * RECORD
        rr = torch.from_numpy(np.stack([flat[o : o + RECORD] for o in off.tolist()]))
        k_hat, v_hat = unpack_kvarn_mla_records(
            rr, kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
        )
        ref = torch.softmax((q[r].float() @ k_hat.T.to(DEV)) * 0.088388, -1) @ v_hat.to(
            DEV
        )
        rel = float(
            (outs[0][r].float() - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
        )
        worst = max(worst, rel)
    assert worst < 0.01, f"MG extend drifted from codec reference: rel {worst:.4f}"
    return worst


def main() -> int:
    worst = test_mg_extend_bit_exact_and_correct()
    print(f"MG extend: bit-exact over {REPS} reps, worst rel {worst:.4f}  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
