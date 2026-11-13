"""Prepared-plan replica of the server's ckv_extend MG launch.

Declares Caps exactly as B12xMLASparseImpl._declare_plan does for
mode="ckv_extend" (num_q_heads=64, page_size=group=256, KVARN_DUAL,
max_chunks_per_row=ceil(topk/64)), publishes scratch, binds, and runs.
Checks (a) determinism across identical launches, (b) correctness vs the
torch codec reference over the same arena bytes.
"""

import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/js/.cache/b12x-kvarn-wt")
sys.path.insert(0, "/home/js/lil/vllm")

_spec = importlib.util.spec_from_file_location(
    "pg", "/home/js/.cache/b12x-kvarn-wt/tests/attention/probe_gather_arena.py"
)
pg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pg)

import b12x.attention.sparse_mla as m  # noqa: E402

DEV = pg.DEV
N = 258
ROWS_PAD = 256
USED = 512
WIDTH = 512
HEADS = 64
QDIM = 512

torch.manual_seed(3)
recs, _ = pg.write_records(N)
buf, arena = pg.build_arena(recs, USED, N)
g = torch.Generator(device="cpu").manual_seed(3)
q = torch.randn((N, HEADS, QDIM), generator=g).to(DEV, torch.bfloat16)

tok = torch.arange(N, device=DEV)
rank = (tok // 4) % 2
local = (tok // 8) * 4 + tok % 4
gath = (rank * ROWS_PAD + local).to(torch.int64)
sel = torch.full((N, WIDTH), -1, dtype=torch.int32, device=DEV)
for r in range(N):
    sel[r, : r + 1] = torch.sort(gath[: r + 1]).values
active = torch.arange(1, N + 1, device=DEV, dtype=torch.int32)
cache_lens = active.clone()

caps_kwargs = dict(
    device=DEV,
    num_q_heads=HEADS,
    max_q_rows=N,
    max_width=WIDTH,
    softmax_scale=pg.SM_SCALE,
    dtype=torch.bfloat16,
    kv_dtype=torch.uint8,
    head_dim=QDIM,
    v_head_dim=512,
    mode="extend",
    max_batch=N,
    max_chunks_per_row=(WIDTH + 63) // 64,
    page_size=pg.GROUP,
    return_lse=False,
    lse_scale="natural",
    model_type=int(m.ModelType.KVARN_MLA),
    scale_format=3,
    kvarn_k_bits=4,
    kvarn_v_bits=4,
    kvarn_group=256,
    cache_record_bytes=536,
)
caps = m.Caps(**caps_kwargs)
plan = m.plan(caps)
try:
    scratch_plan = m.plan_sparse_mla_scratch(caps)
except Exception:
    from b12x.attention.sparse_mla._scratch import plan_sparse_mla_scratch

    scratch_plan = plan_sparse_mla_scratch(caps)
spec = scratch_plan.scratch_specs()[0]
scratch = torch.zeros(spec.shape, dtype=spec.dtype, device=DEV)

outs = []
for rep in range(4):
    binding = m.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=arena,
        selected_indices=sel,
        cache_lengths=cache_lens,
        selected_lengths=active,
    )
    result = m.run(binding)
    out = result[0] if isinstance(result, tuple) else result
    outs.append(out.clone())
    torch.cuda.synchronize()

for i in range(1, 4):
    d = (outs[i].float() - outs[0].float()).abs().max().item()
    print(f"rep {i} vs 0: max abs diff = {d:.6f}")

flat = np.frombuffer(bytes(buf.detach().cpu().untyped_storage()), dtype=np.uint8)
rels = []
for r in range(N):
    sr = sel[r, : active[r]].to(torch.int64)
    off = (sr // pg.GROUP) * pg.TILE_STRIDE + (sr % pg.GROUP) * pg.RECORD
    rr = torch.from_numpy(np.stack([flat[o : o + pg.RECORD] for o in off.tolist()]))
    k_hat, v_hat = pg.unpack_kvarn_mla_records(
        rr, pg.kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")
    )
    ref = torch.softmax((q[r].float() @ k_hat.T.to(DEV)) * pg.SM_SCALE, -1) @ v_hat.to(
        DEV
    )
    rels.append(
        float(
            (outs[0][r].float() - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
        )
    )
rels = np.array(rels)
print("rel: min %.4f med %.4f max %.4f" % (rels.min(), np.median(rels), rels.max()))
print("worst rows:", np.argsort(rels)[-5:].tolist()[::-1])
