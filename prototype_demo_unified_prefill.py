"""Demo: unified paged-prefill prototype — run me and read the output.

    PYTHONPATH=<this worktree> python prototype_demo_unified_prefill.py

What it shows, in order:
 1. resolve() at "engine init" — static, tensor-free, with exclusion reasons
 2. one plan()/run() with the canonical metadata every engine already owns
 3. the same call routed through every runnable backend — same results,
    same LSE contract (base-2, packed), no per-backend metadata code
 4. decode-shaped batch (uniform q_len=1) through the SAME api
 5. a broken call — loud, actionable error instead of plausible garbage
"""

import torch

from flashinfer.attention.unified import UnifiedPagedPrefill, resolve_paged_prefill

torch.manual_seed(0)
dev = torch.device("cuda:0")
cc = torch.cuda.get_device_properties(dev)
print(f"GPU: {torch.cuda.get_device_name(dev)} (sm_{cc.major}{cc.minor})\n")

# ---- the metadata a serving engine already owns --------------------------
B, PAGE, H_QO, H_KV, D = 4, 16, 8, 2, 128
q_lens = torch.tensor([17, 1, 40, 9], dtype=torch.int32)
kv_lens = torch.tensor([170, 93, 256, 41], dtype=torch.int32)
qo_indptr_cpu = torch.cat(
    [torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0, dtype=torch.int32)]
)
pages = (kv_lens + PAGE - 1) // PAGE
pool = int(pages.sum()) + 4
perm = torch.randperm(pool, dtype=torch.int32)
block_tables = torch.zeros(B, int(pages.max()), dtype=torch.int32)
off = 0
for i in range(B):
    block_tables[i, : pages[i]] = perm[off : off + pages[i]]
    off += int(pages[i])

q = torch.randn(int(q_lens.sum()), H_QO, D, dtype=torch.bfloat16, device=dev)
k_cache = torch.randn(pool, H_KV, PAGE, D, dtype=torch.bfloat16, device=dev)
v_cache = torch.randn(pool, H_KV, PAGE, D, dtype=torch.bfloat16, device=dev)

meta = dict(
    qo_indptr=qo_indptr_cpu.to(dev),
    qo_indptr_cpu=qo_indptr_cpu,
    kv_seq_lens=kv_lens.to(dev),
    kv_seq_lens_cpu=kv_lens,
    block_tables=block_tables.to(dev),
    page_size=PAGE,
    max_q_len=int(q_lens.max()),
    max_kv_len=int(kv_lens.max()),
    num_qo_heads=H_QO,
    num_kv_heads=H_KV,
    head_dim_qk=D,
    q_dtype=torch.bfloat16,
    causal=True,
    return_lse=True,
)

# ---- 1. init-time resolution (no tensors, no wrapper) --------------------
print("=== 1. resolve at engine init ===")
res = resolve_paged_prefill(
    device=dev,
    num_qo_heads=H_QO,
    num_kv_heads=H_KV,
    head_dim_qk=D,
    q_dtype=torch.bfloat16,
    page_size=PAGE,
    causal=True,
    need_lse=True,
)
print(res.explain(), "\n")

# ---- 2/3. one call, every backend ----------------------------------------
print("=== 2/3. same call, every runnable backend ===")
results = {}
for name in res.backends:
    attn = UnifiedPagedPrefill(dev)
    attn.plan(**meta, backend=name)
    out, lse = attn.run(q, (k_cache, v_cache))
    results[name] = (out, lse)
    print(
        f"{name:11s} out={tuple(out.shape)} {out.dtype}   "
        f"lse={tuple(lse.shape)} {lse.dtype} (base-2, packed — contract)"
    )
ref_name = res.backends[0]
for name, (out, lse) in results.items():
    d_out = (out.float() - results[ref_name][0].float()).abs().max().item()
    d_lse = (lse - results[ref_name][1]).abs().max().item()
    print(f"{name:11s} vs {ref_name}: max|Δout|={d_out:.4f}  max|Δlse|={d_lse:.4f}")
print()

# ---- 4. decode shape through the same API ---------------------------------
print("=== 4. decode-shaped batch (uniform q_len=1), same API ===")
q1_lens = torch.ones(B, dtype=torch.int32)
qo1 = torch.cat(
    [torch.zeros(1, dtype=torch.int32), q1_lens.cumsum(0, dtype=torch.int32)]
)
meta_d = dict(meta, qo_indptr=qo1.to(dev), qo_indptr_cpu=qo1, max_q_len=1)
attn = UnifiedPagedPrefill(dev)
attn.plan(**meta_d, backend="auto")
out, lse = attn.run(q[:B], (k_cache, v_cache))
print(f"auto chose: {attn._backend}; out={tuple(out.shape)} lse={tuple(lse.shape)}\n")

# ---- 5. broken input → loud error, not plausible garbage ------------------
print("=== 5. broken metadata is rejected, loudly ===")
bad = dict(meta, max_kv_len=64)  # under-claimed: real KV goes to 256
try:
    UnifiedPagedPrefill(dev).plan(**bad, backend="auto")
except ValueError as e:
    print(f"ValueError: {e}")
print(
    "\n(reject-or-correct is machine-checked: tests/attention/test_unified_prefill_fuzzer.py)"
)
