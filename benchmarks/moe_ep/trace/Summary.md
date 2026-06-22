# MegaMoE EP: FlashInfer vs vLLM — Profiling Summary

**Problem:** `world_size=4`, `hidden=4096`, `intermediate=2048`, `num_experts=256`, `topk=6`, `num_tokens=8192`  
**Sources:** `trace.txt`, `profiles/*_nvtx_stats_*.csv` (steady capture), `profiles/*_full_stats_*.csv` (setup)

---

## Steady-state headline

| Metric | FlashInfer | vLLM | Δ |
|--------|-----------|------|---|
| Benchmark steady avg (cuda events) | **1.400 ms** | **1.455 ms** | **+55 μs (~4%)** |

Both backends run the **same two GPU kernels** at the **same speed**. The 55 μs gap is **not** model parallelism (TP/PP/NCCL) — it is **vLLM framework overhead** around the kernels.

---

## Where the 55 μs gap is

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FlashInfer  1.400 ms                                                   │
│  ┌──────────────────────────────┐  ┌──────────┐                         │
│  │  GPU kernels (same)  ~1371 μs │  │ wrapper  │  ~29 μs                │
│  │  mega 1230 + stage 141        │  │ overhead │                        │
│  └──────────────────────────────┘  └──────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  vLLM  1.455 ms                                                         │
│  ┌──────────────────────────────┐  ┌──────────┐                         │
│  │  GPU kernels (same)  ~1384 μs │  │ wrapper  │  ~71 μs   ◄── +42 μs   │
│  │  mega 1244 + stage 141        │  │ overhead │                         │
│  └──────────────────────────────┘  └──────────┘                         │
│         (+14 μs kernel noise)              ▲                            │
│                                            └── entire 55 μs gap         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Layer | FlashInfer | vLLM | Δ | Verdict |
|-------|-----------|------|---|---------|
| `deep_gemm::sm100_fp8_fp4_mega_moe_impl` | 1230 μs avg (1195 med) | 1244 μs avg (1194 med) | +14 μs | Same kernel; medians tied |
| Staging kernel | 141 μs | 141 μs | ~0 | Identical |
| **Wrapper / dispatch overhead** | **~29 μs** | **~71 μs** | **+42 μs** | **Root cause** |

**Not in steady capture:** no NCCL kernels, no TP all-reduces, no extra EP comm — only the two kernels above (`nvtx` steady window, 20 forwards × 4 ranks).

---

## What the vLLM wrapper overhead is

Each timed vLLM forward goes through extra indirection that FlashInfer skips:

```
FlashInfer:  mega.forward() → stage kernel → deep_gemm kernel

vLLM:        set_forward_context()
               → experts.forward()
                 → torch.ops.vllm.deepseek_v4_mega_moe_experts  (custom op)
                   → forward-context module lookup
                   → stage kernel → deep_gemm kernel
```

Evidence from steady `nvtx` capture:

| Signal | FlashInfer | vLLM |
|--------|-----------|------|
| `:forward` host wall (dispatch only) | 218 μs avg | 291 μs avg (+73 μs) |
| `cudaDeviceSynchronize` calls | 89 | 97 (+8) |
| `cuMemUnmap` calls | 5 | 13 |
| `cuLaunchKernelEx` avg | 11.2 μs | 12.3 μs |
| Dominant host syscall (`osrt`) | `ioctl` | `epoll_wait` (background worker threads) |

The extra host latency between kernel enqueues shows up as **GPU stream idle** on the cuda-event timeline (~42 μs). Background `epoll_wait` is ambient noise, not the full gap.

---

## FAQ (steady state)

**Is vLLM doing extra model parallelism?**  
No. TP/PP groups are initialized for vLLM boilerplate, but the timed forward only hits `get_ep_group()` inside the same `deep_gemm` path as FlashInfer.

**Should we optimize the mega kernel?**  
No — same symbol, same template args, median ~1.19 ms both sides.

**Why does `:forward` NVTX show ~0.22–0.29 ms but benchmark shows ~1.4 ms?**  
`cuda.synchronize()` is outside the `:forward` NVTX range. Use cuda-event benchmark timings or kernel sums for latency.

---

## Setup (separate from steady gap)

Full capture is dominated by init — **do not use full kernel % for steady comparison**.

| Phase (`:setup` NVTX, per rank) | FlashInfer | vLLM |
|--------------------------------|-----------|------|
| Setup wall time | **3.2 s** | **13.6 s** (4.3× slower) |

vLLM setup cost: `weight_loader` + `finalize_weights()` + `VllmConfig` / `initialize_model_parallel`. NCCL in full capture is setup-only (4 vs 12 `GroupLaunch` instances).

---

## Next steps

1. Move `cuda.synchronize()` inside `:forward` NVTX so `nvtx_sum` matches benchmark ms.
2. NVTX around `set_forward_context` vs `_run_mega_moe` to split vLLM dispatch cost.
3. Setup optimization (vLLM weight-loading path) — orthogonal to the 55 μs steady gap.

---

## Artifacts

| Capture | FlashInfer | vLLM |
|---------|-----------|------|
| Steady (`nvtx`) | `profiles/flashinfer_moe_ep_ws4_t8192_nvtx.nsys-rep` | `profiles/vllm_moe_ep_ws4_t8192_nvtx.nsys-rep` |
| Full | `profiles/flashinfer_moe_ep_ws4_t8192_full.nsys-rep` | `profiles/vllm_moe_ep_ws4_t8192_full.nsys-rep` |

```bash
bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh stats nvtx both
bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh stats full both
```
