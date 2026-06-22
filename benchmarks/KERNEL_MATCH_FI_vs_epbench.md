# FlashInfer vs ep_bench: same kernels, where the extra time comes from

**Question:** FlashInfer's reported dispatch/combine "kernel" times are much larger than
ep_bench's. Are they running the *same* kernels for the same problem, and if so where is
the extra overhead?

## Side-by-side: kernel time vs host-call time (single node, 8 GPU)

Two numbers per stage: **kernel** = pure GPU device time (ep: CUPTI; FI: nsys); **host-call**
= the per-call `cudaEvent`-bracketed time = host launch path + kernel (ep_bench labels this
"total", FI labels it "kernel-only").

| Case · stage | ep kernel | ep host-call | FI kernel | FI host-call |
|---|---:|---:|---:|---:|
| LL em 8g · dispatch | 43.9 | 50.0 | 45.3 | 108.0 |
| LL em 8g · combine  | 43.8 | 49.8 | 39.3 | 75.0 |
| HT 4096 8g · dispatch | 545.9 | 691.2 | 529.2 | 745.1 |
| HT 4096 8g · combine  | 482.2 | 629.8 | 466.5 | 648.8 |

All µs/call. ep from `RESULTS.md` (CUPTI kernel; host-call = host-observed total — LL derived
from host BW, HT reported directly). FI from `RESULTS_FI.md` (event "kernel-only" = host-call)
+ nsys `cuda_gpu_kern_sum` (kernel).

**Reading it:** the **kernel** columns match (ep ≈ FI — same kernels, same GPU time). The
**host-call** − kernel difference (the per-call host launch path) is tiny for ep LL (~6 µs)
but ~63 µs for FI LL (the Python/TVM-FFI layer), and ~146 µs for ep HT vs ~200 µs for FI HT
(nccl.ep's own GIN host path, plus Python on top). ep_bench is not faster end-to-end — its
headline is just the CUPTI **kernel** column, and it uses **no CUDA graphs**.

---

**Detail:** **Same kernels, same GPU time.** FlashInfer (FI) launches the byte-identical
NCCL-EP kernels and they execute for the same number of microseconds on the GPU. FI's
larger "kernel-only" number is a **measurement artifact**: it brackets the whole Python
`dispatch()`/`combine()` call with CUDA events *after a full `torch.cuda.synchronize()`*,
so the window captures host launch latency (GPU sitting idle while Python issues the
launch) plus auxiliary kernels — not just the dispatch/combine kernel. ep_bench measures
pure kernel device time (CUPTI, by kernel name).

## Evidence: nsys per-kernel GPU time, same single-node problems

Both profiled with Nsight Systems 2026.2.1 on rank 0, hidden=7168, top-k=8, experts=256,
BF16, 8 GPU / 1 node (`prof_fi/*.kern.csv` for FI; `prof_kern/*.kern.csv` for ep_bench).

### Kernel identity — the template instantiations are identical
- LL dispatch: `void nccl_ep::internode_ll::dispatch<(bool)0,(bool)0,(int)7168,(ncclEpLayout_t)1,(bool)1>(...)`
- LL combine: `void nccl_ep::internode_ll::combine<(bool)0,(int)7168,(int)9,(int)4,(ncclEpLayout_t)1>(...)`
- HT: `nccl_ep_jit_ht_dispatch_kernel`, `nccl_ep_jit_ht_combine_kernel`, `nccl_ep_jit_ht_scan_kernel`,
  `hybridep::dense_to_sparse_prob_kernel`, `hybridep::sparse_to_dense_prob_kernel`

Same symbols in both binaries — FI's nccl.ep and ep_bench's nccl.ep are the same library.

### Pure GPU kernel time matches (avg µs/call, nsys)
| Kernel | FlashInfer | ep_bench | Δ |
|---|---:|---:|---:|
| LL `internode_ll::dispatch` (em) | 45.3¹ | 44.9 | ~0 |
| LL `internode_ll::combine` (em) | 39.3¹ | 40.5 | ~0 |
| HT `ht_dispatch_kernel` (4096) | 529.2 | 548.3 | −3% |
| HT `ht_combine_kernel` (4096) | 466.5 | 476.6 | −2% |
| HT `ht_scan_kernel` (1×) | 52.5 | 51.9 | ~0 |
| HT prob (d2s + s2d) | 19.6 | 19.6 | 0 |

¹ FI LL uses the **median** (steady-state); the mean is skewed by one cold first-measured
iteration (5-warmup profiling run). ep_bench numbers are 100-iter CUPTI means.

**The kernels and their GPU execution times are identical within measurement noise.**

## Where FI's extra time comes from

FI's `bench_ep_matrix.py` times each stage like this (lines 260–267):

```python
torch.cuda.synchronize()      # <-- drains the GPU; stream now idle
ev[0].record()
d = do_dispatch()             # handle.dispatch(...)  [Python -> TVM-FFI -> C++ -> launch]
ev[1].record()
torch.cuda.synchronize()
k_disp = ev[0].elapsed_time(ev[1])   # reported as "Dispatch kernel-only"
```

Because the stream is fully drained before `ev[0].record()`, the event marker fires on an
**idle** GPU. The GPU then sits idle while the host walks the Python → FFI → nccl.ep →
`cuLaunchKernel` path; only then does the kernel run. `elapsed_time(ev0,ev1)` therefore =
**host launch latency (GPU idle) + auxiliary kernels + the dispatch kernel** — essentially
the host wall-clock of the call, not the kernel. (Confirmed: FI's own host-observed
wall-clock ≈ its "kernel-only" number — e.g. LL dispatch host=122.9 µs vs event=113 µs.)
ep_bench's CUPTI-by-name measurement sees only the kernel's device duration, so it never
counts that host gap. The combine window additionally spans `handle.complete()`.

### Decomposition (FI event-based "kernel-only" = kernel + overhead)
| Stage | FI "kernel-only" | pure kernel (nsys) | in-bracket aux | host launch / idle |
|---|---:|---:|---:|---:|
| LL em 8g dispatch | 113.3 | 45.3 | — | **~68** |
| LL em 8g combine | 88.0 | 39.3 | — | **~49** (incl. `complete()`) |
| HT 4096 8g dispatch | 737.7 | 529.2 | 17 (prob) | **~191** |
| HT 4096 8g combine | 657.2 | 466.5 | — | **~190** (incl. `complete()`) |

The overhead is ~50–70 µs/stage for LL and ~190 µs/stage for HT. It is **host-side dispatch
path latency** (Python + TVM-FFI + nccl.ep param build + launch), exposed because the
per-iteration `synchronize()` prevents the launch latency of one stage from overlapping
(hiding behind) the kernel of the previous one. It is *not* extra or different GPU work.

### Direct proof from the trace (HT 4096 8g): the gap is real GPU idle, not blocking launches
Querying the nsys SQLite (`gap_analysis.py` / `gap2.py`):
- **`cudaLaunchKernel` host duration: median 4.1 µs** (132 calls) — launches are normal async,
  they do **not** block. (The multi-ms `cuMem*`/`cuLibraryLoadData`/`cuMulticastBindMem` calls
  are all **one-time setup**: buffer registration, window mapping, JIT load — not per-iter.)
- **GPU-idle gap *before* each per-iter kernel** (median):

  | kernel (per iter) | dur µs | idle gap before it µs |
  |---|---:|---:|
  | `dense_to_sparse_prob` (1st kernel of `dispatch()`) | 17.0 | **124.8** |
  | `ht_dispatch_kernel` | 529.2 | 8.4 |
  | `ht_combine_kernel` (after `combine()`+`complete()`) | 466.5 | **196.4** |
  | `sparse_to_dense_prob` | 2.6 | 85.8 |

  So the GPU sits idle **~125 µs** before the dispatch path's first kernel and **~196 µs**
  before the combine kernel, while the host walks the Python→TVM-FFI→nccl.ep prep path. That
  idle — not kernel work and not blocking launches — *is* the overhead.

## Cross-node confirmation (same kernels + same GPU time at 16/32/64 GPU)
Re-profiled FI at multi-node scale and compared median kernel µs to ep_bench's
`RESULTS_KERNELS.md`. Same template instantiations; pure GPU kernel time tracks ep_bench at
every scale (LL 64g not shown — FI's LL GIN fails cross-node at 8 nodes, NCCL err @1491,
unrelated to kernels).

| Case | FI disp | ep disp | FI comb | ep comb |
|---|---:|---:|---:|---:|
| LL em 8g  | 45.3 | 44.9 | 39.3 | 40.5 |
| LL em 16g | 163.4 | 161.0 | 187.8 | 203.8 |
| LL em 32g | 240.8 | 237.0 | 255.0 | 307.3¹ |
| HT 4096 8g  | 529.2 | 548.3 | 466.5 | 476.6 |
| HT 4096 16g | 1412.4 | 1440.1 | 1521.8 | 1554.5 |
| HT 4096 32g | 3639.8 | 3614.8 | 3542.3 | 3517.9 |
| HT 4096 64g | 6032.9 | 6066.5 | 5933.7 | 5976.8 |

¹ Largest gap; cross-node LL combine has high per-iter variance (network jitter — instance
min/max 57/1016 µs) and FI here is a 20-iter median vs ep_bench's 100-iter mean. Dispatch,
which is steadier, matches within ~2% at all scales. **Conclusion: the kernels and their GPU
times are identical single-node and cross-node up to 64 GPU.**

## Ways to reduce the host launch / idle time

The cost is per-iteration GPU idle (~125 µs dispatch, ~196 µs combine for HT) spent in the
host dispatch/combine prep path and exposed by the per-iter full sync. Levers, by impact:

1. **CUDA Graph capture + replay (biggest lever).** Capture `dispatch()` (and ideally
   `dispatch`+sync+`combine`) once; each iteration becomes a single `cudaGraphLaunch`
   (~few µs host) that runs all kernels back-to-back with **no host-in-the-loop gaps**. This
   removes essentially all of the 125/196 µs idle. Requires the nccl.ep GIN kernels to be
   graph-capture-safe — verify capture under `cudaStreamCaptureModeThreadLocal`; device-side
   RDMA may need the stream-ordered/`cudaGraphAddNode` path. LL (no JIT) is the easiest to try first.
2. **Pipeline — drop the per-iteration full sync.** In real serving, issue
   dispatch→combine→next-dispatch on the stream so stage N+1's ~125–196 µs host prep overlaps
   stage N's 0.5 ms kernel and is fully hidden. The dispatch/combine kernels are long enough to
   absorb it. (HT needs a dispatch→combine ordering guarantee, but a lightweight
   `cudaEvent`/stream dependency suffices — a full `torch.cuda.synchronize()` is not required.)
3. **Cut per-call host work in the Python/FFI path.** ~125–196 µs is a lot for "build params +
   launch": reuse `DispatchInputParams`/`CombineInputParams` objects across iters instead of
   reconstructing; skip redundant dtype/shape validation on the hot path; and **collapse the
   metadata `cudaMemcpyAsync` traffic** (the trace shows 2746 small async copies) — batch or
   pre-stage routing metadata on device. A C++ caller (ep_bench) pays µs here; the bulk of FI's
   gap is Python interpreter + TVM-FFI marshaling, so a thin C++/`torch.library` fast path helps.
4. **Make `handle.complete()` lazy/async.** The combine gap (196 µs) spans `combine()` **and**
   `complete()`; if `complete()` forces a host-side wait or redundant teardown each call, defer
   it (only required before the result is consumed) to shrink the combine idle.
5. **Fuse the auxiliary kernels.** `dispatch()` launches `dense_to_sparse_prob` then the GIN
   dispatch kernel (plus a barrier all-reduce); folding the prob conversion into the dispatch
   kernel removes a launch + an inter-kernel gap per iteration.

Note these are *latency/host-efficiency* improvements. The **throughput-relevant** cost is the
pure kernel time, which is already identical between FI and ep_bench — so (1)+(2) (graphs +
pipelining) are what close the host-observed gap without touching kernel code.

## Optimization results (implemented #2 then #3)

Implemented in `bench_ep_matrix.py` behind env flags (LL em 8g, HT 4096 8g, warmup 10 / iters 50):
- **#2 pipeline** — `EP_TIMING=pipeline`: no per-iter `torch.cuda.synchronize()`; per-iter
  CUDA events read once after a single end-of-loop sync, so the host runs ahead and each
  stage's launch prep overlaps the running kernel. `EP_NO_BARRIER=1` additionally drops the
  per-iter cross-rank barrier (safe for LL; the bench notes HT wants it for lockstep).
- **#3 reuse params** — `EP_REUSE_PARAMS=1`: build `Dispatch/CombineInputParams` once, reuse.

All values µs/call. "ep kernel" = ep_bench CUPTI; "FI kernel" = FI nsys pure-kernel device
time; "FI measured" = FI's CUDA-event "kernel-only" number (what the bench reports).

| Stage | ep kernel | FI kernel | FI meas — baseline | FI meas — +#2 | FI meas — +#2+#3 |
|---|---:|---:|---:|---:|---:|
| LL dispatch | 44.9 | 45.3 | 109.3 | **77.3** | 77.1 |
| LL combine  | 40.5 | 39.3 | 75.7  | **72.8** | 72.4 |
| HT dispatch | 548.3 | 529.2 | 733.8 | **731.8**¹ | 728.2 |
| HT combine  | 476.6 | 466.5 | 648.1 | **628.5** | 627.7 |

¹ HT #2 keeps the barrier (correctness). `EP_NO_BARRIER=1` further cuts HT dispatch to ~696 µs
but is not lockstep-safe for HT. LL #2 above uses `EP_NO_BARRIER=1` (total D+C 215→160 µs, −26%).

**What we learned:**
- **#2 (pipeline) helps where the library doesn't self-synchronize.** LL dispatch 109→77 µs
  (−29%), LL D+C total 215→160 µs (−26%). HT gains are small (combine 648→628, dispatch
  flat) because the **nccl.ep HT `dispatch()`/`combine()`/`complete()` calls synchronize
  internally** — the host can't run ahead to overlap, so removing the bench-level sync barely
  moves it. (Confirmed: HT host wall ≈ event time at every config.)
- **#3 (reuse params) is negligible** (<1 µs, <1%) for both LL and HT. The per-call host gap is
  **not** Python object construction — it lives in the TVM-FFI call + nccl.ep C++ host path
  (and the HT internal syncs). Reusing wrappers can't touch that.
- **Implication:** the remaining gap needs lever **#1 (CUDA graphs)** or work *inside* nccl.ep
  (remove the internal per-call host syncs on the HT path; batch the metadata `cudaMemcpyAsync`
  at the C++ level). Bench-level Python changes have hit their floor at ~77 µs (LL) / ~730 µs (HT).
  Graphs are the recommended next step — capture LL first (no JIT, no internal sync), where the
  measured number should drop to ~kernel time (45/39 µs).

## How to make FI match ep_bench (measurement)

The kernels already match; only the measurement differs. To get ep_bench-equivalent
"kernel time" from FI, measure **pure kernel device time** instead of the event-bracketed
host call. Options, best first:
1. **CUPTI by kernel name** (what ep_bench does) — FlashInfer's `bench_gpu_time(...,
   enable_cupti=True)` reports device kernel time directly and excludes the host gap.
2. **CUDA graph capture** of dispatch/combine, then time the graph replay — collapses the
   per-call host launch path so it no longer dominates.
3. **Pipelined event timing** — record events across a long run of back-to-back iterations
   *without* the per-iter `torch.cuda.synchronize()`, so stage N+1's launch latency hides
   behind stage N's kernel. (Note: HT needs the inter-stage sync for correctness, so this
   is only safe for the LL dispatch/combine micro-measurement.)

In a real pipelined serving workload the host launch latency is largely hidden behind
adjacent kernels, so the *throughput-relevant* cost is the pure kernel time — which is
already identical between FI and ep_bench.

## Repro
```bash
RW_FI=/lustre/fsw/coreai_libraries_cudnn/agopal-moe-ep      # FI checkout + flashinfer-ep-pt2605.sqsh
# nsys-profile FI bench (rank 0), reduced iters:
srun --jobid=<jid> -N1 --ntasks-per-node=8 --container-image=$RW_FI/flashinfer-ep-pt2605.sqsh \
  --container-mounts=$RW_FI:/host \
  bash -lc 'EP_SYNC=/host/sync_X PROF_TAG=ll_em_8g_ib NCCL_GIN_TYPE=3 \
    bash /host/flashinfer/benchmarks/run_ep_matrix_one_pt_nsys.sh \
    --algorithm ll --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 --warmup 5 --iters 20'
# extract per-kernel GPU time:
nsys stats --report cuda_gpu_kern_sum --format csv $RW_FI/prof_fi/ll_em_8g_ib.nsys-rep
# host-call durations + GPU-idle gaps (the host-launch/idle proof):
python $RW_FI/gap2.py $RW_FI/prof_fi/ht_4096_8g_ib.sqlite
```
Cross-node: `$RW_FI/run_fi_crossnode_nsys.sh` (-N per case; retry wrapper `retry_fi_cn.sh`
handles transient NCCL bootstrap failures). Wrapper `benchmarks/run_ep_matrix_one_pt_nsys.sh`
nsys-wraps rank 0 (CUDA+NVTX, no CUPTI conflict since FI times with CUDA events). FI traces:
`prof_fi/*.nsys-rep`; parsed `prof_kern_fi/*.kern.csv`. ep_bench reference:
`RESULTS_KERNELS.md`, `prof_kern/*.kern.csv`.
```
