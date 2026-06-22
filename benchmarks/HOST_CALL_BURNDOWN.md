# FI host-call overhead: accurate component burn-down + reduction plan

Per-component **median host wall** (µs/call, steady state, first 15 iters skipped) of the
FlashInfer `nccl.ep` dispatch/combine host path, measured with `EP_PROFILE_HOST=1`
perf_counter probes inside `flashinfer/moe_ep/nccl_ep/handle.py` (rank 0, 8 GPU, 1 node,
hidden 7168, top-k 8, 256 experts, BF16). For reference, the CUDA-event "kernel-only"
(host-call) and nsys pure-kernel numbers are repeated at the bottom of each block.

## Low-Latency (em), 128 tok/rank

| component (host step) | dispatch µs | combine µs | nature |
|---|---:|---:|---|
| recv buffer `torch.empty` | 4.0 | — (caller `out`) | **removable** (cache) |
| build FFI objects (`ep.Tensor` wraps + Inputs/Outputs/LayoutInfo/Config) | 22.8 | 19.8 | **removable** (cache) |
| `nccl.ep dispatch/combine` launch (async) | 5.9 | 6.1 | irreducible (~launch) |
| `nccl.ep complete` (async, LL) | 0.8 | — | irreducible |
| `ExternalStream.synchronize()` | 10.2 | — | **removable** (defer count) |
| `recv_count.sum().item()` host readback | **57.5** | — | **removable** (defer / keep on device) |
| **host-call total (median host wall)** | **101** | **26** | |

Reference: event "kernel-only" = 125 / 78 µs; nsys pure kernel = 45 / 39 µs.

**Reading:** the LL dispatch host-call is dominated by **non-launch Python work** — the
`recv_count.sum().item()` readback (57 µs: a reduce kernel + D2H + host sync just to compute
`num_tokens`) plus FFI-object construction (23 µs) plus the recv `torch.empty` (4 µs) and the
`ExternalStream.synchronize()` (10 µs). The actual `nccl.ep` launch is only ~6 µs. So
**~94 of ~101 µs is removable** Python-side overhead. Combine has no readback/sync — its
26 µs is essentially all FFI-object construction (20 µs), also cacheable.

## High-Throughput (flat), 4096 tok/rank

| component (host step) | dispatch µs | combine µs | nature |
|---|---:|---:|---|
| recv buffers (cached) | 0.7 | — | already cached |
| build FFI objects | 30.2 | 17.3 | **removable** (cache) |
| `nccl.ep dispatch` (BLOCKING: kernel 529 + ~60 C++ host) | **590.2** | — | library-bound |
| `nccl.ep combine` launch | — | 13.9 | ~launch |
| `nccl.ep complete` | 0.9 | 0.8 | ~launch (async) |
| **host-call total (median host wall)** | **622** | **32** | |

Reference: event "kernel-only" = 745 / 650 µs; nsys pure kernel = 529 / 466 µs.

**Reading:** HT dispatch's host call is dominated by the **`nccl.ep dispatch` FFI call
blocking for 590 µs** — it is *synchronous*: it waits on the GPU until the GIN dispatch
completes (≈ the 529 µs kernel) plus ~60 µs of nccl.ep C++ host prep. This is exactly why
opt #2 (pipeline) barely helped HT: the host cannot run ahead because the call itself blocks.
The only Python-removable piece is the ~30 µs of FFI-object construction; the rest lives
*inside* nccl.ep and needs library changes (make dispatch async; trim the C++ host prep).
HT combine, by contrast, is a non-blocking ~32 µs host call (its 650 µs event time is the
combine kernel + complete()'s drain running on the stream, not host time).

## Where the overhead is — summary

| | LL dispatch | LL combine | HT dispatch | HT combine |
|---|---:|---:|---:|---:|
| Python-removable (readback + FFI build + alloc + sync) | **~94 µs** | ~20 µs | ~31 µs | ~17 µs |
| irreducible launch | ~7 µs | ~6 µs | ~6 µs | ~15 µs |
| library-bound (blocking C++/kernel wait) | — | — | **~590 µs** | (drain on stream) |

- **LL is Python-bound and very reducible**: nearly the whole host-call is the recv-count
  readback + FFI-object churn, both removable without touching nccl.ep.
- **HT is library-bound**: the dispatch FFI blocks synchronously; only the ~30 µs FFI build
  is reachable from Python. Real HT wins require nccl.ep to (a) return async from dispatch and
  (b) shrink its per-call C++ host prep.

## Reduction plan (priority order)

**Python-side (this repo, no nccl.ep changes):**
1. **Defer the LL `recv_count.sum().item()` readback (−57 µs, the biggest single item).**
   `num_tokens` is metadata the dispatch kernel doesn't need; return the count tensor on
   device and only `.item()` it if/when a caller actually reads it. Removing it also removes
   the paired `ExternalStream.synchronize()` (−10 µs). EXPERT_MAJOR uses a padded layout, so
   `num_tokens` is often informational — safe to make lazy.
2. **Cache the per-call FFI wrapper objects (−23/−20 µs LL, −30/−17 µs HT).** `ep.Tensor(x)`,
   `DispatchInputs/Outputs/LayoutInfo/Config` (and combine equivalents) are rebuilt every call
   over *stable* tensors (x, out_t, recv_count, weights are fixed after the first iter). Build
   them once in `__init__`/first call and reuse — the inner-FFI analogue of the bench-level
   `EP_REUSE_PARAMS` (which only cached the outer param object and so did ~nothing).
3. **Cache the LL recv buffer (−4 µs).** `_dispatch_ll` does `torch.empty([L, cap*world, H])`
   every call; cache-and-reuse like `_dispatch_ht` already does.

   Expected LL dispatch host-call after 1–3: ~101 → ~12 µs (just FFI launch + residual),
   i.e. event "kernel-only" approaching the 45 µs pure kernel.

**Library-side (nccl.ep, needed for HT):**
4. **Make `nccl.ep` HT dispatch async** (return after launch; drain in `complete()`), so the
   host can run ahead and the 529 µs kernel overlaps the next stage's prep (opt #2 then works).
5. **Trim nccl.ep per-call C++ host prep (~60 µs HT dispatch)** — the GIN op build/schedule
   done on the host each call; hoist invariant setup into handle creation.

## Implemented (EP_FAST_PATH=1) — measured before → after

Wins 1–3 implemented in `nccl_ep/handle.py` behind `EP_FAST_PATH=1` (lazy `num_tokens` via
a thunk in `DispatchOutput.get_num_tokens()`, cached FFI wrappers over stable tensors, cached
LL recv buffer). Round-trip `--validate` passes for LL em and HT. Measured (8 GPU, warmup 10,
iters 100):

### Host-wall component, LL em dispatch (median µs/call)
| step | baseline | EP_FAST_PATH | |
|---|---:|---:|---|
| recv_count `.sum().item()` | 57.8 | **0** | deferred (lazy thunk) |
| `ExternalStream.synchronize()` | 10.1 | **0** | removed with deferral |
| build FFI objects | 23.0 | **9.5** | output/layout/config cached |
| recv buffer alloc | 4.1 | **0.5** | cached |
| ffi dispatch + complete | 6.5 | 5.8 | irreducible |
| **host-wall total** | **101.5** | **16.6** | **−84%** |
(LL combine host-wall 26.4 → 19.2; HT dispatch 618.6 → 597.1 — only FFI build 31→10 helps,
the 585 µs blocking `ffi_dispatch` is library-bound; HT combine 30.9 → 24.7.)

### Bench-reported time (µs/call) — kernel vs measured, baseline → fast
| stage | ep kernel | FI kernel (nsys) | FI measured base | FI measured fast |
|---|---:|---:|---:|---:|
| LL em dispatch | 43.9 | 45.3 | 111.8 | **66.5** (−40%) |
| LL em combine  | 43.8 | 39.3 | 76.8 | **68.6** |
| HT dispatch | 545.9 | 529.2 | 750.9 | **723.8** |
| HT combine  | 482.2 | 466.5 | 652.1 | **643.9** |

LL dispatch+combine host-observed total 218.7 → 163.5 µs (−25%); the LL dispatch measured
time (112 → 67 µs) now approaches the 45 µs pure kernel — the remaining ~21 µs is the per-call
input-tensor FFI wrap + launch. HT is confirmed library-bound: only the ~21/6 µs FFI-build
caching is reachable from Python; the rest needs the nccl.ep changes (#4/#5 below).

## ep_bench vs FI (fast) — kernel & event-measured, single + multi-node

µs/call. **kernel** = pure GPU (ep: CUPTI; FI: nsys). **event measured** = the cudaEvent-
bracketed host-call (ep_bench "total" — LL derived from host BW in `RESULTS.md`, HT reported
directly; FI = bench "kernel-only" with `EP_FAST_PATH=1`). warmup 10 / iters 50.

### Low-Latency (em), 128 tok/rank
| GPUs | stage | ep kernel | ep event meas | FI kernel | FI measured (fast) |
|---:|---|---:|---:|---:|---:|
| 8 | dispatch | 43.9 | 50.0 | 45.3 | 65.0 |
| 8 | combine | 43.8 | 49.8 | 39.3 | 71.1 |
| 16 | dispatch | 170.5 | 177.7 | 163.4 | 217.4 |
| 16 | combine | 200.4 | 207.9 | 187.8 | 236.5 |
| 32 | dispatch | 254.7 | 262.6 | 240.8 | 284.7 |
| 32 | combine | 284.7 | 291.0 | 255.0 | 318.6 |

### High-Throughput (flat), 4096 tok/rank
| GPUs | stage | ep kernel | ep event meas | FI kernel | FI measured (fast) |
|---:|---|---:|---:|---:|---:|
| 8 | dispatch | 545.9 | 691.2 | 529.2 | 714.6 |
| 8 | combine | 482.2 | 629.8 | 466.5 | 645.8 |
| 16 | dispatch | 1426.6 | 1603.5 | 1412.4 | 1627.2 |
| 16 | combine | 1548.8 | 1697.7 | 1521.8 | 1863.9 |
| 32 | dispatch | 3619.1 | 3801.9 | 3639.8 | 3890.6 |
| 32 | combine | 3518.2 | 3666.9 | 3542.3 | 4118.0 |
| 64 | dispatch | 6016.0 | 6220.7 | 6032.9 | 6284.3 |
| 64 | combine | 6024.0 | 6172.8 | 5933.7 | 7116.3 |

(LL 64g omitted — FI LL GIN fails cross-node at 8 nodes, unrelated.)

**Reading:** kernel columns match at every scale (same kernels, same GPU time). **HT FI(fast)
≈ ep_bench on both kernel and event-measured** (dispatch within ~1–3%): the host gap over the
kernel (~145 µs ep / ~185 µs FI) is inherent to the blocking GIN op, present in C++ too — not a
Python artifact. The exception is HT **combine at 16g+** (FI > ep): FI's combine event bracket
includes `handle.complete()`'s scan/drain, whereas ep_bench records combine-end before its
complete — a measurement-boundary difference. **LL FI(fast)** sits ~15–35 µs above ep's
event-measured because ep's C++ caller has near-zero host overhead (event ≈ kernel, ~6 µs
apart) while FI retains ~20 µs/call of Python/FFI (input wrap + launch); still, FI-fast dispatch
fell from ~2.2× ep's event-measured (baseline) to ~1.3× @8g and ~1.08× @32g. Closing the last
~20 µs LL gap needs CUDA-graph replay or a C++/`torch.library` fast path.

## Reproduce
```bash
# instrumented wrapper: flashinfer/moe_ep/nccl_ep/handle.py
EP_PROFILE_HOST=1 EP_PROFILE_SKIP=15 [EP_FAST_PATH=1] <run bench_ep_matrix.py>
# prints "=== EP_PROFILE_HOST (median host wall µs/call ...)" per dispatch/combine path.
```
