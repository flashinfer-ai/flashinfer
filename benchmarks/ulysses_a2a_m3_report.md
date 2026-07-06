# Ulysses all-to-all M3 performance report

Performance contract for the Ulysses communicator work
(`feat/ulysses-p2p-a2a`): the fused-transpose NVLink kernel and its new
public API must not regress the a2a pattern by more than **3% p50** against
the `c83e4204` baseline, measured with the harness in
[`bench_ulysses_a2a.py`](./bench_ulysses_a2a.py).

## Environment

| | |
|---|---|
| Machine | 8x NVIDIA H20 (single node, all-pairs NVLink NV18), hostname `h20-7` |
| Software | torch 2.11.0+cu130, CUDA 13 toolkit, Nsight Systems 2025.3.2.474 |
| Baseline | `c83e420451662c89ca1429e28caff4875e943a9d` (independent git worktree, clean) |
| New | `fec509de719ce6533708ec8e4bef89422fc47a8b` (clean; harness of this commit generated both sides) |
| Harness identity | schema `ulysses-a2a-m3-v2`, script sha `efb057a8ec59` (recorded and cross-checked by `compare`) |

## Methodology

One **sample** = `3 x scatter_heads + 1 x gather_heads` (the q/k/v-in,
output-back pattern of one Ulysses attention layer); the secondary
`e2e_attn` unit inserts a `scaled_dot_product_attention` before the gather.
5 repeats x 30 iters per impl/unit (one CUDA event pair per iteration),
warmup 5 of the same unit; measurement order **rotates per repeat**; each
sample is the **max across ranks**; p50 is the conventional median over the
150 rank-max samples. Workload: B=1, S_global=32760, D=128, bf16; H=40
(Wan2.1-14B) for W=2/4/8, H=48 (divisible standalone workload) for W=6.
Workspace is sized to the exact largest operand (`B*S_local*H*D`).
Correctness preflight: every impl is checked bit-exactly against an
independent all-gather reference before timing.

Reproduce (baseline side runs the *same committed harness* with
`PYTHONPATH` pointing at the baseline worktree):

```bash
python benchmarks/bench_ulysses_a2a.py run --world-size 8 \
    --impls raw,communicator,communicator_nccl,nccl_ref --label new --out new_w8
PYTHONPATH=/path/to/c83e4204-worktree \
python benchmarks/bench_ulysses_a2a.py run --world-size 8 \
    --impls raw,nccl_ref --label baseline --out base_w8
python benchmarks/bench_ulysses_a2a.py compare base_w8.json new_w8.json --threshold-pct 3
```

## Gated results (a2a p50, threshold 3%) — all PASS

| W | raw -> raw (kernel+raw path) | raw -> communicator (public API) | nccl_ref control |
|---|---|---|---|
| 2 | 2.092 -> 2.092 ms (−0.00%) | 2.092 -> 2.093 ms (+0.05%) | 2.625 -> 2.631 ms (+0.24%) |
| 4 | 1.083 -> 1.084 ms (+0.04%) | 1.083 -> 1.083 ms (−0.01%) | 1.685 -> 1.709 ms (+1.40%) |
| 6 | 1.013 -> 1.013 ms (−0.00%) | 1.013 -> 1.024 ms (+1.13%) | 1.466 -> 1.472 ms (+0.39%) |
| 8 | 0.687 -> 0.688 ms (+0.23%) | 0.687 -> 0.682 ms (−0.70%) | 0.998 -> 0.997 ms (−0.18%) |

The NCCL control stayed within **±1.4%** across all world sizes (the earlier
run set spanned −1.3%..+0.35%; a previous summary overstated this as ±0.4%).
The fused kernel remains **1.25x (W=2) to 1.45x (W=8)** faster than the
NCCL reference on the a2a unit.

The pairs in the table above are the *regression contract* (baseline code vs
new code). Pure **public API overhead** is a same-artifact comparison
instead: within the `new` artifacts, `nccl_ref` -> `communicator_nccl`
(identical NCCL algorithm, once inline and once through
`UlyssesCommunicator(backend="nccl")`) measures −1.22%..+0.09% across
W=2/4/6/8 — noise range, i.e. the public API adds no measurable per-call
overhead.

Secondary `e2e_attn` (ungated): flat on most pairs, but individual runs show
±13..23% swings on the raw/communicator paths (e.g. W=4 raw->communicator
+23.1%, W=8 raw->raw +22.2% and raw->communicator −13.2% in the same run)
while the gated a2a numbers and both NCCL controls stay flat; the sdpa proxy
itself (backend/clock variance), not the communication, is unstable at this
sample size.

Artifacts: `/home/claude/m3_results/final/{base,new}_w{2,4,6,8}.{json,csv}`
(full per-sample data, provenance, per-repeat orders); earlier exploratory
runs in `/home/claude/m3_results/`.

## Profiler evidence and optimization attempt (rejected)

`nsys profile --trace cuda,nvtx` (Nsight Systems 2025.3.2.474) over 50
back-to-back scatters + 50 gathers at W=8, `cuda_gpu_kern_sum`:

| kernel | median | mean | launch config |
|---|---|---|---|
| `ulysses_a2a_kernel<bf16, 8, 0>` | 145.3 µs | 149.9 µs | 36 blocks x 512 threads |
| `ulysses_a2a_kernel<bf16, 8, 1>` | 148.1 µs | 149.6 µs | 36 blocks x 512 threads |

Derived *estimates* (payload/time, not measured utilization): 41.9 MB
operand in ~150 µs ≈ 559 GB/s combined read+write, of which ~245 GB/s is
NVLink egress (7/8 of the writes) — well under the link and HBM capabilities
— and the barrier-imposed `kMaxBlocks = 36` cap uses fewer than half of
H20's SMs. Hypothesis: occupancy-limited.

**Attempt**: `kUlyssesThreads` 512 -> 1024
(`include/flashinfer/comm/ulysses_all_to_all.cuh`, one line).

- Isolated same-mode streams at W=8 improved sharply (scatter 0.365 -> 0.196
  ms per op, 1.86x).
- The contract 3+1 unit at **W=8 regressed +14.1% p50, re-tested +13.8%**,
  with the NCCL control flat in both runs (+0.07% / +0.13%) — a real,
  reproducible regression of the mixed pattern.
- W=4/W=6 opt runs also showed Ulysses regressions, but their NCCL controls
  moved too (about +21% / +4% vs the paired new run), so those runs are
  contaminated and are **not** cited as evidence.
- The p95 blow-up in the W=8 opt run (0.71 -> 3.1 ms) appeared in the NCCL
  control of the same run as well (1.01 -> 3.39 ms) and is therefore not
  attributable to the kernel change.

**Conclusion: rejected; reverted to 512 threads (zero kernel diff on the
branch).** All four launches sit on one CUDA stream, so the earlier
"inter-launch overlap" guess does not apply; the mechanism behind
isolated-stream-faster / mixed-unit-slower is an **unidentified mixed-mode /
barrier interaction** pending further profiling. Note for future work: at
W=2 the 1024-thread build improved the full 3+1 unit by ~30% (2.09 -> 1.46
ms), so a per-world-size conditional dispatch of the block size is a
candidate optimization — out of scope for this contract round.
