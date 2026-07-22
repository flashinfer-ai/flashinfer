# CuTeDSL MegaMoE tuning + performance notes

This document collects the performance work on the CuTeDSL mega
backends: the tuning surface (knobs, per-size default profiles, online
autotuning), how tuning resolves at runtime, the measured results for
each backend and combine-leg variant, and the benchmark methodology
behind those numbers — including the measurement pitfalls we hit and how
to avoid them.  It is the companion to `SKILL.md`, which covers the
kernel drop-update workflow.

Unless noted otherwise, all measurements were taken 2026-07-21 on a
single GB200 node (4x GPUs, EP=4) at a DeepSeek-V3-like geometry:
256 experts, top-8, hidden 7168, intermediate 2048.  A full reproduce
recipe (hardware, container, versions, harness invocation) is in the
"Sweep methodology" and "Runbook" sections below.

## Microbenchmark results (2026-07-22, e2e_pipelined p50 µs)

Measured at the `4_5_2-perf-fix` branch tip (`7d538173`) on
**nvidia-cutlass-dsl 4.5.2** — the new default runtime (see "CuTe-DSL
runtime sensitivity" below; the MR!27 WAR makes 4.5.2 the perf floor).
Reproduces the previous reference (2026-07-21, dsl 4.6.1, tip `29e2d2f8`,
CSV `model_shapes_20260721_111314_deepseek_v3.csv`) within run noise at
every cell. Default geometry (7168 hidden / 2048 inter / 256 experts /
top-8), 4x GB200, heuristic knobs, speedup vs `deep_gemm_mega` in parens;
CSV `moe_ep_benchmark/model_shapes/results/model_shapes_20260722_130237.csv`:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 215.0  | 222.2 (0.97x)  | 234.0 (0.92x)  | 228.3 (0.94x)      | 228.8 (0.94x)  |
| 64       | 288.8  | 287.7 (1.00x)  | 314.8 (0.92x)  | 304.2 (0.95x)      | 308.2 (0.94x)  |
| 512      | 346.1  | 359.3 (0.96x)  | 362.5 (0.95x)  | 328.7 (1.05x)      | 334.8 (1.03x)  |
| 1024     | 474.1  | 428.6 (1.11x)  | 431.8 (1.10x)  | **375.8 (1.26x)**  | 384.0 (1.23x)  |
| 2048     | 822.3  | 621.5 (1.32x)  | 615.4 (1.34x)  | **545.8 (1.51x)**  | 573.4 (1.43x)  |
| 4096     | 1501.7 | 1001.5 (1.50x) | 989.2 (1.52x)  | **931.3 (1.61x)**  | 947.5 (1.58x)  |
| 8192     | 3072.9 | 1896.4 (1.62x) | 1878.3 (1.64x) | **1655.3 (1.86x)** | 1752.0 (1.75x) |

[ikr = in_kernel_fc2_reduce; +combine_nvfp4 / +combine_mxfp8 = the fc2
epilogue quantizes each partial in registers to fp4 (e2m1 + bf16 scale per
16) or fp8 (e4m3 + e8m0 scale per 32) just for the wire]

Shape of the curve: **parity with dg through ~512 tok/rank, crossover
between 512 and 1024, win growing to 1.62x at 8192** (1.86x with the fp4
combine wire).  The small-batch regime is weight-load bound and fp4-vs-fp4
there is a wash.

The `mxfp8_cutedsl` backend (latest sweep 2026-07-22 on dsl 4.5.2, CSV
`sweep_20260722_114210_fi_mega.csv`; its >=2048 rows use the re-derived
dispatch-warp default profile) runs 0.63x / 0.82x / 0.88x / 0.96x vs dg at
1024 / 2048 / 4096 / 8192 tok/rank — near dg parity at 8192 at a 3x better
accuracy point (6.4% vs 20.6% rel-L2).

### Real-model geometry sweep (2026-07-21)

The same sweep across the MoE geometries of real models (same recipe,
session, and node as the table above; CSVs
`moe_ep_benchmark/model_shapes/results/model_shapes_20260721_1109*_*.csv`,
generated tables in that repo's `model_shapes/RESULTS.md`).  Pattern holds
everywhere: dg-parity below ~512 tok/rank, fp4 combine-wire best at large
tokens (1.6-1.9x on 7168-hidden shapes).  `e2e_pipelined` p50 µs, speedup
vs `deep_gemm_mega` in parens.

**deepseek_v3** — hidden 7168, inter 2048, 256 experts, top-8 (independent
same-session re-run of the table above; matches within run noise):

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 211.1  | 220.5 (0.96x)  | 233.9 (0.90x)  | 228.3 (0.92x)      | 227.2 (0.93x)  |
| 64       | 286.7  | 285.7 (1.00x)  | 314.4 (0.91x)  | 303.2 (0.95x)      | 306.0 (0.94x)  |
| 512      | 348.2  | 357.4 (0.97x)  | 361.5 (0.96x)  | 326.7 (1.07x)      | 332.8 (1.05x)  |
| 2048     | 830.0  | 612.2 (1.36x)  | 609.2 (1.36x)  | **539.6 (1.54x)**  | 566.2 (1.47x)  |
| 8192     | 3129.4 | 1942.5 (1.61x) | 1916.9 (1.63x) | **1728.0 (1.81x)** | 1784.3 (1.75x) |

**deepseek_v4_flash** — hidden 4096, inter 2048, 256 experts, top-6:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4    | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|------------------:|---------------:|
| 8        | 127.8  | 142.1 (0.90x)  | 148.5 (0.86x)  | 148.3 (0.86x)     | 151.6 (0.84x)  |
| 64       | 176.2  | 190.3 (0.93x)  | 204.7 (0.86x)  | 195.2 (0.90x)     | 197.6 (0.89x)  |
| 512      | 206.8  | 231.1 (0.89x)  | 229.2 (0.90x)  | 216.0 (0.96x)     | 220.2 (0.94x)  |
| 2048     | 401.4  | 344.6 (1.16x)  | 344.1 (1.17x)  | **303.1 (1.32x)** | 318.5 (1.26x)  |
| 8192     | 1294.9 | 1070.1 (1.21x) | 1061.6 (1.22x) | **794.1 (1.63x)** | 901.1 (1.44x)  |

**deepseek_v4_pro** — hidden 7168, inter 3072, 384 experts, top-6:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 286.7  | 304.2 (0.94x)  | 310.4 (0.92x)  | 309.2 (0.93x)      | 312.8 (0.92x)  |
| 64       | 540.7  | 562.2 (0.96x)  | 575.5 (0.94x)  | 566.0 (0.96x)      | 570.1 (0.95x)  |
| 512      | 603.1  | 621.6 (0.97x)  | 621.8 (0.97x)  | 601.1 (1.00x)      | 607.2 (0.99x)  |
| 2048     | 921.1  | 760.8 (1.21x)  | 758.1 (1.22x)  | **695.3 (1.32x)**  | 715.7 (1.29x)  |
| 8192     | 3093.5 | 1897.4 (1.63x) | 1880.4 (1.65x) | **1773.6 (1.74x)** | 1836.5 (1.68x) |

**kimi_k2_6** — hidden 7168, inter 2048, 384 experts, top-8:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 247.8  | 253.0 (0.98x)  | 275.4 (0.90x)  | 261.2 (0.95x)      | 263.2 (0.94x)  |
| 64       | 409.6  | 401.2 (1.02x)  | 440.2 (0.93x)  | 416.8 (0.98x)      | 419.8 (0.98x)  |
| 512      | 472.2  | 469.0 (1.01x)  | 470.4 (1.00x)  | 442.4 (1.07x)      | 449.0 (1.05x)  |
| 2048     | 834.5  | 668.8 (1.25x)  | 662.2 (1.26x)  | **603.1 (1.38x)**  | 619.5 (1.35x)  |
| 8192     | 3188.3 | 2000.3 (1.59x) | 1982.9 (1.61x) | **1701.9 (1.87x)** | 1894.4 (1.68x) |

**qwen3_5_397b** — hidden 4096, inter 1024, 512 experts, top-10:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 129.0  | 145.7 (0.89x)  | 167.8 (0.77x)  | 157.7 (0.82x)      | 161.8 (0.80x)  |
| 64       | 194.6  | 213.0 (0.91x)  | 264.3 (0.74x)  | 234.5 (0.83x)      | 238.6 (0.82x)  |
| 512      | 233.5  | 253.9 (0.92x)  | 256.0 (0.91x)  | 238.6 (0.98x)      | 243.8 (0.96x)  |
| 2048     | 495.6  | 459.0 (1.08x)  | 461.8 (1.07x)  | **361.6 (1.37x)**  | 391.0 (1.27x)  |
| 8192     | 1772.8 | 1506.0 (1.18x) | 1544.7 (1.15x) | **1100.9 (1.61x)** | 1270.8 (1.40x) |

**gpt_oss_120b** — hidden 2880, inter 2880, 128 experts, top-4.  No dg
column: 2880 fails deep_gemm's hard-%128 wire alignment, so this geometry
runs only via the cutedsl %64 relaxation (fp4 variants only):

| tok/rank | dg | nvfp4 bf16 | +ikr   | +combine_nvfp4 | +combine_mxfp8 |
|---------:|:--:|-----------:|-------:|---------------:|---------------:|
| 8        | —  | 111.6      | 113.9  | 115.1          | 116.7          |
| 64       | —  | 125.9      | 132.5  | 132.1          | 135.6          |
| 512      | —  | 150.5      | 150.5  | 150.5          | 150.0          |
| 2048     | —  | 261.1      | 261.0  | **252.7**      | 259.1          |
| 8192     | —  | 697.3      | 678.9  | **551.9**      | 609.3          |

### Accuracy

`acc_loss_pct` (global rel-L2 vs a bf16 dense-MoE reference on random data
— constant across token counts; random data is the worst case, real-model
distributions fare better — see the e2e GSM8K numbers below):

| variant                | acc loss |
|------------------------|---------:|
| deep_gemm_mega         | 20.6%    |
| nvfp4 (bf16 wire)      | 23.2%    |
| nvfp4 `+ikr`           | 23.2%    |
| nvfp4 `+combine_mxfp8` | 23.3%    |
| nvfp4 `+combine_nvfp4` | 25.0%    |
| mxfp8_cutedsl          | 6.4%     |

Note mxfp8's 6.4% vs the fp4-weight backends' ~21-25%: the perf ranking is
not the whole story.

## End-to-end results (vLLM 0.25.1, 2026-07-20)

Measured with this backend integrated into vLLM 0.25.1 (integration lands
as a separate PR): DeepSeek-V4-Flash (4096 hidden / 2048 inter /
256 experts / top-6), 4x GB200 TP4/EP4, CUDA graphs capturing all recurring
step shapes including the 4096-token prefill chunks, per-role offline knob
caches (`python -m flashinfer.moe_ep.tune`, prefill- and decode-tuned).
`fi_nvfp4` consumes the single-quant nvfp4 checkpoint directly via the
prequantized weight-pack path; native vLLM (its built-in deep_gemm mega
path) and `fi_dg` run the mxfp4 checkpoint.

| Workload                                     | native vLLM | fi_dg           | fi_nvfp4            |
|----------------------------------------------|------------:|----------------:|--------------------:|
| Prefill, 8k-token chunks (tok/s)             | 45,701      | 47,534 (1.04x)  | **53,962 (1.18x)**  |
| Decode @ 1024-seq concurrency (output tok/s) | 21,049      | 21,540 (1.02x)  | **22,614 (1.07x)**  |
| GSM8K (200q, greedy)                         | 0.965       | 0.965           | **0.975**           |

- `fi_dg` is the same deep_gemm kernel routed through this integration
  layer — parity-or-better shows the layer itself costs nothing under
  graphs; the nvfp4 deltas above it are kernel-side.
- The decode number requires graph capture to cover the prefill chunk
  shapes (e.g. vLLM `max_cudagraph_capture_size` >= the chunk size).  With
  the default small capture list the chunks run eager, the eager host path
  generates extra inter-rank launch skew that the collective kernel absorbs
  as spin, and decode reads ~4% BELOW native — a config artifact, not a
  kernel property.
- The headline decode cell reproduces at this branch tip within 0.6%.
- **2026-07-22: both headline cells reproduce on nvidia-cutlass-dsl 4.5.2**
  (vLLM 0.25.1's own pin, no force-upgrade — the MR!27 WAR makes 4.5.2 the
  perf floor): prefill-8k native 45,582 / fi_nvfp4 53,623 = 1.176x; decode-1k
  native 32,086 / fi_nvfp4 34,263 total tok/s = 1.068x (vllm_e2e RUNS.md
  runs 41-42, `requick452_*.json`; every cell within 0.7% of the 4.6.1-stack
  reference).

## Historical note: nvfp4 numbers before 2026-07-15 are invalid

All `nvfp4_cutedsl` measurements taken before 2026-07-15 used a broken
weight layout and have been removed (raw CSVs archived under
`moe_ep_benchmark/results/archive_pre20260715_broken_nvfp4_layout/`): the
FI preprocess materialized fc1/fc2 weights N-stride-1 while the frontend
compiles the kernel's TMA layout from the tensor's actual strides, so those
runs used a faster-but-wrong load pattern and produced incorrect results
(the old "uniform 1.5-1.7x at every token count" claim was this artifact).
Caught by the torch-oracle tests
(`tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py`); fixed in
`backends/mega/kernel/nvfp4_cutedsl/weights.py` (K-major transpose views).
`deep_gemm_mega` / `mxfp8_cutedsl` numbers were unaffected.  The corrected
2026-07-15 full sweep reproduces at the 2026-07-21 branch tip within run
noise (<= ~4% per cell); the tables above are the current reference.

## CuTe-DSL runtime sensitivity: `nvidia-cutlass-dsl[cu13]==4.5.2` (2026-07-15) — OBSOLETE

> **OBSOLETE (2026-07-22):** the regression below is fixed by the MR!27
> mainloop WAR (see the follow-up subsection) — 4.5.2 now runs at full
> parity for both nvfp4 and mxfp8, and the perf floor is 4.5.2, not 4.6.1.
> Kept for the record of what unpatched 4.5.2 codegen did.

Same recipe, geometry and runbook as the microbenchmark table above, with
the only change being the CuTe-DSL runtime pinned to 4.5.2 instead of
>= 4.6.1.  Measured 2026-07-15 against a same-day 4.6.1 baseline (which the
2026-07-21 table reproduces within run noise, so the comparison stands);
CSVs `moe_ep_benchmark/results/sweep_20260715_1634*..1641*_fi_mega.csv`:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 1024     | 468.8  | 583.7 (0.80x)  | 580.9 (0.81x)  | **564.2 (0.83x)**  | 566.3 (0.83x)  |
| 2048     | 813.1  | 878.6 (0.93x)  | 869.2 (0.94x)  | **834.6 (0.97x)**  | 842.8 (0.96x)  |
| 4096     | 1538.1 | 1453.1 (1.06x) | 1430.2 (1.08x) | **1385.5 (1.11x)** | 1394.7 (1.10x) |
| 8192     | 3060.7 | 2579.5 (1.19x) | 2529.0 (1.21x) | **2473.0 (1.24x)** | 2492.4 (1.23x) |

Takeaways:

- 4.5.2 **compiles and runs** this kernel drop (4.5.0 fails outright at
  `cute.compile`), and every variant/point completed without error.
- But the 4.5.2-generated code is **34-54% slower** than 4.6.1 across every
  nvfp4 variant and token count (e.g. bf16 wire 583.7 vs 428.5 µs @1024,
  2579.5 vs 1923.5 @8192 on the same-day 4.6.1 baseline;
  `+combine_nvfp4` ~+50% everywhere).  The dg baseline reproduces within
  noise across the two runtimes (≤1%), so the delta is the DSL runtime,
  not the run.
- The perf shape regresses qualitatively too: the dg crossover moves from
  ~512-1024 tok/rank out past 2048 (nvfp4 *loses* at 1024, 0.80-0.83x),
  and the 8192 win shrinks from ~1.6-1.8x to 1.19-1.24x.

Conclusion (2026-07-15): treat **4.6.1 as a performance floor**, not just a
compile-compatibility floor — 4.5.2 is functional but leaves ~1.3-1.5x of
nvfp4 kernel performance on the table.
**Superseded for nvfp4 on 2026-07-22 by the MR!27 mainloop WAR — see the
follow-up below.**

### Follow-up 2026-07-22: MR!27 mainloop WAR recovers nvfp4 on 4.5.2

`cutedsl_megamoe` MR!27 (ported in the `4_5_2-perf-fix` branch, single file
`src/moe_nvfp4_swapab/kernel_fc12.py`) peels the last k-tile out of the
MMA-consumer mainloop when `cutlass.__version__ == 4.5.2` exactly
(`cutlass.const_expr`-gated, so 4.6.1+ traces are unchanged). Rerun of the
same sweep recipe with 4.5.2 pinned (CSVs
`moe_ep_benchmark/results/sweep_20260722_1125*/1126*/1129*_fi_mega.csv`;
e2e_pipelined p50 µs, ratio vs same-run dg):

| tok/rank | dg     | nvfp4 bf16     | +combine_nvfp4     |
|---------:|-------:|---------------:|-------------------:|
| 8        | 214.0  | 222.5 (0.96x)  | 227.3 (0.94x)      |
| 512      | 341.4  | 359.3 (0.95x)  | **327.7 (1.04x)**  |
| 1024     | 469.0  | 428.0 (1.10x)  | **371.7 (1.26x)**  |
| 2048     | 834.0  | 609.3 (1.37x)  | **533.0 (1.56x)**  |
| 4096     | 1576.3 | 996.9 (1.58x)  | **885.8 (1.78x)**  |
| 8192     | 3018.2 | 1859.7 (1.62x) | **1738.7 (1.74x)** |

This matches the 4.6.1 reference table above within run noise at every
point (e.g. nvfp4 bf16 428.0 vs 428.5 @1024, 609.3 vs 625.6 @2048, 1859.7
vs 1923.5 @8192), i.e. the 34-54% regression is fully recovered and the
dg-crossover / large-token win shape is restored. Correctness: the full
`tests/moe_ep/run_tests.sh all` suite passed under pinned 4.5.2 (all 8
sections, 2026-07-22, job 2427254).

mxfp8 needs no WAR: a same-day 4.5.2 sweep
(`sweep_20260722_114210_fi_mega.csv`) matches the 4.6.1 mxfp8 reference
within noise at every point (373.7/604.1/750.4/1007.2/1714.7/3215.3 µs at
8..8192 vs 385.8/634.9/749.6/1006.6/1742.8/3122.2) — the 4.5.2 codegen
regression was specific to the nvfp4 swap-AB fc12 mainloop. With both
dtypes at parity and the suite green, 4.5.2 is a fully acceptable runtime
for this drop; 4.5.0 still fails at `cute.compile`, so <4.5.2 remains
unsupported in practice.

Later the same day the full 5-variant reference sweep and the vLLM e2e
headline pair were re-measured on 4.5.2 and adopted as the current
reference numbers — see "Microbenchmark results (2026-07-22)" and the
end-to-end section above.

The two published versions between 4.5.2 and 4.6.1 were also measured
(dg + nvfp4 @1024/2048/8192, `sweep_20260722_132849/133055_fi_mega.csv`):
**4.5.3 (424.4/613.3/1933.3 µs) and 4.6.0 (427.0/614.1/1922.0 µs) both
match the fast reference** — the codegen regression existed only in
4.5.2 and was fixed upstream in 4.5.3, so the WAR's exact `== 4.5.2` gate
is correct and every published version >= 4.5.2 runs at full performance.

Support statement: **>= 4.5.2 at full performance** (4.5.2 via the MR!27
WAR; 4.5.3 / 4.6.0 / 4.6.1 verified natively fast); < 4.5.2 unsupported
(4.5.0 fails at `cute.compile`, 4.5.1 unmeasured — the shim warns).

The 4.7.0.dev internal drop (`nvidia-cutlass-dsl-internal`
`0.3.0+20260721052439.f58f4a5`) also measures fast: nvfp4
427.0/614.2/1907.7 µs @1024/2048/8192 (`sweep_20260722_142421_fi_mega.csv`)
— compiles and runs cleanly. Note internal wheels report their own version
scheme (`cutlass.__version__ == "0.3.0"`, dist name
`nvidia-cutlass-dsl-internal`), so on internal drops the ==4.5.2 WAR gate
is always OFF and the shim's <4.5.2 warning silently skips (dist lookup
misses) — both are the right behavior as long as internal drops carry the
4.5.3+ codegen fix, which this one does.

## The knob system (`shim/tuner.py`, `shim/autotune.py`)

- `tuner.py` mirrors the kernel team's `tester/solvers/inference_solver.py`
  knob taxonomy (re-audit on every drop).  Correctness knobs change a code
  path/output (`token_back_mode`, `mma_tiler_mnk`, `in_kernel_fc2_reduce`,
  ...); perf knobs are output-invariant (`group_hint`, `flag_batch`,
  `epi_flag_batch`).
- `default_knobs(num_tokens, dtype=...)` — the measured per-size profiles
  (provenance in the profile dict comments).  NVFP4 has FOUR profiles
  (re-validated 2026-07-15 on the corrected weight layout: the online
  autotuner confirmed all four non-ikr NVFP4 defaults within run noise;
  only ikr candidates beat them, -4..5% at >=2048, kept opt-in for
  determinism); the dominant axis is `token_back_mode`:
  - `epi_warps` wins at small batch but falls off a cliff mid-range
    (measured pre-fix as +18% at 512 tokens / +35% at 1024 with every
    dispatch-warp candidate ahead; the 07-15 re-validation kept the same
    per-bucket choices).
  - tile (256x128 vs 256x256) and `flag_batch` are second-order (~1-5%).
  MXFP8 has TWO profiles (re-derived 2026-07-15): fb4 + epi_warps below
  2048 tokens, fb4 + reuse_dispatch_warps at >=2048 (-14.5% at 2048:
  1010.6 vs 1181.6 us kernel-mode; supersedes the 07-14 "dispatch-warp is
  ~5% slower for MXFP8" reading, which conflated it with fb8).
- Backend configs (`Nvfp4/Mxfp8CutedslMegaMoeConfig.knobs`): explicit dict
  overrides the heuristic ENTIRELY (pin every knob you care about);
  `"auto"` runs the online autotuner at the first forward.
- `autotune.py` — collective online tuner: every EP rank compiles+times the
  same candidate list in lockstep, per-candidate medians are all-reduced
  MAX (slowest rank = collective latency), argmin winner applied
  identically everywhere.  Cost: one `cute.compile` per candidate
  (~1-2 min), once per session.  Candidates mirror the tester sweep
  restriction; for NVFP4 that now INCLUDES `in_kernel_fc2_reduce`
  (24 candidates — the symm buffer's output is always sym-heap allocated,
  so the knob flips per-compile).  ikr is ~par with the bf16 wire at the
  FI default geometry at >=1024 tok/rank and slower at small batch — see
  "Measured results" below; it stays a sweep candidate rather than a
  default because the tuner keeps it only if it wins the live problem.  An ikr winner makes the output accumulation
  order nondeterministic — pin `in_kernel_fc2_reduce=False` via explicit
  knobs if bit-reproducibility matters.  MXFP8 keeps ikr config-owned
  (its kernel rejects ikr + dispatch-warp token-back).
- The kernel-repo tester remains the wide-sweep tool
  (`torchrun -m tester.tester --mode Perf --sweep --use_knob ...`); winners
  transfer via the `knobs=` dict.  Its problems (`nvfp4_perf.jsonl`) are
  top-6 with different inter/experts — never compare its numbers to
  FI-default-geometry runs directly.

## How a forward gets its knobs (runtime flow)

Knobs are **compile-time** kernel parameters, resolved **once per session**
at workspace allocation and keyed on the buffer capacity `num_max_tokens` —
never on the runtime token count.  `knobs=None` resolves through the
persistent knob cache (`shim/knob_cache.py`, offline-tuned winners keyed on
device/dtype/world-size/geometry/wire/token-bucket; path via
`FLASHINFER_MOE_EP_KNOB_CACHE`) before falling back to the built-in
`default_knobs` heuristic — a pure dict lookup, no compiles or collectives.
Populate it with the offline CLI (`python -m flashinfer.moe_ep.tune`) or a
one-off `knobs="auto"` run outside the engine.  There is no per-shape
tuning and no shape→kernel dispatch.  Explicitly, what does NOT happen when
a shape comes in:

- **No kernel lookup by shape.**  The frontend holds exactly ONE compiled
  kernel (single-slot cache: `_mega` + `_mega_key` in `shim/nvfp4.py`),
  compiled for `num_max_tokens`.  Every incoming token count launches that
  same kernel and slices the padded buffer `[:num_tokens]`.  The per-launch
  "cache" (`_launch_cache_key`: raw data_ptrs + token count + stream) only
  decides whether validation + cute-view rebuild can be skipped — it never
  selects among kernels.  Consequence: a session sized for 2048 max tokens
  runs the throughput profile even for an 8-token decode step; size the
  buffer to the workload (or pin `knobs=`) if that matters.
- **No in-engine tuning.**  Persistence is write-once/offline: rank 0
  records `"auto"` winners into the knob cache, and later `knobs=None`
  sessions look them up — but nothing tunes inside a serving process
  (the ~24 `cute.compile`s + collective barriers of a live sweep are why
  `"auto"` is unusable in-engine).  `flashinfer/autotuner.py` /
  `FLASHINFER_AUTOTUNER_LOAD_FROM_FILE` are NOT wired into this path.
- The single slot is deliberate: each `_CompiledMega` pins symmetric-heap
  (NVSHMEM) workspaces, so keeping N candidates alive would multiply
  symmetric memory.  `_mega_compile_key()` already covers every knob, so a
  per-key dict (e.g. small+large profile selected per-launch by token
  count) is the natural upgrade if per-shape dispatch is ever needed;
  today the deployment pattern is per-role knob caches instead (one
  prefill-tuned and one decode-tuned cache file, selected per engine role
  via `FLASHINFER_MOE_EP_KNOB_CACHE` — validated in the vLLM e2e runs).

### Session setup + steady-state forward (`knobs=None` / explicit dict)

```mermaid
sequenceDiagram
    autonumber
    participant L as MoEEpMegaLayer<br/>(modes/mega_layer)
    participant B as Backend<br/>(nvfp4_cutedsl)
    participant S as get_symm_buffer_for_mega_moe<br/>(shim)
    participant T as tuner.py
    participant F as Frontend<br/>(shim nvfp4/mxfp8)
    participant C as cute.compile

    Note over L,C: session setup — knobs bind HERE, once,<br/>keyed on num_max_tokens
    L->>B: _allocate_workspace(fleet_params)
    B->>S: get_symm_buffer_for_mega_moe(..., ikr, combine_dtype, knobs)
    alt knobs is an explicit dict
        S->>T: with_knobs(cfg, knobs) — overrides heuristic ENTIRELY
    else knobs is None (how the backend's auto mode arrives too)
        S->>T: knob-cache lookup (knob_cache.lookup_knobs),<br/>else with_knobs(cfg, default_knobs(num_max_tokens, dtype))
        Note over S,T: a quantized combine_dtype auto-adjusts the default<br/>token_back_mode to reuse_dispatch_warps
    end
    Note over S: in_kernel_fc2_reduce / combine_dtype are config<br/>params (not knobs) — output_activation always sym-heap
    S-->>B: MegaMoESymmBuffer (frontend + staging tensors)

    Note over L,C: every forward (any num_tokens up to capacity)
    L->>B: stage_inputs() then compute()
    B->>F: nvfp4_mega_moe(...) -> frontend.run()
    alt compile-key miss (first launch / config changed)
        F->>C: compile Sm100MegaMoEKernel @ num_max_tokens
        C-->>F: _CompiledMega (single slot)
        F->>F: validate + build cute views, cache launch entry
    else launch-cache hit (same ptrs + num_tokens + stream)
        Note over F: pointer-tuple compare only —<br/>skip validation + view rebuild
    end
    F->>F: ikr only: enqueue output_activation.zero_()<br/>(accumulate-from-zero contract)
    F->>F: enqueue kernel (sync=False), copy y[:num_tokens]
```

### `knobs="auto"`: collective online sweep at the FIRST forward

The backend defers tuning to the first `compute()` (weights + staged inputs
exist there); the symm buffer is built with the heuristic default in the
meantime.  The default candidate list is session-aware
(`nvfp4_candidates(combine_format, allow_in_kernel_fc2_reduce)`: 24 for the
bf16 wire including the ikr axis; quantized wires prune to the valid
subset).  Candidates are compiled **serially and destructively** — each
`apply_knobs` frees the previous candidate's sym workspace and nulls the
compiled slot, so nothing accumulates and the winner is recompiled once
more after the sweep (unless it happened to be timed last).

```mermaid
sequenceDiagram
    autonumber
    participant B as Backend.compute()<br/>(first forward only)
    participant A as autotune_knobs<br/>(shim/autotune.py)
    participant F as Frontend
    participant C as cute.compile
    participant D as torch.distributed
    participant G as GPU

    B->>A: autotune_nvfp4_mega_moe(y, weights, symm_buffer)
    loop each candidate — same list + order on EVERY rank
        A->>F: apply_knobs(candidate)
        Note over F: frees sym workspace,<br/>nulls the compiled slot
        A->>D: barrier()
        A->>F: warmup launches (sync=True)
        F->>C: recompile kernel for this candidate
        F->>G: warmup x3
        A->>D: barrier()
        A->>G: timed launches x10 -> median
        Note over A: ctor/compile failure -> score inf<br/>(deterministic, ranks stay aligned)
        A->>D: barrier()
    end
    A->>D: all_reduce(scores, MAX) — slowest rank = collective latency
    A->>A: winner = argmin(scores)
    A->>F: apply_knobs(winner)
    Note over F: slot nulled AGAIN — the winner's<br/>sweep compile is not reused
    A-->>B: winner knob dict (rank 0 records it in the knob cache)
    B->>F: nvfp4_mega_moe(...) — the real forward
    F->>C: compile winner (second time, unless timed last)
    F->>G: launch
    Note over B,G: all later forwards: fast-path launch of this ONE<br/>kernel — no lookup, no retune, until the process exits
```

## Launch-path work (why FI now matches the bare kernel)

Steady-state cost of the full FI forward over the bare kernel launch is now
**2-13 µs** (mostly the output copy at large tokens).  What it took:

1. **Launch-kwargs cache** (`_CompiledMega.launch_*`): `frontend.run()`
   used to rebuild all 12 cute tensor views (`from_dlpack`) + a
   `SymBufferHost` on EVERY launch.  Now keyed on input data_ptrs + token
   count + stream; a hit also skips input validation (validated when the
   entry was built).  Any config change nulls the cache.
2. **No per-launch workspace reset**: workspaces are allocated zeroed and
   the kernel TAIL-CLEANS its own counters/flags (the kernel-team drivers
   and tester never host-reset; the NVLink phase slots must NOT be reset).
   `run(reset_counters=False)` is the default; `True` only recovers an
   aborted launch.  Guarded by repeated-forward bit-exact tests.
3. **Async wrappers**: `nvfp4/mxfp8_mega_moe(sync=False)` default — kernel
   + output copy are stream-enqueued, no host sync (matches dg).  Anything
   timing with `perf_counter` must pass `sync=True` (the autotuner does).
4. **Launch thunks** (`{nvfp4,mxfp8}_mega_launch_thunk`): prebuilt bare
   launch closures matching the tester's `perf_run` timed region — for
   benchmark loops, tuners, and (future) CUDA-graph capture.

## Benchmarking lessons (how the 2x-slower misread happened)

Use `moe_ep_benchmark` (`RUNBOOK.md` there) with `MEGA_TIMING`:

- `kernel` — tester-parity bare launch (back-to-back, per-iter events,
  L2 flush outside the window).  Compare THIS against the kernel team's
  tester numbers, at MATCHED geometry.
- `e2e_pipelined` — full FI forward, back-to-back.  The serving-relevant
  number.  Caveat: the bench prestages inputs, so the per-forward
  activation-quantize/staging cost is NOT in the timed region (true for
  every backend, so rankings hold; absolute layer time in a real serving
  stack is higher for all columns).
- `e2e` — full FI forward, each iter from a global barrier + idle GPU.
  A cold-start stress case: it adds ~85-90 µs of **from-idle collective
  start skew** (the collective dispatch stalls on the slowest rank's
  from-idle launch; dg shows the same effect at ~34 µs).  A pipelined
  workload never pays this; if a workload truly launches from global idle,
  CUDA-graph capture of the thunk is the fix.

The original misread combined (a) barrier-cold e2e timing of a then-heavy
launch path against dg's thin wrapper, and (b) a geometry mismatch vs the
tester problems.  Rule: match BOTH the problem shape and the timed region
before comparing MoE backends.

## TRT-LLM-import knobs (landed)

Imported from TRT-LLM PR #16190, both idea families are
plumbed through `get_symm_buffer_for_mega_moe` and the backend configs
(`Nvfp4CutedslMegaMoeConfig`):

- `in_kernel_fc2_reduce` — in-flight top-k combine via cross-rank REDG
  atomic-add; its main win is that the multi-GB per-topk combine staging
  disappears from `shared_workspace` (latency is geometry-dependent — see
  "Measured results").  Contract:
  `output_activation` lives on the sym heap (now unconditional) and is
  zeroed before EVERY launch by `frontend.run()` / the launch thunk
  (accumulate-from-zero); output accumulation order is nondeterministic AND
  the K terms accumulate in bf16 (vs fp32 in the explicit reduce), so where
  large terms nearly cancel the agreement is bounded by the bf16 ULP of the
  largest TERM — validate with a row-scaled band (tests'
  `_assert_ikr_close`: K x 2^-8 x safety 8 x row max), never a flat
  atol/rtol.  Requires `apply_topk_in_fc1=True` + bf16 combine wire.
- `combine_dtype` — quantized cross-rank combine wire (`"mxfp8"` =
  `32e4m3xe8m0`, 2x less NVLink combine traffic; `"nvfp4"` = `16e2m1xbf16`,
  4x less).  Numerics tradeoff (wire-quantizes the per-topk fc2 outputs);
  requires the explicit-reduce path with dispatch-warp token-back (the
  default knobs are adjusted automatically; explicit knobs must comply).

### What the variants mean (all share identical dispatch + fp4xfp4 compute)

The MoE forward has three legs: dispatch (tokens to experts), compute (the
two expert GEMMs), and combine (fc2 partials back to each token's home
rank + top-k sum).  The knobs only change the COMBINE leg, and the public
output is 2D `(T, hidden)` **bf16 in every variant**:

- **bf16 wire (default)** — partials travel as bf16, staged per-topk on
  the home rank, summed in fp32, cast once to bf16.  Exact terms, fp32
  sum, deterministic.
- **`+combine_nvfp4` / `+combine_mxfp8`** — the fc2 epilogue quantizes
  each partial IN REGISTERS to fp4 (e2m1 + bf16 scale per 16) or fp8
  (e4m3 + e8m0 scale per 32) *just for the wire* (4x / 2x fewer combine
  bytes); the home rank dequantizes and still sums in fp32.  Lossy terms
  (one quant round-trip per partial), fp32 sum, deterministic.
- **`+ikr`** (`in_kernel_fc2_reduce`) — no staging: each bf16 partial is
  REDG-atomic-added into the output as it arrives.  Exact terms, bf16
  unordered sum, nondeterministic; deletes the multi-GB staging region.

### Measured results

The full per-variant table lives in the microbenchmark section above (one
table, 8-8192 tok/rank, all combine-leg variants); accuracy in the table
right below it.

Takeaways at this (single-node NVLink) geometry, from the 2026-07-21 sweep:
- Small batch (8-64 tok/rank): plain nvfp4 is dg-parity (219.5/283.6 µs vs
  dg 212.5/284.7) and every variant is slightly slower than plain there —
  ikr +14/+29 µs, the quantized wires +7/+19 µs — so the bf16 wire is the
  small-batch default.
- `combine_nvfp4` is the throughput winner from 512 tok/rank up (-12..-14%
  vs the bf16 wire at 1024-8192; 1.78x vs dg and ~19.5 Mtok/s at 8192),
  buying that for ~1.8pt of extra quantization loss (25.0% vs 23.2%
  rel-L2).  Expect larger wins multi-node where combine bytes dominate.
- `combine_mxfp8` sits between the bf16 and fp4 wires up to 4096 and ties
  the fp4 wire at 8192, at ~0.1pt extra loss.
- `+ikr` is ~par with the bf16 wire at >=1024 with the default profiles
  and slower at small batch; it stays an autotune candidate rather than a
  default, and in the vLLM e2e runs at DSV4-Flash geometry it lost both
  phases — its value is geometry-dependent (and it deletes the multi-GB
  combine staging region, which can be the point).  An ikr winner makes
  the output accumulation order nondeterministic — pin
  `in_kernel_fc2_reduce=False` if bit-reproducibility matters.
- The quantized wires also did NOT transfer to the vLLM e2e geometry
  (4096 hidden / top-6: the wire-forced dispatch-warp token-back costs
  more than the combine-traffic saving there) — the bf16 wire is the e2e
  production default at DSV4; re-check the wires at higher
  combine-traffic geometries.

### Sweep methodology + environment (reproduce recipe)

**Hardware / software.**  One GB200 node, 4x NVIDIA GB200 (sm_100, cc
10.0) over NVLink, driver 580.173.02.  Pyxis container image
`flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh` (NGC 26.05 base):
torch `2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2, `cuda-bindings 13.2.0`,
`triton 3.7.0+nv26.5`, `deep_gemm 2.5.0+891d57b`,
`nvshmem4py-cu13 0.3.1`; `nvidia-cutlass-dsl` upgraded in-container to
**4.6.1** (>=4.5.2 suffices since the 2026-07-22 MR!27 WAR; the reference
numbers were measured on 4.6.1 and 4.5.2 reproduces them within noise).
FlashInfer = this branch,
editable-installed inside the container
(`PIP_CONSTRAINT="" BUILD_NIXL_EP=0 pip install --no-build-isolation -e .`).

**Harness.**  `moe_ep_benchmark/bench_moe_ep_mega.py` via
`SECTION=fi_mega GPUS=4 SEQ_LENS="1024 2048 4096 8192" run_sweep.sh`
(see that repo's RUNBOOK.md), one `torch.multiprocessing.spawn` of 4 EP
ranks per (variant, token-count) point — every point pays a fresh
`cute.compile` (amortized by cute's on-disk cache).  Variants selected
with the bench env knobs `MEGA_IKR=1` / `MEGA_COMBINE_DTYPE=nvfp4|mxfp8`
(mapped onto `Nvfp4CutedslMegaMoeConfig`); knobs left at the default
per-size heuristic profiles (`tuner.default_knobs`), which the quantized
wires auto-adjust to dispatch-warp token-back.

**Problem.**  DeepSeek-V3-like: hidden 7168, inter 2048 (post-SwiGLU),
256 experts, top-8, EP=4, `gate_up_clamp` default.  Random bf16
activations quantized at staging; random (uneven) routing from
`torch.topk` over randn scores; weights random bf16 quantized by
`preprocess_mega_weights`.

**Timed region** (`MEGA_TIMING`):
- `kernel` — tester-parity: a prebuilt zero-arg launch thunk
  (`nvfp4_mega_launch_thunk`; for `+ikr` the thunk is the required
  `output.zero_()` + launch, so the contract cost is included), 20
  warmup + 50 timed iters enqueued back-to-back with no host sync,
  per-iter CUDA events, 300 MB L2 flush outside each event window.
- `e2e_pipelined` — the FI forward compute path (launch-cache/arg handling,
  kernel, output copy), same 20/50 back-to-back event scheme.  Inputs are
  prestaged once outside the loop, so per-forward activation
  quantize/staging is NOT included (for any backend).
Reported number = rank-0 median of the 50 iters (p50); min/max in the
CSVs.  `deep_gemm_mega` has no thunk API, so its "kernel" number loops
`compute()` (includes its thin FI wrapper).

### Runbook (rerun the sweep)

The harness lives at <https://github.com/mhoqueanik/moe_ep_benchmark>
(bench env-knob reference in its README.md; SLURM/container details in
its RUNBOOK.md).  `MEGA_IKR` / `MEGA_COMBINE_DTYPE` land in bench commit
`a15fb01`.  On a SLURM + pyxis cluster with the image above:

```bash
export ROOT=/path/to/workspace          # holds the image, the bench repo,
export REPO=$ROOT/flashinfer-moe_ep     # and the flashinfer checkout
srun -A <account> -p batch -N 1 --ntasks-per-node=1 --time=04:00:00 \
  --container-image="$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh" \
  --container-mounts="$ROOT:$ROOT" --container-workdir="$REPO" \
  bash -lc '
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e .
    python -m pip install --upgrade "nvidia-cutlass-dsl[cu13]"   # >= 4.5.2
    export SECTION=fi_mega GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3
    export SEQ_LENS="1024 2048 4096 8192"
    for MODE in kernel e2e_pipelined; do
      export MEGA_TIMING=$MODE
      MEGA_LIST="deep_gemm_mega nvfp4_cutedsl" \
        bash '"$ROOT"'/moe_ep_benchmark/run_sweep.sh              # baseline
      MEGA_LIST=nvfp4_cutedsl MEGA_IKR=1 \
        bash '"$ROOT"'/moe_ep_benchmark/run_sweep.sh              # +ikr
      MEGA_LIST=nvfp4_cutedsl MEGA_COMBINE_DTYPE=nvfp4 \
        bash '"$ROOT"'/moe_ep_benchmark/run_sweep.sh              # +combine_nvfp4
      MEGA_LIST=nvfp4_cutedsl MEGA_COMBINE_DTYPE=mxfp8 \
        bash '"$ROOT"'/moe_ep_benchmark/run_sweep.sh              # +combine_mxfp8
    done
  '
```

Each `run_sweep.sh` invocation writes its own timestamped CSV under
`moe_ep_benchmark/results/`; the variant is identified by the
`compute_kernel` column suffix (`+ikr`, `+combine_nvfp4`, ...).  Notes:
- The editable install lives in the container overlay — rerun it in
  every fresh container.  Verify `import flashinfer` resolves to the
  checkout, not the image copy.
- Every (variant, token-count) point pays a `cute.compile`, amortized by
  cute's on-disk cache; the 8-invocation sweep above completed in ~17 min
  warm (budget ~2 h cold).
- Geometry knobs (`HIDDEN INTER NUM_EXPERTS TOP_K GPUS`) and
  `MEGA_KNOBS` (explicit knob dict / `auto` online autotune) are
  documented in the bench README.

## Next levers

CUDA-graph capture of the launch thunk (landed — `MoEEpMegaLayer.warmup()`
+ capture guards, `tests/moe_ep/test_mega_cuda_graph*.py`; with ikr
the thunk is zero+launch, both graphable), streaming weight reload
(on demand — from TRT-LLM PR #16190: a load/reload lifecycle with
partial-group coverage tracking for serving-oriented live weight updates;
import only if the FI serving integration needs reload).
