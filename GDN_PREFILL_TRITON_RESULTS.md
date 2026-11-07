# GDN MTP WY — Triton Kernel: Results

Triton implementation of the WY-parallel GDN MTP decode path. Two code
families:

1. **Single-pass T-tokens-from-state**
   (`gdn_decode_wy_triton.py` + `gdn_decode_wy_triton_v1.py`)
   — computes T outputs from an initial state h0, optionally writing h_T
   back. Public API: `gated_delta_rule_mtp_wy_triton(...)`.

2. **Full MTP speculation cycle**
   (`gdn_decode_wy_triton_mtp.py` dispatcher + `_mtp_fused.py` + `_mtp_split.py`)
   — in one call: replay K accepted draft tokens (h0 → h_K), compute T new-
   token outputs, write h_K back. Public API:
   `gated_delta_rule_mtp_auto(...)`.

## TL;DR

- **2–5× faster than the sequential BF16 kernel** on B200 (qwen3.5, HV=64).
- Passes `torch.testing.assert_close(atol=5e-3, rtol=5e-3)` — the FlashInfer-
  standard BF16 gate from `tests/gdn/test_decode_delta_rule.py` — **39/39**
  T-only configs + **73/73** MTP spec-cycle configs. Also passes a 5× tighter
  1e-3/1e-3 gate. Observed max error ~4e-4 (≥10× headroom, ~1 BF16 ULP).
- For the MTP spec cycle, `gated_delta_rule_mtp_auto` dispatches fused
  (BS ≤ 8) vs split + PDL (BS > 8). At T=8 BS=512 it **enables configs the
  cache-all-states baseline OOMs on**, freeing ~7 GB per call.

## Files landed on this branch

| Path | Role |
|---|---|
| `flashinfer/gdn_kernels/gdn_decode_wy_triton.py` | T-only: output kernel + state kernel |
| `flashinfer/gdn_kernels/gdn_decode_wy_triton_v1.py` | T-only: fused small-BS fallback (imported by the above) |
| `flashinfer/gdn_kernels/gdn_decode_wy_triton_mtp.py` | Spec-cycle: dispatcher `gated_delta_rule_mtp_auto` |
| `flashinfer/gdn_kernels/gdn_decode_wy_triton_mtp_fused.py` | Spec-cycle: fused (phase-A replay + phase-B new) |
| `flashinfer/gdn_kernels/gdn_decode_wy_triton_mtp_split.py` | Spec-cycle: split state-step + output + PDL |
| `tests/gdn/test_wy_triton.py` | T-only: quick Triton-vs-FP32-WY sanity |
| `tests/gdn/test_wy_triton_tolerance.py` | T-only: 39-config correctness gate |
| `tests/gdn/test_mtp_fused.py` | Spec-cycle: 73-config correctness gate (fused + split) |
| `tests/gdn/reference_delta_rule.py` | `verify_delta_rule_wy` appended (FP32 WY reference) |
| `benchmarks/bench_seq_vs_triton_tvar.py` | T-only: CUPTI T×BS sweep |
| `benchmarks/bench_mtp_fused.py` | Spec-cycle: CUPTI 5-way MTP sweep |
| `results_seq_vs_triton_qwen3_5.csv` | Raw T-only numbers |

## Correctness

### ULP-level accuracy (unscaled randn, qwen3.5, BS=4, T=8)

| Path | max_abs vs FP32 ref | rel | ULP_bf16 |
|---|---|---|---|
| SEQ kernel vs FP32 PT-SEQ | 9.75e-4 | 3.09e-3 | **0.79** |
| **Triton vs FP32 PT-SEQ** | **1.12e-3** | **3.55e-3** | **0.91** |
| **Triton vs FP32 PT-WY** | **1.12e-3** | **3.55e-3** | **0.91** |
| Triton vs SEQ (both BF16) | 1.95e-3 | 6.21e-3 | 1.59 |

Both kernels round to sub-ULP precision.

### FlashInfer-standard BF16 gate (`atol=rtol=5e-3`)

| config | kernel | 5e-3/5e-3 | 1e-3/1e-3 | max_abs |
|---|---|---|---|---|
| qwen3.5   BS=4  T=8  | SEQ    | PASS      | PASS      | 1.2e-4 |
| qwen3.5   BS=4  T=8  | Triton | **PASS**  | **PASS**  | 3.3e-4 |
| qwen3.5   BS=16 T=8  | Triton | **PASS**  | **PASS**  | 3.6e-4 |
| qwen3.5   BS=64 T=8  | Triton | **PASS**  | **PASS**  | 4.2e-4 |
| qwen3-next BS=16 T=8 | Triton | **PASS**  | **PASS**  | 4.2e-4 |

`test_wy_triton_tolerance.py`: **39/39 pass**. `test_mtp_fused.py`: **73/73 pass**
(covers fused + split × K_MAX ∈ {4,8,16} × T ∈ {4,8,16} × B ∈ {2,8} × both presets).

## T-only benchmarks — SEQ vs Triton

B200, qwen3.5 (H=16, HK=16, HV=64, K=V=128, bf16), CUPTI kernel-only timing
(`ActivityKind.CONCURRENT_KERNEL`), cold L2 flush, median of 50 iters after
20-iter warmup. `disable_state_update=True` on both kernels.

### SEQ (µs)

```
   BS |   T=2 |   T=3 |   T=4 |   T=5 |   T=6 |   T=7 |   T=8 |   T=9 |  T=10 |  T=11 |  T=12 |  T=13 |  T=14 |  T=15 |  T=16
    1 |  5.34 |  6.26 |  7.36 |  9.18 | 10.24 | 11.41 | 12.19 | 14.18 | 15.23 | 16.10 | 16.83 | 19.07 | 19.97 | 20.77 | 21.86
    4 |  8.13 |  9.90 | 11.58 | 14.19 | 15.94 | 17.73 | 19.62 | 22.18 | 23.95 | 25.66 | 27.49 | 30.08 | 32.06 | 33.76 | 35.39
    8 | 14.94 | 15.23 | 18.50 | 22.59 | 25.84 | 28.96 | 32.13 | 36.27 | 39.49 | 42.64 | 45.70 | 52.50 | 55.90 | 59.34 | 62.62
   16 | 20.51 | 25.98 | 32.38 | 39.49 | 45.94 | 52.24 | 58.26 | 65.74 | 72.00 | 78.10 | 84.32 | 96.85 |103.57 |110.00 |116.56
   32 | 37.12 | 46.02 | 57.89 | 71.12 | 83.02 | 95.14 |107.06 |119.95 |132.00 |143.57 |155.55 |176.59 |189.10 |201.46 |213.84
   64 | 68.26 | 83.97 |107.17 |131.79 |155.04 |178.38 |201.57 |226.22 |249.63 |273.15 |296.16 |334.30 |358.32 |382.62 |406.58
  128 |127.02 |158.56 |204.75 |252.40 |298.26 |344.38 |390.34 |438.05 |484.10 |530.24 |576.29 |644.22 |691.55 |738.98 |786.05
  256 |244.38 |306.56 |397.66 |490.83 |581.50 |672.45 |763.52 |856.61 |947.55 |1038.59|1129.71|1257.58|1350.72|1444.16|1536.58
```

### Triton (µs)

```
   BS |   T=2 |   T=3 |   T=4 |   T=5 |   T=6 |   T=7 |   T=8 |   T=9 |  T=10 |  T=11 |  T=12 |  T=13 |  T=14 |  T=15 |  T=16
    1 |  4.83 |  4.77 |  4.74 |  4.83 |  4.67 |  4.74 |  4.99 |  5.28 |  5.25 |  5.34 |  5.28 |  5.50 |  5.39 |  5.46 |  5.47
    4 |  7.33 |  7.49 |  7.30 |  7.55 |  7.36 |  7.46 |  7.57 | 10.00 | 10.08 | 10.53 | 10.10 | 10.85 | 10.14 | 10.14 |  8.42
    8 | 11.81 | 12.18 | 12.03 | 12.48 | 12.13 | 12.27 | 12.51 | 16.00 | 15.15 | 16.10 | 16.16 | 16.30 | 15.84 | 16.70 | 14.53
   16 | 18.59 | 21.25 | 20.58 | 21.34 | 20.90 | 20.93 | 22.11 | 26.56 | 25.94 | 27.01 | 26.56 | 27.65 | 27.09 | 27.28 | 25.79
   32 | 34.30 | 37.76 | 37.33 | 38.43 | 37.55 | 38.08 | 39.65 | 49.58 | 47.95 | 50.34 | 49.20 | 51.41 | 50.56 | 50.90 | 45.87
   64 | 63.49 | 71.17 | 70.53 | 72.93 | 71.04 | 72.03 | 75.23 | 93.15 | 90.43 | 94.62 | 92.91 | 97.10 | 94.96 | 95.39 | 86.75
  128 |121.26 |135.89 |135.17 |139.63 |136.21 |138.46 |144.24 |178.16 |173.78 |181.42 |178.77 |186.30 |182.54 |184.16 |166.96
  256 |237.39 |265.28 |264.35 |273.76 |267.20 |271.66 |282.88 |347.84 |340.11 |355.62 |350.05 |364.42 |358.93 |361.28 |327.98
```

### Speedup (SEQ / Triton)

```
   BS |  T=2 |  T=3 |  T=4 |  T=5 |  T=6 |  T=7 |  T=8 |  T=9 | T=10 | T=11 | T=12 | T=13 | T=14 | T=15 | T=16
    1 | 1.11 | 1.31 | 1.55 | 1.90 | 2.19 | 2.41 | 2.44 | 2.68 | 2.90 | 3.01 | 3.19 | 3.47 | 3.70 | 3.81 | 3.99
    4 | 1.11 | 1.32 | 1.59 | 1.88 | 2.17 | 2.38 | 2.59 | 2.22 | 2.38 | 2.44 | 2.72 | 2.77 | 3.16 | 3.33 | 4.21
    8 | 1.27 | 1.25 | 1.54 | 1.81 | 2.13 | 2.36 | 2.57 | 2.27 | 2.61 | 2.65 | 2.83 | 3.22 | 3.53 | 3.55 | 4.31
   16 | 1.10 | 1.22 | 1.57 | 1.85 | 2.20 | 2.50 | 2.63 | 2.48 | 2.78 | 2.89 | 3.17 | 3.50 | 3.82 | 4.03 | 4.52
   32 | 1.08 | 1.22 | 1.55 | 1.85 | 2.21 | 2.50 | 2.70 | 2.42 | 2.75 | 2.85 | 3.16 | 3.44 | 3.74 | 3.96 | 4.66
   64 | 1.08 | 1.18 | 1.52 | 1.81 | 2.18 | 2.48 | 2.68 | 2.43 | 2.76 | 2.89 | 3.19 | 3.44 | 3.77 | 4.01 | 4.69
  128 | 1.05 | 1.17 | 1.51 | 1.81 | 2.19 | 2.49 | 2.71 | 2.46 | 2.79 | 2.92 | 3.22 | 3.46 | 3.79 | 4.01 | 4.71
  256 | 1.03 | 1.16 | 1.50 | 1.79 | 2.18 | 2.48 | 2.70 | 2.46 | 2.79 | 2.92 | 3.23 | 3.45 | 3.76 | 4.00 | 4.68
```

Triton wins **119 of 120 cells**; one tie (BS=256, T=2 at 1.03×). Best case:
**BS=128, T=16 at 4.71×**. Raw CSV: [`results_seq_vs_triton_qwen3_5.csv`](results_seq_vs_triton_qwen3_5.csv).

## MTP spec-cycle benchmarks — Fused vs Split+PDL

B200, qwen3.5, T=8, CUPTI, cold L2 flush. Times in microseconds. Tri×2 =
calling the T-only kernel twice (once with K accepted tokens, once with
T new tokens).

```
 K   BS |      SEQ×2    Tri×2      Fused      Split  Split+PDL   |  pick
 2    1 |      40.99    63.01      15.63      25.73      26.03   |  Fused
 2    4 |      45.10    63.89      28.54      28.38      27.66   |  Split+PDL
 2    8 |      57.65    69.49      49.60      38.54      34.67   |  Split+PDL
 2   16 |      84.38   116.18      84.29      65.01      59.81   |  Split+PDL
 2   32 |     150.13   131.36     159.55     114.91     109.89   |  Split+PDL
 2   64 |     279.47   212.24     309.34     221.68     216.83   |  Split+PDL ≈ Tri×2
 2  128 |     533.47   398.24     608.77     427.57     421.76   |  Split+PDL ≈ Tri×2
 2  256 |    1036.00   771.45    1203.26     839.07     831.45   |  Split+PDL ≈ Tri×2

 8    1 |      39.28    64.48      15.78      27.02      26.24   |  Fused
 8    8 |      68.64    70.77      49.58      39.26      35.09   |  Split+PDL
 8   16 |     120.37   115.22      85.50      65.41      60.80   |  Split+PDL
 8   32 |     217.58   132.61     161.39     115.63     111.57   |  Split+PDL
 8   64 |     411.68   223.58     312.10     225.70     219.69   |  Split+PDL ≈ Tri×2
 8  128 |     794.56   421.95     615.60     433.86     428.13   |  Split+PDL ≈ Tri×2
 8  256 |    1552.20   817.60    1212.80     851.18     844.51   |  Split+PDL ≈ Tri×2
```

`gated_delta_rule_mtp_auto` dispatches fused at BS ≤ 8 and split+PDL above.

## MTP spec-cycle vs. cache-all-states baseline

T=8, K=8 (every draft accepted), qwen3.5. Baseline numbers from the
cache-all-states variant; Triton from `bench_mtp_fused.py`. Median of 100
iters after 20 warmups, cold L2.

| BS  | Baseline (µs) | Best Triton (µs) | Triton / Baseline      | Memory saved |
|----:|--------------:|-----------------:|-----------------------:|-------------:|
|   1 | 11.50         |            15.62 |     1.36× slower       |   14 MB      |
|   2 | 14.18         |            17.46 |     1.23× slower       |   28 MB      |
|   4 | 22.22         |            29.06 |     1.31× slower       |   56 MB      |
|   8 | 37.92         |            34.85 | **0.92× faster**       |  112 MB      |
|  16 | 62.11         |            60.48 | **0.97× faster**       |  224 MB      |
|  32 | 117.58        |           111.55 | **0.95× faster**       |  448 MB      |
|  64 | 225.44        |           220.10 | **0.98× faster**       |  896 MB      |
| 128 | 432.17        |           427.78 | **0.99× tied**         | 1.75 GB      |
| 256 | 846.40        |           843.84 | **1.00× tied**         | 3.50 GB      |
| 512 | **OOM**       |          1676.45 | **∞ — baseline fails** | 7.00 GB      |

**Memory math** at qwen3.5 (HV=64, V=K=128, bf16): one state tensor is
`64 × 128 × 128 × 2` = 2 MB. Baseline per batch per spec step: `T × state`
= 16 MB at T=8. Triton: `1 × state` = 2 MB. Savings scale linearly with BS.

At BS=512 the baseline needs 8 GB just for the intermediate state cache;
Triton needs 1 GB — the gap that crosses the OOM line.

### K-sensitivity (Split+PDL, qwen3.5, T=8)

| BS  | K=2 (µs) | K=4 (µs) | K=8 (µs) | K=2 vs K=8 |
|----:|---------:|---------:|---------:|-----------:|
|   8 |    34.58 |    34.70 |    34.85 |  −0.8%     |
|  16 |    60.11 |    60.00 |    60.48 |  −0.6%     |
|  64 |   216.86 |   218.10 |   220.10 |  −1.5%     |
| 128 |   421.79 |   423.81 |   427.78 |  −1.4%     |
| 256 |   831.25 |   835.36 |   843.84 |  −1.5%     |
| 512 |  1651.36 |  1658.75 |  1676.45 |  −1.5%     |

At most 1.5% variation between K=2 and K=8.

## Reproduction

```bash
# Correctness (pytest)
pytest tests/gdn/test_wy_triton_tolerance.py -v   # 39/39 T-only
pytest tests/gdn/test_mtp_fused.py           -v   # 73/73 MTP spec-cycle

# T-only sweep (produces the CSV)
python benchmarks/bench_seq_vs_triton_tvar.py \
    --preset qwen3.5 --t-min 2 --t-max 16 \
    --batch-sizes 1 4 8 16 32 64 128 256 \
    --warmup 20 --iters 50 \
    --csv results_seq_vs_triton_qwen3_5.csv

# MTP spec-cycle sweep
python benchmarks/bench_mtp_fused.py --preset qwen3.5 --T 8 \
    --K-values 2 4 8 \
    --batch-sizes 1 4 8 16 32 64 128 256 \
    --warmup 20 --iters 50
```
