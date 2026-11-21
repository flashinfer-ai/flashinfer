# Whole-kernel modeled SOL

`T_SOL = max(F_useful/P_nom, Q_weight_min/B_HBM, Q_directed_NIC_min/B_cut, other established resource bounds)`, with ideal overlap. `F_useful/rank=36*M*4096*2048`; each expert has 1536 weight tiles. Nominal peaks are 316.29312 TFLOP/s, 4.814304 TB/s HBM, 60 MiB L2 capacity and a declared 8x400 Gb/s directed node-cut envelope. Cold weights use `U/B`; the optimistic warm-cache necessary-byte bound uses `max(U-60MiB,0)/B`. Cold percentages are conditional on cold weights; timings are warm graph replays.

Unknown L2/NVLink/fixed-startup lower-bound contributions are conservatively omitted. A separately measured isolated two-visit software-service calibration is about 29 us in-kernel / 30.5 us by events; it is an engineering sensitivity, not an irreducible alpha for this fused kernel. These modeled bounds do not prove attainability. Own-format efficiency and the common compact8448 optimization target are both shown; FP8 expansion and padded GEMM work cannot raise the common target. `Attainment=100*SOL_us/kernel_us`. The 60% common-SOL optimization goal remains open.

<details>
<summary>All measured EP-side SOL comparisons (cold/hot pairs; us and %)</summary>

| Scope/request | Storage EP8/EP16 | EP | Kernel us | Own SOL us cold/hot | Own % cold/hot | Common8448 SOL us cold/hot | Common % cold/hot |
|---|---|---:|---:|---:|---:|---:|---:|
| per-rank/1 | fp8/fp8 ; fp8 | 8 | 65.489 | 31.364/18.296 | 47.89/27.94 | 16.172/3.104 | 24.69/4.74 |
| per-rank/1 | fp8/fp8 ; fp8 | 16 | 94.390 | 31.364/18.296 | 33.23/19.38 | 16.172/3.104 | 17.13/3.29 |
| per-rank/4 | fp8/fp8 ; fp8 | 8 | 168.004 | 125.455/112.387 | 74.67/66.90 | 64.688/51.620 | 38.50/30.73 |
| per-rank/4 | fp8/fp8 ; fp8 | 16 | 155.077 | 83.637/70.569 | 53.93/45.51 | 43.125/30.057 | 27.81/19.38 |
| per-rank/8 | fp8/fp8 ; fp8 | 8 | 221.898 | 167.274/154.205 | 75.38/69.49 | 86.250/73.182 | 38.87/32.98 |
| per-rank/8 | fp8/fp8 ; fp8 | 16 | 161.665 | 83.637/70.569 | 51.73/43.65 | 43.125/30.057 | 26.68/18.59 |
| per-rank/16 | fp8/fp8 ; fp8 | 8 | 224.582 | 167.274/154.205 | 74.48/68.66 | 86.250/73.182 | 38.40/32.59 |
| per-rank/16 | fp8/fp8 ; fp8 | 16 | 168.196 | 83.637/70.569 | 49.73/41.96 | 43.125/30.057 | 25.64/17.87 |
| per-rank/32 | fp8/fp8 ; fp8 | 8 | 230.549 | 167.274/154.205 | 72.55/66.89 | 86.250/73.182 | 37.41/31.74 |
| per-rank/32 | fp8/fp8 ; fp8 | 16 | 178.266 | 83.637/70.569 | 46.92/39.59 | 43.125/30.553 | 24.19/17.14 |
| per-rank/64 | fp8/fp8 ; fp8 | 8 | 249.792 | 167.274/154.205 | 66.97/61.73 | 86.250/73.182 | 34.53/29.30 |
| per-rank/64 | fp8/fp8 ; fp8 | 16 | 197.258 | 83.637/70.569 | 42.40/35.77 | 61.106/61.106 | 30.98/30.98 |
| total/16 | fp8/fp8 ; fp8 | 8 | 96.738 | 62.728/49.659 | 64.84/51.33 | 32.344/19.276 | 33.43/19.93 |
| total/16 | fp8/fp8 ; fp8 | 16 | 94.685 | 31.364/18.296 | 33.12/19.32 | 16.172/3.104 | 17.08/3.28 |
| total/64 | fp8/fp8 ; fp8 | 8 | 222.137 | 167.274/154.205 | 75.30/69.42 | 86.250/73.182 | 38.83/32.94 |
| total/64 | fp8/fp8 ; fp8 | 16 | 154.669 | 83.637/70.569 | 54.07/45.63 | 43.125/30.057 | 27.88/19.43 |
| total/128 | fp8/fp8 ; fp8 | 8 | 224.213 | 167.274/154.205 | 74.60/68.78 | 86.250/73.182 | 38.47/32.64 |
| total/128 | fp8/fp8 ; fp8 | 16 | 161.667 | 83.637/70.569 | 51.73/43.65 | 43.125/30.057 | 26.68/18.59 |
| total/256 | fp8/fp8 ; fp8 | 8 | 232.172 | 167.274/154.205 | 72.05/66.42 | 86.250/73.182 | 37.15/31.52 |
| total/256 | fp8/fp8 ; fp8 | 16 | 168.193 | 83.637/70.569 | 49.73/41.96 | 43.125/30.057 | 25.64/17.87 |
| total/512 | fp8/fp8 ; fp8 | 8 | 250.030 | 167.274/154.205 | 66.90/61.67 | 86.250/73.182 | 34.50/29.27 |
| total/512 | fp8/fp8 ; fp8 | 16 | 178.662 | 83.637/70.569 | 46.81/39.50 | 43.125/30.553 | 24.14/17.10 |
| total/1024 | fp8/fp8 ; fp8 | 8 | 277.131 | 167.274/154.205 | 60.36/55.64 | 122.212/122.212 | 44.10/44.10 |
| total/1024 | fp8/fp8 ; fp8 | 16 | 197.555 | 83.637/70.569 | 42.34/35.72 | 61.106/61.106 | 30.93/30.93 |
| per-rank/1 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 8 | 70.285 | 16.172/3.104 | 23.01/4.42 | 16.172/3.104 | 23.01/4.42 |
| per-rank/1 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 16 | 88.476 | 16.172/3.104 | 18.28/3.51 | 16.172/3.104 | 18.28/3.51 |
| per-rank/4 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 8 | 155.532 | 64.688/51.620 | 41.59/33.19 | 64.688/51.620 | 41.59/33.19 |
| per-rank/4 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 16 | 146.411 | 43.125/30.057 | 29.45/20.53 | 43.125/30.057 | 29.45/20.53 |
| per-rank/8 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 8 | 207.721 | 86.250/73.182 | 41.52/35.23 | 86.250/73.182 | 41.52/35.23 |
| per-rank/8 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 16 | 163.282 | 43.125/30.057 | 26.41/18.41 | 43.125/30.057 | 26.41/18.41 |
| total/16 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 8 | 95.392 | 32.344/19.276 | 33.91/20.21 | 32.344/19.276 | 33.91/20.21 |
| total/16 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 16 | 88.619 | 16.172/3.104 | 18.25/3.50 | 16.172/3.104 | 18.25/3.50 |
| total/64 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 8 | 207.769 | 86.250/73.182 | 41.51/35.22 | 86.250/73.182 | 41.51/35.22 |
| total/64 | mxfp4_compact/mxfp4_compact ; mxfp4_compact | 16 | 146.201 | 43.125/30.057 | 29.50/20.56 | 43.125/30.057 | 29.50/20.56 |
| per-rank/1 | fp8/mxfp4_compact ; fp8 | 8 | 65.858 | 31.364/18.296 | 47.62/27.78 | 16.172/3.104 | 24.56/4.71 |
| per-rank/1 | fp8/mxfp4_compact ; mxfp4_compact | 16 | 88.599 | 16.172/3.104 | 18.25/3.50 | 16.172/3.104 | 18.25/3.50 |
| per-rank/4 | fp8/mxfp4_compact ; fp8 | 8 | 167.853 | 125.455/112.387 | 74.74/66.96 | 64.688/51.620 | 38.54/30.75 |
| per-rank/4 | fp8/mxfp4_compact ; mxfp4_compact | 16 | 146.545 | 43.125/30.057 | 29.43/20.51 | 43.125/30.057 | 29.43/20.51 |
| per-rank/8 | fp8/mxfp4_compact ; fp8 | 8 | 221.770 | 167.274/154.205 | 75.43/69.53 | 86.250/73.182 | 38.89/33.00 |
| per-rank/8 | fp8/mxfp4_compact ; mxfp4_compact | 16 | 163.507 | 43.125/30.057 | 26.38/18.38 | 43.125/30.057 | 26.38/18.38 |
| total/16 | fp8/mxfp4_compact ; fp8 | 8 | 96.559 | 62.728/49.659 | 64.96/51.43 | 32.344/19.276 | 33.50/19.96 |
| total/16 | fp8/mxfp4_compact ; mxfp4_compact | 16 | 89.069 | 16.172/3.104 | 18.16/3.48 | 16.172/3.104 | 18.16/3.48 |
| total/64 | fp8/mxfp4_compact ; fp8 | 8 | 221.524 | 167.274/154.205 | 75.51/69.61 | 86.250/73.182 | 38.93/33.04 |
| total/64 | fp8/mxfp4_compact ; mxfp4_compact | 16 | 146.419 | 43.125/30.057 | 29.45/20.53 | 43.125/30.057 | 29.45/20.53 |
| total/128 | fp8/mxfp4_compact ; fp8 | 8 | 223.482 | 167.274/154.205 | 74.85/69.00 | 86.250/73.182 | 38.59/32.75 |
| total/128 | fp8/mxfp4_compact ; mxfp4_compact | 16 | 163.569 | 43.125/30.057 | 26.37/18.38 | 43.125/30.057 | 26.37/18.38 |

The 46 rows cover 12 FP8/FP8 pairs, 5 compact/compact pairs and 6 FP8/compact pairs. High-M compact plans are excluded from the executable matrix; their compact representation still defines the shared optimization target.

</details>
