# Recurrent KDA prefill: CuTe DSL vs Cake

Status: final reference for the measured source tree. Rerun after a code,
compiler, driver, benchmark-protocol, or hardware configuration change that can
affect recurrent-KDA performance.

## Provenance

- Measured source commit: `e9edc98be160a85c5a54dd96bbb14aa5254ccd61`
- Measured source tree: `ceca88abb7a62b957563aa2e16474a8bb52e5604`
- Branch base: `27a5a2945a2af3a4aaa0d1f659c6933d411bdfed`
- Benchmark: `benchmarks/bench_recurrent_kda_prefill.py --case-set all`
- Case count: 12 (the unchanged six legacy cases plus six H12 preset cases)
- Public API: `recurrent_kda`, with `--backend cake` or `--backend cute-dsl`
- Timing: CUPTI device time, cold L2, no CUDA Graph
- Warmup / measurement target: 20 ms / 100 ms
- Run order on each allocation: Cake A, CuTe DSL A, CuTe DSL B, Cake B
- Reported value: median of the two run medians
- Common tensors: BF16 Q/K/V/G/beta and state, K/V head dimension 128,
  in-kernel QK L2 normalization and gate, beta logits, lower bound -5.0
- B200: NVIDIA B200, CC 10.0
- B300: NVIDIA B300 SXM6 AC, CC 10.3

`Speedup` is Cake / CuTe DSL, so values above 1 mean CuTe DSL is faster.

## B200 results

| Case | Exact sequence shape | CuTe route | Cake A (us) | Cake B (us) | Cake (us) | CuTe A (us) | CuTe B (us) | CuTe (us) | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h96_fixed8192 | fixed `[1,8192,96,128]` | engine | 516.374 | 515.287 | 515.831 | 487.384 | 487.735 | 487.559 | 1.058x |
| h96_mixed | packed H96, lengths `[1300,547,2048,963,271,3063]` | engine | 382.585 | 382.137 | 382.361 | 359.322 | 359.226 | 359.274 | 1.064x |
| h96_uniform | packed H96, lengths `8x1024` | engine | 417.337 | 417.305 | 417.321 | 401.465 | 401.465 | 401.465 | 1.039x |
| h64_fixed8192 | fixed `[1,8192,64,128]` | decomp | 470.504 | 470.680 | 470.592 | 425.913 | 426.297 | 426.105 | 1.104x |
| h64_mixed | packed H64, lengths `[1300,547,2048,963,271,3063]` | engine | 295.947 | 296.091 | 296.019 | 253.852 | 254.075 | 253.964 | 1.166x |
| h64_uniform | packed H64, lengths `8x1024` | engine | 284.315 | 284.219 | 284.267 | 271.323 | 271.547 | 271.435 | 1.047x |
| h12_packed_512x32 | packed H12, lengths `32x512` | engine | 271.612 | 271.707 | 271.659 | 116.478 | 116.574 | 116.526 | 2.331x |
| h12_packed_128x8 | packed H12, lengths `8x128` | engine | 35.168 | 35.167 | 35.167 | 17.376 | 17.407 | 17.392 | 2.022x |
| h12_fixed_512 | fixed `[1,512,12,128]` | decomp | 83.710 | 83.678 | 83.694 | 35.679 | 35.680 | 35.680 | 2.346x |
| h12_fixed_8192 | fixed `[1,8192,12,128]` | decomp | 1084.365 | 1085.710 | 1085.037 | 291.515 | 291.707 | 291.611 | 3.721x |
| h12_packed_mixed | packed H12, lengths `[1300,547,2048,963,271,3063]` | decomp | 445.912 | 446.457 | 446.184 | 143.166 | 143.262 | 143.214 | 3.116x |
| h12_packed_1024x8 | packed H12, lengths `8x1024` | engine | 162.669 | 162.717 | 162.693 | 71.327 | 71.391 | 71.359 | 2.280x |

- CuTe DSL wins: 12 / 12
- Geometric-mean speedup: 1.667x

## B300 results

| Case | Exact sequence shape | CuTe route | Cake A (us) | Cake B (us) | Cake (us) | CuTe A (us) | CuTe B (us) | CuTe (us) | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h96_fixed8192 | fixed `[1,8192,96,128]` | engine | 480.945 | 481.568 | 481.256 | 465.089 | 464.993 | 465.041 | 1.035x |
| h96_mixed | packed H96, lengths `[1300,547,2048,963,271,3063]` | engine | 399.168 | 399.201 | 399.184 | 342.945 | 342.977 | 342.961 | 1.164x |
| h96_uniform | packed H96, lengths `8x1024` | engine | 417.153 | 417.314 | 417.234 | 384.129 | 384.257 | 384.193 | 1.086x |
| h64_fixed8192 | fixed `[1,8192,64,128]` | decomp | 442.561 | 442.465 | 442.513 | 401.793 | 401.793 | 401.793 | 1.101x |
| h64_mixed | packed H64, lengths `[1300,547,2048,963,271,3063]` | engine | 276.033 | 276.096 | 276.064 | 241.153 | 241.249 | 241.201 | 1.145x |
| h64_uniform | packed H64, lengths `8x1024` | engine | 280.608 | 280.513 | 280.561 | 257.376 | 257.345 | 257.361 | 1.090x |
| h12_packed_512x32 | packed H12, lengths `32x512` | engine | 256.033 | 255.969 | 256.001 | 108.865 | 108.865 | 108.865 | 2.352x |
| h12_packed_128x8 | packed H12, lengths `8x128` | engine | 32.832 | 32.704 | 32.768 | 16.448 | 16.416 | 16.432 | 1.994x |
| h12_fixed_512 | fixed `[1,512,12,128]` | decomp | 79.136 | 79.072 | 79.104 | 33.792 | 33.792 | 33.792 | 2.341x |
| h12_fixed_8192 | fixed `[1,8192,12,128]` | decomp | 1029.539 | 1029.251 | 1029.395 | 275.873 | 275.872 | 275.873 | 3.731x |
| h12_packed_mixed | packed H12, lengths `[1300,547,2048,963,271,3063]` | decomp | 399.841 | 399.809 | 399.825 | 135.616 | 135.616 | 135.616 | 2.948x |
| h12_packed_1024x8 | packed H12, lengths `8x1024` | engine | 148.256 | 148.096 | 148.176 | 66.368 | 66.368 | 66.368 | 2.233x |

- CuTe DSL wins: 12 / 12
- Geometric-mean speedup: 1.674x

The largest difference between paired run medians is 0.391%.
