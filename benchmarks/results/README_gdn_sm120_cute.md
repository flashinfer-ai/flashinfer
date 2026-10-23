# RTX6000D public-call results

Measured commit: `3cd6473fd4a30e117982fd193384bac3517e544b`.
46 cases, 27600 CUPTI cold-L2 samples; 24 faster, 18 slower, 4 within 2% of official default. Maximum output/final-state relative L2: 0.00538853.

Both calls use the same native FP32 log gates, matching CPU offsets, BF16 normalized QKV and nonzero FP32 initial state. Preallocated output/state buffers are supplied to both; experimental buffer copies and official gate materialization are included. CPU overhead/JIT and input generation are excluded. TP describes head shapes, without communication. All experimental matrix calls were traced to the CuTe backend, not a fallback.

Baseline: public `chunk_gated_delta_rule(backend="flashinfer", use_cp="auto")`; candidate: same public call with `backend="sm120_cute"`. This is not a comparison against the minimum of forced CP/fused timings. Linear/raw-gate calls and checkpoint/state-pool calls use the official fallback and are not claimed as optimized.

| TP | Batch | Per-sequence lengths | Official route | Official ms | Experimental ms | Speedup |
|---|---:|---|---|---:|---:|---:|
| 1 | 1 | 2048 | cp | 0.275247 | 0.202703 | 1.358x |
| 1 | 1 | 4096 | cp | 0.403823 | 0.324862 | 1.243x |
| 1 | 1 | 8192 | cp | 0.792909 | 0.683598 | 1.160x |
| 1 | 1 | 16384 | cp | 1.500139 | 1.336747 | 1.122x |
| 1 | 1 | 32768 | cp | 3.848770 | 3.173749 | 1.213x |
| 1 | 2 | 2048,2048 | fused | 0.218128 | 0.298591 | 0.731x |
| 1 | 2 | 4096,4096 | fused | 0.401167 | 0.598878 | 0.670x |
| 1 | 2 | 8192,8192 | fused | 1.116604 | 1.174124 | 0.951x |
| 1 | 2 | 16384,16384 | fused | 2.878584 | 2.717752 | 1.059x |
| 1 | 2 | 32768,32768 | fused | 5.986354 | 5.638323 | 1.062x |
| 1 | 4 | 2048,2048,2048,2048 | fused | 0.416799 | 0.617679 | 0.675x |
| 1 | 4 | 4096,4096,4096,4096 | fused | 0.764142 | 1.161901 | 0.658x |
| 1 | 4 | 8192,8192,8192,8192 | fused | 2.186619 | 2.294747 | 0.953x |
| 1 | 4 | 16384,16384,16384,16384 | fused | 5.496085 | 4.577398 | 1.201x |
| 1 | 4 | 32768,32768,32768,32768 | fused | 10.007934 | 9.178403 | 1.090x |
| 1 | 8 | 2048,2048,2048,2048,2048,2048,2048,2048 | fused | 0.712399 | 1.098910 | 0.648x |
| 1 | 8 | 4096,4096,4096,4096,4096,4096,4096,4096 | fused | 1.912877 | 2.140541 | 0.894x |
| 1 | 8 | 8192,8192,8192,8192,8192,8192,8192,8192 | fused | 4.280298 | 4.232108 | 1.011x |
| 1 | 8 | 16384,16384,16384,16384,16384,16384,16384,16384 | fused | 8.665433 | 8.498442 | 1.020x |
| 1 | 8 | 32768,32768,32768,32768,32768,32768,32768,32768 | fused | 16.506999 | 21.770727 | 0.758x |
| 1 | 2 | 2048,4096 | fused | 0.412832 | 0.498704 | 0.828x |
| 1 | 4 | 8192,2048,4096,1024 | fused | 1.175888 | 1.128096 | 1.042x |
| 1 | 2 | 2049,4097 | fused | 0.420512 | 0.506000 | 0.831x |
| 2 | 1 | 2048 | cp | 0.191344 | 0.214895 | 0.890x |
| 2 | 1 | 4096 | cp | 0.284688 | 0.263392 | 1.081x |
| 2 | 1 | 8192 | cp | 0.418768 | 0.412207 | 1.016x |
| 2 | 1 | 16384 | cp | 0.826607 | 0.791646 | 1.044x |
| 2 | 1 | 32768 | cp | 1.481903 | 1.542159 | 0.961x |
| 2 | 2 | 2048,2048 | cp | 0.277296 | 0.203360 | 1.364x |
| 2 | 2 | 4096,4096 | cp | 0.402096 | 0.329792 | 1.219x |
| 2 | 2 | 8192,8192 | cp | 0.781936 | 0.676143 | 1.156x |
| 2 | 2 | 16384,16384 | cp | 1.475888 | 1.356559 | 1.088x |
| 2 | 2 | 32768,32768 | cp | 3.844528 | 2.707056 | 1.420x |
| 2 | 4 | 2048,2048,2048,2048 | fused | 0.220161 | 0.296640 | 0.742x |
| 2 | 4 | 4096,4096,4096,4096 | fused | 0.403328 | 0.597392 | 0.675x |
| 2 | 4 | 8192,8192,8192,8192 | fused | 1.058289 | 1.183600 | 0.894x |
| 2 | 4 | 16384,16384,16384,16384 | fused | 2.914658 | 2.723890 | 1.070x |
| 2 | 4 | 32768,32768,32768,32768 | fused | 5.998359 | 5.628934 | 1.066x |
| 2 | 8 | 2048,2048,2048,2048,2048,2048,2048,2048 | fused | 0.417473 | 0.616385 | 0.677x |
| 2 | 8 | 4096,4096,4096,4096,4096,4096,4096,4096 | fused | 0.765121 | 1.160482 | 0.659x |
| 2 | 8 | 8192,8192,8192,8192,8192,8192,8192,8192 | fused | 2.400628 | 2.279972 | 1.053x |
| 2 | 8 | 16384,16384,16384,16384,16384,16384,16384,16384 | fused | 5.450202 | 4.549257 | 1.198x |
| 2 | 8 | 32768,32768,32768,32768,32768,32768,32768,32768 | fused | 10.005226 | 9.392774 | 1.065x |
| 2 | 2 | 2048,4096 | cp | 0.307248 | 0.306192 | 1.003x |
| 2 | 4 | 8192,2048,4096,1024 | fused | 1.028563 | 0.635906 | 1.617x |
| 2 | 2 | 2049,4097 | cp | 0.379985 | 0.312353 | 1.217x |

These are GPU operator measurements on RTX6000D only, not model end-to-end gains or a guarantee for other GPUs. Public correctness tests and sanitizer logs are separate from this performance matrix; maintainer CI is still required.

Raw per-round samples and source manifests are in `gdn_sm120_cute_rtx6000d_tp1.json.gz` and `gdn_sm120_cute_rtx6000d_tp2.json.gz`; gzip compression keeps the review diff focused on code. The JSON/CSV summary contains every case.
