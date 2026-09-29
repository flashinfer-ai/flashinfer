# Changelog

## 2026-09-28

### SM100 FP4 routed-MoE decode

The FC2 terminal-buffer recycling update preserves the routed-MoE pipeline, NVFP4 representations, routing weights, clamped SwiGLU with limit 10.0, and BF16 output.

B200 validation covers every integer token count 1–32 at H=4096, I=2048, 256 routed experts and top-k 6. All 32 strict reference comparisons passed at atol=rtol=0.01, together with 576 strict timing postchecks, 8 dynamic-routing fixtures and 7 API/graph cases.

Compared with `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` on identical physical inputs, geometric-mean speedup is **1.021482434×**. All 32 shapes win; minimum speedup is **1.004732110× at T=14**, with no ties or regressions. Measurements use CUPTI, cold L2 and matched complete-call timing boundaries, with six balanced captures per arm.

| T | Candidate (µs) | FlashInfer baseline (µs) | Speedup |
|---:|---:|---:|---:|
| 1 | 26.387665 | 29.407550 | 1.114443037× |
| 2 | 41.583749 | 44.756510 | 1.076298114× |
| 3 | 56.335622 | 59.976458 | 1.064627597× |
| 4 | 68.719672 | 71.460615 | 1.039885852× |
| 5 | 79.242466 | 81.903009 | 1.033574715× |
| 6 | 90.970399 | 92.421005 | 1.015945910× |
| 7 | 100.111570 | 102.271917 | 1.021579391× |
| 8 | 108.884969 | 110.191307 | 1.011997414× |
| 9 | 121.039818 | 121.801850 | 1.006295710× |
| 10 | 127.055646 | 127.818044 | 1.006000504× |
| 11 | 134.602436 | 135.252960 | 1.004832932× |
| 12 | 144.255821 | 145.081703 | 1.005725116× |
| 13 | 153.775485 | 154.943467 | 1.007595376× |
| 14 | 159.930292 | 160.687099 | 1.004732110× |
| 15 | 167.605098 | 168.473853 | 1.005183347× |
| 16 | 177.204607 | 178.063016 | 1.004844168× |
| 17 | 182.618485 | 186.756839 | 1.022661203× |
| 18 | 187.130155 | 190.772834 | 1.019466015× |
| 19 | 192.548644 | 196.388681 | 1.019943206× |
| 20 | 200.266476 | 204.143575 | 1.019359703× |
| 21 | 208.058486 | 211.711656 | 1.017558381× |
| 22 | 218.890141 | 223.204720 | 1.019711163× |
| 23 | 222.186324 | 227.204805 | 1.022586814× |
| 24 | 234.409988 | 238.415085 | 1.017085865× |
| 25 | 242.127158 | 246.201867 | 1.016828795× |
| 26 | 248.366962 | 251.828828 | 1.013938512× |
| 27 | 253.812662 | 257.402025 | 1.014141782× |
| 28 | 261.673974 | 264.964460 | 1.012574755× |
| 29 | 268.911818 | 272.527208 | 1.013444519× |
| 30 | 278.458135 | 281.999030 | 1.012716075× |
| 31 | 282.244813 | 285.871512 | 1.012849479× |
| 32 | 286.580986 | 291.407602 | 1.016842068× |

Validation worker runtime was 5243.462 seconds; submission-to-completion turnaround was 5371.736 seconds. Per-shape GPU timings are listed above. Hardware-limit evidence and final promotion checks remain in progress.

Related public request: [FlashInfer issue #5184](https://github.com/flashinfer-ai/flashinfer/issues/5184).
