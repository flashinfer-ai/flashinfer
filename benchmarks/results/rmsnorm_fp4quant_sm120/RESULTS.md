# SM120 CuTe RMSNorm/NVFP4 full-chain measurements

2026-09-16; RTX 5090, Python 3.12, Torch 2.13.0+cu130, CuTe DSL 4.6.2, CUDA 13.0.88, driver 595.71.05.

Scope: actual fused RMSNorm/NVFP4 producer followed by cuBLASLt GEMM, with synthetic inputs. No model or serving integration is claimed. See [caller and protocol](../../README_rmsnorm_fp4quant_gemm.md).

## Primary: identical preallocated output addresses

120 paired cases (15 shapes × two seeds × two global scales × forward/reverse). Geometric-mean paired speedup: **1.0196x**. CuBLASLt algorithm ID 70 was selected for all cases; the entire algorithm instance/workspace is shared within each pair.

Each row below reports median absolute latency over eight cases and the geometric mean of the eight paired ratios. The ratio of the displayed medians is not the aggregation rule. Min/max expose individual-case variation.

| M | N | K | Original CuTe + GEMM (µs) | Optimized CuTe + GEMM (µs) | Paired geomean | Min–max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 4096 | 18.484 | 18.442 | 1.0024x | 1.0022–1.0026x |
| 2 | 4096 | 4096 | 18.519 | 18.442 | 1.0108x | 1.0024–1.0402x |
| 3 | 4096 | 4096 | 18.475 | 17.446 | 1.0654x | 1.0032–1.1224x |
| 7 | 4096 | 4096 | 18.506 | 18.442 | 1.0048x | 1.0025–1.0087x |
| 8 | 4096 | 4096 | 18.484 | 18.428 | 1.0030x | 1.0029–1.0031x |
| 24 | 4096 | 4096 | 17.035 | 16.353 | 1.0423x | 1.0410–1.0436x |
| 128 | 4096 | 4096 | 17.387 | 16.876 | 1.0303x | 1.0299–1.0309x |
| 512 | 4096 | 4096 | 20.834 | 20.153 | 1.0364x | 1.0336–1.0397x |
| 1024 | 4096 | 4096 | 39.697 | 38.763 | 1.0242x | 1.0225–1.0258x |
| 16384 | 4096 | 4096 | 605.645 | 596.158 | 1.0164x | 1.0154–1.0177x |
| 1 | 2048 | 7168 | 21.151 | 21.135 | 1.0006x | 0.9987–1.0026x |
| 4 | 2048 | 7168 | 20.533 | 20.532 | 1.0000x | 1.0000–1.0001x |
| 16 | 2048 | 7168 | 20.548 | 20.556 | 0.9992x | 0.9973–1.0001x |
| 256 | 2048 | 7168 | 24.515 | 24.525 | 0.9994x | 0.9980–1.0000x |
| 4096 | 2048 | 7168 | 144.075 | 135.704 | 1.0619x | 1.0599–1.0647x |

Independent invocation geomeans:

- forward, seed 73, scale 1: 1.0154x.
- forward, seed 73, scale 32: 1.0172x.
- forward, seed 109, scale 1: 1.0194x.
- forward, seed 109, scale 32: 1.0153x.
- reverse, seed 73, scale 1: 1.0224x.
- reverse, seed 73, scale 32: 1.0222x.
- reverse, seed 109, scale 1: 1.0224x.
- reverse, seed 109, scale 32: 1.0223x.

Worst paired case: [16, 2048, 7168], reverse, seed 109, scale 1: 20.618 → 20.674 µs (0.9973x). 0 cases have speedup < 0.98x. Improvements near 1x should be treated as approximately flat, not a universal win.

## Separate control: automatic producer-output allocation

- forward, seed 73, scale 32, 15 cases: geomean **1.0150x**; worst [4, 2048, 7168]: 20.532 → 20.532 µs (1.0000x).
- reverse, seed 73, scale 32, 15 cases: geomean **1.0120x**; worst [1, 2048, 7168]: 21.232 → 21.901 µs (0.9695x).

These graph measurements allocate outputs during capture, not inside the timed replay. Distinct output addresses can alter cache behavior. This control is not pooled with the primary comparison.

## Numerical and runtime evidence

- Integrated CuTe test suite: **1,027 passed**, including 20 new padded-row / graph-replay cases at widths 64, 80, 528, 4112 and 7168.
- Compute Sanitizer memcheck: **65 passed, zero errors**. Racecheck: **20 passed, zero hazards/errors/warnings**. These are targeted checks, not the complete repository GPU suite or upstream CI.
- Repository-wide pre-commit checks passed; changed tests and evidence were checked again after the final additions.
- Maximum full-output relative RMS against the original producer+GEMM: **0.02312%**.
- Maximum sampled GEMM relative RMS against FP64 dot products of decoded quantized operands: **0.18049%**. This is not an unquantized-model error measurement.
- Six alternating timing rounds per pair; 50 complete chains per graph, 100 replays per round. Raw JSONL files retain all samples.
- Resident-input repeated CUDA Graphs, no locked GPU clock, no cold-cache or service benchmark. Compilation and weight quantization excluded.

## Prior experiments and limitations

The earlier shared-staging/register-reuse variants were development experiments. A 64-thread choice for K=7168 regressed a development case by about 8.6% and was rejected. The final choice retains 128 threads for that width.

The preceding candidate (before removal of the final unused CTA barrier) measured 1.0133x over 120 public-allocation cases, including four reverse-order M=1,N=2048,K=7168 cases around 0.968x. Those regressions motivated the identical-address control. This does not establish cache/address placement as the sole cause; the candidate and protocol both changed, so the current matrix is not a clean barrier-removal ablation.

The earlier Triton ~1.14x preprocessing result uses different arithmetic and does not describe this CuTe implementation. Current numbers establish a bounded single-layer benefit on SM120 only. They do not establish a serving gain or justify extrapolation to B300/Hopper.

## Source identity

```json
{
  "candidate_sha256": "9b57b20126b56029fc717c7c6591a63447ac5c021fbd0498e1299a5865447c8b",
  "baseline_sha256": "ec32fae9254adb9b888c0affd99822c89806a5881de04db0b0a755d81f6f90a3",
  "bridge_sha256": "3ca4abc5d3401b0965891b075aecff1254c7242d354ceea6ba5fd49808f93e09"
}
```
