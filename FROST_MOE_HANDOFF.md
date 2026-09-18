# Frost / FlashInfer MoE: capabilities, measured benefits and attribution

This consolidated pathfinding draft integrates open-source cuDNN Frost with
FlashInfer for BF16 and calibrated-FP8 MoE. It adds explicit preparation and
configuration contracts, CUDA Graph execution, and BF16 coverage on SM100/SM120.
Candidate enumeration has no ranking heuristic. Yanqin and Yihua can continue
reviewing and distilling the companion drafts: [Frontend #1080](https://github.com/NVIDIA/cudnn-frontend/pull/1080)
and [FlashInfer #5250](https://github.com/flashinfer-ai/flashinfer/pull/5250).

The work combines contributions across NVIDIA Frost, TensorRT-LLM, CUTLASS and
FlashInfer. Comparisons describe measured behavior and complementary coverage.
This public summary records impact and provenance; detailed experimental methods
and implementation notes are maintained separately for the handoff.

## Contributions developed in this effort

Kernel Factory is part of this effort's optimization workflow. Its discoveries,
independent validation and integration count as project contributions. The
following rows describe newly implemented improvements on the existing
infrastructure, without claiming that the underlying techniques are new to the
field. Configuration search is excluded from these implementation benefits.

| Contribution | Origin | Full-MoE latency reduction on the measured B200 fixture |
|---|---|---:|
| Small-batch routing path | Local implementation | T1: 8.31–8.38%; T64: 1.37–1.38% |
| Small-batch GEMM execution efficiency | Kernel Factory discovery, local validation and integration | T1: 9.66–9.82% |
| Additional small-batch execution specialization | Kernel Factory discovery, local validation and integration | T1: 1.00–1.17% |
| GEMM implementation for larger batches | Kernel Factory discovery, local validation and integration | T8/T64: 1.19–1.53% |

These are separate matched synthetic BF16 operator ablations on a 148-SM,
1000 W B200, E=128, top-k=8, H=2048, I=768. They use fixed configurations
within each comparison and include the complete MoE operator with precomputed
routing. Weight preparation, router-logit/top-k computation and model execution
are excluded. The percentages must not be added or compounded. They are not an
overall attribution percentage or a model-level speedup claim.

## Reuse and acknowledgements

- NVIDIA Frost/CuTeDSL and FlashInfer supply the kernel and integration foundations.
- NVIDIA TensorRT-LLM/FI authors retain credit for reused native finalization and
  fallback routing. Follow-on adaptations build on those contributions. The
  measured packed-T1 finalization followup reduced complete-MoE latency 2.124%.
- TensorRT-LLM's preparation design motivated separate Frost work. This is
  credited inspiration, not an independently originated design direction.
  Its independently measured complete-MoE benefits are 1.29–1.31% and 0.80–1.06%
  for the respective stage-specific adaptations; these also have separate baselines.
- Yanqin Zhai (@yanqinz2) authored the incorporated [Frontend #1090](https://github.com/NVIDIA/cudnn-frontend/pull/1090)
  contributions. Yanqin and Yihua's parallel distillation and integration work
  retain their own credit.
- NVIDIA CUTLASS example 113 and the NVIDIA CuTeDSL MegaMoE team's implementation
  informed distinct parts of the exploration. Referenced designs and reused
  source retain their original authorship and license headers.
- Kernel Factory campaigns contributed the project improvements listed above.
  Candidates without independently validated benefit are not counted.

API support, correctness repairs, validation and configuration selection are
valuable engineering work, but are not counted as new optimization discoveries.

## Current matched reference comparisons

Each row is its own matched run; differences between rows are not optimization
deltas. Preparations are excluded for both paths. B200 reference searches cover
704 joint choices per shape; RTX PRO 6000 Server searches cover 128 exported
choices, including documented unsupported configurations.

| GPU / fixture | Tokens | Frost | NVIDIA reference implementation |
|---|---:|---:|---:|
| B200 1000 W, synthetic, unpacked | 8 | 98.818 us | TRT-LLM: 96.078 us |
| B200 1000 W, synthetic, unpacked | 64 | 198.294 us | TRT-LLM: 195.270 us |
| RTX PRO 6000 Server 600 W, synthetic, packed | 1 | 82.795 us | CUTLASS: 87.114 us |
| RTX PRO 6000 Server 600 W, synthetic, packed | 8 | 400.965 us | CUTLASS: 407.425 us |
| B200 1000 W, trained Qwen layer MoE, unpacked | 8 | 78.428 us | TRT-LLM: 75.462 us |
| B200 1000 W, trained Qwen layer MoE, unpacked | 64 | 136.540 us | TRT-LLM: 131.617 us |

The Qwen rows use Qwen3-30B-A3B layer-0 trained weights, actual activations and
routing captured from a causal-prefill systems prompt. Credit Qwen and Hugging
Face Transformers. They measure the MoE subgraph and exclude attention and
router-logit/top-k computation; they are not decode-with-KV-cache or model E2E.
Frost uses specified configurations, without claiming an exhaustive Frost search.

## Validation and limitations

Reported comparisons passed target-GPU numerical, actual-route, CUDA Graph,
live-input/weight, memcheck and racecheck gates, followed by fresh symmetric
measurement processes. The trained-Qwen comparison additionally checks direct
Hugging Face outputs: 10170 raw checks, 7312 changed-reference controls and
69408 timing spans across the two shapes. Reference comparisons and contribution
ablations retain their separate source snapshots and evidence records.

An existing paired-FC2 accumulation-fidelity failure on strongly cancelling
BF16 dot products remains unresolved. Original and updated stage-depth versions
reproduce it; no tolerance was relaxed. Passing reported fixtures does not prove
accuracy on arbitrary ill-conditioned inputs. Full repository CI, broad workload
coverage, deployment benefit and a performance roof remain unclaimed.
