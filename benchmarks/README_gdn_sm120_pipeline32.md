# SM120 GDN pipeline32

The opt-in p32 path uses a precompute kernel and a 32-token consumer with four
warp groups (state, delta values, output, and loading). The consumer grid is
`(2, heads, sequences)`: two V=64 CTAs per head/sequence, each with 512 threads.
This is grid decomposition, not a claim of two resident CTAs on one SM.
Shared-memory inputs use two slots with ready/free mbarriers. The default
ldmatrix path uses eight-element row padding (about 76.8 KiB of shared-memory
fields per CTA before final structure alignment). Accumulations are FP32.

## Public API

With this checkout installed on SM120:

```python
import os
from flashinfer.gdn_prefill import chunk_gated_delta_rule

os.environ["FLASHINFER_GDN_SM120_PIPELINE32"] = "1"
o, state = chunk_gated_delta_rule(
    q, k, v, g=alpha, beta=beta, initial_state=initial_state,
    cu_seqlens=cu_seqlens, output_final_state=True,
    use_cp=False, backend="flashinfer",
)
```

Q/K/V must be contiguous FP16/BF16 token-major tensors with head dimension 128.
Gates are multiplicative FP32; state is contiguous FP32 `[N,H,V,K]`.
State pools and in-kernel QK normalization are rejected on this path.
The public API retains its existing checkpoint interval restrictions.
Turning off the environment variable restores the official fused route;
CP and other architectures keep their existing dispatch. This kernel targets
SM120 specifically, not SM121. The legacy
`VLLM_GDN_FLASHINFER_PIPELINE32` environment alias is retained for the supplied
integration; the FlashInfer variable takes precedence.

## Benchmark and accuracy

```bash
CUDA_VISIBLE_DEVICES=4 python benchmarks/bench_gdn_sm120_pipeline32.py \
  --model '27B TP1' --seqlen 1x --fi-path all --check-acc \
  --fla-chunk-size 64 --warmup 10 --repeats 100 --backend cudagraph \
  --json /tmp/gdn_pipeline32.json

# Tail/variable-length smoke case
python benchmarks/bench_gdn_sm120_pipeline32.py --seqlen 31+33+65 --check-acc
pytest tests/gdn/test_prefill_pipeline32.py -v
python tests/gdn/test_pipeline32_host.py -v
```

The standalone benchmark compares only FLA Triton, official FI fuse/CP, and
p32. The `cp32` alias means p32, not context-parallel execution. It disables
optional overlay selectors for the official baselines and calls p32 directly,
so a missing dispatch patch cannot silently benchmark fuse as p32. Baseline
files must come from a clean installation; loaded source hashes are recorded.

All paths share the same input per shape. FLA receives log gates and KV state;
FI receives exp(g) and transposed VK state. Timed FI calls include those
conversions and the final-state transpose; p32 includes both kernels. The same
explicit FLA chunk size is used for accuracy and timing. Accuracy measures
relative L2 for both output and final state, rejecting nonfinite errors and
returning a nonzero status on failure. Warmup/repeats are call/replay counts;
PyTorch CUDA events measure the mean time.

## Historical evidence and scope

`results/gdn_sm120_pipeline32_rtx6000d.json` contains author-provided historical
measurements on RTX 6000D, Hqk=16, Hv=48, D=128, BF16, single sequences of
2K–32K tokens. They compare **FlashInfer 0.6.18**, not today's main, with FLA
0.5.2. p32 takes 0.227–3.272 ms versus 0.391–6.095 ms for fuse,
0.362–3.990 ms for CP, and 0.271–5.148 ms for FLA. All 15 FI-vs-FLA checks
pass relative L2 <= 0.02. These are not fresh runs of this PR.

The historical harness used TileLang profiler 0.1.9 and requested FLA chunk=32
for accuracy but the default for timing. This standalone harness unifies that
choice and uses a different timing driver, so new numbers must be reported
separately. The original log contains neither a source hash nor the
ldmatrix/padding settings. Revalidate this exact commit on SM120 before merge.

The author's MMA-throttling observation motivated the scheduling change.
Timing alone does not identify the limiting pipeline or prove elimination of
throttling; Nsight Compute counters and controlled variants are still needed
for attribution. No default-dispatch or full-model speedup is claimed.

The GPU tests add FP64 recurrent-reference checks for FP16/BF16, chunk tails,
empty/ragged sequences, public dispatch, and graph replay. These tests and
compute-sanitizer checks require a CUDA SM120 host and were not run on the
local macOS development machine. Checkpoint and broader head-layout coverage
remain to be validated before merge.
