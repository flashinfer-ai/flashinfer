# W4A16 with NVFP4 activation prefill

Expert-presence marking, NVFP4-CSF scale expansion, and deterministic dynamic
route reduction have registered mutating operators. ATen tracing preserves
these launches and their caller-owned outputs for subsequent compilation and
CUDA graph capture. Scale expansion retains its supplied prepared CuTe program;
route reduction uses the current capture stream.

This experimental SM120 path reuses ModelOpt NVFP4 W4A16 weights for
NVFP4 activation GEMMs during prefill. Activation quantization changes model
outputs and requires model quality validation before deployment. The default
remains W4A16.

Set `B12X_W4A16_A4_PREFILL=1` before preparing weights and execution plans to
enable this capability. Preparation requires finite,
positive input and intermediate calibration scales; absent scales and a pair
of all-one placeholder scales retain W4A16. The shared input global scale is
the minimum across experts. Intermediate scales remain per expert. Ordinary
A4 weight preparation retains its per-expert scales.

Supported weights use packed ModelOpt NVFP4 with ordinary or compressed
NVFP4-CSF block scales, BF16 input/output, SiLU gating, hidden size divisible
by 256, and intermediate size divisible by 64. FC1 masks the unused half of
a final 128-channel intermediate tile and reads gate/up scale halves from
native packed scale storage. A positive `swiglu_limit` uses
the W4A16 activation contract: clamp the BF16 GEMM gate above the limit and the
up input to the symmetric interval, round SiLU and up to BF16, multiply, then
round the intermediate to BF16 before quantization. Expert maps, input router
weighting, activation-amax collection, and unsupported layouts retain W4A16.

The execution binding accepts `a4_prefill`:

- `True` selects prepared A4 launches.
- `False` or `None` selects W4A16.

The caller selects precision from request semantics: prefill may use A4;
decode and speculative verification use A16. Token count does not select
hybrid precision, and `ActivationSpec.a16_max_tokens` does not override an
explicit `a4_prefill=True`. Each call must fit its prepared capacity and
caller-owned scratch buffers. When an exact decode variant lacks the padded
A4 workspace, a composite plan reuses a larger fitting prepared variant in
the same scratch arena. A plan with no fitting A4 variant retains W4A16.

Compressed NVFP4 and MXFP4 scales use `csf_32x2_v1` resident blocks. MoE
preparation exposes `MoeDecodeConfig.csf_inline_scales` as a tuning choice for
compatible FP4 kernels. `False` expands scales into caller-owned native buffers
before compute; small calls expand routed experts and long prefills expand all
experts. `True` reconstructs scales inside the compute kernel's shared-memory
pipeline stages. Both choices read the same resident
payload and preserve native scale bytes, normalization, and arithmetic.

The autotuner measures the complete invocation, including expansion, route
planning, synchronization resets, and reduction. Token capacity is part of the
tuning query. Explicit config overrides select either choice; preparation
without autotuning defaults to expansion. Inline readers cover M16/M32 NVFP4
W4A4 and MXFP4 W4A8 tiles, and packed W4A16 kernels. Scalar and compact micro
kernels, M64/M128 activation-quantized tiles, materialized NVFP4 W4A4,
source-native W4A16, dense GEMM, and the optional
W4A16 activation-prefill path consume expanded scales. Unsupported inline
overrides are rejected during planning.

M16/M32 inline candidates are omitted from autotuning when routed rows exceed
the expert count times the tile height. Explicit overrides can still measure
these kernels with larger batches. Compact MXFP4 W4A8 retains reconstructed
K128 scale words across both K64 pipeline steps.

Expansion reconstructs each selected expert's scales once per invocation.
Inline reconstruction repeats for every compute tile that consumes those
scales. Inline can reduce memory traffic when many experts are active and each
has few token tiles. With few active experts, expanded scales can remain in L2;
with many token tiles per expert, repeated inline decoding competes with compute
while expansion is amortized. NVFP4 has one scale byte per 16 weights, twice
MXFP4's scale density. Kernel staging, routing distribution, and cache residency
determine whether the saved traffic pays for decoding and synchronization.

The benchmark option `--scale-compression csf_32x2_v1` uses these resident
blocks. `--moe-config '{"backend":"w4a16","route_planner":"internal",
"max_active_clusters":null,"w4a16_route_mode":"packed",
"csf_inline_scales":true}'` forces inline W4A16. Omit `--moe-config` to tune
both choices. `--time-graph-replay` measures the captured invocation, including
scale expansion when selected.

Pre-routed single-op benchmarks default to `--routing-workload mixture`.
The synthetic scenarios cover diffuse expert popularity, concentrated popularity,
and correlated preferences within groups of four tokens. Expert IDs are sampled
without replacement within each token. `--tuning-samples` selects independent
seeds per scenario, and `--routing-samples` selects held-out evaluation seeds.
No recorded model inputs are required. `--routing-workload default` uses the
profile's input generator; individual workload names allow controlled comparisons.

Candidate races minimize the arithmetic mean of invocation latencies across the
synthetic scenarios. Equal scenario weights provide distribution coverage;
they are not measured production traffic frequencies. Invocation-frequency
weights, when known, belong in this arithmetic mean. Geometric mean speedup can
favor a configuration that consumes more total GPU time. Evaluation reports
individual scenarios and their arithmetic mean. Each benchmark process uses a
fresh selection cache so a previous routing distribution cannot supply its winner.

`expand_scales(experts)` expands every NVFP4 expert on the current stream.
A later `bind(..., scales_expanded=True)` skips per-call expansion, except for
micro kernels whose synchronization reset is fused with expansion. Callers
must order expansion after the previous use of shared scale scratch and before
compute. Concurrent execution streams require disjoint scratch or explicit
stream dependencies.

`uses_expanded_nvfp4_scales(plan, num_tokens=..., a4_prefill=...,
route_ids_dtype=..., scales_expanded=...)` reports whether a prepared call has
an expansion consumer. It returns `False` for an inline plan, allowing callers
to skip scale prefetch. The query performs no GPU work.

`B12X_W4A16_A4_PREFILL_TERMS=1` uses one NVFP4 activation plane. A value of `2`
adds an independently quantized residual plane. `B12X_W4A16_A4_PREFILL_WARPS`
accepts `8` (default) or `16`. These controls and the capability flag are immutable
execution-plan inputs and are part of the tuning query. Changing the
environment requires preparing another execution plan. Kernel compilation
retains device-owned callables; replay performs no compilation or tensor
allocation. A4 always applies router weights in the FP32 final sum. Set
`B12X_W4A16_FP32_TOPK_WEIGHTS=1` before importing B12X when comparing with an
A16 baseline that also uses FP32 router weighting.

Run the synthetic quantization-contract and changed-input graph checks on an
SM120 GPU from the FlashInfer repository:

```sh
PYTHONPATH="$PWD:$PWD/flashinfer/experimental/b12x/_compat" \
  .venv/bin/python -m pytest \
  tests/experimental/b12x/moe/test_w4a16_a4_prefill.py -v
```

The two-device ownership check requires two visible SM120 GPUs. These checks
verify the specified lossy arithmetic and replay behavior; they do not measure
model accuracy or serving throughput.
