# Development tracking: GLM-4.7-Flash FP8 MLA on SM120

Development request / fork owner: [sjtushenhai](https://github.com/sjtushenhai).
Branch: `fp8_mla_sm120_dev`. This note tracks the requested source publication;
no upstream maintainer ownership or release commitment is implied.

Use case: lower the memory footprint and operator latency of paged absorbed MLA
on a workstation Blackwell GPU, for decode and prefill. SM120 cannot use the
SM90/SM100-specific FlashMLA kernels evaluated in the accompanying study.

Reason for experimental status: new architecture-specific FP8 MMA path, narrow
shape contract, synthetic-only precision evidence, and a research ABI.

Before proposing admission to upstream experimental CI, create/link a GitHub
tracking issue, confirm a maintainer owner, and use the Experimental Track PR
checkbox. This branch push does not open a PR or create an issue. A public
tracking issue number is therefore not claimed here.

Graduation candidates for review within four weeks (2026-11-06; no promised
release):

1. Validate GLM outputs and task metrics with model-derived activations, including
   outliers, long context, RoPE and a calibrated quantization policy.
2. Integrate explicit backend selection and the scale/cache contract into the
   existing MLA API with deferred imports and experimental warnings. Keep
   automatic selection gated and never register the experimental backend in AOT.
3. Replace the research ctypes ABI with the standard JIT/TVM-FFI layer, finish
   stream/error contracts, and add serving cache writes and allocator integration.
4. Compare absorbed versus expanded-K/V prefill including projection costs;
   improve short-query utilization and short-prefill latency.
5. Add model throughput, broader devices/shapes, API/trace documentation, and
   stable tests before considering a release target.

Completed evidence and reproducible commands are linked from the
[study overview](../../../benchmarks/mla_fp8_sm120/README.md).
