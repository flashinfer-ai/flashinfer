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

Compressed NVFP4 and MXFP4 scales expand into caller-owned native scale
buffers before compute. Small calls expand only routed experts; long prefills
expand all experts. MoE and dense GEMM kernels accept native scales and use the
same arithmetic and tuning candidates as resident native weights. W4A16
preparation preserves its scale normalization and packed layout in the expander.

`expand_scales(experts)` expands every NVFP4 expert on the current stream.
A later `bind(..., scales_expanded=True)` skips per-call expansion, except for
micro kernels whose synchronization reset is fused with expansion. Callers
must order expansion after the previous use of shared scale scratch and before
compute. Concurrent execution streams require disjoint scratch or explicit
stream dependencies.

`uses_expanded_nvfp4_scales(plan, num_tokens=..., a4_prefill=...,
route_ids_dtype=..., scales_expanded=...)` reports whether a prepared call has
compressed NVFP4 scales. All such calls consume expanded scales, independent
of activation precision and token count. The query performs no GPU work.

`B12X_W4A16_A4_PREFILL_TERMS=1` uses one NVFP4 activation plane. A value of `2`
adds an independently quantized residual plane. `B12X_W4A16_A4_PREFILL_WARPS`
accepts `8` (default) or `16`. These controls and the capability flag are immutable
execution-plan inputs and are part of tuning query schema 28. Changing the
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
