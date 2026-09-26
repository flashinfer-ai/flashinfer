# cuDNN Frost-selected MoE grouped GEMM kernels

This package contains Frost-selected MoE grouped GEMM kernels.
This experimental backend follows Cake's source-distribution model. cuDNN Frost is
an offline generator: FlashInfer ships standalone Python/CuTe DSL kernels and a
SHA-256 manifest, with no precompiled `.o` files. Required cuDNN Frost device helpers
are included in each generated source. The deployed process does not import
`cudnn-frontend`, construct a cuDNN Frost graph, or run the cuDNN Frost graph compiler.

On first use, `runtime.py` fills the selected template's geometry constants and
TMA-store expansion, then compiles the instantiated source through
`flashinfer.jit.cute_dsl_core.build_and_load_cute_dsl_kernel` and exposes native
TVM-FFI functions to the MoE adapter. Later processes reuse the writable
FlashInfer JIT cache. Cache identities include target architecture, the complete
CuTe DSL compiler stack and source contents; persisted tactic identities also
include the compiler stack. Upgrading CuTe DSL therefore recompiles kernels
and invalidates old cuDNN Frost tactic choices. Compilation and native plan setup
must finish before CUDA Graph capture.

The sources use `cutlass.experimental.primitives` and the experimental tensor
map API. Admission checks compiler capabilities instead of imposing a version
number: `capabilities.py` reads the frozen sources without executing them and
checks their imported symbols, nested enum members, and statically specified
calls against Python signatures where available. It also checks native
`sm_107a` compile-option support, TVM-FFI support, and conflicting
`CUTE_DSL_ARCH` overrides. Probe results are cached per architecture and source
set; no kernel is compiled or launched by the probe.

Missing capabilities skip cuDNN Frost in automatic MoE selection and raise an
actionable `NotImplementedError` for explicit cuDNN Frost use. Source integrity errors
and later compiler/runtime failures remain errors; static capability checks do
not guarantee compiler correctness. Validation uses the installed internal DSL
build, not a public 4.8 wheel; public 4.8+ remains the planned deployment baseline.
This backend does not change the repository-wide dependency floor.
MoE runners additionally require PyTorch's
`CUDAGraph.get_currently_capturing_graph()` and
`CUDAGraph.retain_object(..., synchronize_before_release=True)` APIs, verified
with PyTorch 2.14. Missing graph ownership capabilities decline automatic
admission and raise `NotImplementedError` for explicit MoE runner use.
The native routing/finalize adapter additionally needs a CUDA toolkit supporting
the target architecture. Set `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` to compile
without persisting CuTe DSL artifacts; neither mode writes compiled objects
into the installed package.

An explicit `FLASHINFER_CUDNN_FROST_PTXAS=/path/to/ptxas` selects an external
assembler for every Frost dtype. CuTe still generates the original PTX and
host/TVM-FFI wrapper. Frost assembles that PTX with the selected executable,
replaces the GPU binary initializer in standard compiler IR, and exports a new
host object through CuTe's existing interface. The source, tensor descriptors,
launch arguments and kernel algorithm are unchanged. The resolved executable
path, version and SHA-256 participate in memory, disk and tactic cache identities.
Disabled disk caching and failed persistence also execute the external binary.

The SM107 MXFP8, NVFP4 and MXFP8 × MXFP4 validation uses CUDA 13.5 PTXAS:
the installed DSL's bundled CUDA 13.4 assembler produced incorrect dynamic
tensor-map dimension updates in the original MXFP8 kernels. Setting `CUDA_HOME`
alone does not select the
external assembler. Without the explicit Frost option, the existing bundled
compiler path remains in use; it is not covered by this compiler workaround.

The original operation in this prototype is cuDNN Frost's fused dual grouped GEMM1:

```text
gate = grouped_tokens @ gate_weight[expert].T
up   = grouped_tokens @ up_weight[expert].T
out  = silu(gate) * up * scale
```

`grouped_tokens` must already be materialized in contiguous group order.
`first_token_offset[g]` is the first row of group `g`; the last group ends at
`S`, and group `g` uses expert `g % E`. Routing, permutation, GEMM2, and final
scatter are outside this API.

An independent FC2 grouped-GEMM PoC and a full `CudnnFrostBf16MoeRunner` are also
implemented. The old cuDNN Frost-FC1/CUTLASS-FC2 hybrid integration has been removed
completely; the new runner does not call an existing backend.

## Runtime layout

All data types have a sibling package (`bf16/`, `mxfp8/`, `nvfp4/`,
`mxfp8_mxfp4/`) with a
`runtime.py` for their numerical contracts and launch arguments. The shared
root `runtime.py` handles source verification, template materialization,
compilation, caching and CUDA streams. Artifact paths are always selected by
an explicit dtype through `artifact_root(dtype)`.

Each dtype package provides `moe.py` for full MoELayer integration and a
lightweight `support.py` for numerical-contract admission and deferred runner
construction. The root `support.py` contains the shared model/top-k and token-range
policy. BF16 also provides its standalone `fc2.py`; MXFP8's grouped FC1 and FC2 share the same
block-scale runtime.

## Artifact layout

Artifacts are grouped by operand dtype under a common root:

```text
artifacts/
  bf16/
    cudnn_frost_selected_kernels.json
    moe_shortlists.json
    sources/
  mxfp8/
    cudnn_frost_selected_kernels.json
    moe_shortlists.json
    sources/
  nvfp4/
    cudnn_frost_selected_kernels.json
    moe_shortlists.json
    sources/
  mxfp8_mxfp4/
    cudnn_frost_selected_kernels.json
    moe_shortlists.json
    sources/
```

Manifest source paths are relative to their dtype directory. Runtime discovery
selects that directory; export/benchmark `--output-dir`, `--artifacts`, and
`--selected-dir` refer to a dtype directory, not the shared parent. Source
packaging and generated-file exclusions cover the same layout for future dtypes.

## MXFP8 block-scale grouped pipelines

`mxfp8/runtime.py` provides explicit prepared execution of frozen MXFP8 × MXFP8
FC1/activation and FC2 pipelines on SM107a. Both operands use E4M3 data with E8M0
scales per 32 K elements. Both grouped stages produce BF16 output.
`mxfp8/moe.py` and `csrc/moe_mxfp8.cu` compose them into the automatic
`cudnn_frost_mxfp8` MoELayer candidate:

```text
precomputed top-k ids/weights
  -> histogram + prefix + gather E4M3 data and E8M0 scales
  -> frozen FC1 + activation -> BF16
  -> per-32-element MXFP8 requantization + segmented scale packing
  -> frozen FC2 -> BF16
  -> FP32 weighted reduction -> BF16 output
```

The runner consumes the existing `cutlass_mxfp8` weight view, including its
packed per-expert weight scales. Activation scales may use linear or swizzled
F8_128x4 layout. Gated weights retain the canonical **[up, gate]** data layout;
their scale buffers are split into the frozen kernels' contiguous expert layout
during preparation. Per-expert FC1/FC2 input scales must be one. Bias, shared
experts, expert parallelism, non-default activation parameters and logits-based
routing are unsupported.

Automatic admission requires SM107a, MXFP8 operands, BF16 finalized output,
precomputed routing and a matching measured shortlist. As for BF16, each
`(activation,E,H,I,top_k,tokens)` profile supplies two FC1/activation and two
FC2 configurations, giving four complete plans to autotune against the original
eligible backends. Intermediate token counts use the next measured profile;
native plans use the exact input shape. Calls beyond the largest token profile
or without a matching table entry retain their original candidates.

Prepare and autotune outside CUDA Graph capture. Activation data, activation
scales, expert ids and routing weights may change between graph replays.
Prepared weights and weight scales must remain static. For ordinary tensors,
changing weight scales and repacking refreshes the copied scale buffers, and
existing graphs must be captured again. Inference tensors have no version
counter: replace the tensor object when changing prepared weights or scales.
Keep the layer alive for its graphs and use one instance per stream/thread.
Preparation lookup caches retain at most 32 exact-shape plans, four packed
FC1 scale views, and 32 scale-validation records per runner. Scale keys weakly
reference their source tensors; unused records disappear when their sources
die. Active packed inputs and captured PyTorch graphs own the resources they
use independently of lookup eviction. Graph reset/destruction synchronizes its
replay streams before releasing these resources. Replaying through raw CUDA
graph handles bypasses that PyTorch lifetime tracking and is unsupported.
A shape or scale version evicted from lookup must be prepared again outside
capture before capturing a new call. MoELayer keeps at most 128 winner entries;
`reset_winner()` clears them as before.

BF16 admission and packing share a 128-entry selection cache keyed by artifact
roots and cache version, architecture, exact geometry, and activation. It holds
only kernel metadata; tensor layouts, alignment, and semantics are checked on
each call. `bf16.runtime.clear_artifact_cache()` invalidates this metadata when
refreshing artifacts. Measure ordinary warm calls separately from CUDA Graph
replay, which bypasses Python admission and packing.

The `artifacts/mxfp8/` pool retains 169 selected configurations in 39 source
templates, with 400 measured two-by-two profiles. The offline pool covers all
44 families: ten FC1 activations and one shared FC2, each with normal/swap-AB
and STG/TMA output variants. Only families used by the selected profiles ship.
Geometry records share these templates. CTA/MMA/cluster geometry,
pipeline stages, scale-factor geometry, and persistent grid parameters live
in `source.parameters`, not in duplicated Python implementations. At runtime
`source_template.py` injects them before JIT; they remain compiler constants,
not dynamic device branches. The exporter requires AST equality between
every reconstructed source and the original Frost-rendered source.

Export success alone does not establish correctness or performance. Validate
and select candidates on the deployment GPU: shared-memory, L2 and persistent
grid budgets are part of the compiled specialization. Each dtype has its own
manifest and shortlist under the same artifact layout.

All ten default activation contracts are included: gated `SwiGLU`, `GeGLU`,
`GeGLUTanh`, `SwiGLUStep`, `SiTU`; non-gated `Identity`, `ReLU`, `ReLU2`,
`GELU`, `SiLU`. SiTU binds gate/linear scales 4/25 in the generated auxiliary
argument order; SwiGLUStep uses limit 7. FC1 fuses activation on FP32
accumulators before the BF16 output conversion. The optional `scale` argument
multiplies the activation output for every FC1, including non-gated variants.

Export a geometry using the same exporter as BF16:

```bash
python -m flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.export \
  --dtype mxfp8 --op grouped_gemm1_swiglu \
  --config CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma \
  --cta-group 1 --store-mode stg \
  --s 1024 --n 256 --k 128 --experts 8 \
  --output-dir "$ARTIFACT_DIR" --cudnn-frost-revision "$FROST_REVISION"
```

Use `--op grouped_gemm2` for FC2. Changing geometry while retaining the same
operation, dtype, orientation and output store reuses the source file; an
unexpected source difference is an error rather than a new geometry-named file.

For prepared grouped execution:

```python
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8 import runtime as mxfp8
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.runtime import (
    _arch_for, _dimension_matches,
)

# x: grouped E4M3 [S,K]; gate/up: E4M3 [E,N,K]; offsets: int32 [G].
# x_sf: logical E8M0 bytes [S,K/32]; gate_sf/up_sf: [E,N,K/32].
# Choose an artifact matching the operation, architecture and shape contract.
arch = _arch_for(x.device)
geometry = dict(
    s=x.shape[0], n=gate.shape[1], k=x.shape[1],
    experts=gate.shape[0], groups=offsets.numel(),
)
kernel = next(
    (k for k in mxfp8.discover()
     if k.fc1 and k.activation == "swiglu" and not k.swap_ab
     and k.arch == arch
     and all(_dimension_matches(value, k.contract.get(name))
             for name, value in geometry.items())),
    None,
)
if kernel is None:
    raise RuntimeError(
        f"No packaged MXFP8 SwiGLU kernel matches {arch}, {geometry}; "
        "export a matching artifact before preparing this grouped GEMM."
    )
plan = mxfp8.PreparedMxfp8GroupedGemm(
    kernel, x, (gate, up), offsets,
    mxfp8.pack_token_scales(x_sf, offsets),
    (mxfp8.pack_weight_scales(gate_sf), mxfp8.pack_weight_scales(up_sf)),
    output,  # BF16 [S,N], caller-owned
)
plan()  # launch on the current stream; supports subsequent CUDA Graph capture
```

Non-gated FC1 and FC2 take one weight/scale tuple. Group `g` uses expert `g % E`; empty and
uneven groups are supported. N and K must be divisible by 128 in this initial
runtime contract. Scale packing preserves E8M0 bytes, pads each token group
independently to 128 rows, and applies F8_128x4 ordering. The token allocation
reserves capacity for every partition with the same S/G. Preparation reads
offsets on the host and must run outside capture. If group boundaries change,
repack token scales for those boundaries before replaying the plan. Each
concurrent stream needs its own plan/workspace. No cuDNN import is required
for discovery, scale packing, JIT, or execution.

### MXFP8 benchmarks

The `benchmark` subcommand measures the complete MoELayer pipeline and compares
the original eligible backend pool, the same pool with automatic Frost
admission, and Frost alone. Enable experimental automatic selection when running
the full-MoE benchmarks for any dtype:

```bash
export FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1
python benchmarks/bench_cudnn_frost_moe_mxfp8.py benchmark \
  --activation swiglu --experts 8 --hidden 4096 --intermediate 14336 \
  --top-k 2 --tokens 1,16,128,512,2048,4096,8192,12288 \
  --routing uniform --verify-locked-clocks --output "$RESULT_DIR/mxfp8_e2e.jsonl"
```

The benchmark preserves the original backend tactics, checks all four Frost
plans eagerly and through CUDA Graph replay, and records actual autotune winners.
It alternates timing order over batched graph replays and reports numerical
error, raw timings and GPU telemetry. Use `--artifacts DIR` to benchmark another
selected pool containing `moe_shortlists.json`.
Backend eligibility
follows the existing registry: on SM107, the current original MXFP8 pool is
CUTLASS; TRT-LLM MXFP8 supports SM100/SM103. Stage timing cannot establish an
end-to-end improvement.

`benchmarks/bench_cudnn_frost_moe_mxfp8.py` validates and times the prepared
grouped stages. Select an idle GPU with `CUDA_VISIBLE_DEVICES` and set
`RESULT_DIR` to a directory for results:

```bash
python benchmarks/bench_cudnn_frost_moe_mxfp8.py \
  --activation all --stage both --tokens 128,1024 \
  --experts 8 --hidden 1024 --intermediate 512 --top-k 2 \
  --output "$RESULT_DIR/mxfp8.jsonl"
```

This sweeps all matching geometry, normal/swap-AB and STG/TMA candidates for
all ten FC1 activations and the shared FC2. FC2 runs once per shape, independently
of the selected FC1 activations. `--tokens` counts tokens before routing;
each stage processes `tokens * top_k` grouped rows. Use `--activation NAME`
and `--stage fc1|fc2` to narrow the sweep, `--artifacts DIR` to select another
MXFP8 artifact directory, or repeat `--kernel-id ID` to restrict its candidates.
The output must be a new file. JSONL `candidate` records contain each artifact's
geometry, timing samples and numerical error; `best` records rank candidates
separately for each stage, activation and shape.

Every candidate is checked against a dequantized FP64 GEMM reference with FP32
activation and BF16 output, both eagerly and through CUDA Graph replay, before
timing. FC1 and FC2 use independent random inputs. Timings include workspace
reset, grouped GEMM and the fused FC1 activation; they exclude routing, scale
packing, intermediate requantization, finalization, reference calculation and
compilation. Repeated graph replay uses hot inputs. These stage timings must
not be summed into a full MoE latency or compared directly with the BF16
script's complete routed MoE measurements.

## NVFP4 block-scale grouped pipelines

`nvfp4/runtime.py`, `nvfp4/moe.py` and `csrc/moe_nvfp4.cu` provide the
corresponding NVFP4 × NVFP4 pipeline and `cudnn_frost_nvfp4` automatic
candidate. Both operands contain two E2M1 values per byte, with E4M3 block
scales per 16 logical K elements. Logical N and K must be divisible by 128;
packing halves the tensor's last storage dimension, not its logical K.
Both grouped stages produce BF16. The native pipeline gathers packed inputs,
runs fused FC1/activation, requantizes the intermediate to NVFP4, runs FC2,
and reduces routed outputs in FP32 before the final BF16 conversion.

The runner consumes the canonical `cutlass_nvfp4` weight view. Activation
scales may use linear or F8_128x4 layout. FC1 dequantization multiplies the
FP32 GEMM accumulators **before** activation; the standalone grouped API
accepts separate per-group gate and up multipliers through `gemm_scales`.
FC2 applies its dequantization multiplier before BF16 conversion. The
intermediate quantizer uses `fc2_act_global_scale`, whose reciprocal is
already part of the canonical `fc2_dequant_scale`. Global scales must be
positive and finite. Default activation contracts and shape/shortlist
admission follow the same policy as MXFP8.

Automatic admission requires the caller to prepare the `cutlass_nvfp4`
weight view. The default NVFP4 backend list does not prepare this view;
Frost skips weight packs that lack it. Use
`CutlassNvfp4Config.prepare_weights(...)` and
`weight_pack.prepare_for("cutlass_nvfp4", view)` before calling the layer
inside `flashinfer.autotune()`. An already usable backend selection can remain
unchanged. `MoELayer` construction still requires a configured native backend
that supports the activation; explicitly include `CutlassNvfp4Config()` for
activations unsupported by the default backend list. If this view is already
present, Frost reuses it without another
copy; otherwise, retaining it alongside other backend views consumes
additional weight memory. Frost does not convert other backends' views.

`artifacts/nvfp4/` retains 154 selected configurations in 34 geometry-parameterized
source templates and 400 measured two-by-two profiles. The offline sweep
covers all 44 operation/orientation/store families, including every default
gated and non-gated activation and the shared FC2. Only templates referenced
by the selected profiles are packaged.

Use `--dtype nvfp4` with the shared exporter. The corresponding stage and
full-layer benchmark is `benchmarks/bench_cudnn_frost_moe_nvfp4.py`, with
the same command-line structure as the MXFP8 benchmark. Its default input
standard deviation is 1.0: tiny inputs combined with unit global scales
can underflow E4M3 intermediate scales after gated activations. Full-layer
validation checks both other eligible backends and an independent
dequantized reference. On SM107, the benchmark checks lower-level activation
support as well as registry declarations before including a backend. It also
autotunes and times each original backend separately, preserving its complete
tactic pool and including GPU work performed by `pack_inputs`.

## MXFP8 × MXFP4 block-scale grouped pipelines

`mxfp8_mxfp4/runtime.py`, `mxfp8_mxfp4/moe.py` and
`csrc/moe_mxfp8_mxfp4.cu` provide the mixed pipeline and
`cudnn_frost_mxfp8_mxfp4` automatic candidate. Activations use E4M3 data;
weights pack two E2M1 values per byte. Both operands use E8M0 block scales
per 32 logical K elements. Logical N and K must be divisible by 128.
FC1/activation and FC2 produce BF16, with MXFP8 intermediate requantization.

The runner reuses the canonical `cutlass_mxfp8_mxfp4` weight view, including
its packed weights, F8_128x4 scales and unit input multipliers. Input scales
may be linear or F8_128x4. Automatic admission requires this view; callers
with other backend views must first add it using
`CutlassMxfp8Mxfp4Config.prepare_weights(...)` and
`weight_pack.prepare_for("cutlass_mxfp8_mxfp4", view)`. As with NVFP4, an
additional view consumes weight memory, while an existing view is reused.
`MoELayer` still requires at least one usable configured native backend;
include `CutlassMxfp8Mxfp4Config()` for activations unsupported by the default
backend list. Automatic Frost candidates are added when the layer is called.
The quantization configuration uses
`QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)`.

`artifacts/mxfp8_mxfp4/` retains 169 selected configurations in 42
geometry-parameterized source templates and 400 measured two-by-two profiles.
The offline pool covers all 44 operation/orientation/store families.

Use `--dtype mxfp8_mxfp4` with the shared exporter and
`benchmarks/bench_cudnn_frost_moe_mxfp8_mxfp4.py` for stage sweeps or full
MoELayer benchmarks. The benchmark preserves the complete eligible backend
pools and also times each original backend individually. On SM107, CUTLASS
supports all ten activations and TRTLLM supports SwiGLU, GeGLU and ReLU2;
the current CuTe DSL runner rejects W4A8 on this architecture.

## Activation sources

The SM107 selected source pool additionally supports default `GeGLU()`,
`GeGLUTanh()`, `ReLU2()`, `SiTU()`, `SwiGLUStep()`, `GELU()`, `ReLU()`,
`SiLU()`, and `Identity()`. This backend supports only SM107a (Rubin).
Availability is checked
against the architecture, geometry and activation in the manifest.

Set `MoEConfig(activation=...)` using the existing typed activation API.
Non-default scalar parameters and per-expert overrides are rejected rather
than silently ignored. In particular, selected SiTU uses gate scale 4 and
linear scale 25; SwiGLUStep uses limit 7. FC1 applies the activation to FP32
accumulators before storing BF16. Gated activations fuse two grouped GEMMs;
non-gated activations use one. FC2 sources are shared across activations.
The standalone `cudnn_frost_grouped_gemm1_swiglu` API remains SwiGLU-specific.

The native plan carries the generated argument order, including SiTU's scalar
inputs, for both normal and swap-AB layouts. Activation-specific source hashes
and artifact IDs isolate persisted tactics. Automatic selection still compares
against eligible existing backends; it does not assume that every new activation
is faster. Benchmark output names the activation and only includes baseline
backends that support it (TRT-LLM BF16 supports SwiGLU and ReLU2).

## Independent full MoE and automatic selection

The full path is:

```text
precomputed top-k ids/weights
  -> histogram + prefix + vectorized gather
  -> JIT-compiled cuDNN Frost FC1 + configured activation
  -> JIT-compiled cuDNN Frost FC2
  -> vectorized FP32 weighted reduction -> BF16 output
```

`bf16/moe.py` owns support validation, compound tactics, native plans and workspace.
`csrc/moe_bf16.cu` owns routing/finalize and calls the two native exported functions
directly, without a Python callback between stages. Both the small embedding
module and the GEMM kernels are compiled on first use through FlashInfer's JIT
infrastructure. Workspace counters
are reset on the current stream on every call and every graph replay. FC2
reuses the grouped-input buffer after FC1 finishes consuming it. Exact-shape
host plans share power-of-two-sized workspace allocations on the layer's
stream; varying the token count does not allocate a large buffer per shape.

Enable experimental automatic selection before using the original API:

```bash
export FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1
```

```python
with flashinfer.autotune():
    output = layer(act_pack, weight_pack)  # existing MoELayer and packs
output = layer(act_pack, weight_pack)      # cached winner
print(layer.winner_backend)               # "cudnn_frost_bf16" if cuDNN Frost won
```

Automatic admission is deliberately narrow:

- SM107a (Rubin), BF16 input/weights/output, finalized output;
- matching source artifacts for the requested activation and architecture;
- on SM107a, all ten supported activations use registered stage shortlists for
  `(E,H,I,top_k)` = `(12,7168,3072,1/2/4)`, `(8,4096,14336,2)`, or
  `(64,2048,1408,6)`, with `1 <= num_tokens <= 12288`;
- `PackedPrecomputed` routing, int32 ids and FP32 routing weights;
- an existing contiguous row-major `cutlass_bf16` weight view containing only
  `fc1_expert_weights[E,2I,H]` in **[up, gate]** order for gated activations
  (or `[E,I,H]` for non-gated activations), and
  `fc2_expert_weights[E,H,I]`. This is layout reuse, not CUTLASS execution;
- no bias, per-expert activation overrides, quantization scales, expert
  parallelism, shared experts, or in-kernel routing from logits.

The dispatcher first admits cuDNN Frost during `autotune()`, compares its best
compound tactic against the original backend pool, and reuses the winner in
inference. Cold non-autotuned calls keep the original behavior. Unsupported
calls keep the original candidates and a separate winner-cache entry. cuDNN Frost's
exact-shape plans are never reused across rounded token-count buckets.

Automatic candidates are registered lazily through
`fused_moe/auto_candidates.py`; the layer does not contain cuDNN Frost-specific shape
checks, runner fields, or cache branches. Each dtype's `support.py` owns admission
and the deferred runner factory. Winner keys include the eligible automatic candidate
set and exact input shape, so incompatible weight views cannot reuse a cuDNN Frost
winner. All four cuDNN Frost registrations require
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`, including reuse of cached
runners and winners after autotuning. Disabling the gate removes these
candidates from subsequent calls while retaining resources used by captured graphs.

### Rubin artifacts and swap AB

Runtime matching requires the exact SM107a device architecture. The routing/finalize
adapter is also compiled for SM107a.
cuDNN Frost's `CONFIG_sm100_*` names describe its template family, which includes SM107;
they do not specify the source kernel's target architecture.

The SM107a pool has 400 shape profiles across SwiGLU, GeGLU, GeGLUTanh,
SwiGLUStep, SiTU, GELU, Identity, ReLU, ReLU2 and SiLU.
`artifacts/bf16/moe_shortlists.json` registers two FC1 and two FC2 artifact identities
per profile. Only those kernels are compiled and combined: online autotuning
sees at most four complete MoE plans, then compares the winning Frost plan
against every original eligible backend.

Stage candidates were selected offline using the geometric mean of normalized
stage latency across uniform and skew routing, giving both distributions equal
weight. No routing classification or device-to-host routing inspection is added
at inference. The selected pairs are validated with actual routed inputs and
CUDA Graph replay. The table records no timings or machine-specific paths.

Profiles include tokens `1,16,128,512,2048,4096,8192,12288` for each supported
model/top-k. For intermediate token counts, the next measured token profile is
used as a shortlist heuristic; native launch plans still use the exact input
shape. Table entries are not claims that a plan is optimal for every routing
distribution. Autotuning chooses between the four candidates on the caller's
routing inputs, and caches their source-qualified tactic identities.

Small-shape configurations cover normal/swap AB and STG/TMA for graph tests.
Legacy gated and FC2 configurations remain available through explicit selection.
The BF16 pool contains 458 configurations and 44 source templates: ten FC1
activations plus FC2, each with normal/swap AB and STG/TMA variants. Each family
has one maintained implementation; historical producer bodies are removed.
The templates use the MoE implementation in cuDNN Frontend revision
`667fe4ce8ce437866217066f075fd4dcecad6eac`, including scheduler synchronization
and stream-ordered counter initialization. Geometry contracts, workspace sizes
and launch ABIs remain unchanged. Older configurations gain the explicit
unaligned-token promise required by this implementation.

This upgrade changes older kernel bodies; the earlier migration's AST equality
with historical sources is not a correctness claim for the upgraded kernels.
GPU correctness and performance must be revalidated with a compatible environment.
Template paths name only the family. A later producer update requires explicit
replacement and refresh of all affected manifest records, rather than adding
revision-suffixed files.

Geometry is supplied through `source.parameters` before JIT compilation, rather
than being a dynamic GPU launch argument. This includes CTA/CGA/MMA tiles, MMA
multiplicity, CTA group and the producer-derived pipeline/resource constants.
The exporter also factors the geometry-unrolled TMA-store epilogue and descriptor
box dimensions; changing only the leading tile constants would be insufficient.
New exports automatically use this representation and reject a template whose
instantiation changes the concrete source AST. The deployed process needs no
cuDNN installation to instantiate a template.

Both stages independently support normal and swap-AB artifacts. Swapped kernels
use explicit `cudnn_frost_grouped_gemm1_swiglu_swap_ab_v1` and
`cudnn_frost_grouped_gemm2_swap_ab_v1` ABIs, with matching `tactic.swap_ab` metadata.
The adapter exchanges operand order, M/N problem dimensions and output strides;
the external [up, gate]/down weight layout and grouped workspace layout stay the
same. Normal-ABI records without `swap_ab` remain valid. Contradictory ABI
metadata is rejected before loading a kernel.

`benchmarks/bench_cudnn_frost_moe_bf16.py` is the BF16 manual benchmark and tuning
entry point. Its `benchmark` command measures complete routed MoE calls;
the [MXFP8 benchmark](#mxfp8-benchmarks) provides both grouped stage measurements
and a complete MoELayer `benchmark` command.
Run the BF16 script with `--help`, or `<command> --help` for command-specific options:

| Command | Purpose |
| --- | --- |
| `benchmark` | Compare complete MoE execution against the original CUTLASS/TRTLLM pool. |
| `export` | Generate FC1/FC2 candidates, including independent normal/swap-AB configurations. |
| `sweep` | Validate and time stage candidates, then shortlisted complete-MoE pairs. |
| `select` | Copy the union of complete-MoE winners from a completed sweep. |

These commands are not invoked by inference, runtime autotuning or CI. Only
`export` needs the cuDNN Frost compiler; normal benchmarking JIT-compiles packaged sources.

The offline sweep covers both FC1 and FC2, normal/swap orientation, STG/TMA
stores, CTA/MMA tiles and clusters. It checks every exported stage against an
FP32 PyTorch reference, shortlists candidates separately for each orientation,
then checks and times all shortlisted FC1 x FC2 pairs as complete MoE calls.
Every shortlisted pair is replayed with changed routing to verify state resets.
The ranking is a staged search, not an exhaustive search of all possible pairs.

```bash
python benchmarks/bench_cudnn_frost_moe_bf16.py export \
  --cudnn-frost-source ../cudnn-frontend --artifacts /tmp/cudnn_frost-sm107 \
  --output /tmp/cudnn_frost-sm107-export.jsonl
python benchmarks/bench_cudnn_frost_moe_bf16.py sweep --source-jit \
  --artifacts /tmp/cudnn_frost-sm107 --output /tmp/cudnn_frost-sm107-sweep.jsonl --adaptive
python benchmarks/bench_cudnn_frost_moe_bf16.py export --wide-swap \
  --cudnn-frost-source ../cudnn-frontend --artifacts /tmp/cudnn_frost-sm107 \
  --output /tmp/cudnn_frost-sm107-wide-export.jsonl
python benchmarks/bench_cudnn_frost_moe_bf16.py sweep --source-jit --adaptive \
  --artifacts /tmp/cudnn_frost-sm107 --refine-from /tmp/cudnn_frost-sm107-sweep.jsonl \
  --output /tmp/cudnn_frost-sm107-refine.jsonl
python benchmarks/bench_cudnn_frost_moe_bf16.py select \
  --artifacts /tmp/cudnn_frost-sm107 --results /tmp/cudnn_frost-sm107-refine.jsonl \
  --selected-dir flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm/artifacts/bf16 \
  --output /tmp/cudnn_frost-sm107-selection.jsonl
```

Run on an idle target GPU with a CUDA toolkit that accepts `compute_107a`.
JSONL records include GPU/clock settings, all timings, errors and artifact IDs.
Generation and timing are separate commands; do not run them concurrently.
Only selected, validated source kernels are copied into the packaged manifest.
`--adaptive` reduces repeats for slow stage candidates, retaining all rounds
and at least one batched graph replay per sample; each record stores its repeat
count. Complete MoE timing always uses the requested repeat count. `--resume`
continues an interrupted sweep while preserving completed cases and RNG draws.
The optional refinement adds wider swap tiles to the previous stage finalists
and re-ranks complete MoE pairs; it does not remeasure previously rejected tiles.
The SM107 search uses E/H/I/top-k = 12/7168/3072/2, 8/4096/14336/2 and
64/2048/1408/6; tokens 1,16,128,512,2048,4096,8192; and uniform/skew routing.
It exports 372 model configurations (62 configurations per geometry and stage).
That original search uses default SwiGLU. New searches accept `--activation`
and use the corresponding gated or non-gated FC1 geometry and reference.
`export --config NAME` can be repeated to choose an explicit tile pool; both
STG and TMA stores are tried for each normal or swap-AB config.

Earlier sweep timings and validation records describe the precompiled-object
implementation. Use the full-MoE benchmark below to measure the source/JIT
implementation with the deployed compiler; historical speedups do not establish
performance after a compiler upgrade. The original sweep recorded one
unexplained E64/16-token/skew illegal-memory-access failure; subsequent reruns
and memchecks passed. The source-distribution change does not establish a fix
for that historical failure.

Automatic admission follows the experimental-auto gate. Explicit experimental
API calls remain an opt-in and do not require the environment variable. A
once-per-backend experimental warning is issued if cuDNN Frost actually wins.
Release admission/ownership and the tracking issue remain to be settled before
publishing this research integration.

Keep the layer alive while its CUDA graphs are used, and use one instance per
thread/stream. Input contents, including expert ids, may change between
replays. Routing ids outside `[0,E)` are masked by this private runner rather
than used as addresses. Profiling cache keys include both artifact identities.

The full-MoE tests cover all 16 small-shape tactics, top-k 1/2/4,
uneven/empty experts, non-default streams, changed routing during graph replay,
interleaved packs, unsupported semantic overrides, and the unchanged
`MoELayer` API. Every packaged model pair is also checked on ragged partial tiles and graph
replay. Forced-winner tests verify real cuDNN Frost dispatch at each admitted model
geometry independently of machine performance.

Reproduce the full-MoE comparison, not a PyTorch stage baseline:

```bash
python benchmarks/bench_cudnn_frost_moe_bf16.py benchmark --source-jit --tokens 8192,12288
python benchmarks/bench_cudnn_frost_moe_bf16.py benchmark --source-jit --tokens 4096,8192 --top-k 2 --routing skew
```

The benchmark autotunes the original CUTLASS/TRTLLM pool and the same pool plus
cuDNN Frost, reports both winners, and separately times the best full cuDNN Frost plan even
if it loses. Measurements use rotated order, batched CUDA graphs and medians;
output checks use relative L2 against the original winner. It checks for other
compute processes on the same GPU before tuning and between measurement
rounds, aborting rather than reporting contaminated results. `--source-jit` is
only for source checkouts whose editable-install data symlinks are incomplete.

### Validating additional artifacts

Use the offline exporter described in [Generate artifacts](#generate-artifacts)
to export both stages into an explicit research directory. A successful export
is not a correctness or performance result.

```bash
python benchmarks/bench_cudnn_frost_moe_bf16.py benchmark --source-jit \
  --experts 8 --hidden 4096 --intermediate 14336 --top-k 2 \
  --tokens 4096,8192 --probe-cudnn-frost --artifact-root /tmp/cudnn_frost-candidates \
  --sweep-cudnn-frost
```

`--probe-cudnn-frost` and `--artifact-root` are benchmark-only overrides: they do not
change the installed runtime's artifact search or automatic admission policy.
An explicit artifact root replaces the packaged source pool for that benchmark process;
export both stages for the requested geometry into it.
To evaluate the packaged E64 Rubin pool outside automatic admission, use
`--experts 64 --hidden 2048 --intermediate 1408 --top-k 6 --probe-cudnn-frost`.
`--sweep-cudnn-frost` checks every compound tactic against the original winner and
separately ranks full-MoE CUDA graphs. Those diagnostic rankings are **not** the
automatic API result: use the `with_cudnn_frost` winner and its interleaved timing to
judge whether users actually benefit. Only validated, useful source kernels should be
selected for packaging.

## Standalone cuDNN Frost FC2 PoC

`bf16/fc2.py` implements an internal prepared launcher for:

```text
grouped_intermediate[S,I] @ down_weight[expert,H,I].T -> grouped_output[S,H]
```

All data tensors are contiguous BF16. Offsets are int32[E], one group per
local expert, with an implicit final end at S. Empty/uneven expert groups
are supported. This stage does not apply routing weights or finalize/scatter.
Preparation validates offsets and compiles or reloads the native TVM-FFI function outside
CUDA Graph capture. `PreparedFc2.run()` resets the caller-owned descriptor /
scheduler workspace and invokes that Function, without importing cuDNN Frost or
compiling kernels during execution. Tensor metadata is prepared once; tensor contents may
change between calls. Use distinct workspaces/plans for concurrent streams.

The FC2 configurations target SM107a with dynamic S and are shared across
activations. FC2's N is the **hidden size**, whereas FC1's N is the intermediate
size. FC2 does not apply an activation; its configurations are selected
independently from FC1.

The FC2 records use `op=grouped_gemm2` and ABI `cudnn_frost_grouped_gemm2_v1`;
the manifest lists their Python source paths and digests. FC1 discovery ignores
these records, retaining its separate candidate pool.

Example build-box export (repeat with the other tile/store modes and shape):

```bash
python -m flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.export \
  --op grouped_gemm2 \
  --output-dir flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm/artifacts/bf16 \
  --cudnn-frost-revision 667fe4ce8ce437866217066f075fd4dcecad6eac \
  --config CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma \
  --cta-group 2 --store-mode stg --experts 12 --n 7168 --k 3072
```

The exporter renders and compiles the candidate with the chosen cuDNN Frost revision,
then freezes `generated_path` and its device helpers into standalone Python.
Device function bodies are retained; cuDNN Frost imports and its secondary compiled
cache are removed. Sources use a `cudnn_frost_` prefix and configuration-based
artifact names, preserving the operation, architecture, geometry, tile and
store mode in the filename,
including `swapAB` for swapped configurations. `select` verifies source hashes
and refuses to replace different bytes.

`tests/experimental/test_cudnn_frost_selected_kernels.py` focuses on the complete
**cuDNN Frost FC1 + SwiGLU -> cuDNN Frost FC2 -> finalize** path: every compound tactic on
small shapes and both auto-admitted model geometries, empty and uneven groups,
non-default streams, changed routing and repeated graph replay. The independent
runner tests disable the cuDNN Frost compiler; the small-shape tests also forbid a
CUTLASS MoE launch. The suite additionally covers request-state isolation,
unsupported semantics, automatic admission/fallback and original `MoELayer`
execution with a selected cuDNN Frost winner. It does not assert performance rankings.

## Backend isolation

The old mixed cuDNN Frost/CUTLASS MoE runner, native GEMM1 plan adapter/interface,
associated tests and full-MoE benchmark have been deleted. The CUTLASS CUDA
sources and its Python runner remain identical to HEAD. Their workspace
planning, tactic pools and cache keys are not modified for cuDNN Frost. Other backend
runners do not import or inspect the cuDNN Frost manifest during execution. The only
execution-dispatch change in core is `MoELayer` adding a separate candidate;
backend-specific implementation and shape admission stay in this directory.

The standalone FC1 API, offline exporter, FC2 prepared launcher and packaged
source kernels remain. Core correctness and integration tests live in
`tests/experimental/test_cudnn_frost_selected_kernels.py`; redundant standalone and
auxiliary mock tests are omitted. BF16 benchmarking and offline tuning share
`benchmarks/bench_cudnn_frost_moe_bf16.py`; the earlier standalone FC1/FC2 benchmark
scripts have been removed. MXFP8 grouped-stage measurements use
`benchmarks/bench_cudnn_frost_moe_mxfp8.py`. These tools do not restore the removed
hybrid hooks.

## Generate artifacts

Run on SM107a in the build environment containing the matching cuDNN Frost and
CuTe DSL revisions:

```bash
python -m flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.export \
  --output-dir flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm/artifacts/bf16 \
  --cudnn-frost-revision 667fe4ce8ce437866217066f075fd4dcecad6eac \
  --config CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma \
  --store-mode tma \
  --experts 8 --n 256 --k 128
```

Run the command repeatedly with other tile/CTA/scheduler/store configurations to add
candidates. Select tile configurations using measurements for the target
problem geometry and activation.
FlashInfer sees every matching stable artifact id as a tactic and autotunes
among them using the caller's real grouped offsets.

Use `--store-mode stg` to invoke cuDNN Frost's `force_stg_epi()` path for the same
geometry. TMA-store and STG are distinct artifacts and distinct autotune
tactics.

## Execute

```python
import torch
import flashinfer

workspace_bytes = flashinfer.cudnn_frost_grouped_gemm1_swiglu_workspace_size(
    grouped_tokens, gate_weights, up_weights, first_token_offset, scale
)
workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
out = flashinfer.cudnn_frost_grouped_gemm1_swiglu(
    grouped_tokens,
    gate_weights,
    up_weights,
    first_token_offset,
    scale,
    workspace,
)
```

To profile all matching packaged artifacts and cache the winner, run the call
once in FlashInfer's autotuning context. Eager mode measures the complete
per-call cost without requiring the delay-kernel utility:

```python
policy = flashinfer.MeasurementPolicy(execution_mode="eager", cold_l2=False)
with flashinfer.autotune_v2(mode="tune", measurement_policy=policy):
    out = flashinfer.cudnn_frost_grouped_gemm1_swiglu(
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
        workspace,
    )
```

The first tune for an exact tensor-shape signature profiles the caller's real
grouped offsets. Later calls reuse the cached winning stable artifact id.

## Artifact metadata

Each schema-v2 manifest record contains a `source.path`/`source.sha256` pair,
optional `source.parameters` for a shared template, and exact `E/N/K/group-count`, architecture, workspace size,
cuDNN Frost revision, template, tile, CTA group, scheduler, and SHA-256. `S` remains
dynamic because the generated cuDNN Frost host kernel receives it at launch.

The package source directory is read-only at runtime. Generating or replacing
source artifacts is an offline operation. Instantiated sources are atomically
written under `FLASHINFER_GEN_SRC_DIR`; compiled objects and cache metadata live
under `FLASHINFER_JIT_DIR`. Both are in the writable JIT workspace. Source and
tactic cache identities use the rendered source digest, including geometry, so
two configurations sharing a template cannot alias one specialization.
Schema-v1 precompiled-object manifests are intentionally rejected; re-export
research candidates with this source exporter before sweeping them.
