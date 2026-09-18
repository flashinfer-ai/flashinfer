# cuDNN Frost-selected BF16 MoE kernel PoC

This experimental backend follows Cake's source-distribution model. cuDNN Frost is
an offline generator: FlashInfer ships standalone Python/CuTe DSL kernels and a
SHA-256 manifest, with no precompiled `.o` files. Required cuDNN Frost device helpers
are included in each generated source. The deployed process does not import
`cudnn-frontend`, construct a cuDNN Frost graph, or run the cuDNN Frost graph compiler.

On first use, `runtime.py` compiles the selected sources through
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
`sm_100a` / `sm_107a` compile-option support, TVM-FFI support, and conflicting
`CUTE_DSL_ARCH` overrides. Probe results are cached per architecture and source
set; no kernel is compiled or launched by the probe.

Missing capabilities skip cuDNN Frost in automatic MoE selection and raise an
actionable `NotImplementedError` for explicit cuDNN Frost use. Source integrity errors
and later compiler/runtime failures remain errors; static capability checks do
not guarantee compiler correctness. Validation uses the installed internal DSL
build, not a public 4.8 wheel; public 4.8+ remains the planned deployment baseline.
This backend does not change the repository-wide dependency floor.
The native routing/finalize adapter additionally needs a CUDA toolkit supporting
the target architecture. Set `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` to compile
without persisting CuTe DSL artifacts; neither mode writes compiled objects
into the installed package.

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

## Independent full MoE and automatic selection

The full path is:

```text
precomputed top-k ids/weights
  -> histogram + prefix + vectorized gather
  -> JIT-compiled cuDNN Frost FC1 + SwiGLU
  -> JIT-compiled cuDNN Frost FC2
  -> vectorized FP32 weighted reduction -> BF16 output
```

`moe.py` owns support validation, compound tactics, native plans and workspace.
`csrc/moe.cu` owns routing/finalize and calls the two native exported functions
directly, without a Python callback between stages. Both the small embedding
module and the GEMM kernels are compiled on first use through FlashInfer's JIT
infrastructure. Workspace counters
are reset on the current stream on every call and every graph replay. FC2
reuses the grouped-input buffer after FC1 finishes consuming it. Exact-shape
host plans share power-of-two-sized workspace allocations on the layer's
stream; varying the token count does not allocate a large buffer per shape.

Users keep calling the original API:

```python
with flashinfer.autotune():
    output = layer(act_pack, weight_pack)  # existing MoELayer and packs
output = layer(act_pack, weight_pack)      # cached winner
print(layer.winner_backend)               # "cudnn_frost_bf16" if cuDNN Frost won
```

Automatic admission is deliberately narrow:

- SM100a or SM107a (Rubin), BF16 input/weights/output, default SwiGLU, finalized output;
- `(E, hidden size, intermediate size)` in `(12,7168,3072)` or `(8,4096,14336)`;
- `num_tokens * top_k >= 8192` expanded rows;
- `PackedPrecomputed` routing, int32 ids and FP32 routing weights;
- an existing contiguous row-major `cutlass_bf16` weight view containing only
  `fc1_expert_weights[E,2I,H]` in **[up, gate]** order and
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
checks, runner fields, or cache branches. `support.py` owns admission and the
deferred runner factory. Winner keys include the eligible automatic candidate
set and exact input shape, so incompatible weight views cannot reuse a cuDNN Frost
winner. Other registrations require the normal experimental-auto gate; the
branch-local cuDNN Frost exception described below remains explicit.

The selected SM100a source configurations provide:

| E / H / I | FC1 configs | FC2 configs | Full-MoE tactics |
| --- | ---: | ---: | ---: |
| 12 / 7168 / 3072 | 4 | 6 | 24 |
| 8 / 4096 / 14336 | 4 | 4 | 16 |

Both stages include N128/1-CTA and N256/2-CTA, each in STG and TMA-store
modes. E12 additionally includes FC2 N256/2-CTA with cluster 2x2 in both store
modes. All use M128/K128B and CLC. cuDNN Frost profiling warms and samples multiple
captured graph replays; it does not alter another backend's tuning settings.
The small E=8/H=128/I=256 artifacts remain available for standalone correctness
tests but are not admitted to auto. Other SM100a model geometries are not auto-admitted.

### Rubin artifacts and swap AB

SM107a has a separate source configuration pool. Runtime matching uses the exact device
architecture; an SM100a kernel is never selected on Rubin. The routing/finalize
adapter is also compiled for the selected device architecture and cached separately.
cuDNN Frost's `CONFIG_sm100_*` names describe its template family, which includes SM107;
they do not specify the source kernel's target architecture.

The selected SM107a pool contains the union of complete-MoE winners from the
cuDNN Frost configuration sweep. E64 is retained for explicit/research use; it is not
automatically admitted because the original TRTLLM backend was faster:

| E / H / I | FC1 configs | FC2 configs | Full-MoE tactics |
| --- | ---: | ---: | ---: |
| 12 / 7168 / 3072 | 9 | 11 | 99 |
| 8 / 4096 / 14336 | 8 | 9 | 72 |
| 64 / 2048 / 1408 | 6 | 9 | 54 |

Eight additional small-shape configurations cover normal/swap x STG/TMA in each stage.
The source migration preserves all 26 SM100a and 60 SM107a configuration IDs,
producer revisions, geometry contracts, workspace sizes and launch ABIs. Each
record has its own `sources/cudnn_frost_<artifact_id>.py`, using the previous
`.o` basename with a `cudnn_frost_` prefix and a `.py` extension. These 86 files
contain 57 distinct source bodies; the manifest retains a SHA-256 digest for
every file.

Both stages independently support normal and swap-AB artifacts. Swapped kernels
use explicit `cudnn_frost_grouped_gemm1_swiglu_swap_ab_v1` and
`cudnn_frost_grouped_gemm2_swap_ab_v1` ABIs, with matching `tactic.swap_ab` metadata.
The adapter exchanges operand order, M/N problem dimensions and output strides;
the external [up, gate]/down weight layout and grouped workspace layout stay the
same. Normal-ABI records without `swap_ab` remain valid. Contradictory ABI
metadata is rejected before loading a kernel.

`benchmarks/bench_cudnn_frost_moe_bf16.py` is the single manual benchmark and tuning
entry point. Run it with `--help`, or `<command> --help` for command-specific
options:

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
  --selected-dir flashinfer/experimental/cudnn_frost_selected_kernels/artifacts \
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
The search uses default SwiGLU, matching this backend's contract; model-specific
clamped/OAI/SiTU activations remain unsupported.

Earlier sweep timings and validation records describe the precompiled-object
implementation. Use the full-MoE benchmark below to measure the source/JIT
implementation with the deployed compiler; historical speedups do not establish
performance after a compiler upgrade. The original sweep recorded one
unexplained E64/16-token/skew illegal-memory-access failure; subsequent reruns
and memchecks passed. The source-distribution change does not establish a fix
for that historical failure.

This branch intentionally allows automatic admission without an environment
variable, per the requested unchanged-user-code PoC. This is a scoped exception
to the usual experimental-auto gate, not a change to the policy for other
experimental backends. A once-per-backend experimental warning is issued if
cuDNN Frost actually wins. Release admission/ownership and the tracking issue remain
to be settled before publishing this research integration.

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

`fc2.py` implements an internal prepared launcher for:

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

The fourteen original FC2 configs target SM100a with dynamic S:

| Model contract | FC2 GEMM N / K | Configs |
| --- | --- | --- |
| E=8, H=128, I=256 | 128 / 256 | N128/1-CTA and N256/2-CTA, each STG/TMA |
| E=12, H=7168, I=3072 | 7168 / 3072 | N128/1-CTA, N256/2-CTA, and N256/2-CTA/cluster-2x2, each STG/TMA |
| E=8, H=4096, I=14336 | 4096 / 14336 | N128/1-CTA and N256/2-CTA, each STG/TMA |

All use CTA-M128/K128B and the CLC scheduler. FC2's N is the **hidden size**,
whereas FC1's N is the intermediate size. FC2 has no SwiGLU, so FC1's usual
preference for N128 must not be assumed for this stage.

The FC2 records use `op=grouped_gemm2` and ABI `cudnn_frost_grouped_gemm2_v1`;
the manifest lists their Python source paths and digests. FC1 discovery ignores
these records, retaining its separate candidate pool.

Example build-box export (repeat with the other tile/store modes and shape):

```bash
python -m flashinfer.experimental.cudnn_frost_selected_kernels.export \
  --op grouped_gemm2 \
  --output-dir flashinfer/experimental/cudnn_frost_selected_kernels/artifacts \
  --cudnn-frost-revision c61a5e4eb5c5d26a920d8504c655fa9a00919484 \
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
auxiliary mock tests are omitted. Benchmarking and offline tuning share
`benchmarks/bench_cudnn_frost_moe_bf16.py`; the earlier standalone FC1/FC2 benchmark
scripts have been removed. These tools do not restore the removed hybrid hooks.

## Generate artifacts

Run in the build environment containing the matching cuDNN Frost and CuTe DSL
revisions:

```bash
python -m flashinfer.experimental.cudnn_frost_selected_kernels.export \
  --output-dir flashinfer/experimental/cudnn_frost_selected_kernels/artifacts \
  --cudnn-frost-revision 6f539860d6e737c59004c1fc3c1c02b2fe25b64e \
  --config CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma \
  --store-mode tma \
  --experts 8 --n 256 --k 128
```

Run the command repeatedly with other tile/CTA/scheduler/store configurations to add
candidates. CTA-N=128 is typically the strongest choice for SwiGLU, so it is
the recommended first/default artifact, not a compatibility restriction.
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

Each schema-v2 manifest record contains a `source.path`/`source.sha256` pair and exact `E/N/K/group-count`, architecture, workspace size,
cuDNN Frost revision, template, tile, CTA group, scheduler, and SHA-256. `S` remains
dynamic because the generated cuDNN Frost host kernel receives it at launch.

The package source directory is read-only at runtime. Generating or replacing
source artifacts is an offline operation. Compiled objects and cache metadata
are created only under `FLASHINFER_JIT_DIR` in the writable JIT workspace.
Schema-v1 precompiled-object manifests are intentionally rejected; re-export
research candidates with this source exporter before sweeping them.
