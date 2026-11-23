# Experimental GLM-5 TP8 decode

Owner: Long Cheng (@thynics). Tracking issue: flashinfer-ai/flashinfer#TBD.

Status: experimental API and experimental backend, JIT-only. Calling an API in
`flashinfer.glm5_tp8_decode` is the opt-in; no environment variable is
involved. The feature takes no part in automatic routing: it is never selected
by `backend="auto"`, autotuning or trace-apply, so
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` does not affect it.

This backend exposes six specialized decode kernels through two collective
operator families. It is intended for one node with eight NVLink-connected
B300 GPUs (SM103, 148 SMs each), batch one, query length one or four, hidden
size 6144, and contiguous BF16 caches. It has no serving-framework dependency.
The kernels compile on demand and are not registered in the AOT package.
Use a source checkout or a nightly wheel that includes experimental sources;
release wheels built with `FLASHINFER_WHEEL_EXCLUDE_EXPERIMENTAL_SOURCES=1`
omit these CUDA files.

The fused stages share a collective workspace and fixed model geometry, which
form one explicit experimental API contract. Proposed graduation target:
FlashInfer 0.8.0, subject to maintainer release planning. The first lifecycle
review takes place within four weeks of admission, as recorded in the tracking
issue; if admission moves, the review moves with it. That review covers
deployment-independent ownership, graph replay, numerical bounds, reproducible
operator timings, and whether the fixed geometry should remain a model-specific
API or move behind broader operations.

## Operators and ownership

- **MoE:** RMSNorm, replicated router, top-eight routed experts, one shared
  expert, tensor-parallel reduction and residual addition. All eight ranks
  participate; the intermediate dimension per rank is 256. There are 256 routed
  experts and 257 weight banks including the shared expert.
- **Sparse attention:** logical rank zero computes the q4 indexer or publishes a
  supplied selection; ranks 1–7 compute MLA and update their latent/positional
  caches. Each has ten head slots; rank 7 contains four real heads and six
  zero-padded slots. The operator includes output projection, tensor-parallel
  reduction and residual addition. q1 reuses an accepted selection; q4 can use
  full indexing or a supplied selection. Full indexing selects the exact top
  2048 of actual computed logits. For unchanged inputs, the selected set is stable;
  its slot order is unspecified.

Create a workspace collectively with `create_glm5_tp8_decode_workspace` and a
logical rank in an eight-rank communication group. Physical CUDA ordinals may
be permuted or masked. The workspace checks distinct GPU identities, device
geometry and native peer atomic support. A Gloo process group is suitable for
metadata and IPC-handle exchange. `operators=("moe",)` or `("attention",)` avoids
allocating and compiling the other family.

Pack weights once outside capture from the layer's checkpoint tensors (see
[Weight packing](#weight-packing)), then prepare plans with fixed tensor
addresses. Preparation validates rank-local arguments collectively before any
peer-waiting kernel launches. Packed weights own their storage. Treat their
objects, dictionaries and contents as immutable while plans use them. Plans
retain their operands and private device descriptors. The caller owns inputs,
outputs, caches and RoPE tables; a workspace owns communication records,
generation counters and operator scratch.

A plan call launches on the current stream and returns its output without a
host synchronization. Capture that call and update input contents, positions
and selections in place between replays. Keep the plan and workspace alive for
all replays; CUDA graphs do not retain Python pointer owners. Never resize or
replace bound tensors during replay. Destroy the workspace collectively after
all work finishes and before destroying its process group. Destroying a workspace
invalidates its plans. `memory_usage()` reports symmetric physical allocation
bytes and local scratch/descriptor bytes; caller-owned weights and caches are
excluded.

All ranks must execute the same operator sequence. Do not overlap these
persistent peer-waiting grids, including grids from separate workspaces, on the
same GPUs. Eager calls on one workspace order successive streams. The caller
must order graph replays across streams, order input producers before execution,
and order output consumers after execution. Independent workspaces may be
interleaved when their launches are serialized. `reset_request()` clears the
selection-consumption channel while preserving collective generations; it is a
collective operation outside capture. There is no per-call external NaN fill.

## Tensor contracts

Inputs and outputs are contiguous, 16-byte-aligned BF16 `[q_len,6144]` tensors.
All ranks provide the same residual input. Output may not alias inputs, weights,
cache or workspace scratch. Autograd is unsupported. Every launch is a CUDA
cooperative launch, so a grid that cannot be fully resident fails to launch
instead of running partially. Programmatic dependent launch is enabled by
default; `enable_pdl=False` uses plain stream ordering. Initialization checks
the kernel's resource occupancy before execution.

### Weight packing

Both packers take one decoder layer of the FP8 checkpoint and a logical rank:
`pack_glm5_tp8_attention_weights(weights, *, rank, device=None)` and
`pack_glm5_tp8_moe_weights(weights, *, rank, device=None)`. `weights` maps
the Hugging Face GLM-5 tensor names, without the `model.layers.{i}.` prefix,
to the layer's full (unsharded) tensors. Each packer reads only the entries
that its rank needs and ignores all others, so every rank can pass the same
mapping, and it takes the rank's tensor-parallel slices itself. A missing
entry, or a wrong shape, dtype, layout or device, raises before any large
allocation and names the entry.

- FP8 weights are E4M3FN codes. Every `*.weight_scale_inv` is the FP32 scale of
  a 128 x 128 block and multiplies the decoded codes. Packing rearranges bytes
  only: there is no quantization, requantization or scale conversion.
- 1-D norm weights and biases and `mlp.gate.e_score_correction_bias` may be
  BF16 or FP32 and are widened exactly to FP32. `mlp.gate.weight` and
  `self_attn.indexer.weights_proj.weight` must be BF16; FP32 is rejected
  because narrowing would round.
- All required tensors are strided and on one CPU or CUDA device.
  Noncontiguous views and aliasing between entries are accepted; inputs are
  only read. Rank-local slices are taken on that device and only they are
  copied to `device` (default: the input device), where the layouts are built.
  The packed weights live on `device`, own their storage and alias no input.
- For MoE, pass CPU tensors (for example memory-mapped safetensors) with
  `device="cuda"`: about one eighth of the 9.7 GB of expert codes per layer is
  then copied. Whole-layer CUDA inputs also work, at about 9.7 GB of transient
  memory per GPU. An attention layer is 175 MB, of which `o_proj` is 101 MB.

Attention entries on every rank:

| Name | Dtype | Shape |
|---|---|---|
| input_layernorm.weight | BF16 or FP32 | [6144] |
| self_attn.q_a_proj.weight / .weight_scale_inv | E4M3FN / FP32 | [2048,6144] / [16,48] |
| self_attn.q_a_layernorm.weight | BF16 or FP32 | [2048] |

Additional entries on rank zero (DSA indexer), used whole:

| Name | Dtype | Shape |
|---|---|---|
| self_attn.indexer.wk.weight / .weight_scale_inv | E4M3FN / FP32 | [128,6144] / [1,48] |
| self_attn.indexer.wq_b.weight / .weight_scale_inv | E4M3FN / FP32 | [4096,2048] / [32,16] |
| self_attn.indexer.weights_proj.weight | BF16 | [32,6144] |
| self_attn.indexer.k_norm.weight / .bias | BF16 or FP32 | [128] / [128] |

Additional entries on ranks 1–7 (MLA). Rank `r` owns the global heads
`h0 = 10 * (r - 1)` through `h0 + hv - 1`, with `hv = min(10, 64 - h0)`, in
ten head slots; the packer fills rank 7's six absent slots with zero codes and
zero scales.

| Name | Dtype | Full shape | Rank-local part |
|---|---|---|---|
| self_attn.kv_a_proj_with_mqa.weight / .weight_scale_inv | E4M3FN / FP32 | [576,6144] / [5,48] | whole: 512 latent rows, then 64 RoPE key rows |
| self_attn.kv_a_layernorm.weight | BF16 or FP32 | [512] | whole |
| self_attn.q_b_proj.weight / .weight_scale_inv | E4M3FN / FP32 | [16384,2048] / [128,16] | rows [256·h0, 256·(h0+hv)) / scale rows [2·h0, 2·(h0+hv)) |
| self_attn.kv_b_proj.weight / .weight_scale_inv | E4M3FN / FP32 | [28672,512] / [224,4] | rows [448·h0, 448·(h0+hv)) / scale rows [3.5·h0, 3.5·(h0+hv)) |
| self_attn.o_proj.weight / .weight_scale_inv | E4M3FN / FP32 | [6144,16384] / [48,128] | columns [256·h0, 256·(h0+hv)) / scale columns [2·h0, 2·(h0+hv)) |

Each head's 448 `kv_b_proj` rows are 192 rows of W_UK (key without RoPE),
which the packer transposes to absorb the query into the latent space, and
256 rows of W_UV (value), which project the latent attention output. The head
boundaries fall on 128-row scale blocks for every rank. On layers without
indexer tensors (GLM-5.2 indexer type `"shared"`), rank zero does not pack
attention weights and prepares shared mode with
`weights=None, caches=None, rope=None`.

MoE entries, 1545 in total. Rank `r` keeps intermediate channels
`[256·r, 256·r + 256)` of the shared expert and of every routed expert; bank
zero of the packed weights is the shared expert and bank `e + 1` is routed
expert `e`. The dense MLP layers are not served by this operator.

| Name | Dtype | Full shape | Rank-local part |
|---|---|---|---|
| post_attention_layernorm.weight | BF16 or FP32 | [6144] | whole |
| mlp.gate.weight | BF16 | [256,6144] | whole |
| mlp.gate.e_score_correction_bias | FP32 or BF16 | [256] | whole |
| mlp.shared_experts.{gate_proj,up_proj}.weight / .weight_scale_inv | E4M3FN / FP32 | [2048,6144] / [16,48] | rows [256·r, 256·r+256) / scale rows [2·r, 2·r+2) |
| mlp.shared_experts.down_proj.weight / .weight_scale_inv | E4M3FN / FP32 | [6144,2048] / [48,16] | columns [256·r, 256·r+256) / scale columns [2·r, 2·r+2) |
| mlp.experts.{e}.… for e in 0..255 | as for the shared expert | | |

### Caches, positions and selections

Cache dictionaries contain `indexer_k_cache` BF16 `[rows,128]` on rank zero
when full indexing is used, and `kv_cache` BF16 `[rows,512]` plus `pe_cache`
BF16 `[rows,64]` on other ranks. All caches have at least `workspace.max_seq_len+8` rows; cache
rows may include padding. RoPE is the complete FP32 `[rows,64]` table, with at
least as many rows as the caches. Pass its base address, not a current-query
slice. `cur_pos` is a device int32 scalar, 4-byte aligned, holding the first query
position. Its value must satisfy `0 <= cur_pos` and
`cur_pos+q_len <= workspace.max_seq_len` on **every replay**. Preparation checks
the initial value; subsequent replay values are a caller precondition.

The default capacity is 202752 tokens. An explicit `max_seq_len` supports
up to 1052672 tokens, allowing a 1M-token prefix and 4096 decode tokens. The
private communication arenas have a fixed size; public caches and the
workspace's logits/candidate scratch grow with the requested capacity. A
single layer uses approximately 576 MiB of KV/PE cache on each attention rank
at 512K tokens, or 1152 MiB at 1M tokens, excluding padding and weights.

Full q4 indexing returns the exact top 2048 of the computed FP32 logits at
every supported row length: tokens are ranked by their complete FP32 scores,
with -0.0 below +0.0 and the smaller token ID winning equal scores, and
candidates are never truncated. Rows above 202760 scored tokens use a long-row
selector over per-group score summaries kept in the workspace's candidate
scratch; it preserves the same contract, and `csrc/attention/long_topk.cuh`
describes its design. The slot order of the selected IDs is unspecified.

Shared mode takes rank-zero `selection` int32 `[q_len,2048]`. For sparse queries,
IDs must be unique, in cache bounds and causal for their query. Dense queries
use their causal prefix and ignore unused selection entries. Runtime values
inside captured tensors are not host-validated on replay. Finite inputs and
scale values are required. Counter generations are 32-bit. The supported
lifetime is fewer than 2^24 calls per query-length counter; recreate workspaces
before reaching that limit to avoid scaled-counter wraparound.

## Numerical and protocol boundaries

Weights retain FP8 codes and FP32 scales. MoE is W8A8: the normalized input
and the expert intermediate activations are quantized dynamically to FP8 E4M3,
with one FP32 scale per 128 values, before the expert GEMMs; the router stays
BF16. Attention and indexer projections multiply FP8 weights by BF16
activations, except the BF16 indexer head-weight projection. BF16 intermediate
boundaries, FP32 accumulation/reduction order, approximate sigmoid routing and
RoPE arithmetic are part of the operator contract. This implementation uses
precise compile flags and does not enable `--use_fast_math`.

MoE publication reserves exact F32 and BF16 pending encodings. Computed NaNs
retain their numerical class and are published with a different canonical
encoding. Each polled component uses a matching scalar access width; consumers
wait for every component. Finite inputs can still overflow internally or
normalize zero routing mass and produce nonfinite numerical results.

Each rank emits BF16 partials. Wire records are summed in logical rank order in
FP32, the residual is added in FP32, and one final BF16 rounding produces output.
Tests independently decode current-generation records and require bitwise
agreement with that reduction. Full indexing uses atomic tickets to pack IDs,
so the same selected set can reach attention in a different order on replay.
Consequent floating-point reduction-order changes can alter output bits across
full-indexer calls. Tests require stable sorted ID sets and validate every call
against the numerical contract; fixed-selection replays keep bitwise checks.
Projection and MoE stage tests use bounded BF16 error metrics and explicitly
handle analytically ambiguous router ties. A compound high-precision end-to-end
comparison is diagnostic only; the stage-level checks define the contract.

Communication uses aligned scalar 128-bit system-scope record loads/stores and
native peer atomics. Lifetime credits prevent reuse while peers still read a
previous generation. Initialization and NaN sentinels are internal to the
workspace/kernel; callers must not reset protocol storage between layers.

## Running

From a source checkout on eight assigned B300 GPUs:

```bash
torchrun --standalone --nproc-per-node=8 examples/glm5_tp8_decode.py
```

The example uses zero expert weights to demonstrate ownership and graph replay.
Test commands and numerical coverage are documented in
`tests/experimental/glm5_tp8_decode/README.md`. Run that package with eight
processes; single-process invocation skips the GPU tests. Tests include changed
inputs/positions, q1/q4 alternation, interleaved workspaces, lifecycle failures,
first graph exchange under lazy module loading, and physical-device permutation.
Standalone operator validation does not certify a complete serving model's
startup time, memory headroom or end-to-end determinism.
