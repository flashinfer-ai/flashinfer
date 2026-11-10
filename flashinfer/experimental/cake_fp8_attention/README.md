# Experimental FP8 attention

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection.

## Contract

```python
from flashinfer.prefill import prepare_fp8_attention, fp8_attention_workspace_bytes

runner = prepare_fp8_attention(q, k, v, out, backend="cake")  # softmax_scale=None, workspace=None
runner.launch()  # or runner(): quantizes q, k, v on the device and writes out
```

Supported on SM90 GPUs. `q`, `k`, `v` and `out` are BF16 tensors of the
logical shape `[B, H, S, 128]` with any strides: the channel stride is 1, the
batch, head and token strides are multiples of 16 elements (32 bytes), no two
rows alias. An `[B, S, H, 128]` buffer is passed as its `[B, H, S, 128]` view
(`x.transpose(1, 2)`) and is read or written in place; the view is the
contract (strides alone cannot tell the two storage orders apart when
`S == H`). Any `B, H, S >= 1` with `B * H * S * 128 < 2**31` (the kernels
index their operands with 32-bit integers). `softmax_scale` defaults to
`128 ** -0.5` and must be positive. The route computes dense, non-causal
multi-head attention with `Hq == Hkv`:
`out = bf16(softmax(q k^T * softmax_scale) v)`.

Not supported: causal or windowed masks, GQA/MQA, paged KV, returning the
log-sum-exp, backward, head dimensions other than 128, dtypes other than BF16.

Preparation validates the tensors, carves the workspace and binds the
arguments of the four programs of the chain. Nothing is compiled, allocated or
synchronised at launch. Every launch re-reads `q`, `k` and `v` --
quantization is part of the launch -- so their values are free between
launches while their identities, shapes and strides are bound. Prepare a new
runner when a tensor, a shape or a stride changes.

### Workspace

`workspace` is a caller-owned contiguous `uint8` tensor on the inputs' device,
256-byte aligned, of at least `fp8_attention_workspace_bytes(B, H, S, device)`
bytes; without one the runner allocates it at prepare (never at launch). The
workspace holds the statistics and the quantized operands of one launch, in
256-byte slots, carved in the order below (the order of the registry's
workspace rule). With `N = ceil(S / 128)` and `Sp = N * 128`:

| Region | Bytes |
| -- | -- |
| `partials` (FP32, per 128-row tile: the channel sums of K, the channel maxima of abs V) | `B*H*N*2*128*4` |
| `mean_k` (FP32, the K channel mean per (batch, head)) | `B*H*128*4` |
| `v_scale` (FP32, one per channel) | `B*H*128*4` |
| `q8`, `k8` (INT8 codes, `[B*H, Sp, 128]`) | `B*H*Sp*128` each |
| `vt8` (transposed E4M3 V codes, `[B*H, 128, Sp]`) | `B*H*128*Sp` |
| `q_scale` (FP32, one per 128-row block) | `B*H*N*4` |
| `k_scale` (FP32, one per 64-key half, index `2*tile + half`) | `B*H*2*N*4` |

Every region is rounded up to a 256-byte slot; rows and keys past `S` hold
zero codes with finite scales. `runner.workspace_bytes` and
`runner.route_metadata["workspace_bytes"]` report the total. A caller pooling
workspaces sizes the pool with the largest problem's figure.

### Launch, CUDA graphs, determinism

A launch is four kernels on the current torch stream of the inputs' device, in
order:

| Stage | Work | CTAs (threads) |
| -- | -- | -- |
| `stats_partials` | per (batch, head, 128-row tile): the channel sums of K and the channel maxima of abs V | `B*H*N` (256) |
| `stats_finalize` | per (batch, head): the K channel means and the V channel scales from the partials | `B*H` (256) |
| `quantize` | per (batch, head, 128-row block): the INT8 codes and FP32 block scale of Q, the INT8 codes and two FP32 half scales of the mean-centred K, the transposed, per-channel scaled E4M3 V, written to the workspace | `B*H*N` (256) |
| `attention` | per (batch, head, 64-row Q tile): the quantized attention, BF16 output; one warpgroup per CTA, the Q tile and a two-stage K / V^T ring in a 73728-byte pool behind the launcher's 1,024-byte control prefix (74752 bytes of dynamic shared memory at launch), three CTAs per SM | `ceil(S/64)*B*H` (128) |

`stats_finalize`, `quantize` and `attention` launch with programmatic
dependent launch and wait for their producer inside the kernel;
`stats_partials` launches plainly. The chain keeps no counters and no tables:
every launch rewrites every region it reads, so a launch leaves the workspace
ready for the next one.

The runner is CUDA-graph capturable; capture belongs to the caller. Replays are
bitwise repeatable: every reduction has a fixed order and the chain carries no
device-side state across launches. `runner.launch_stage(name)` launches one
stage alone, for diagnostics. `runner.route_metadata` reports the
architecture, the recipe id, the QK operand format, the registry record of
each stage, the launch grids, the softmax scale, the workspace size and the
logical shape; `runner.out` is the caller's `out`.

### Numerics

| | SM90 |
| -- | -- |
| Recipe id | `fp8-sage-v1` (placeholder) |
| QK operands (`int8_blk`) | INT8 Q with one FP32 scale (`amax / 127`) per 128-row block, Q not centred; INT8 K centred by the per-(batch, head) channel mean with one FP32 scale (`amax / 127`) per 64-key half; no offset row. `Q K^T` accumulates exactly in INT32 |
| PV operands | E4M3 `P` (`P <= 256`) against E4M3 V with one FP32 scale (`max abs V / 448`) per channel |
| Q centring | none: the recipe smooths K only, so there is no block-mean compensation operand |
| K centring | K centred per (batch, head) by its channel mean (`sum / S`) |
| Softmax | FP32 online softmax in the log2 domain over 128-key tiles with one scale constant per (Q block, 64-key half); `P <= 256` quantized to E4M3; two-level FP32 accumulation (every tile's `P V` from a zero accumulator, then folded into the running output in FP32) |
| Tensor core | Hopper `wgmma m64n128k32`: INT8 for `Q K^T` (exact INT32 accumulation), E4M3 for `P V` |
| Accumulator (documented deviation) | the Hopper FP8 `wgmma` accumulator of `P V` keeps about 22 bits: per k32 step the 32 products and the incoming accumulator are truncated toward zero to multiples of `2^(E-13)` of the step's largest exponent `E` and the step result to 14 significant bits. The recipe emulation models it exactly, so the route agrees with the emulation to the last bit on its own tests; the per-tile fold into an FP32 running output confines the truncation to one tile. `Q K^T` is exact |
| `P` floor (documented deviation) | `P < 2^-10` becomes E4M3 `0` while the row sum keeps the unrounded `P`: per key at most `2^-18` of the row's maximum weight, in the worst case about `12.5 %` of the row mass on a flat row at `S = 32760`; the `spread` regime of the package tests measures the actual loss, and the emulation models the floor exactly |

Error bands (**pending**, filled in from the validation run): against the
recipe emulation (the same quantization in FP32 with the tensor-core
accumulator modelled) the route is expected to agree within
`atol 1e-2 + rtol 1e-2`; against exact BF16 attention on real activations the
validation reports the cosine similarity and relative L2 error per shape. The
package tests check every launch against the dequantized-operand oracle of
the recipe at `atol 1.0, rtol 0.1` (the tolerance of the kernels' own unit
tests) over the regimes `randn`, `v_x8`, `onehot`, `outlier32`,
`outlier32_q` and `spread`, tails (`S % 128 != 0`), `S < 128`, `S == 1`,
`B > 1` and both storage orders, and the quantized operands the chain leaves
in the workspace against the recipe in torch (the codes and scales exactly,
from the kernel's own statistics).

## Package layout

The backend is a self-contained package. `cake_jit.py` is the registry: a role
table of `MODULES` records, one per stage (`stats_partials_sm90`,
`stats_finalize_sm90`, `quantize_sm90`, `attention_sm90`), each carrying the
host plan (the program's library, the architectures it is delivered for, the
source module and build keywords it was rendered from, its role and stage, the
launch record, the argument plan of its `run` entry, the kernel-parameter to
logical-argument map, the element type of every buffer parameter, the grid
rule, the selection rule and the workspace rule) and the loader glue
(`records_for`, `launch_grid`, `launch_args`, `load`, `run_module`). What to
compile is recorded once in `build.json`: the generated CUDA sources under
`csrc/`, their digests and nvcc flags, one entry per (program, architecture).
`cake_build.py` loads a prebuilt library from `aot/` when a builder compiled
one and otherwise JIT-builds the manifest entry through FlashInfer from the
same sources and flags.

The package registers nothing outside its own directory and the thin
`flashinfer.prefill` entries: wheels built from this tree do not ship its
`build.json` and `csrc/`, so the route is available from a source checkout
(or a build that adds the package's data files to its own package table).

`cake_backend.py` holds the host: tensor validation, the workspace layout
(`workspace_layout`), the logical argument values of every record and the
runner. `tests/experimental/test_cake_fp8_attention.py` holds the oracle, the
validated shape set, the workspace-operand check, the graph-replay and
workspace contracts and the manifest contract;
`tests/experimental/test_cake_fp8_attention_api.py` the public entry;
`benchmarks/bench_cake_fp8_attention.py` times the complete call per shape
(CUPTI, cold L2, graph replay); `examples/cake_fp8_attention.py` is a
complete example.

## Relation to the NVFP4 attention route

| | NVFP4 route (`flashinfer.experimental.nvfp4_attention`) | This route |
| -- | -- | -- |
| GPUs | SM100, SM103, SM120 | SM90 |
| Entry | `prepare_nvfp4_attention(q, k, v, out, backend="cake", softmax_scale=, workspace=)`; `nvfp4_attention_workspace_bytes` | `prepare_fp8_attention(...)` with the same keywords; `fp8_attention_workspace_bytes` |
| Contract | logical `[B,H,S,128]` with any strides, any `S >= 1`, `softmax_scale > 0`, quantization inside every launch, graphs captured by the caller | same |
| Stages | `stats`, `quantize`, `attention` (three launches, PDL on `quantize` and `attention`) | `stats_partials`, `stats_finalize`, `quantize`, `attention` (four launches, PDL on the last three) |
| Workspace | counters region reset by the chain | no counters; the finalize kernel reduces the partials |
| Operands | E2M1 codes with UE4M3 block scales | INT8 Q / K codes with FP32 per-128-row-block / per-64-key-half scales; E4M3 V codes with FP32 per-channel scales |
| Numerics | NVFP4 recipe (see that README) | the FP8 recipe above; outputs of the two routes differ by design |

Both runners expose `launch()`, `launch_stage(name)`, `workspace_bytes` and
`route_metadata`; a caller that selects the route by the device's
architecture can hold one of each and treat them alike.
