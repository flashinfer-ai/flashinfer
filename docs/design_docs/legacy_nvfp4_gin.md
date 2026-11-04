# NCCL GIN support for legacy NVFP4 MegaMoE

This change ports the legacy MegaMoE NCCL GIN path into FlashInfer's
`MoEEpLayer` API. It enables NVFP4 expert-parallel execution across two
independent NVLink domains, including caller-managed CUDA Graph capture and
replay after collective warmup.

The integration is **disabled by default** and currently supports one fixed
configuration. Existing intranode behavior remains behind the default
`internode=False` setting. Correctness checks have passed on the configurations
listed below. Historical intranode performance gates passed; the native-donor
internode performance comparison remains inconclusive because its baseline was
not stable enough for acceptance.

## 1. API changes and feature coverage

### Public configuration

`Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig` gains two keyword-only fields:

| Field | Default | Behavior |
| --- | --- | --- |
| `internode` | `False` | Master switch for the new legacy NCCL GIN path. Set it when constructing the layer; changing a live session is unsupported. |
| `ranks_per_node` | `None` | Ranks per NVLink domain. Must be explicitly set to `2` when GIN is enabled; ignored when GIN is disabled. |

Existing callers can omit both fields. With `internode=False`, the backend does
not enter GIN preflight, control agreement, or session creation. The switch
applies to both eager and Graph execution.

GIN requires `kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl"`. Combining it with
the `_cutedsl_next` selector is rejected. This port does not change provider
selection or add non-GIN functionality. FI's existing activation, scaling,
finalization, and Graph support remain available on the intranode path.
In particular, the existing `MEGA_F4CVT_USE_MANUAL` FP4 decoder control is
preserved; the unused imported `f4cvt_use_manual` option and `_f4man` name suffix
are omitted from this subset port.

### Supported GIN configuration

N2R2 means two independent NVLink domains with two ranks per domain. Ranks must
be contiguous within each domain: `0,1` and `2,3`. Two hostnames alone do not
establish this topology; the runtime checks NCCL rail and LSA capabilities.

| Property | Supported value |
| --- | --- |
| Process group | Default Torch process group, world size `4`; no subgroups |
| Fleet | `max_tokens_per_rank=32`, `token_hidden_size=7168`, `num_experts=48` |
| Live tokens | Independently `0 <= T <= 32` on each rank, including empty ranks |
| Expert configuration | `intermediate_size=1536`, `top_k=4`, 12 experts per rank |
| Input | BF16 with FI quantization, or packed NVFP4 with caller-provided scale factors |
| Weights | NVFP4 at launch; BF16 `MoEWeightPack` can be preprocessed by FI |
| Computation | Standard SwiGLU, early routing weights, default fast math, `input_norm_const=1.0` |
| Output | BF16 combine followed by a separate TopkReduce |
| Finalization | `FinalizeMode.AUTO` or `DEFAULT`, both resolving to the separate default reducer |
| Compile bucket | `compile_tokens_per_rank=None` only; even explicit `32` is rejected |
| Tuning | `knobs=None`; no custom tile, cluster, schedule, or autotuning |

| Execution interface | Coverage added by this port |
| --- | --- |
| `MoEEpLayer.forward()` | BF16 and packed inputs; full, partial, and empty-rank cases |
| `stage_inputs()` / `compute_staged()` | Separate staging and compute, including capture after preparation |
| CUDA Graph | Public forward, split staging/compute, and a prepared low-level launch thunk |
| Distributed validation | Common rejection before device work for supported metadata, mode, bucket, and initialization errors |
| Resource lifecycle | Session ownership, coordinated construction cleanup, and job-fatal cleanup failure propagation |

GIN does not support custom activations or activation clamps, epilogue scales,
FC1 per-token scales, quantized combine, in-kernel FC2 reduction, unfinalized
outputs, or custom finalizers. These restrictions apply to the new GIN path.
They do not narrow existing intranode support.

## 2. Calling the layer

Bind each process to its CUDA device before initializing the default process
group. GIN needs PyTorch Gloo for host control. A default group configured with
`cpu:gloo,cuda:nccl` is reused directly. For an NCCL-only default group, GIN
initialization collectively creates and caches one Gloo control group; it is
owned by the default process group and is never created during capture/replay.
GIN uses a separate NCCL communicator and does not own the caller's Torch group.

The following configuration is used by the
[complete executable example](../../examples/nvfp4_gin.py). The example also
provides imports, weights, inputs, and successful-path teardown.

```python
backend = MegaConfig(
    megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=1536,
        top_k=4,
        internode=True,
        ranks_per_node=2,
    ),
    preprocess_weights=True,
    quantize_input=True,
)
```

Use `FleetParams(num_experts=48, max_tokens_per_rank=32,
token_hidden_size=7168)` and `BootstrapConfig(world_size=4, rank=rank,
auto_bootstrap=False, device=local_rank)`. The caller initializes the default
process group before constructing the layer.

### Tensor contract

All inputs must be contiguous and on the bound rank's CUDA device.

| Input | Shape and dtype |
| --- | --- |
| BF16 `hidden_states` (`quantize_input=True`) | `[T, 7168]`, `torch.bfloat16` |
| Packed `hidden_states` (`quantize_input=False`) | `[T, 3584]`, `torch.float4_e2m1fn_x2` |
| Packed-input `MoEEpTensors.scales` | `[T, S]`, `torch.float8_e4m3fn`, with `S >= 448`; one scale per 16 hidden elements in plain row layout |
| `topk_ids` | `[T, 4]`, CUDA int32/int64; global expert IDs `0..47`, with `-1` for invalid routes |
| `topk_weights` | `[T, 4]`, CUDA FP32; invalid routes contribute neither payload nor score |

Input scale factors use the plain activation layout, not the blocked/swizzled
weight-scale layout. For BF16 weight preparation, use local
`MoEWeightPack(w13=..., w2=...)` with `w13=[12,3072,7168]` (gate then up) and
`w2=[12,7168,1536]`. The public intermediate size is `1536`; FC1's combined
gate/up width is `3072`. FI quantizes and transforms these weights when
`preprocess_weights=True`.

Existing prequantized weight entry points remain available, subject to the
packed/logical shape, scale layout, interleave, and K-major requirements in
[the backend weight contract](../../flashinfer/moe_ep/backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/weights.py).
Matching the dtype alone is insufficient
to reuse another kernel's weights.

Run the example on each of two independent domains with `NODE_RANK=0` or `1`:

```bash
torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 examples/nvfp4_gin.py
```

Every default-group rank must enter initialization, staging/forward, and
lifecycle operations in the same order, even when its live token count is zero.
On successful completion, synchronize outstanding device work, call
`layer.destroy()`, then destroy the default process group. Outputs that borrow
workspace storage and prepared thunks must not outlive that workspace/session;
copy outputs to independent storage when they must survive subsequent calls.

The lower-level shim factory `get_symm_buffer_for_mega_moe` also gains
keyword-only `internode=False`, `ranks_per_node=None`, and `process_group=None`.
These wire the frontend to the session; they do not extend the public support
allowlist. Normal integrations should use `MoEEpLayer`.

## 3. CUDA Graph contract

The caller creates the Graph with PyTorch/CUDA. FI provides capture-compatible
staging and launch interfaces; the GIN kernel itself does not create a Graph.

| Phase | Host work | Device work |
| --- | --- | --- |
| Collective eager warmup | Validate topology, create/register resources, compile, prepare launch arguments, and validate current routes | Initialize state and execute eager work |
| Capture | Validate metadata, phase/session identity, capture mode, and prepared bindings over CPU/Gloo | Record staging, mega kernel, TopkReduce, output copy, and surrounding GPU operations |
| Replay | No FI Python forward or CPU/Gloo preflight runs again | Execute recorded operations and advance device communication state |

For a layer and static input tensors prepared as in section 2:

```python
static = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
layer.warmup(static)  # All session ranks, with the same bindings used for capture.
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = layer.forward(static)

# All ranks finish capture before replaying in the same order.
dist.barrier(group=dist.group.WORLD)
graph.replay()
```

Keep each Graph's tensor addresses, shapes, and bindings fixed. Input values
may change in place, including activations, scales, route IDs, and
`topk_weights`. A rank's shape may be zero, but cannot change within one Graph. A change
of address or shape requires warmup and capture for the new bindings; an old
Graph cannot automatically adopt them.

Capture does not copy GPU route values to the CPU, and replay cannot raise a
host validation error for new values. The caller must ensure every replay uses
valid route IDs in `-1..47`; device-side safe rejection of invalid inputs is
not part of the contract. Updating values in fixed buffers does not require
reinitializing topology or NCCL GIN resources.

Order input updates, execution, and output reads on one stream, or establish
explicit event/`wait_stream` dependencies. Python submission order across
streams is insufficient. Multiple Graphs and eager calls may share a session
only serially; they must not concurrently reuse its workspace. Consume or copy
a workspace-backed result before that storage is reused. Every session rank
must capture and replay in the same order and count, including empty ranks.

Keep the layer, prepared quantized weights, workspace/session, and Graph
input/output storage alive until all replays finish and the Graph will no
longer be used. Updating the original BF16 weight pack does not update prepared
weights. Dynamic expert-weight replacement within a Graph is not supported.
Initialization, compilation, argument rebuilding, and destruction stay outside
capture. Discard partial Graphs after a recoverable capture rejection, fix the
cause, then capture again. Device/NCCL failures are job-fatal.

Low-level callers can collectively prepare
`nvfp4_mega_launch_thunk(w1, w2, workspace)` outside capture, where `w1/w2` are
kernel-ready `(weight, scale)` tuples, then invoke it during capture. The GIN
thunk binds the current stream at invocation. Reprepare after changing its
bindings or lifetimes. This does not extend the capture contract of the ordinary
low-level `nvfp4_mega_moe` call.

## 4. Implementation and code changes

### Backend integration and default-off isolation

The call path is `MoEEpLayer -> MoEEpMegaLayer ->
Nvfp4CutedslMegaKernelBackend -> package exports -> NVFP4 shim / GinSession ->
legacy device kernel`. Only the shim imports raw vendor sources.

`MegaKernelBackend.requires_collective_forward_validation` defaults to `False`.
The GIN backend enables it so `MoEEpMegaLayer` can agree on errors before
rank-local validation or staging would let ranks diverge. Other backends retain
their existing path. GIN does not use NVSHMEM peer-pointer initialization.

Prepared launch arguments are cached by workspace, weight identity, and compiled
callable. GIN excludes the rank-local stream from the cache key: a stream switch
on only one rank must not send it into preparation while peers enter launch.
The thunk binds the current stream instead. Intranode caching remains keyed by
stream. Steady-state public GIN forward uses two control agreements, reduced
from five; first compilation and cache misses can require additional preparation.
Simple intranode cases use the donor's lean `launch_without_gin` entry. Existing
FI extensions such as persistent finalization and FC1 per-token scaling retain
the full entry.

### Registered arena and ownership

`shim/gin.py` partitions one complete contiguous CUDA uint8 allocation into
128-byte-aligned regions:

```text
registered root
  gin_transport at offset 0
  packed activation
  activation scale factors
  top-k weights
  compiled-kernel shared workspace
```

NCCL registers the entire root. Every peer-visible view comes from that root,
so the same-domain peer-base deltas apply to all views. Independently allocated
tensors cannot be mixed into those deltas. Transport starts at zero to preserve
the donor's window-relative wire offsets. Host validation checks shape, dtype,
storage extent, alignment, overlap, and bounds.

The session broadcasts a UID over the default process group, creates its NCCL
communicator, validates rail/LSA support, allocates the arena, creates a device
communicator, registers the window, and maps local peers. `GinSession` owns these
resources; workspaces and thunks borrow them. Weights and non-peer-visible local
state need not reside in the registered root.

### Communication and generation state

The port retains legacy dispatch, FC1/activation/FC2 scheduling, and token-back,
with relay/return communication across domains. Combine follows the defined
BF16 term and node order before the separate TopkReduce. An exact protocol
oracle checks transport and rounding independently of the tolerant Torch math
oracle.

A local token-back generation fix waits for `internode_barrier_counter+1`
instead of sampling `combine_issue_counter+1`. A fast relay producer could
advance the latter before a slow consumer sampled it, making that consumer
wait for the next launch and deadlock. The barrier generation advances only
after token-back retirement, giving a stable target within the current launch.
[VENDOR.md](../../flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/VENDOR.md)
records this as a pending-upstream correctness change.

### Distributed errors and cleanup

At initialization boundaries, local validation/allocation failures are agreed
before peers can enter the next resource operation. Both the workspace factory
and direct session creation reject initialization during capture collectively.
After session creation, views, route/output allocations, epilogue vectors, and
workspace construction are protected by a common construction phase. The
workspace is published only after all ranks succeed.

On a recoverable initialization or workspace-construction rejection, all ranks
clean up and agree on the cleanup result. Any cleanup failure, or failure of
that agreement, becomes `GinFatalError` on all participating ranks. Dependencies
of failed resources remain retained. Successful cleanup preserves the common
`GinPreflightError`, allowing a corrected call to retry.

Eager public forward and `stage_inputs()` validate metadata, routes, and device
binding on every call. Capture checks metadata and prepared bindings, with route
values remaining the caller's responsibility. Mode, bucket, phase, configuration,
and session mismatches are agreed before staging/launch. Unsupported explicit
buckets are rejected in the existing input agreement; they add no steady-state
or replay control rounds.

CPU/Gloo preflight uses fixed-size headers with 128-bit BLAKE2b fingerprints of
phase/signature. Every rank sees the same headers and, on any error or mismatch,
collectively takes a detailed object-report fallback. All boundaries use this
same protocol, so different phases cannot split ranks between incompatible
collectives. An NCCL-only default group uses the precreated Gloo control group;
capture never submits a CUDA/NCCL control collective.

`GinPreflightError` is a recoverable `RuntimeError` subclass only when rejection
happens before unsafe communication and coordinated cleanup succeeds. Do not
treat arbitrary `RuntimeError` instances as recoverable. Resource/control
collective failures, CUDA errors, and NCCL/device communication failures require
whole-job termination through the launcher/timeout; communicator recovery is
not promised. Successful destruction drains the device and releases window,
device communicator, communicator, then root storage in dependency order.

`FLASHINFER_GIN_GUARD_DIAGNOSTICS=1` optionally adds fixed source-site prints
before device guards trap. It defaults to off and is read at JIT compilation.
Guards remain active when it is off. The variable is registered in the root
`CLAUDE.md`; it is separate from the `internode` feature switch.

### Review map and provenance

Paths below are relative to the repository root; package-relative paths use
`flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/`.

| Area | Change and review focus |
| --- | --- |
| `flashinfer/moe_ep/backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/{config,backend}.py` | Public flags, allowlist, collective validation, warmup/capture readiness, launch cache |
| `flashinfer/moe_ep/core/kernel/base.py`, `flashinfer/moe_ep/modes/mega_layer.py` | Opt-in validation capability and ordering before local staging |
| Package `shim/gin.py`, `shim/gin_debug.py` | Registered arena, NCCL/LSA lifecycle, error agreement, protocol snapshots |
| Package `shim/nvfp4.py`, `shim/autotune.py`, exports | Runtime wiring, session ownership, lean intranode entry, tuning rejection |
| Package `src/src/`, `src/common/`, `src/moe_nvfp4_swapab/` | Targeted merge of 17 device-source files; wire ABI, generation, barriers, retained FI changes |
| CPU contracts and `tests/moe_ep/test_nvfp4_cutedsl_gin_*.py` | Distributed errors, real N2R2 oracles, public Graph paths, intranode regressions |
| `examples/nvfp4_gin.py`, `benchmarks/moe_ep/bench_legacy_nvfp4_gin.py` | Executable example and scoped, reproducible timing |
| This document, package `VENDOR.md` / `SKILL.md`, root `CLAUDE.md` | API/design, source provenance, maintenance rules, environment-variable reference |

This is a targeted import, not a replacement of the vendor snapshot:

| Source | Fixed revision |
| --- | --- |
| Original vendor snapshot | `fa662f345980f75776ae0c85d2e5c188918ded26` |
| Native legacy donor | `f863d7540614bd546a42867c21adbf51d573061a` |
| FI receiving/intranode baseline (F_BASE) | `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34` |

FI sparse decode, masked reduction, per-token scales, custom activation,
singleton TMA, and BF16/MXFP8 compatibility changes are retained. Future drops
must reapply the local differences listed in `VENDOR.md`.

## 5. Correctness and qualification

GPU results below qualify source `a0ff9eb5e`, not subsequent host-error changes.
Both GB200 jobs completed with exit code zero; their 100 Python source hashes
matched that qualification source.

| Validation | Result and scope |
| --- | --- |
| Host contracts at session-cleanup fix `9b3dcd35b` | 25 passed; no Torch/GPU required |
| GB200 N2R2, job `3595270`, `lyris0120/0129` | Per rank: 2 new-case screening tests, 40 full protocol/math/public tests, and 21 NCCL-only checks; no skips |
| GB200 N1R4, job `3595271`, `lyris0131` | 10 intranode checks per rank: original 6 regressions, 2 FP4 decoder environment/PTX cases, 2 default-off eager/Graph isolation cases |
| Exact protocol | 24 calls per rank, 96 transport/L1 byte checks across the job; partial/empty ranks, sparse holes, and invalid routes with NaN scores |
| Independent Torch math | 93 nonempty batches; maximum relative L2 `0.0026485608` against unchanged tolerance `0.02` |
| Graph coverage | BF16/packed, partial/empty, three entry points, 48 bitwise replay comparisons per case, including 12 queued replays |
| NCCL-only control | One Gloo group created outside capture and reused |

Graph tests poison staged inputs and outputs between calls, compare independent
snapshots after queued replays, and assert one mega kernel plus one reducer in
the public replay trace. These checks distinguish real replay from stale eager
results. Default-off isolation replaces GIN preflight/session entry points with
functions that fail if called, then exercises eager and Graph with
`ranks_per_node=None` and `2`. Both existing FP4 decoder environment branches
pass all 16 FP4 code reductions bitwise and produce the expected PTX conversion
path. Five core intranode reducer methods have identical ASTs to F_BASE.

Failure tests cover rank-local bucket errors, initialization capture, and
post-session allocation failure with cleanup/rebuild. OOM is injected on one
rank at view, route, output, and final epilogue allocation sites; the tests do
not exhaust real device memory. The later session-initialization cleanup fix
has three additional host tests invoking the actual factory with CUDA/NCCL
resource doubles. They inject LSA mapping failure, cleanup failure on another
rank, and cleanup-control failure. All three fail before the fix and pass after
it. CUDA/NCCL multinode tests were not rerun for that later fix, and destructive
NCCL cleanup failures were not injected on GPUs.

Run host contracts without importing the GPU package:

```bash
python -m unittest discover -s tests/cpu/moe_ep -p test_legacy_nvfp4_gin_contracts.py -v
```

Run the strong/public suite on two real independent NVLink domains:

```bash
torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 -m pytest \
  tests/moe_ep/test_nvfp4_cutedsl_gin_oracles.py \
  tests/moe_ep/test_nvfp4_cutedsl_gin_multinode.py -v -x
```

The qualified environment was GB200/SM100, CUDA 13.3,
Torch `2.13.0a0+9186a08b2c.nv26.07`, CuTe DSL 4.6.2, nccl4py 0.3.1,
NCCL 2.30.7, and driver 580.196. These are measured versions, not a claim that
all other versions are supported. Ordinary single-node four-rank CI cannot
replace the two-domain lane. A topology skip is not a pass. Earlier broad
EP1/2/4 and BF16/MXFP8 matrices remain historical evidence, not reruns at the
current head.

## 6. Performance and acceptance limits

### Scope and baselines

The following measurements are historical results at their respective source
revisions. The later host error-handling fixes do not constitute a new
performance measurement. No comparison uses the next implementation.

| Scope | Baseline and timing boundary |
| --- | --- |
| Intranode regression | Fixed F_BASE legacy implementation versus receiver |
| Internode migration | Actual native legacy donor measurement from `pre_unification/moe_nvfp4_swapab/mega_runner.py`, not an estimate |
| Raw pipeline | CUDA events around the mega kernel, separate TopkReduce, and intervening launch gap; not one fused-kernel duration |
| Full operator | Completed wall time including public validation, staging, allocation, and launches |

The native donor uses a prepared adapter and does not perform the same public
validation as FI. Its full-call timing is therefore not equivalent to the FI
public API's full-call timing.

### Historical GB300 eager measurements

Hardware/software: GB300/SM103 (152 SMs/device), Torch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, CuTe DSL 4.6.2, nccl4py 0.3.1,
NCCL 2.30.7, driver 580.173.10. GPUs were exclusive and clocks were not locked.
Each paired comparison used the same physical GPU allocation and input payload;
this does not mean the two implementations shared CUDA memory.

Source revisions reported by the measurement manifests are listed here so
these numbers cannot be mistaken for a benchmark of the current PR head.

| Measurement | Job IDs | Receiver revision(s) |
| --- | --- | --- |
| Intranode gates, EP1 | `3578010` | `e930de841` |
| Intranode gates, EP2/4 | `3578087`, `3578088` | `38f650e63` |
| Paired GIN host-overhead fix | `3577960` | Before `c597d35ba`, after `efb7b3d29` |
| Final native-donor comparison | `3578176` | `74c036287` |

Five rounds alternated sources, with five warmups and 20 samples per round:
100 samples per source, taking the slowest rank per iteration. Warm and cold
results are separate; cold runs evicted 512 MiB/rank outside timing. The gate
allows at most 5% median regression, but only when baseline round-median CV is
at most 5%; otherwise the result is inconclusive. All accepted samples passed
bit/payload checks. Internode routing used seeded random top-k, not a guaranteed
balanced assignment.

For H2048, public I1024, E8, K4, all **24 intranode gates passed** across
EP1/2/4, T64/1024, warm/cold, and raw/full. Each cell is F_BASE -> receiver,
in microseconds; 4,800 samples cover both sources across all gates.

| EP | T/rank | Warm raw | Cold raw | Warm full | Cold full |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 91.44 -> 87.50 | 95.68 -> 90.91 | 211.91 -> 207.97 | 202.06 -> 199.99 |
| 1 | 1024 | 132.40 -> 127.07 | 136.98 -> 131.42 | 237.33 -> 234.87 | 238.56 -> 231.43 |
| 2 | 64 | 106.98 -> 101.52 | 106.46 -> 102.11 | 226.71 -> 217.63 | 217.39 -> 212.37 |
| 2 | 1024 | 152.82 -> 146.02 | 156.82 -> 152.78 | 264.48 -> 256.95 | 259.99 -> 254.90 |
| 4 | 64 | 129.73 -> 124.43 | 130.90 -> 124.03 | 249.62 -> 241.95 | 246.21 -> 240.35 |
| 4 | 1024 | 184.62 -> 181.89 | 186.27 -> 180.62 | 299.97 -> 293.81 | 296.12 -> 287.70 |

For the fixed GIN configuration, a paired host-overhead fix reduced FI public
full-call time from **6155.76 to 2208.60 us**. PTX/cubin bytes matched on all
four ranks in that before/after comparison. A later two-agreement version
measured `1630.15 us` warm full on a different allocation; it cannot form a
paired speedup ratio with `6155.76 us`.

| Final standard raw | Native legacy donor (us) | Receiver (us) | Baseline round CV | Decision |
| --- | ---: | ---: | ---: | --- |
| Warm | 350.37 | 349.31 | 6.38% | Inconclusive |
| Cold | 363.98 | 356.43 | 6.39% | Inconclusive |

The native donor's prepared full-call warm median was `402.74 us`, versus FI public
`1630.15 us`. FI still pays public validation costs; these figures establish
neither full-call parity nor a fourfold device-kernel regression. An earlier
raw +18.8% regression reversed direction on other allocations. GPU-local,
disjoint CPU placement reduced EP4 launch variance but did not eliminate N2R2
variance. The unstable donor baseline prevents formal internode raw-pipeline
acceptance; unfavorable results remain part of the evidence.

The [benchmark entry point](../../benchmarks/moe_ep/bench_legacy_nvfp4_gin.py)
runs sources in separate processes with shared fixture files and emits per-rank
samples/manifests. For intranode comparisons, select `--mode off --impl
base/port` from the corresponding checkout and the same `--root`; run EP1/2/4
for five rounds (T64/1024 are included). For GIN, first generate the fixture
using `--mode on --impl port --prepare`, then alternate port and donor rounds
with the same `--root` and different `--round` values. Point donor processes'
`PYTHONPATH` to the fixed donor checkout so they do not import FI. Use
`--skip-profiles` for primary samples and `--profile-only` afterward. Do not
mix warm/cold summaries. The final resident experiment used a separate
controller; its results are not presented as executions of this entry point's
independent-process protocol.

### CUDA Graph: compare the same GPU work

GB200 job `3593508` compared eager/replay at source `200576d1c`, on
`lyris0057/0126`, with the same N2R2 configuration, layer, weights, inputs, and
preallocated outputs. It completed with exit code zero under CUDA 13.3,
Torch 2.13dev, CuTe DSL 4.6.2, NCCL 2.30.7, and driver 580.196.

Each sample queued 32 calls, synchronized once at the end, and reported
completed wall time per call. Three rounds alternated modes, with two warmup
blocks and 12 measured blocks per round: 36 samples per mode, using the
slowest rank for each sample. Initialization, compilation, capture, and entry
CPU/Gloo barriers were outside timing. Raw excludes staging, public validation,
and output copy. Prepared GPU includes identical staging and output copy on
both sides; it is a timing adapter, not a change to public eager validation.

| Input | Matched work | Eager (us) | Graph (us) | Eager / Graph |
| --- | --- | ---: | ---: | ---: |
| BF16 | Raw mega + reduce | 231.26 | 231.63 | 0.998x |
| Packed | Raw mega + reduce | 222.75 | 222.28 | 1.002x |
| BF16 | Prepared GPU, including staging/copy | 239.39 | 234.08 | 1.023x |
| Packed | Prepared GPU, including staging/copy | 241.62 | 237.30 | 1.018x |

Raw queued medians are effectively equal; prepared GPU differs by about 2%.
Neither result establishes a kernel arithmetic speedup or passes the separate
native-donor migration gate.

Earlier job `3593453` reported public full-call BF16 `1505.30 -> 308.11 us`
and packed `1803.99 -> 359.94 us`. That apparent 4.9-5.0x difference compares
checked public eager calls with replay. Eager also executes four route-check
kernels, a `.item()` synchronization, two CPU control agreements, and default
output allocation; replay does not repeat that host path. Even preallocating
outputs in job `3593508` left BF16 `1511.28 -> 234.12 us` and packed
`1523.54 -> 236.01 us`. This is public-entry overhead, not equivalent GPU work
or mega-kernel acceleration. The native donor's roughly 350 us raw timing is
also not comparable to the 1.5-1.8 ms public full-call scope.

Separate single-call samples (three rounds of 60, ten warmups per round)
measured raw BF16 `397.95 -> 327.07 us`, packed `388.11 -> 297.95 us`, and
prepared GPU `470.63 -> 342.63 us`, `489.88 -> 359.12 us`. These intervals still
include host submission gaps and cross-rank waiting. Do not mix single-call
and 32-call results to calculate a speedup.

All 2,592 samples/rank were present. Traces confirmed the same compute-kernel
sequence for prepared eager/replay and public replay; raw contains one mega
and one reducer. BF16 Graph output copy can appear as `memcpy32_post`; packed
paths contain five D2D copies. A kernel-count comparison alone cannot establish
copy coverage. Profiling followed a CPU barrier and ran separately from primary
timing. Poisoned-input/output and timing-block results matched bitwise. Raw
samples, tails, per-round medians, CV/P95, and traces remain in the experiment
records.

## 7. Review boundaries and evidence

This PR delivers the default-off, fixed-configuration GIN integration described
above. Additional shapes/topologies, removal of public eager validation,
changes to reduction precision/order, autotuning, and formal internode
performance promotion require separate work. Correctness success does not
imply performance acceptance.

The design and provenance records ship with the PR; investigation logs and
large measurement artifacts do not. An earlier migration/eager investigation
was archived in the internal `qiangf/agent-trace` repository at
`2cf0a06da12654a3ab4089224523822f93de9807`, under
`cases/legacy-nvfp4-gin-blackwell-01a19f6e-r6`. That frozen archive predates
Graph qualification. Graph and later error-path evidence is identified by the
source revisions and job IDs above, with logs retained in the separate local
experiment records. The archive is not required to build or run this code.
