# `kernel_src/sm120/cake_mxfp4_megamoe/` — Cake SM120 MXFP4 x MXFP8 hybrid-transport MegaMoE

The kernel package behind the moe_ep backend `sm120_mxfp4_mxfp8_bf16_cake`
(`flashinfer/moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_cake/`).  `src/` holds the
**generated, checksum-attested** device sources of the Cake persistent MegaMoE kernel for
consumer Blackwell (`sm_120a`, CC 12.0) plus two hand-written files (the host binding and the
cubin carrier); `shim/cake_jit.py` is the JIT loader.  Humans do not edit the generated device
files (formatting them breaks the manifest digests — see the `clang-format` exclusion note
below).

One operator serves every EP topology: each ordered pair of EP ranks moves its dispatch
payloads and expert results over the transport a per-pair table assigns to it —

* **LSA** (load/store over NVLink or PCIe peer mappings of the symmetric NCCL windows,
  with dedicated copier CTAs streaming the bulk copies), or
* **GIN** (NCCL GPU-initiated networking puts with signals),

so a PCIe-only host and a multi-host deployment (LSA inside each host, GIN across hosts)
run the same kernel (validated on PCIe hosts and two hosts over GIN; NVLink peers take the
same load/store path but were not available for testing).  The table is pinned by the caller (the backend's
`transport_table`, default all-GIN), looked up in the tuned knob cache, or chosen by a setup
probe (`"auto"`).

Tables without an LSA pair run the copier-free instances (every SM computes).  Any LSA pair
selects the copier instances: they reserve **one copier CTA per LSA destination of a rank**
(`MOE_LSA_COPIERS` = the largest number of LSA cells in a table row).  Copier `k` of a rank
serves its `k`-th LSA destination (found at run time from the table): it announces and pushes
that destination's dispatch payload and streams its expert results back with TMA bulk copies;
CTA0 announces the GIN and self destinations; an LSA destination beyond the pool raises
`protocol_error` (the host refuses such tables first).  The copier instances use the same M128
GEMM schedule as the copier-free ones (two-fragment B lookahead; K-step unroll at EP 8).

## Contents (`src/`)

| File | Origin | Role |
|---|---|---|
| `cake_sm120_megamoe_binding.cu` | hand-written | TVM-FFI binding: transport session (adopt `ncclComm_t`, `ncclMemAlloc` + `ncclCommWindowRegister` symmetric windows, `ncclGetPeerDevicePointer` LSA peer pointers, `ncclDevCommCreate` when GIN is available), the setup-time transport probe kernels, TMA descriptor encoding into caller-owned 128-byte carriers, cubin load + `cuLaunchCooperativeKernel` with a kernel-ordered argument list |
| `cake_sm120_megamoe_kernel.cuh` + `cake_sm120_megamoe_kernel_part0NN.cuh` | generated | **the one parametric device source** (split below the repository's 500 KB file limit; the entry file includes the parts) |
| `cake_sm120_megamoe_{m128,m64}_{copier,nocopier}.cu` | generated | three-line instance wrappers: `MOE_INSTANCE` + kernel symbol, then `#include` the source |
| `cake_sm120_megamoe_geometry.json` | generated | host geometry: every switch, constant, buffer extent, guard and the kernel ABI as integer expressions of the parameters |
| `cake_sm120_megamoe_manifest.json` | generated | SHA-256 seal of all of the above (schema below) |
| `cake_sm120_megamoe_cubins.cu` | hand-written | carrier of the two embedded cubins of one device module (`cubin_image(task_m)`); not part of the generated closure |

### Parameters

The device source is compiled per deployment with

| `-D` | meaning | constraint (checked by the source and by the backend) |
|---|---|---|
| `MOE_RANKS` | EP world size | 2..32 (the generator-covered region; GPU-validated at 4, 8, 16; 32 compile-checked) |
| `MOE_EXPERTS` | global routed experts (physical experts under EPLB) | `% MOE_RANKS == 0`, <= 128 per rank, <= 1024 |
| `MOE_TOPK` | routed experts per token | 1..9 |
| `MOE_MAX_ROWS` | tokens per rank per call (capacity) | multiple of 128, <= 32640 (the row fields of the signed 32-bit `active_rows` kernel argument) |
| `MOE_HIDDEN` | hidden size | multiple of 512, <= 7680 |
| `MOE_INTERMEDIATE` | expert intermediate size | multiple of 128 |
| `MOE_SHARED_EXPERT` | 1: one in-kernel FP8 e4m3 shared expert added to the output | 0 / 1 |
| `MOE_SWIGLU_LIMIT` | SwiGLU clamp `L`: `gate <= L`, `-L <= up <= L` (10 = DeepSeek-V4); 0 = unclamped SwiGLU (DeepSeek-V3, Qwen3) | integer >= 0, default 10 |
| `MOE_CODEC_OVF_BLOCKS` | raw-overflow capacity of the lossless BF16 return codec, in 64-value blocks per row (see below) | 16..256, default 16 |
| `MOE_LSA_COPIERS` | copier CTAs of the copier instances (one per LSA destination of a rank) | 1..min(`MOE_RANKS` - 1, 16), default 1; set by the backend from the transport table |

and `MOE_INSTANCE` from the wrapper: two row schedules (`m128`: 128-row tasks; `m64`: 64-row
tasks, launched when no rank holds more than 512 tokens in the call) times two copier flavors
(`copier`: with the LSA copier CTAs; `nocopier`: for tables without any LSA pair, which frees
those SMs).  A workspace compiles and loads the two schedules of the flavor its transport table
needs.  Parameter-dependent code paths are selected by derived switch macros (`MOE_SW_*`)
inside the source; a parameter set outside the generator's covered region fails at compile time
with `#error` / `static_assert`, and every derived integer is range-checked, so an overflowing
instantiation does not compile.

**Return codec capacity.** Expert results return as 64-value BF16 blocks coded to 96 bytes
(lossless delta code) or kept raw (128 bytes) in an overflow area behind each return chunk.
The area holds `rows x MOE_CODEC_OVF_BLOCKS` raw blocks; a chunk that needs more sets the
kernel's `protocol_error` flag; that call's output is then not valid.  The flag is per call (the next
launch clears it) and the backend's forward path does not read it: a caller that needs the check reads
it with the backend's `protocol_error(workspace)` (a host synchronization) after the call.  After a call
that set it, discard the workspace: a rejected dispatch header leaves the per-peer signal bookkeeping
out of step with the senders, so later calls on the same workspace are not reliable.  The default
16 is 25 % of a row's blocks at hidden 4096 and 14 % at 7168; real DeepSeek-V4 activations need
about 0.3 %, random test data stays far below the capacity.  Each extra block per row costs
`128 x max_tokens_per_rank x top_k x world_size x 2` bytes of result window per rank (the
backend's `codec_overflow_blocks`; e.g. +2.9 GB at DeepSeek-V3 EP 8 / 8192 rows going from 16
to 28 blocks).

**Wait deadline.** Every device-side wait of the kernel that can depend on another rank (GIN
and LSA signal waits, LSA slot arbitration) and every wait on another CTA's mailbox, queue or
ring carries a wall-clock deadline of 600 s (`%globaltimer`, armed by the first unsatisfied
probe, so a satisfied wait never reads the timer).  A wait still unsatisfied at the deadline
executes `trap`: the launch fails on that rank with a CUDA error (`unspecified launch failure`)
and its peers fail at their own deadlines, so a rank that stalls or dies for longer than the
deadline makes the call fail on every rank instead of hanging it; a shorter stall only delays
the call.  The CUDA context of a failed launch is unusable (as after any device fault): the
process has to be restarted.  Not covered: the session's first launch, which starts with an
NCCL world barrier (`ncclGinBarrierSession`; released by `ncclCommAbort`), and device-local
counter / mbarrier / grid waits, whose producers on the same GPU either progress or trap.

The backend derives the parameters from the moe_ep configuration (`BootstrapConfig.world_size`,
`FleetParams.num_experts / token_hidden_size / max_tokens_per_rank` rounded up to 128, the
kernel config's `intermediate_size / top_k / shared_expert / swiglu_limit / codec_overflow_blocks`,
and `MOE_LSA_COPIERS` from the transport table) and evaluates the geometry (guards
first, as a `MoEEpConfigError` in `validate_init`).

## JIT and AOT modules

Both kinds of modules are ordinary FlashInfer `JitSpec`s (`shim/cake_jit.py`), cached in the JIT
directory and packaged ahead of time by `flashinfer/aot.py`:

* `cake_sm120_megamoe_binding_<digest>`: the host binding, one per source revision.
* `cake_sm120_megamoe_{copier,nocopier}_{proxy,gdaki,gpi}_<digest>`: the device instances of one
  parameter set, transport flavor and NCCL GIN backend.  Its `embedded_cubin_factory` compiles the
  flavor's M128 and M64 wrappers (`nvcc -cubin --generate-code=arch=compute_120a,code=sm_120a
  -std=c++17 -O3`, about 10 s each, in parallel) and the build embeds both cubins into the carrier;
  the digest covers the sealed sources, the wrappers, the `-D` parameters, the flags and the NCCL
  version.  A workspace builds or loads only the module of the flavor its transport table needs and
  of its communicator's GIN backend (`ncclGinType_t` of the transport session; proxy when the
  communicator has no GIN, where no GIN call executes).

**GIN backend builds.** The NCCL Device API inlines every enabled GIN backend (proxy, GDAKI, GPI)
into each GIN call behind a runtime switch on the context's backend.  The device modules compile
only the session's backend (`NCCL_GIN_{PROXY,GDAKI,GPI}_ENABLE`; `GIN_BACKENDS` in
`shim/cake_jit.py`), which lowers the persistent kernel's register use by 10-20 registers and
removes the 8-byte stack frame of the GPI fence.  `validate_init` runs before the transport
session exists, so it compile-checks the candidate instances of all three backends (in parallel,
cached); the workspace loads the one it needs.

`flashinfer/aot.py` registers the binding and the DeepSeek-V4-Flash routed-expert defaults
(`AOT_DEVICE_PARAMETERS`: E256 / H4096 / I2048 / top-k 6, in-kernel FP8 shared expert,
`swiglu_limit=10`, 4096 tokens per rank, all-GIN table, EP 4 and EP 8, each GIN backend) when the target list has
SM120 and NCCL 2.30.7 (headers + library) is found at build time.  Any other parameter set compiles
on first use; with `FLASHINFER_DISABLE_JIT=1` an unpackaged one raises `MissingJITCacheError`.  The
`MOE_MAX_ROWS` capacity is a compile-time parameter, so a packaged module is used only by a
deployment with the same `max_tokens_per_rank` (rounded up to 128).

**Provenance.** The source is the generator's preprocessor-selected merge of its traced
emissions over the parameter space: for every parameter set the selected text is byte for
byte the generator's emission at that point.  At the qualified DeepSeek-V4-Flash points
(E256 / H4096 / I2048 / top-k 6 / 8192 rows; EP 4 / 8 / 16; both variants; all four
instances) the source differs from the qualified per-configuration kernels only by the copier
instances' LSA-bound copier pool and by the wait deadline above (the qualified kernels capped
their mailbox/queue waits by an iteration count and waited for GIN signals without a bound).  Apart from the license headers, the generator's tensor-map carrier structs
and infinity macro are renamed for the public export (`CakeTensorMap`, `CakeTensorMap64`,
`CakeTensorMapPack`, `CAKE_INF`); the export tool checks that the inverse substitution restores
the generator's text.

Register use at the shipped test geometries plus DeepSeek-V4.1-Flash (E384 / H5120 / I2304 /
top-k 6), EP 4 / 8 / 16 / 32, both variants, every copier count the tables select (480 instances
per GIN backend): every instance of every GIN backend build compiles spill-free with each CUDA 13
release listed under *Runtime requirements*, at 130-168 registers (the `m128` instances run at
the 168-register cap), local 0, stack 0 for the proxy and GDAKI builds and 8 bytes for the GPI
build.  Two choices keep the kernel under the cap on newer toolkits: the `m128` instances issue
one MMA K step per main-loop iteration (the `m64` instances keep four, which their throughput
needs; unrolling `m128` measured within noise on CUDA 13.0 and lets later compilers keep more
operand fragments in flight), and above 16 ranks the task builder keeps its per-source tables in
the `tb_pub` workspace instead of rank-indexed register arrays.  Register allocation depends on
the toolkit, so `validate_init` builds the device modules a deployment may launch (the same
modules the workspace loads) at every rank count and raises `MoEEpConfigError` on any instance
that spills, naming its GIN backend, registers, stack and local bytes and the nvcc release; the
workspace re-checks every loaded instance (`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES <= 8`, the
GPI fence's stack frame).

**libcu++ automatic-storage check.** The device source defines
`_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE` before it includes `nccl_device.h`.  The libcu++ headers
of CUDA 13.3 test every `cuda::atomic_ref` operation for a local-memory (automatic storage)
address and carry a local-memory fallback for it.  NCCL's Device API uses `cuda::atomic_ref` for
its proxy queues, signals and counters, and with the fallback 34 of the 352 instances above
spill one register at EP 8 / 16 (`m64_copier`, `m128_nocopier`, `m128_copier` at the
DeepSeek-V3, Qwen3 and DeepSeek-V4.1-Flash geometries).  Those atomics only ever address
NCCL-owned device or host-pinned memory.  The headers of CUDA 13.0 and 13.1 skip the check
unconditionally and do not know this macro, so their output is unchanged by it.

## Manifest schema (`cake-sm120-megamoe-manifest/3`)

```json
{
  "format": "cake-sm120-megamoe-manifest/3",
  "arch": "sm_120a",
  "binding": {"path": "cake_sm120_megamoe_binding.cu", "sha256": "<hex>"},
  "device": {
    "entry": "cake_sm120_megamoe_kernel.cuh",
    "sources": {"cake_sm120_megamoe_kernel.cuh": "<hex>", "cake_sm120_megamoe_kernel_part001.cuh": "<hex>", "...": "..."},
    "instances": [{"instance": 0, "name": "m128_copier", "path": "cake_sm120_megamoe_m128_copier.cu",
                   "symbol": "cake_megamoe_m128_copier", "task_m": 128, "copiers_active": true, "sha256": "<hex>"}, "..."],
    "parameters": {"MOE_RANKS": "...", "...": "..."},
    "threads": 384,
    "renamed_identifiers": ["CAKE_INF", "CakeTensorMap", "CakeTensorMap64", "CakeTensorMapPack"],
    "license_header_prepended": true
  },
  "geometry": {"path": "cake_sm120_megamoe_geometry.json", "sha256": "<hex>"},
  "provenance": {"generator_commit": "...", "single_file_sha256": "...", "unrenamed_sha256": "...",
                 "geometry_module_sha256": "..."}
}
```

The geometry file holds the derived-switch ladder, the buckets of structural switches, one
table of constant / buffer / guard expression indices per bucket (as deltas against the first)
and the 130-argument kernel ABI.  The backend evaluates its expressions with a whitelisted
integer interpreter (`+ - *`, comparisons, `not`, conditional, `max`, `min`, C-truncating
division and remainder); nothing is `eval`-ed.

## Runtime requirements

* NCCL **2.30.7** with the Device API on the EP communicator
  (`ncclCommQueryProperties(...).deviceApiSupport`); GIN is optional (an all-LSA table needs
  none).  The binding fails closed on any other NCCL version: the device code inlines the
  Device API of the headers it is compiled against, and only 2.30.7 is validated.  Headers and
  library are found as `flashinfer/jit/comm.py` finds NCCL: `NCCL_HOME`, then the
  `nvidia-nccl-cu13==2.30.7` wheel.
* CUDA 13 toolkit (`cuTensorMapEncodeTiled` via the driver entry point; `cuLaunchCooperativeKernel`;
  `-lcuda`).  The device build uses the compiler and headers of one release, found through
  `CUDA_HOME`.  All multi-GPU tests run with **13.0.88**.  The register grid above compiles
  spill-free with 13.0.48, 13.0.88, 13.1.80, 13.1.115, 13.2.51, 13.2.78, 13.2.86, 13.3.33,
  13.3.73, 13.4.59 and 13.4.92 (compile only for the releases other than 13.0.88).  Code built
  by the releases after 13.0 ran about 5-8 % slower in our EP 8 DeepSeek-V4-Flash measurements
  (most at 512-1024 tokens per rank, within a few percent at 4096), with or without spills, so
  13.0 remains the recommended device compiler.  On any other release `validate_init` refuses a
  spilling build with the nvcc release in the message; point `CUDA_HOME` at a release listed
  above and clear the JIT cache.
* An initialized `torch.distributed` **NCCL** process group for the EP domain (the session adopts
  its `ncclComm_t`).
* Window memory per rank grows with world size x rows x top-k (the result windows dominate:
  about 3.3 GB at EP4 / E256 / top-k 6 / 4096 rows, 13 GB at EP16); size `max_tokens_per_rank`
  to the real per-call token count.
* Per call the host needs a rank-uniform bound on the per-rank token counts (row schedule,
  grid, skip of an all-empty call).  A framework that knows the largest per-rank token count of
  the call (e.g. data-parallel attention's per-rank token counts) passes it as the token bucket
  of the invocation, the same value on every rank:
  `layer.stage_inputs(t, compile_tokens_per_rank=bucket)` + `layer.compute_staged(output=...)`
  (the `MoEEpMegaLayer` hook `sm120_mxfp4_mxfp8_bf16_cutedsl` uses for its graph buckets).  The
  call is then free of host synchronization and the grid is sized from the bound.  Without it
  (`layer(t)`) the backend all-gathers `[rows, expert histogram]` and reads it back: one host
  synchronization per call, which stops the caller's CPU from running ahead of the GPU.  A bucket
  above the true largest count is correct but may pick the M128 schedule where M64 would do and
  spends per-call setup on rows that are not there; the exact largest count reproduces the
  exchanged schedule bit for bit.

## CUDA graphs

A bucketed call has no host-side dependency on device data, so it can be captured:

1. Run the layer once eagerly on the workspace (`layer(t)` or `layer.warmup(t)`; the first call
   uploads the weights' TMA descriptors).
2. Capture one graph per token bucket, each with its own static input tensors:
   `with torch.cuda.graph(g): layer.stage_inputs(static, compile_tokens_per_rank=bucket); out = layer.compute_staged(output=None)`.
   Every rank captures the same buckets.
3. Replay: copy the new `hidden_states / topk_ids / topk_weights` into the bucket's static
   tensors (the row count of a graph is the one it was captured with) and replay.  All ranks
   must replay the same bucket sequence, in the same order, like eager calls; graphs of different
   buckets may be interleaved freely (`test_cuda_graph_buckets_replay_in_mixed_order` replays
   three buckets, both row schedules, in a mixed order and checks the eager bits).
4. The output view of `compute_staged(output=None)` is the workspace's output buffer, shared by all
   graphs and eager calls on that workspace: consume or copy it before the next call, or pass
   `output=`.

The kernel keeps its call epoch in device memory, so replays and eager calls can be mixed.
Capture without a bucket (`layer(t)` inside capture) is refused with `MoEEpConfigError`.

* **Pin NCCL's topology on EP8 / EP16 hosts** (`NCCL_TOPO_FILE`): without it the launch-time LSA
  team partition of a PCIe host can vary between launches, and with it which pairs are LSA
  reachable.  Pin the transport table too (`transport_table=` in the backend config) where the
  probe's candidates are within its 3 % decision margin.
* The setup probe scores the candidate tables at one full-capacity call of the deployment:
  `max_tokens_per_rank x top_k / world` routes per destination of the build's own dispatch-record
  and encoded-result-row sizes (capped at 32 MiB per destination), GIN puts of the kernel's
  minimum dispatch chunk.

## Formatting

The generated files are excluded from the formatting hooks in `.pre-commit-config.yaml`
(`sm120/cake_mxfp4_megamoe/src/cake_sm120_megamoe_(?:kernel.*\.cuh|m(?:128|64)_(?:no)?copier\.cu|geometry\.json)`);
the JIT loader verifies their digests.  The binding and carrier TUs **are** formatted with the
repository's `clang-format`.
