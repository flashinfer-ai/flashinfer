# NCCL EP 0.3 integration

The package requires `nccl-extensions>=0.2.0`, which ships EP library 0.3.
Admission uses the loaded EP library version. The adapter additionally admits
LL hidden sizes 3072 and 3584 on that
library; older providers retain the legacy seven-size guard. EP 0.3 uses runtime
JIT compilation, so provide its kernel headers, NCCL device headers, a compatible
`nvcc`, and a writable `NCCL_EP_JIT_CACHE_DIR`. Other JIT sizes remain outside
the adapter's admitted set. The CUDA-major requirement remains unchanged. The
8192-token HT capacity also matches the EP 0.3 provider limit.

Pass `FleetAlgoKnobNcclEpConfig` in the Fleet's `algo_knobs` to set a group-wide
`num_topk` bound or shared `max_num_sms`. The new release also accepts independent
`dispatch_num_sms`, `combine_num_sms`, `ht_flow` values `auto`, `push_pull`, or
`pull_push`, and `meta_exchange_policy` values `auto`,
`require_fused`, or `force_scan`. Unsupported fields fail early on older
providers. A positive group top-k bounds every handle created by that Fleet.

HT uses Flat layout. Explicit `pull_push` requires all ranks to belong to one
NCCL LSA team; `require_fused` has the same topology restriction. The default
leaves flow selection to the provider. A handle-time received-token counter
requires metadata before dispatch and is incompatible with `require_fused`; under
`auto` the provider can select its unfused fallback.

Set `eager_recv=True` on EP0.3 to size receive payload and routing arrays from
the actual received-row count. This mode reads that count on the host and
allocates compact arrays for expert compute. An empty receiver uses one inactive
row with masked routing. Handle creation, updates and dispatch reject CUDA graph
capture in this mode. The default fixed-capacity mode supports capture and
avoids the host count read. Eager sizing is HT-only and cannot use
`require_fused`. It changes the provider's receive bound to AUTO, which can
increase internal staging even though FI's receive arrays become smaller.
For example, four ranks with capacity 128 and top-k 2 use an AUTO bound of
1024 rows instead of the fixed mode's 512. Account for this reservation when
enabling eager sizing; group creation can run out of memory sooner.

Set `dispatch_zero_copy=True` to register a persistent receive window for LL
Rank-Major dispatch on EP0.3. Warm up outside CUDA graph capture; registration
is collective and the payload shape, dtype and device must stay fixed for the
Fleet. All handles on that Fleet share the receive allocation, so finish using
one dispatch's output before issuing another. Destroy captured graphs and
synchronize before destroying the Fleet. Its teardown closes the registrations.
It also destroys communicators FI created itself. A communicator supplied
through `BootstrapConfig.nccl_comm` remains caller-owned and usable afterward;
NCCL can retain registration overhead until its owner destroys it.

The provider selects direct dispatch on NVLink-only groups. This option keeps
its internal staging reservation and does not alter combine. Receive tensors
use the NCCL allocator, whose physical reservation is not included in PyTorch's
allocated-memory counter. This is an opt-in path; its latency and memory cost
depend on the workload.

Set `dispatch_quantization="forward"` to forward quantized values and their
scale rows separately using NCCL's FWD recipe. Through `MoEEpSplitLayer`, this
requires MXFP8 activation dispatch and `FusedMoeKernelConfig.mxfp8_dispatch=False`;
FI quantizes before dispatch and supplies the received scales to expert compute.
The low-level handle takes `DispatchInputParams(x=[values, scales])`, with
contiguous, matching row counts and 16-byte-aligned rows. Split-layer MXFP8
forwarding therefore requires hidden size divisible by 512. In fixed-capacity
mode, receive payload and scale geometry, dtype and device stay fixed for the
Fleet, including without receive windows. A changed geometry requires a new Fleet; dispatch
rejects it before releasing buffers that captured graphs may still reference.
LL Expert-Major, LL Rank-Major and HT Flat support forwarding; HT eager sizing also
applies. Window-backed forwarding retains the LL Rank-Major restriction above.
Fused provider DeepSeek FP8 dispatch quantization remains unwrapped.

Set `combine_quantization="nvfp4"` to use experimental NVFP4 return communication
on EP 0.3 with LL Rank-Major layout and hidden size divisible by 512. Expert
outputs must be contiguous 3D BF16 tensors. FI computes FP32 per-row scales, including zero
rows, and NCCL quantizes, transfers and combines those outputs into BF16 token
results. This option changes communication precision independently of expert
weight precision. Quantization error is expected; downstream model quality and
performance depend on the workload. It supports capture after ordinary warmup.

Both quantization fields default to `"none"`. Their availability does not
change the existing BF16 path or select them automatically. HT Expert-Major,
HT MXFP8 combine, HT window paths, AFD roles, prepared-handle import and an
overflow-policy selector remain outside the adapter. The provider's default
overflow policy still applies. CUDA 12, other LL JIT sizes, and opt-in CFT/NUCS
remain outside qualification. These limits concern the split NCCL backend;
native MegaMoE uses its own communication path.

Eager forwards using a Fleet graph state must run on the stream that created
the state. A different eager caller stream is rejected before input preparation
or transport. Capture may use another stream because the transport follows
the active capture stream.

Fault-tolerance query/update use the C ABI because their Python counterparts
take an EP Tensor and a host NumPy array rather than pointers. Mask updates keep
their pinned host input unchanged until its asynchronous copy finishes; reuse,
topology replacement and destruction wait for that copy when needed.
`update_topology()` preserves the Fleet's fault-tolerance settings; create a
new Fleet to change them. A native replacement failure leaves no live group.
Handle creation and group access then fail until a successful topology retry.
Normal LL group destruction still barriers across the original communicator
and can block after a peer exits.
