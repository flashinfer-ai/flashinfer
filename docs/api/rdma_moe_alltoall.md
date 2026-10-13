# RDMA MoE AlltoAll on SM120

`flashinfer.comm.RdmaMoeAlltoAll` provides expert-major
GPU-initiated RDMA dispatch/combine for SM120 GPUs with RDMA-capable NICs.
NVFP4 communication reduces activation traffic, with BF16 inputs and expert
outputs; BF16 and FP8 communication are also available. All non-self
messages use NCCL GIN/GDAKI RDMA, including messages between GPUs on the
same host. No NVLink, GPU peer-memory mapping, or NVSHMEM runtime is required.

This is a regular FlashInfer communication API. The implementation is JIT-only
because it builds against the caller's external NCCL host/device SDK and runtime;
see [dependencies](../development/rdma_moe_alltoall_dependencies.md).
`RdmaMoeAlltoAllConfig` fixes precision, reduction and launch strategy for a
buffer. `communication_mode="auto"` selects a launch strategy within RDMA.

Operator benchmarks establish communication performance, not model quality
or end-to-end decode improvement.

## Supported scope

- SM120, hidden sizes **4096, 6144, 7168**, top-k **1–8**, positive fixed
  per-rank capacity, and expert count divisible by the EP group size.
  `world_size * capacity` must be divisible by four; at most 1024 experts
  are assigned to each rank.
- BF16, E4M3 FP8 with FP32 block scales per 128 values, and NVFP4 E2M1 with
  E4M3 block scales per 16 values. Communication dtype is fixed per buffer.
- Audited **NCCL 2.30.7 GDAKI** host/device headers and runtime, with a working
  GPUDirect RDMA provider. Version/provider checks fail before communication
  when these requirements are unmet. InfiniBand/RoCE device setup is external.
- Collective setup/teardown via the caller's process group, then GPU-initiated
  communication through an independent NCCL communicator. Self contributions
  use local copies. A Gloo setup group is sufficient.
- `legacy`, `coalesced` and capacity-based `auto` launch selection. Unknown
  hardware/shape profiles retain legacy defaults; manual per-operation
  SM/context budgets remain available for measured tuning.

On the measured 110-SM device (CUDA name `NVIDIA Graphics Device`), auto is
calibrated for H4096 and E288. For EP8, NVFP4 uses 32-SM/8-context coalesced
dispatch for fixed capacity <=32 and combine for capacity <=16; BF16 uses
limits 8 and 16 respectively. For EP4, only NVFP4 is calibrated: dispatch
uses coalesced 32-SM/8-context settings for capacity <=16, and combine for
capacity <=4. Larger capacities retain legacy defaults; EP4 BF16 and FP8
also retain legacy defaults.
The separate RTX 6000D profile is unchanged. These are capacity-based
choices made before Graph capture, not rank-local token-count decisions;
small actual batches inside a larger capacity may need explicit tuning.

The current validation targets are four- and eight-GPU groups on RTX PRO 5000
SM120 hosts with 400 Gb/s ConnectX-8 RoCE ports. Earlier GIN tests covered RTX
6000D EP4/8. On the measured 110-SM device, EP4 unconfigured legacy launches
resolve to 96 CTAs and 72 contexts. Report actual
settings separately from manually tuned configurations.
Cross-node/EP16, ARM/Grace and SM121 are not validated by this feature's
current FlashInfer tests. Compile support alone is not performance evidence.

## Setup and example

Use a source FlashInfer installation with normal JIT dependencies and an
SM120-capable CUDA compiler. Set `NCCL_HOME` to matching host/device headers
and runtime. If the framework would load another NCCL version, preload the
matching library **before Python starts**:

```bash
export NCCL_HOME=/path/to/nccl-2.30.7
export LD_PRELOAD="$NCCL_HOME/lib/libnccl.so.2"
export NCCL_GIN_TYPE=3
# Choose the RDMA devices appropriate to the machine, if needed:
# export NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1'
torchrun --standalone --nproc-per-node=8 examples/rdma_moe_alltoall.py
# BF16 with a dispatch-owned route/weight plan (also supports four ranks):
torchrun --standalone --nproc-per-node=8 examples/rdma_moe_alltoall.py \
  --dtype bf16 --combine-reduction bf16_local_partial_planned
```

Set each process's CUDA device and initialize its process group before
construction. All ranks must use the same configuration, operation order and
fixed capacity. Construct buffers and warm up quantization helpers before
Graph capture. Construction and `close()` are collective.

## Tensor and numerical contract

Let `T` be this rank's token count, `H` the hidden size,
`E_local = num_experts / world_size`, and
`C = world_size * num_max_dispatch_tokens_per_rank`.
`topk_idx` is contiguous INT32 `[T,K]`; negative entries are skipped.
Nonnegative entries must be in range and unique within a token's route.
`topk_weights` is contiguous FP32 `[T,K]`. Device route-value validation
belongs to the caller before Graph capture; metadata validation does not
scan route values or synchronize the hot path.

| Communication dtype | Dispatch payload `[T,…]` | Dispatch scales `[T,…]` | Combine global scale |
|---|---|---|---|
| BF16 | BF16 `[T,H]` | `None` | `None` |
| FP8 | UINT8 `[T,H]` (E4M3 bytes) | FP32 `[T,H/128]` | `None` |
| NVFP4 | UINT8 `[T,H/2]` (packed E2M1) | UINT8 `[T,H/16]` (E4M3 bytes) | FP32 `[E_local,C,1]` |

By default, `dispatch(x, topk_idx)` accepts BF16 inputs and quantizes them
for the configured wire format. Use `input_dtype="prequantized"` with
`x_scales=` to transport packed bytes without modifying them. It returns
`DispatchResult(data, scales, counts, handle, recv_hook)` with payload and
scale shapes `[E_local,C,…]`, INT32 counts `[E_local]`, and an opaque handle.
BF16 receives an empty final scale dimension. Only `counts[e]` rows are valid;
padded rows are not expert inputs. A handle belongs to its originating buffer
and dtype. Keep it intact; converting it to a plain tuple or using it with
another buffer is rejected.

For BF16 legacy dispatch with fixed capacity greater than eight and at least
two experts, routes from one token to the same remote rank share one full
payload. Additional expert routes carry 16-byte references; the receiver
expands them into the same expert-major output after all referenced payloads
arrive. Values and the combine reduction order are unchanged. This reduces
payload bytes, while retaining one communication request per expert route.

Typed NVFP4 legacy dispatch reuses payloads when fixed capacity is greater
than 32 and there are at least two experts. With four or eight ranks, it sends
one complete packet per token/destination rank: a 48-byte header containing
the original token index and expert IDs, followed by the unchanged packed
payload and block scales. The receiver expands each packet into contiguous
rows for each source/expert pair. Other world sizes retain the full-payload
plus 16-byte expert-reference scheme. Quantization, scale conventions, the
expert-major output/handle contract and FP32 combine order are unchanged.
The guard uses fixed buffer capacity and the resolved legacy mode, not the
current token count or a requested auto mode. Packet dispatch retains the
expert-capacity arena needed by NVFP4 combine; it does not use the compact
BF16 planned arena.

The buffer's `communication_dtype` selects the wire format. BF16 wire sends
the input directly; FP8 and NVFP4 wire quantize BF16 rows on the current stream
before transport, including on each CUDA Graph replay.

```python
from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig

buffer = RdmaMoeAlltoAll(
    group, num_max_dispatch_tokens_per_rank=8, hidden=4096, num_experts=288,
    config=RdmaMoeAlltoAllConfig(communication_dtype="nvfp4"),
)
received = buffer.dispatch(
    x_bf16, topk_idx, nvfp4_global_scale=256.0, return_recv_hook=True,
)
received.recv_hook()
# Consume received.data / received.scales up to received.counts per expert.
# Supply BF16 expert output and per-row NVFP4 combine global scales:
combined = buffer.combine(
    expert_output, received.handle, topk_idx=topk_idx,
    topk_weights=topk_weights, global_scale=combine_global_scale,
)
output = combined.data
# All ranks retire GPU/Graph users, then call buffer.close() collectively.
```

Empty tensors, including BF16's UINT8 `[E_local,C,0]` receive scales and
zero-token inputs, retain the same device, rank and dtype checks as nonempty
tensors. Their strides need no contiguity check because no elements are
accessed; nonempty tensors must remain contiguous.

**NVFP4 dispatch does not transmit its global scale.** All ranks and experts
must agree on the same external dispatch global scale; the references use
256.0. Rank- or token-dependent global scales need a separate metadata path
and are outside this contract. The BF16-input path accepts that agreed value
as the positive finite host scalar `nvfp4_global_scale` (default 256.0), fixed
when the Graph is captured; it does not compute an external global-scale
reduction. The static quantization helpers return payload
and block scales in the transport layouts; these are not automatically
reformatted into a GEMM backend's preferred scale layout.

With the default `combine_reduction="fp32"`, combine takes contiguous BF16
expert outputs `[E_local,C,H]`, quantizes each
valid expert contribution for FP8/NVFP4, communicates it, dequantizes and
accumulates in FP32 in the original top-k order with `topk_weights`, then
writes BF16 `[T,H]`. For NVFP4, the caller supplies positive per-row global
scales; `2688 / amax` for nonzero rows (1 for zero rows) is the test convention.
Quantization, dequantization and weighted reduction are part of combine;
expert computation is separate. Dispatch input quantization is part of
dispatch only when the BF16-input path is selected; prequantized dispatch
keeps its original timing boundary.

`out=` reuses caller-owned allocations. Dispatch accepts
`(recv_x, recv_scales, counts, src_info, ranges)`. Reusing these outputs overwrites their old
contents, including handle metadata. Independently retained handles require
independent metadata allocations.

## Explicit BF16 partial reductions and planned combine

The default `fp32` mode applies combine-time weights and accumulates decoded
expert contributions in the original top-k order in FP32 before the final
BF16 cast. Two opt-in modes, available only with `communication_dtype="bf16"`,
reduce communication by forming one partial result per token/expert-owner rank:

- `bf16_local_partial` receives current routes and weights at combine time and
  exchanges that metadata as part of combine.
- `bf16_local_partial_planned` requires `topk_weights=` during dispatch.
  Dispatch builds a private plan, and `combine(expert_x, handle)` uses
  its frozen route and weight snapshots without a second metadata exchange.

Both partial modes have an explicit rounding contract: each weighted expert
contribution is rounded to BF16; contributions on one owner are accumulated in
original top-k order in FP32 and rounded to a BF16 partial; those partials are
accumulated in first-owner top-k order in FP32 and rounded to BF16. This can
differ from the default FP32 combine, even though the communication dtype is
BF16. Exact staged-reference checks and error against the original FP32 sum
are separate measurements; neither establishes model-quality equivalence.

```python
buffer = RdmaMoeAlltoAll(
    group, num_max_dispatch_tokens_per_rank=128, hidden=4096, num_experts=288,
    config=RdmaMoeAlltoAllConfig(
        communication_dtype="bf16",
        combine_reduction="bf16_local_partial_planned",
        communication_mode="coalesced",
    ),
)
received = buffer.dispatch(x_bf16, topk_idx, topk_weights=topk_weights)
# Identity experts; only received.counts[e] rows are meaningful.
output = buffer.combine(received.data, received.handle).data
```

Changing caller-owned routes or weights after dispatch does not change an
existing planned handle. Run dispatch again to build a new plan. Each handle
owns its private plan storage; pending receive hooks retain it as well.
Different handles have independent plans, and a combine-only Graph can reuse
one fixed plan with changing expert outputs. A Graph containing dispatch
rebuilds its plan on every replay from the current device inputs. Keep handles
and their expert outputs alive while eager operations or captured Graphs use
them. Reusing dispatch `out=` overwrites the old expert outputs and public
handle metadata; it does not make those old outputs safe to reuse, even though
private plans are independent.

Typed planned BF16 with coalesced dispatch uses a compact registered arena
indexed by rank and capacity. Its send region and two receive regions reserve
rank-token records instead of expert-token records. Public expert-major output
buffers, the per-handle plan tensors, and appended native metadata retain
their original sizes. Other modes keep their existing arena layout. A smaller
native arena is not an equal reduction in total process memory or a latency
guarantee; `backend_info()` reports the actual native `arena_bytes`, separately
from Torch allocations.

These modes are explicit API choices. The existing auto policy does not enable
planned combine or select its tuned per-operation budgets. To specify budgets,
use `dispatch_num_sms`, `dispatch_num_contexts`, `combine_num_sms` and
`combine_num_contexts` together with explicit modes, and inspect the resolved
CTA counts in `backend_info()`.

## Streams, hooks, and Graphs

Operations run on the current CUDA stream. A buffer's workspace cannot serve
concurrent operations; use stream events or synchronization to order operations
when switching streams. A receive hook
allows the send and receive phases to be separated by independent work on
that stream. Only one hook may be outstanding per buffer, it must be invoked
once on the same stream, and it must finish before another operation or close.

GIN retains two host-selected receive slots. Repeated Graphs must contain an
even number of dispatch/combine API calls and preserve slot alternation on
every rank. A serial full dispatch→combine cycle is supported. Odd-operation
GIN Graphs and arbitrary eager/cross-Graph interleaving are unsupported;
these scheduling restrictions are the caller's responsibility. Retire every
Graph referencing the buffer before collective close. If initialization
cleanup itself fails, restart the worker; the failed constructor does not
provide a public resource-recovery handle.

## Tests and benchmark

CPU dependency and launch-policy tests live in `tests/cpu/comm/`. The distributed
suite is organized by behavior under `tests/comm/rdma_moe_alltoall/`, with shared
numerical references. It covers transport, quantization, route boundaries,
planned and dynamic reductions, retained handles, receive hooks and Graph replay.
Unsupported hardware skips distributed tests.

```bash
pytest tests/cpu/comm/test_rdma_moe_alltoall_policy.py \
  tests/cpu/comm/test_rdma_moe_alltoall_jit.py

torchrun --standalone --nproc-per-node=8 -m pytest \
  tests/comm/rdma_moe_alltoall/ -q --rdma-moe-dtype nvfp4

# BF16 input quantization is included in dispatch timing.
torchrun --standalone --nproc-per-node=8 benchmarks/bench_rdma_moe_alltoall.py \
  --dtype nvfp4 \
  --hidden 4096 --experts 288 --topk 8 \
  --tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
  --operations dispatch combine
```

For planned BF16 use `--dtype bf16 --combine-reduction
bf16_local_partial_planned --communication-mode coalesced`. Dispatch timing
includes plan construction; standalone combine reuses that plan.
Combine includes quantization, communication, decoding and weighted reduction;
expert GEMM and caller-prepared global scales are outside that timer.

The benchmark reuses allocations with warm caches and fixed `1 / top_k` weights.
It uses FlashInfer's CUDA-event Graph timing helper, with paired operations to
preserve alternating slots. Samples include GPU waits and inter-rank launch skew;
the reported P50/P95 use the slowest rank per sample. Rank zero prints JSON lines
with the shapes, resolved strategy and timing settings. Correctness and dynamic
Graph replay are covered by the tests, separately from this latency measurement.

## API tracing and distribution

Public dispatch/combine use FlashInfer API logging. Standalone trace replay
cannot reconstruct remote expert data, the collective communicator, or the
handle's route snapshot. The trace integration explicitly declines unsupported
local replay; distributed correctness is checked by the multi-rank tests.
Initialization, lifetime management and communication remain collective even
when the setup process group uses Gloo.
