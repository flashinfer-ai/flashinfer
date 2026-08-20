# SM120 MXFP4 x MXFP8 Split MegaMoE

This package splits the SM120 MXFP4-weight/MXFP8-activation MegaMoE path into
three device phases:

1. `kernel_dispatch_fc1.py`: dispatch, mixed W4A8 FC1, SwiGLU, amax, and
   MXFP8 output quantization.
2. `kernel_fc2_combine.py`: FC2 and direct token-back to the source rank.
3. `kernel_combine_reduce.py`: top-k reduction into the final BF16 output.

K1 publishes the existing HBM FC1-output ready counters with GPU-scope release
semantics. K2 consumes those counters with acquire polling and can start FC2
bundle-by-bundle before K1 finishes all down-projection columns. K2 executes a
cross-rank completion barrier after its peer stores, so K3 cannot observe a
partially written remote combine buffer.

Weights use packed E2M1 storage (`torch.float4_e2m1fn_x2`) with two logical K
elements per byte. Activations and the FC1-to-FC2 handoff use E4M3, while both
weight and activation scales use E8M0 with K32 block scaling. FC1 and FC2 emit
SM120 mixed `E2M1 x E4M3` block-scaled QMMA instructions; communication payloads
and the BF16 token-back contract are unchanged from the W8A8 split path.

## Validated Environment

The current SM120 path is validated with:

- public `nvidia-cutlass-dsl==4.6.0`;
- CUDA Toolkit 13.3;
- NVSHMEM library 3.7.0 and NVSHMEM4Py 0.3.1; and
- Python 3.12 on RTX Pro 5000-class SM120 GPUs.

The hybrid transport compatibility guard currently requires NVSHMEM 3.7.0.

## Launch Modes

`mega_runner.py` provides two launch modes:

- `green_graph` (default): captures K1, K2, and K3 into one native CUDA Graph on
  ordinary streams, rebinds the K1 and K2 kernel nodes to disjoint driver
  Green Contexts, instantiates the graph once, and replays it from the caller
  stream. K3 remains on the primary context and depends on both K1 and K2.
- `sequential`: runs K1, host synchronization, K2, then K3. This mode is kept
  only for bring-up and debugging.

`heuristic.py` resolves explicit K1/K2/TX/RX SM counts before JIT compilation.
The benchmark runner can override them with `--k1_sms`, `--k2_sms`,
`--tx_sms`, and `--rx_sms`; all four counts must cover the physical device's
SMs. The runtime queries the Green Context minimum partition and co-schedule
alignment, then scales the measured 110-SM presets proportionally for other
SM120 devices. The FlashInfer-facing path performs no online candidate
compilation or timing.

The selector uses expected routed rows per local expert,
`tokens_per_rank * EP * topk / total_experts`, as its tile key: N32 below 30
rows, N64 for 30--63 rows, and N128 from 64 rows onward. On a 110-SM RTX Pro
5000, same-NUMA P2P uses K1/K2 `80/30` for N64 and `72/38` otherwise;
cross-NUMA EP uses hybrid P2P + IBGDA with K1/TX/RX/K2 `48/16/16/30`.
Other SM120 devices preserve these ratios after Green Context alignment.

`dispatch_chunk_tokens` describes only the standalone staged transport used
by the cross-NUMA hybrid path; it is not the direct-P2P dispatch granularity.
The hybrid selector uses 32- or 64-token chunks. It chooses 32 tokens for the
512--1024-token latency band and when a 64-token wide-hidden IBGDA payload
would exceed the configured chunk-byte limit; otherwise it uses 64. Hybrid
dispatch uses two channels and two slots. Hybrid combine uses 16-row chunks,
two channels, and normally two slots (four slots for the small cross-NUMA
window at 128 tokens/rank or below). IBGDA uses one RC per PE with warp RC
mapping.

The same-NUMA `p2p_direct` path does not packetize dispatch into those staged
chunks. Its persistent K1 dispatch group uses token-strided pull: one dispatch
warp pulls one routed token's E4M3 activation with one 1-D TMA transaction,
loads that token's E8M0 scale words and selected top-k weight, stores the row
in the local expert pool, and then advances to another routed token. K1 work
is published only when the selected FC1 N tile is complete, so the visible
ready-work granularity is N32, N64, or N128 according to the tile heuristic,
not a fixed 64 tokens. DP and model names are deliberately absent from the
selector; identical EP shape and topology produce the same specialization.

K2 defaults to the compact eight-warp CTA selected by
`--k2_warp_count 8`: four compute warps, two TMA producer warps, one
scheduler warp, and one FC2-ready auxiliary warp. It has no dispatch warp
group. Compute warps 0-3 are reused only for K2's final rank-release/reset
protocol after FC2 completes. `--k2_warp_count 12` retains the bring-up
topology as a controlled performance baseline.

## Framework API and JIT Cache

Framework integrations import `api.py`; they do not import `mega_runner.py`.
The production flow is:

1. Describe the local EP-rank problem with `MegaMoEProblemSpec`.
2. Call `select_compile_spec(...)` with topology and device SM properties.
3. Cache the resulting kernel set by `MegaMoECompileSpec.cache_key`.
4. On a miss, call `build_split_kernels(spec)` and allocate the returned
   local and symmetric workspace byte counts.

The cache key is a canonical SHA-256 digest over model dimensions, DP/TP/EP
semantics, transport/Green Context heuristic, tile/stage/warp choices, host
build options, and `Sm120JitConfig`. The cache ABI constant in `api.py` must be
bumped when generated device ABI or opaque workspace layout changes. Legacy
benchmark environment variables are parsed once by
`Sm120JitConfig.from_environment()`; production callers should construct the
dataclass explicitly.

Public CuTeDSL 4.6 CUDA-dialect executors do not expose their `CUfunction`
handles. For the current `cluster_shape_mnk=1,1,1` persistent path,
`green_graph` identifies K1 and K2 graph nodes by their distinct captured
grid volumes, which equal the two Green Context SM allocations. Other
cluster shapes are rejected until the executor exposes stable function
identity.

## Configurable DP / TP / EP

The one-node runner exposes `--data_parallel_size` and
`--tensor_parallel_size`; EP is derived as:

```text
EP = WORLD_SIZE / (DP * TP)
```

Global ranks are laid out as contiguous EP worlds:

```text
global_rank = ((dp_rank * TP + tp_rank) * EP) + ep_rank
```

The commands below use the DSV4-flash shape and 8192 input tokens per rank.
They assume an eight-GPU host whose `CUDA_VISIBLE_DEVICES` order places GPUs
0--3 on one NUMA node and GPUs 4--7 on the other. Every GPU pair within one
NUMA node must support CUDA peer access (`cuDeviceCanAccessPeer == 1`) and a
usable NVSHMEM peer mapping. Cross-NUMA pairs are expected to use IBGDA.

The runtime does not infer NUMA membership from ordinal ranges: it gathers
each selected GPU's PCI BDF and `/sys/bus/pci/devices/<BDF>/numa_node`, then
builds the per-EP same-NUMA/cross-NUMA mask. The ordinal ordering above is
still required for these DP/TP examples to form the intended NUMA-local EP
groups. The current selector does not independently validate every
same-NUMA pair with `cuDeviceCanAccessPeer`, so deployments with a different
topology must validate peer access before launch.

```bash
# DP1 x TP1 x EP8 (original path)
torchrun --standalone --nproc_per_node=8 \
  -m moe_sm120_mxfp8_split.mega_runner \
  --num_tokens_per_rank 8192 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 1 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape

# DP2 x TP1 x EP4: two complete expert replicas
torchrun --standalone --nproc_per_node=8 \
  -m moe_sm120_mxfp8_split.mega_runner \
  --num_tokens_per_rank 8192 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 2 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape

# DP1 x TP2 x EP4: NUMA-local dispatch/combine plus TP all-reduce
torchrun --standalone --nproc_per_node=8 \
  -m moe_sm120_mxfp8_split.mega_runner \
  --num_tokens_per_rank 8192 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --data_parallel_size 1 \
  --tensor_parallel_size 2 \
  --route_distribution balanced \
  --enable_static_expert_shape
```

TP partitions the configured full intermediate dimension. FC1 output
channels and the matching FC2 input channels are sharded, K3 runs locally,
then matching EP ranks sum their `(tokens, hidden)` BF16 partial output over
the TP group. With TP2 x EP4, TP pairs are `(GPU0,GPU4)`, `(GPU1,GPU5)`,
`(GPU2,GPU6)`, and `(GPU3,GPU7)`.

`--intermediate` is the full pre-TP gate+up width (4096 for the DSV4-flash
commands above), so TP2 runs a 2048-wide shard on each GPU. TP currently supports the separate
form-A K3 path; the in-kernel form-B top-k reduce is rejected until it has a
TP-aware accumulation-order reference.

The package-local process-group and NVSHMEM bootstrap lives in
`runtime/parallelism.py`; public `src/` is unchanged.
