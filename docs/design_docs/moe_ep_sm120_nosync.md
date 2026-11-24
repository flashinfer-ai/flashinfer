# SM120 MXFP4 × MXFP8 nosync MegaMoE

`sm120_mxfp4_mxfp8_bf16_nosync` is an expert-parallel MegaMoE kernel
for SM120 (compute capability 12.0: RTX PRO 6000, RTX 6000D) aimed at decode-size
batches, at most 128 tokens per rank. It takes MXFP4 (E2M1, UE8M0 per 32) expert
weights and MXFP8 (E4M3, UE8M0 per 32) activations and returns BF16. One
persistent kernel per layer exchanges the routes, dispatches tokens, runs FC1,
clamped SwiGLU and FC2, pushes the partials back, and combines them. Ranks never
wait on a barrier: each producer writes an epoch-tagged flag into the consumer's
memory after its data, and the consumer polls its own memory. A group fits one
host (up to 8 ranks) or spans hosts (up to 32 ranks, e.g. EP16 on two 8-GPU
hosts, EP32 on four); the other hosts' ranks are reached over the NIC (NCCL GIN).

It is normally used as a small-batch option of a layer that runs the cake or
cutedsl SM120 MXFP4 backend: calls within `max_tokens_per_rank` tokens per rank
and `max_tokens_per_expert` tokens per expert run this kernel, larger calls run
the configured backend unchanged. Sources:

| Part | Path |
| --- | --- |
| Kernel | `include/flashinfer/megamoe/sm120_mxfp4_mega_moe.cuh` |
| Launcher, GIN session | `csrc/megamoe/megamoe_sm120_mxfp4.cu`, `csrc/megamoe/megamoe_sm120_gin_session.cu` |
| JIT specs | `flashinfer/jit/megamoe.py` (`gen_megamoe_sm120_mxfp4_module`, `gen_megamoe_sm120_gin_session_module`) |
| Backend, weights, buffer | `flashinfer/moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_nosync/` |
| Layer option | `flashinfer/moe_ep/modes/nosync_layer.py` (`MoEEpNosyncLayer`, `NosyncOptions`) |
| Tests, benchmark | `tests/moe_ep/test_moe_ep_sm120_w4a8_nosync_mega.py`, `benchmarks/bench_megamoe_mxfp4_sm120.py` |

## Public API

```python
layer = MoEEpLayer(
    bootstrap, fleet_params,
    PrequantizedMoEWeights(w13=..., w2=..., w13_scale=..., w2_scale=...),
    backend=MegaConfig(megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(...), ...),
    nosync=NosyncOptions(),  # max_tokens_per_rank=128, max_tokens_per_expert=128, transport=None (cake's table)
)
```

- The configured layer is built exactly as without the option; `forward`,
  `stage_inputs` / `compute_staged`, `warmup` and other attributes behave as on it.
  With `transport=None` on cake, the nosync kernel is built on cake's table at
  `warmup()` or the first eager call, so the default workspace can still be
  configured (`create_workspace`) after construction; call `warmup()` on every
  rank before capturing.
- Every rank must take the same kernel for a call, so the switch uses a
  rank-uniform bound: `compile_tokens_per_rank` of `stage_inputs` when given, the
  captured token count during CUDA graph capture, otherwise the largest token
  count of any rank (one all-reduce).
- Where the kernel cannot run (compute capability other than 12.0, weights that
  are not pre-quantized MXFP4, cake with its in-kernel shared expert
  (`shared_expert="fp8_in_kernel"`; the nosync kernel computes the routed experts
  only), an unsupported geometry) the option warns once and every call runs the
  configured backend.
- A call runs the nosync kernel when its bound is at most `max_tokens_per_rank`
  and the routes it bounds average at most `max_tokens_per_expert` per expert:
  `bound * ep_size * top_k / num_experts <= max_tokens_per_expert`. The per-expert
  load, not the per-rank count, decides where the kernel stops winning. The
  default 128 keeps every call up to 128 tokens per rank on the kernel through
  EP32 for DeepSeek-V4.1-Flash and GLM-5.3-Flash. Measured on RTX 6000D with
  balanced routing and ConnectX-8 switches, GLM-5.3-Flash at EP16 (57 tokens per
  expert at 128 tokens per rank) takes 397 / 469 µs per layer at 96 / 128 tokens
  per rank on the kernel, 534 / 608 µs when a limit of 32 hands them to cake;
  past 57 tokens per expert (measured with a larger per-rank cap) the kernel
  and cake are within 1.5%. **For the best performance,
  measure the switch point on the deployment and set `max_tokens_per_expert`
  yourself**: where the kernel stops winning depends on the model, the routing
  and the interconnect (e.g. 12–24 tokens per expert for EP8 over the NIC, beyond
  28 with every pair P2P), so no single value is best everywhere.
- `NosyncOptions` fields: `max_tokens_per_rank` (1–128) and
  `max_tokens_per_expert` (default 128), both the same on every rank;
  `transport` and `tokens_per_item` (below), and `top_k` / `swiglu_limit`, which
  default to the configured cake or cutedsl backend's and must be set for any
  other backend.
- The kernel is also a plain MegaMoE backend
  (`Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig`) for deployments that only run
  decode.

## Weight contract

The kernel reads the FP4 data in the layout the cake and cutedsl SM120 backends
already hold: row-major packed E2M1 (`[rows][K/2]` bytes, element `2i` in the low
nibble of byte `i`) with the FC1 rows interleaved per 16-row block (gate rows
`8b..8b+7`, then up rows `8b..8b+7`), so each SwiGLU pair lands in one thread.
With either backend configured, the layer passes their packed tensors to this
kernel (same pointers) and adds only its own UE8M0 scale arrays, regrouped per
16-row block and K128 (1/16 of the FP4 bytes). Any other MXFP4 backend gets a
packed copy, whose size is logged.

## Kernel

One cooperative launch per call; the grid is the device's full occupancy, so
every CTA is resident (CTAs wait on each other's flags). Phases:

1. **Route exchange.** One CTA sends every peer (a thread per peer) its routes
   to each of that peer's local experts and the number of its tokens routed
   there, so every rank knows its full expert load before any token arrives.
2. **Weight prefetch at t = 0.** The rank's own tokens already show some of its
   active local experts; while tokens and the other ranks' routes are in flight,
   the CTAs that neither dispatch nor send the routes pull the head of those
   experts' FC1 weights into L2 (64 MB budget, when the rank has at most 64
   local experts). Once the expert map exists and it has more than 8 tiles, each
   CTA also pulls the weights and scales of its static first work item (FC1 or
   FC2).
3. **Dispatch.** One CTA per token: the staged MXFP8 row (codes and scales) goes
   to each destination rank's expert pool with 16-byte P2P stores (each copy's
   pool slot counts the earlier routes to its expert). The last CTA out flags
   the P2P ranks, a thread per rank, then puts each GIN rank its rows.
4. **Expert map.** Each rank builds the per-expert token lists from the exchanged
   counts (same slot order as the dispatch) and waits for every source's rows.
5. **FC1 / FC2 work queue.** A work item is one expert's 128 weight rows times a
   tile of its tokens; the first item of each CTA is static (the last CTAs take
   the head of the queue: the first ones dispatch and arrive last), the rest
   come from an atomic counter. The queue holds FC1 of the first `lag` tiles,
   then FC1 of tile t + `lag` followed by FC2 of tile t, then the last FC2s
   (`lag` = one wave of CTAs plus an expert's own tiles), so partials leave
   during the whole weight stream. An FC2 item waits until every FC1 item of
   its expert is done.
6. **Push-back and combine.** FC2 writes each (token, k) partial in BF16 into the
   source rank's combine buffer; the source sums its top-k partials with the
   routing weights once every expert rank has signalled.

**Work-item width (`tokens_per_item`).** An item multiplies 8 tokens per weight
load, 16 once the experts average at least 3 routed rows each
(`tokens * top_k / local_experts`), and 32 from 8 rows; 24 can be forced. Items
wider than 8 stage their tokens' activations in shared memory once per CTA (the
8 warps share them, one K step ahead with `cp.async`) and skip the MMAs of their
empty groups of 8, so the last, partly empty tile of an expert costs little. On
RTX 6000D with cold L2 (DeepSeek-V4.1-Flash, balanced routing, EP8 on one host
and EP16 across two) 32 is 1–9% faster than 16 from 8 rows; below it 16 is as
fast or faster.

**GEMM core.** `mma.sync ... kind::mxf8f6f4.block_scale` with swap-AB: 16 weight
rows × 8 tokens per MMA. Each lane loads 16 contiguous bytes of a weight row per
K128 with streaming loads (`__ldcs`) and the four lanes of a quad transpose them
into MMA fragments; within each 32-wide scale block K is permuted identically
for both operands. Each warp also prefetches its rows 128 B ahead into L2. With
8-token items (one MMA per weight fragment) the loop waits on DRAM, so the next
step's weights and scales are loaded into a second register buffer while the
current step's MMAs run. Wider items keep the single-buffer loop: under the
128-register budget the second buffer made 64–128 tokens per rank 3–5% slower.

**Occupancy.** The kernel is built for two CTAs per SM (`__launch_bounds__(256,
2)`, 128 registers) and the grid follows the occupancy. With one CTA per SM
(the 8-token loop at ~168 registers) an expert's 8-token items did not fill one
wave of load slots: EP8 at 1 token per rank was 207 µs, 195 µs with two. A
change in register allocation still changes the grid; check `cuobjdump
--dump-resource-usage` of both builds after kernel edits.

**Routes.** Every CTA reads the rank's top-k ids into shared memory once at
start (int16, masked or repeated ids as -1), a thread per route: the routing
counts, the prefetch, the dispatch slots and the combine all read that copy.
Reading the int64 ids where each phase needed them put up to top-k dependent
memory latencies on the routing CTA's path (GLM-5.3-Flash EP8, 1 token per
rank: routes sent at 8 µs instead of 3).

**Synchronization and errors.** The data is fenced with `__threadfence_system`
and the flag then written with a plain store (a release store adds a second
system fence, 1–3 µs per flag), read with `ld.acquire.sys`; there are no remote
atomics. For each wait on every rank (routes, rows, partials) one CTA's warp 0
polls the peers' flags and GIN signals, a lane per rank, and republishes the
epoch on a local word that the other CTAs wait on: with every CTA polling, the
loads delayed the flags' and signals' arrival. Messages to every rank (routes, the "done" after the last
FC2 item) go out a thread per rank. Every spin
times out (30 s) and records an error code instead of hanging;
`backend.protocol_error(workspace)` reads it. The last CTA to exit resets the
per-call counters and advances the epoch, so CUDA graph replay needs no host
work.

## Transports

`transport` selects how a pair of ranks reaches each other:

- `None` (the `NosyncOptions` default): with the cake backend configured, the
  per-pair table cake resolved (its `transport_table`, including `"auto"` /
  `"tuned"`), so both kernels reach each pair of ranks the same way: P2P where
  cake uses LSA, GIN where it uses GIN; with any other backend `"p2p"`. The table
  is read from the layer's default workspace at `warmup()` or the first call.
- `"p2p"` (the default of `Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig`): PCIe
  peer stores between every pair (torch symmetric memory).
- `"auto"`: the buffer is one NCCL symmetric window; pairs NCCL maps for
  load/store use P2P, the others use NCCL GIN puts over the NIC.
- `"nic"`: GIN for every other rank.
- a flat tuple of `world_size * world_size` cells in cake's encoding (`0` self,
  `1` P2P, `2` GIN, symmetric); a P2P cell must be a pair NCCL maps.

`"auto"`, `"nic"` and tables with GIN cells need NCCL 2.30.7 with the Device API and a GIN-capable
network, and every rank must use the same value. The GIN path is a separate JIT
module per GIN backend (proxy / GDAKI / GPI) with the NCCL device code in the
device pass only. A token goes once to each GIN rank's inbox, whatever number of
that rank's experts it routes to: the rows for one GIN rank are staged packed
and put once after the dispatch (one put per token queued on the NIC with many
GIN ranks), with the slot table mapping the rank's pool slots to the packed rows
right before them in the same buffer, so one put carries both; partials come
back one put per run of finished rows of an FC2 tile. A CTA posts its puts to one peer on one context
and spreads peers over the contexts' queue pairs, so puts to several peers go
out in parallel. The receiver counts GIN signals per source instead of polling
a flag.

**Which pairs `"auto"` sends over the NIC depends on the host.** Three 8-GPU
hosts measured during development:

| Host | PCIe fabric | Cross-socket P2P (SM stores) | NICs | EP8 with `"auto"` |
| --- | --- | --- | --- | --- |
| RTX 6000D, 2× EPYC 9575F | GPU pairs under Broadcom PEX890xx switches | fast: 8 GPUs storing to each other at once, 41.7 GB/s per GPU | 4 ports up, not all reachable from each other | every pair P2P |
| RTX 6000D, 2× EPYC 9575F | GPU pairs under ConnectX-8 built-in switches (four bridge levels, relaxed ordering off) | a GPU storing to two or more remote switches at once drops to ~1.5 GB/s (8 GPUs at once: 0.8 GB/s) | 8 ports up, all reachable | P2P within a socket, GIN across, **only when `NCCL_IB_HCA` names all NICs**; with one NIC NCCL maps every pair and the kernel stays on the slow P2P path |
| RTX PRO 6000, 2× Xeon 8480+ | GPU pairs under Broadcom PEX890xx switches | ~5 GB/s, NCCL does not map these pairs | 4 ports up | P2P within a socket, GIN across |

EP2 and EP4 within one socket never store to two remote switches at once and run
the same on both RTX 6000D hosts.

## Tests and benchmark

```bash
# correctness; FLASHINFER_TEST_MEGAMOE_SM120_NIC=1 adds the GIN tests (nic at EP 2/4/8, auto and cake's tables at EP8,
# the 32-rank build at EP8), FLASHINFER_TEST_SM120_CUTEDSL=1 the cutedsl fallback (NVSHMEM; not on RTX 6000D)
pytest tests/moe_ep/test_moe_ep_sm120_w4a8_nosync_mega.py

# across hosts (EP16 on two 8-GPU hosts; --nnodes 4 for EP32), on every host:
FLASHINFER_TEST_MEGAMOE_SM120_MULTIHOST=1 torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d \
  --rdzv-endpoint <host0>:29500 -m pytest -x tests/moe_ep/test_moe_ep_sm120_w4a8_nosync_mega.py -k multihost

# timing: one MoE layer per call, CUDA graph, cold L2, identical seeded routes across backends
B="python benchmarks/bench_megamoe_mxfp4_sm120.py --ep 4 --shape flash --capacity 128 --tokens 1 2 8 16 32 64 128"
C="--cold-l2 --repeats 1 --rounds 20"
$B --backend nosync --save-routing /tmp/rt $C   # cake fallback + NosyncOptions() (cake's table); EP8: add --transport lsa_where_reachable
$B --backend cake   --load-routing /tmp/rt $C   # EP8: --transport lsa_where_reachable
```

The environment variables the tests read are listed in
[environment variables](../development/environment_variables.md).

## Performance summary

Commit `3ffd6c87`, all backends from that tree, cold L2, µs per MoE layer, DeepSeek-V4-Flash routed experts
(H 4096, I 2048, 256 experts, top-6); speedup is cake ÷ this kernel, cake with its fastest table on each host
(RTX 6000D EP8: `lsa_where_reachable` with `NCCL_IB_HCA==mlx5_0`, every pair LSA). The full
tables (three model shapes, cutedsl and DeepEP + CUTLASS columns) are in the PR
description of [#8806](https://github.com/flashinfer-ai/flashinfer/pull/8806).

| Host | Config | 1 token / rank | 8 | 32 | 128 |
| --- | --- | --- | --- | --- | --- |
| RTX 6000D (Broadcom) | EP4 | 125.7 (2.35×) | 448.3 (1.49×) | 696.8 (1.35×) | 776.0 (1.24×) |
| RTX 6000D (Broadcom) | EP8, every pair P2P | 142.8 (2.21×) | 309.8 (1.65×) | 397.9 (1.51×) | 544.3 (1.20×) |
| RTX PRO 6000 | EP4 | 124.9 (2.41×) | 389.1 (1.56×) | 599.7 (1.40×) | 700.4 (1.25×) |
| RTX PRO 6000 | EP8, cake `lsa_where_reachable` (P2P within a socket, GIN across) | 172.6 (1.73×) | 371.0 (1.40×) | 435.7 (1.33×) | 703.5 (0.91×) |

## Limits and known gaps

- Compute capability 12.0; `hidden % 256 == 0`, `hidden <= 7936`,
  `intermediate_size % 256 == 0`; top-k ≤ 8; at most 128 tokens per rank; up to
  8 ranks with ≤ 256 local experts, or 9–32 ranks with ≤ 64 local experts (a
  second build of the kernel, `FLASHINFER_MEGAMOE_SM120_MAX_WORLD=32`, loaded for
  groups of more than 8 ranks: its per-call bookkeeping in shared memory holds
  ranks × local experts ≤ 2048). Across hosts every pair between hosts is a GIN
  pair: `transport` must be `"auto"`, `"nic"` or a table (cake's by default).
- Spanning hosts does not pay off at 1 token per rank: the cross-host round
  trips dominate, and EP16 over two hosts is 33–88% slower per layer than EP8 on
  one host at the same 1 token per rank (DeepSeek-V4.1-Flash and GLM-5.3-Flash,
  random and balanced routing, RTX 6000D hosts with ConnectX-8 switches). From 8
  tokens per rank, EP16 is faster with random routing (9–30%).
- When EP8 goes over the NIC across sockets (`"auto"` on the RTX PRO 6000 host
  and on the ConnectX-8-switch RTX 6000D host), cake can be faster at 128 tokens
  per rank (on RTX PRO 6000: 0.91× for DeepSeek-V4-Flash, 0.85× for GLM-5.3-Flash; on the ConnectX-8 host
  0.87× for DeepSeek-V4-Flash and 0.81× for GLM-5.3-Flash). Phase stamps place
  the extra time in the GIN path: on the ConnectX-8 host in receiving the
  dispatched tokens (the per-token puts queue behind each other), on RTX PRO 6000
  in waiting for the pushed-back partials. On such hosts `max_tokens_per_expert=16`
  hands these calls to cake. Hosts where every pair is P2P (the Broadcom RTX 6000D
  host) stay faster up to 28 tokens per expert, which the default keeps on the
  kernel.
