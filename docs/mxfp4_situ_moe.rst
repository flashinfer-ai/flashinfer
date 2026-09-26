Planned native MXFP4 SiTU MoE
=============================

The CuTe DSL planned runner targets SM100 and SM103, including NVIDIA B300.
It consumes native packed E2M1 weights and UE8M0 scales with block size 32,
FP8 E4M3 activations and BF16 output. Quantization is explicit metadata.

.. code-block:: python

   import torch

   from flashinfer.fused_moe.prepare import (
       prepare_cute_dsl_mxfp4_weights, shard_cute_dsl_mxfp4_weights,
   )
   from flashinfer.fused_moe.cute_dsl import (
       CuteDslMxfp4MoEWrapper, Mxfp4MoEParallelLayout,
   )
   from flashinfer.tllm_enums import ActivationType

   # Explicit placement: this process is rank 3 of 8 expert-parallel ranks.
   layout = Mxfp4MoEParallelLayout("expert_parallel", size=8, rank=3)

   # Load time: canonical unsharded W1 is [up, gate]; all four tensors are
   # uint8 bytes. Slice this rank's shard, then shuffle it for the kernels.
   shard = shard_cute_dsl_mxfp4_weights(
       w1, w1_scale, w2, w2_scale, ep_size=layout.size, ep_rank=layout.rank,
   )
   weights = prepare_cute_dsl_mxfp4_weights(*shard)
   runner = CuteDslMxfp4MoEWrapper(
       num_experts=896, top_k=16, hidden_size=7168, intermediate_size=3072,
       parallel_layout=layout,
       quantization="mxfp4_w4a8", activation_type=ActivationType.Situ,
   )
   assert runner.layout.num_local_experts == 112
   assert runner.layout.local_expert_offset == 336
   workspace = torch.empty(runner.get_workspace_size(T), device="cuda", dtype=torch.uint8)
   output = torch.empty((T, 7168), device="cuda", dtype=torch.bfloat16)
   plan = runner.plan(
       x, x_scale, topk_ids, topk_weights, *weights,
       beta=beta, linear_beta=linear_beta, workspace=workspace, output=output,
   )

   # Forward: input, routing and parameter contents may change in-place.
   plan.run()

``plan`` compiles and runs warmup with valid inputs before serving or graph
capture, writing the supplied output and workspace. It binds tensor addresses
and compiled launch arguments for one token count. Create the required plans
before serving; their workspaces can be reused only when execution does not
overlap. ``run`` enqueues on the caller's current CUDA stream and returns the
same output storage. No auxiliary/default stream is used. Use distinct output
and workspace buffers for concurrent streams, with neither buffer overlapping
an input. Captured graphs retain the bound addresses; change values in place
before replay.

SiTU parameters are contiguous CUDA FP32 tensors with one element or one
element per local expert. The caller must supply finite, positive values;
planning validates tensor metadata without reading these values on the host.
They are read at execution, not specialized into compiled constants:

.. code-block:: text

   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
   up_out = up                                  # linear_beta is None
   up_out = linear_beta * tanh(up / linear_beta)  # otherwise
   activation = gate_out * up_out

``ActivationType.Situ`` is the default. ``ActivationType.Swiglu`` is also
supported without SiTU parameters. GEMM1 quantizes the activation directly
from FP32 to group-32 MXFP8 before GEMM2. Router weights are applied after the
expert computation. Output is this rank's partial sum (its expert interval or
its intermediate shard, see `Parallel layouts`_); cross-rank reduction is
external.

Mixed MXFP8/MXFP4 SiTU uses native approximate FP32 tanh for the gate and
optional up clamp. Runtime beta, sigmoid evaluation and intermediate
quantization remain explicit. Values near a quantization boundary can change
an intermediate byte; the numerical report includes that effect. W4A4/W4A16
SiTU and unrelated GELU/tanh helpers retain their existing arithmetic.

Native layouts
--------------

Canonical W1 is ``[local_experts, 2*I, H/2]`` uint8, with two E2M1 values per
byte, low nibble first. Its scales are ``[local_experts, 2*I, H/32]`` uint8.
Canonical W2 is ``[local_experts, H, I/2]`` with
``[local_experts, H, I/32]`` scales. UE8M0 byte ``127`` encodes unity and
``255`` encodes NaN; an E4M3 tensor containing numeric ones is not the same
representation.

The load-time helper interleaves W1 up/gate in 64-row groups and rearranges
scale bytes without dequantizing or requantizing. W2 storage is reused.
Prepared scales have logical axes ``[32,4,M/128,4,K/128,local_experts]`` and
strides ``[16,4,(K/128)*512,1,512,(M/128)*(K/128)*512]``. The helper retains
no source cache. Preparation temporarily needs both the source and transformed
W1/scales; release canonical W1 and linear scales afterward if they are no
longer needed. The plan retains only the prepared weights. The prepared tuple
keeps the reused W2 storage alive.

Activations use ``[T,H]`` E4M3 plus linear ``[T,H/32]`` UE8M0 bytes. Routing
uses global int32 expert IDs and BF16 or FP32 weights, each ``[T,top_k]``.
Passing ``topk_weights=None`` accepts packed int32 routing with expert ID in
the high 16 bits and BF16 weight bits in the low 16. FP32 weights are bound
directly; BF16/packed routing is converted into preallocated workspace on the
caller stream.

For T=1..16, one CuTe kernel combines routing unpack/conversion with output
clearing and the local expert histogram/prefix/scatter (on the dense path only
when PDL is disabled). The default decode forward then launches GEMM1 and
GEMM2 with finalize: three device-kernel launches in total. Expert IDs must be
unique within each token and lie in ``[0, num_experts)``, as in standard top-k
routing. On the dense path, PDL-enabled decode retains separate native
sorting. The fused sort covers every T with ``T * top_k <= 8192``
(``FUSED_ROUTE_MAX_ROUTES``; T <= 512 for Kimi K3) and ``T * top_k <= 16384``
(T <= 1024) on ranks with at most 128 local experts or where the split form
runs (see "Kernel selection" below); above that the same conversion + output-
clear kernel runs without the fused sort (it is grid-strided over any token
count), followed by the native ``moe_sort``; PDL-enabled prefill keeps the
torch conversion and a separate zero-fill of the output, which since round 15
runs on an auxiliary stream forked after the native sort and joined before
GEMM2 on expert-parallel ranks (``MXFP4_DENSE_ASYNC_MEMSET``: ``ep`` default,
``1`` every layout, ``0`` never; the shard keeps the in-order launch, see the
roofline subsection); the native sort's single-cluster permutation kernel
serves ``T <= 1024`` and the cooperative histogram + offsets path every larger
``T`` (round 15; it was the cluster kernel up to 8192 tokens). The split form
of the MoE-TP shard (T = 128..1024, below) adds two launches to the chain.

Deferred finalize
~~~~~~~~~~~~~~~~~

``plan(..., do_finalize=False)`` delivers the deferred form the feature
request describes for fusing finalization with caller-owned collectives:
rank-local GEMM2 rows, the router weights and the assignment-to-row map,
mirroring ``trtllm_fp4_block_scale_routed_moe(do_finalize=False)``.

* ``output`` is a caller-owned contiguous BF16 ``[rows, H]`` buffer with
  ``rows >= get_deferred_output_rows(T)`` (the padded permuted-row capacity
  for any routing at that token count; it is a host-side formula, no GPU
  work). GEMM2 writes ``alpha * acc`` for every valid permuted row and never
  applies the route weight; rows not referenced by the map are padding and
  hold unspecified values.
* ``plan.expanded_idx_to_permuted_idx`` (int32 ``[T, top_k]``) gives the
  output row of each (token, slot) assignment, or ``-1`` when the expert is
  not local to this rank; ``plan.route_weights`` (FP32 ``[T, top_k]``) are
  the converted router weights. Both are workspace-resident, fixed-address
  tensors valid after ``run``. The finalized output is
  ``sum_k route_weights[t, k] * output[map[t, k]]`` over the local slots.
* The deferred form is provided by the swap-AB path (SiTU, with or without
  the caller's PDL) at
  every token count, in both parallel layouts: routing preprocessing (plus
  ``moe_sort`` above T=256), GEMM1 and GEMM2 with the plain-store epilogue.
  ``mxfp4_moe_capability(..., do_finalize=False)`` reports it through
  ``deferred_output`` and ``get_workspace_size(T, do_finalize=False)``
  sizes its workspace. The plan is CUDA Graph capturable exactly like the
  finalized one; the EP rank-sum and TP shard-sum identities hold after the
  caller's finalization.
* Tests: ``tests/moe/test_cute_dsl_mxfp4_situ_deferred.py`` compares the
  gathered (token, slot) rows with an FP64 row oracle
  (``reference_moe_rows``), with trtllm-gen ``do_finalize=False`` rows
  gathered through its own map, with the finalized plan output after
  reduction, and replays T=1..16 graphs in both layouts.
  ``benchmarks/bench_mxfp4_situ_moe.py --deferred`` produces the paired
  timing table of this form against trtllm-gen ``do_finalize=False``.

On B300, SiTU T1..16 additionally uses a ballot/popcount routing prefix and
a 16-row A transfer in GEMM2 when PDL is disabled, routing tile size is 128,
and both GEMMs use M128/N128, cluster1, non-raster tactics. Unique IDs bound
each expert to at most T valid rows. The transfer retains full shared stage
strides, scale-factor transfers, MMA shape and valid-row output predicates.
Routing and finalize caches include their specialization modes. SM100, T17+,
other activations, PDL and other tactics retain the existing routing/finalize
paths; native SiTU arithmetic still applies to other supported W4A8 shapes.

Parallel layouts
----------------

``num_experts`` and ``intermediate_size`` passed to the capability query and
the runner are always the global model values. The rank-local geometry is
selected from explicit metadata; tensor shapes are validated against it and
never used to infer it. Two forms exist and may be combined when they agree:

* ``parallel_layout=Mxfp4MoEParallelLayout(mode, size, rank)`` with ``mode``
  ``"single"``, ``"expert_parallel"`` or ``"moe_tensor_parallel"``.
  ``Mxfp4MoEParallelLayout.from_sizes(ep_size=, ep_rank=, moe_tp_size=,
  moe_tp_rank=)`` builds it from engine-style sizes.
* ``num_local_experts``/``local_expert_offset``: an explicit contiguous global
  expert interval. This is the expert-parallel form; every expert at offset 0
  is ``single``. Any interval is accepted, not only uniform rank slices.

``resolve_mxfp4_moe_layout`` derives and validates the rank-local values; the
runner exposes them as ``runner.layout`` (an ``Mxfp4MoERankLayout``) and as
``num_local_experts``, ``local_expert_offset`` and ``intermediate_shard``.
Passing both forms with different results, an out-of-range rank, a
non-divisible expert count or intermediate size, or a shard that is not a
multiple of 128 is rejected with a message naming the conflict. Hybrid
expert/tensor parallelism is not representable and ``from_sizes`` rejects
sizes that both exceed one; the corresponding sharding helper does the same.

.. list-table:: Kimi K3 rank-local layouts (E=896, I=3072, top-k 16)
   :header-rows: 1

   * - Mode
     - Rank ``r`` owns
     - Local experts
     - Intermediate shard
     - GEMM1 / GEMM2 per expert
   * - ``expert_parallel`` (EP=8)
     - global experts ``[112r, 112r+112)``
     - 112 at offset ``112r``
     - 3072
     - ``[m,7168]x[7168,6144]`` / ``[m,3072]x[3072,7168]``
   * - ``moe_tensor_parallel`` (MoE TP=8)
     - intermediate columns ``[384r, 384r+384)`` of every expert
     - 896 at offset 0
     - 384
     - ``[m,7168]x[7168,768]`` / ``[m,384]x[384,7168]``

Routing is global in both modes. An expert-parallel rank ignores routes
outside its interval (they produce no rows and no tiles) and returns zeros
for tokens without local routes. MoE-tensor-parallel ranks receive identical
inputs and routes and each returns a partial ``[T, H]`` BF16 result. In both
modes the sum over ranks equals the unsharded result up to the BF16 rounding
of each partial; the caller performs that reduction (all-reduce for TP,
reduce or all-to-all combine for EP). FlashInfer performs no communication.

``shard_cute_dsl_mxfp4_weights(w1, w1_scale, w2, w2_scale, ep_size=,
ep_rank=, moe_tp_size=, moe_tp_rank=)`` slices the canonical unsharded
``[E, 2I, H/2]``/``[E, 2I, H/32]``/``[E, H, I/2]``/``[E, H, I/32]`` tensors
into one rank's canonical shard, which is then passed to
``prepare_cute_dsl_mxfp4_weights``. With ``Is = I / moe_tp_size`` the
tensor-parallel shard takes W1 up rows ``[r*Is, (r+1)*Is)`` and gate rows
``[I + r*Is, I + (r+1)*Is)``, W2 packed columns ``[r*Is/2, (r+1)*Is/2)`` and
W2 scale columns ``[r*Is/32, (r+1)*Is/32)``; the expert-parallel shard takes
experts ``[r*E/ep_size, (r+1)*E/ep_size)``. Slicing is deterministic and
depends only on these arguments. Expert-parallel shards are views of the bank
(``copy=True`` clones the shard only); tensor-parallel shards are shard-sized
copies because their rows and columns are not contiguous. No full-size
duplicate is created; release the canonical bank when no other rank on the
device needs it. Group-32 scale blocks are never split, so a shard's MXFP8
intermediate quantization equals the unsharded one block for block.

The kernels place no additional constraint on the shard beyond the existing
multiple-of-128 rule: GEMM1 N=768 and GEMM2 K=384 satisfy the gather kernel's
N-tile and 16-byte/128-element alignment checks and the finalize kernel's
``N % 128`` and alignment checks. For T=1..16 with 896 local experts the fused
routing kernel uses its 1024-thread variant; that is one additional compiled
routing callable relative to a 112-local-expert inventory, and no new GEMM
variant. Per-forward launch counts are unchanged in both modes.

Workspace and kernel selection
------------------------------

``get_workspace_size(T)`` needs no GPU allocation. Workspace is a contiguous,
256-byte-aligned CUDA uint8 buffer. Its regions hold worst-case padded route
indices, the FP8 GEMM1 result and scales, routing conversion, unit per-expert
GEMM scales, and (for T>1024) expert-count scratch. Output is separate.
The padded-row bound is ``tile_size * max_tiles``, where
``max_tiles = T*top_k`` if ``T*top_k <= local_experts``, otherwise
``(T*top_k + (tile_size-1)*local_experts) // tile_size``.

The runner accepts an offline tactic table mapping token-count upper bounds
to existing W4A8 tactic tuples. It chooses the smallest covering bucket using
host shape metadata; without a covering bucket the SiTU dense path uses a
B300-measured per-shard table keyed by token count
(``B300_SITU_DENSE_TACTIC_TABLE_WIDE`` for intermediate shards of at least
1024 columns, ``..._NARROW`` below). The base tactic is
``(128, ((128, 256), (1, 1)), ((128, 256), (1, 1)))`` (256-wide MMA N halves
the GEMM2 tile count against the conservative M128/N128 tactic); the tables
switch GEMM2 to a 192-wide tile with a two-CTA cluster where the finalize
reduction dominates (wide shard at T=16384, narrow shard from T=16384: 1-7%),
give the narrow shard a two-CTA-cluster 256-wide GEMM2 for
T in (4096, 7168] (hot routing -4%) and the two-CTA (``cta_group::2``)
M256 GEMM1 tile with the two-CTA 256-wide GEMM2 for T in (7168, 14336]
(same GPU, T=8192 balanced/empty/hot -9.1/-4.2/-0.4% against the M128
tactic of the previous bucket: GEMM1 910 -> 760 us). The M256 tile wins
where the rows of an expert pad to the same 256-row multiple under both
tiles, i.e. when ``ceil(rows / 128)`` is even: 146 rows per expert
(balanced routing at T=8192) pad to 256 either way and the two-CTA tile
streams each weight tile once, while 73 rows (T=4096) pad 3.5x and 293 rows
(T=16384) pad 1.75x against 1.31x, where M128 wins by 7-16%. The
two-CTA form had been excluded because compute-sanitizer synccheck
reported "Missing wait" records from the gather GEMM1's MMA warp; the
cause was the empty side of the pipeline that relays "A landed in both
CTAs" to the leader MMA warp, whose producer never waits on it (it is
throttled by the A pipeline, released by the same ``tcgen05.commit``), so
the leader's release arrived on barriers nobody waits on. That release
and its tail wait are removed; synccheck and memcheck report zero errors
on the two-CTA form (T=8192, both layouts). Which M tile wins follows the
routing, not the token count: at T=8192 M256 gains 17/13/5% on the
balanced/hot/empty routings of the expert-parallel rank but loses 34% on
the remote-dominated one (73 rows per expert), and at T=16384 the reverse
(-18% remote-dominated, +16/+6% balanced/hot). Above T=7168 the routing
kernel therefore chooses the tile on the device (``MXFP4_DENSE_DUAL_TILE``):
it pads every local expert to the 128- and to the 256-row tile, takes the
two-CTA M256 tile when its padded total is within 1.10 x of the M128 one
(the measured break-even lies between 1.0 and 1.137: at ratio 1.0 M256
wins 4-17%, at 1.137 and above M128 wins 2-14%, T=8192..32768, both
layouts), writes the 128-granularity tile list, which is always valid, plus
the 256-granularity list and one active tile count per variant, and both
grouped GEMMs are launched once per variant on the same permuted rows and
output; the variant the routing did not choose reads a zero count and
exits. The choice replays inside a captured graph. Its cost is the two
launches of the unchosen variant, 2.6-3.9 us each plus their launch gaps
(about 8 us per row, measured EP=8 T=8192..32768, interleaved A/B/A/B,
30 replays): below 0.5% on every MoE-TP shard row above T=7168 (>= 1.07
ms) but 1-4% of the wide rank's empty routings (86-160 us), which must
not run slower than the single-tile plan. The rule was therefore the
default on shards up to 512 columns (``MXFP4_DENSE_DUAL_TILE_MAX_SHARD``)
and opt-in on the wide rank, where it takes the T=8192 balanced/hot rows
down 17/14% and the T=16384 remote-dominated row down 16% (same-GPU
A/B/A); since round 15 the limit defaults to 3072 columns, so the wide
rank uses it too (its T=8192 / 16384 ``empty`` rows pay 5-8%, every row
stays ahead of the phase-2 baseline). Launching the alternate
variant as a programmatic dependent (``MXFP4_DUAL_ALT_PDL``) does not
recover the cost: the base kernels trigger their dependents at their end,
so the alternate grid launches into the base kernel's tail and its CUPTI
duration grows to 26-61 us when unchosen while the GPU span does not
improve. Skipping the unchosen launches on the device (graph conditional
nodes set by the routing kernel) is the open item that would make the rule
free on the wide rank. Other activations keep the conservative tactic. It
never inspects device routing on the host or autotunes during execution.

Kernel selection by token count is host-side and layout-aware. Both parallel
layouts run the swap-AB grouped GEMMs (weights as MMA-M, ``n_tile``-row
expert groups as MMA-N) up to ``swapab_max_tokens`` (1024 on an
expert-parallel rank, 2048 on a MoE-TP shard narrower than 1024 columns,
where T > 1024 takes the hybrid form described below) and the dense grouped
GEMMs above. The group width follows the expected rows
per local expert: with the balanced routing of the benchmarks every expert
receives T/56 rows, and as soon as that exceeds the group width every expert
needs a second group, which streams its weight tile a second time and costs
both swap GEMMs 40-50% on B300 (the per-tile cost is set by the tile count,
not by the bytes). A wider group also caps the weight re-streaming of a hot
expert (T=128 hot: 16 groups of 8 rows re-read 34 MB of weights each). The
defaults are 16 rows up to T=128 and 32 above for an expert-parallel rank;
8 up to T=128, 16 up to T=256, 32 up to T=1024 and 64 above for a 384-wide
MoE-TP shard, whose rank sees eight times the local rows. Both follow
``swapab_tile_policy``/``SWAPAB_TILE_POLICY`` and
``swapab_max_tokens``/``SWAPAB_MAX_TOKENS``. On an expert-parallel rank
(full 3072-wide shard) GEMM2 streams its K in 4-K-block (128-wide) stages up
to T=256 (a deeper pipeline for the single-group case, 6-10 us) and in one
384-wide stage above (``SWAP_GEMM2_SHORT_STAGE_MAX_TOKENS``); the 384-wide
MoE-TP shard always uses one stage. On an expert-parallel rank the dense
path is faster above T=1024.

From T=1025 to T=2048 the MoE-TP shard uses the hybrid form
(``SWAPAB_HYBRID=0`` disables it): ``moe_sort`` groups the permuted rows in 128-row tiles, a
single-block dispatch kernel (``csrc/moe_swapab_dispatch.cu``, about 3 us)
lists the occupied 64-row halves of those tiles, and the swap-AB GEMM1 runs
only those halves (its cp.async gather warps fetch 64 activation rows per
weight tile; two 64-row halves of a full tile re-stream the weight tile
once more, which the row operand's smaller stage keeps cheaper than a
128-row swap tile on B300) and writes its MXFP8 rows with the tcgen05
block-scaled scale-factor layout. GEMM2 is then the dense contiguous
grouped GEMM with the bulk-reduce finalize (``cp.reduce.async.bulk`` of
each token's 256-byte hidden segment into the zero-filled output), which
runs at the L2 reduction throughput its 128-row tiles reach (TP8 T=2048
balanced 314 us) instead of the swap GEMM2's permuted partial rows plus the
finalize gather (295 + 94 us). On B300 at TP8 T=2048 the hybrid form takes
0.795 ms (balanced) against 0.841 ms for the all-swap form and 0.844 ms
for the dense path, and 0.415 ms against 0.506 / 0.291 ms on the ``empty``
routing (16 active experts); a swap GEMM1 over 128-row tiles was measured
and rejected (369 us against 225 us for two 64-row halves), as was a
per-column ``red.global.add`` or bulk-reduce finalize in the 64-row swap
GEMM2 epilogue (1009 / 808 us against 389 us for the two-stage form).

The GEMM1 tiles of the hybrid form are mixed per sort group
(``SWAPAB_HYBRID_DENSE_MIN_ROWS``, default 64; 128 disables it): the
dispatch kernel lists the 128-row groups with more than 64 valid rows
separately, and the dense gather GEMM1 runs exactly those groups through
its scheduler (a compacted work list of group indices; expert, row limit,
activation rows and output rows follow the group) while the swap GEMM1
keeps the sub-tiles of the partially filled groups. Both write the same
MXFP8 rows and block-scaled scale layout, so GEMM2 is unchanged. A full
group streams its expert's weight tile once in the dense tile against once
per 64-row half in the swap form, and the dense tile pads nothing there.
Same-GPU candidate-only medians on B300 (TP8 rank 0): ``empty`` routing
T=2048 414 -> 295 us (GEMM1 249 -> 127 us dense + 4 us for the empty swap
list) and T=1536 326 -> 222 us; ``hot`` T=2048 850 -> 816 us and T=1536
816 -> 787 us (the experts above 64 rows move to dense tiles, the rest stay
swap); ``balanced`` (36 rows per expert, no full group) 828 -> 820 us and
792 -> 795 us, i.e. within the run-to-run band, the dense GEMM1 launch that
finds an empty list costing 3 us. Thresholds 64 and 96 measure the same;
32 sends the balanced groups to dense tiles and loses 1-2%. At
T=4096 the hybrid form loses to the dense path (its 64-row GEMM1 re-streams
every expert's weights for two to three groups: 0.80-0.83 against
trtllm-gen), so the shard returns to the dense grouped GEMMs there.

From T=16 up the swap GEMM2 of the narrow MoE-TP shard (K = 384) runs two
weight M-tiles per work item with 128-wide (4 K-block) stages
(``SWAPAB_TP_MGROUP2`` / ``SWAPAB_TP_MGROUP2_MIN_TOKENS``, defaults 2 and
16): the token stage is fetched once for both weight tiles and the short K
loop keeps two tiles in flight. Same-GPU candidate-only medians on B300
(TP8 rank 0) against one tile per item: every row from T=16 up is faster
(balanced/hot -0.7..-6.6 %, ``empty`` -2..-7.3 %: T=16 balanced 199 -> 186
us, T=512 empty 211 -> 199 us, T=1024 empty 388 -> 360 us). Below T=16 the
balanced and hot rows would gain 2-5 % as well, but the one-expert
``empty`` rows lose 0.2-0.8 us (56 weight tiles become 28 items, halving
the CTAs that stream them), so the shard keeps one tile there. The wide
expert-parallel shard (K = 3072, multi-stage GEMM2) loses 7-11 % on its
24-40 us decode and ``empty`` rows with the same grouping and keeps one
tile per item.

Below the hybrid cap the swap form can also run **mixed GEMM1 tiles**
(``SWAPAB_MIXED=1``, opt-in): the sort groups become 128 rows, the dispatch
kernel emits the wide / narrow lists as in the hybrid form plus the list of
every occupied sub-tile for the swap GEMM2 (which then reads the
block-scaled scale layout the dense tiles write), and a device-side rule
(``SWAPAB_MIXED_WIDE_PERMILLE``, e.g. 500) keeps every group narrow unless
the full groups hold that share of the valid rows, so only concentrated
routings pay for dense tiles. Measured on B300 (TP8 rank 0, same GPU, rule
500): ``empty`` T=128/256/512/1024 145 -> 94, 171 -> 103, 211 -> 152,
388 -> 247 us (0.60-0.72 x), while ``balanced``/``hot`` rows pay for the
128-row groups. On the fused-routing path (T <= 1024 on the shard) the
routing kernel emits the three lists itself (``SWAPAB_MIXED_FUSED_LISTS``, default 1;
the separate dispatch launch remains for ``moe_sort``-routed token
counts), which the per-kernel breakdown shows is not where the cost is:
at T=256 balanced the lists add 2.7 us to the routing kernel against the
4.4 us dispatch launch they replace, the dense GEMM1 that finds an empty
list costs 2.8 us, but the swap GEMM2 over the 128-row groups' sub-tiles
takes 220 us against 209 us with 16-row groups and the swap GEMM1 405
against 402 us, so the row reads 649 against 629 us (+3.1 %; T=128
balanced/hot +1.4 %, same-GPU A/B/A, the two default arms within 0.1 %).
The grouping itself, not a launch, is the price of the mixed form, so it
stays opt-in; the open item is a dense tile that works on the default
16-row group layout (a row-start work list and a per-row scale epilogue
in the gather GEMM1), which would leave the balanced and hot rows at the
default form's cost.

Every weight TMA load (swap-AB and dense) can carry an L2 eviction hint.
The dense GEMMs mark their B/SFB streams ``EVICT_FIRST`` (CUTLASS SM90 TMA
cache-hint encoding; ``MXFP4_DENSE_L2HINT`` = ``none``/``first``/``last``)
so the streamed expert weights stop evicting the activation rows that every
tile re-reads: TP8 T=2048 balanced 892 -> 843 us and T=4096 1049 -> 930 us
on B300. The swap-AB GEMMs apply the same hint per layout and token count
(``_swap_weight_l2_hint``, ``SWAPAB_L2HINT`` overrides): always for decode
(EP8 T=1 29.4 -> 26.1 us, T=16 194 -> 184 us; TP8 T=16 209 -> 194 us) and for
every prefill T on the MoE-TP shard (T=128..1024 -5..-14 us), but not for an
expert-parallel rank at T=17..255 (``SWAPAB_EP_L2HINT_SKIP_MAX_TOKENS``),
where a hot expert's 8-16 groups re-read its weights from L2 and the hint
costs +10 us (balanced +4 us); from T=256 the 32-row groups make it a
5-14 us win there too. All deltas are same-GPU paired medians.

The swap-AB kernel chain is launched with programmatic dependent launch
(``SWAPAB_PDL``, default 1) whatever the caller's ``enable_pdl``: the fused
routing kernel triggers its dependents at entry, and every swap GEMM (and
the hybrid form's dense GEMMs) executes ``griddepcontrol.wait`` after its
prologue (TMEM allocation, barrier initialisation, descriptor prefetch) and
before its first read of a routing output, so the launch latency and the
prologue of each GEMM overlap the previous kernel. The first launch of the
op stays a plain launch, so the caller-visible stream order is unchanged;
with ``enable_pdl=True`` SiTU decode now also takes this path instead of
the dense one. Measured per row on B300 (graph replay, CUPTI, same
process): GPU span -2.1..-2.6 us on every decode row of both layouts
(EP8 T=1 hot 25.1 -> 23.0 us, TP8 26.9 -> 24.6 us) with bitwise-identical
output; the summed kernel durations rise by 0.4-1.3 us because a dependent
kernel's clock runs while it waits, so overlapped rows are reported by GPU
span. Triggering the dependents right after the wait instead of at the end
of the epilogue gains nothing further (the dependent CTAs cannot become
resident before the working CTAs leave their SMs) as long as the
dependent waits in every warp. In the plain chain (routing -> GEMM1 ->
GEMM2, no split / hybrid / mixed launches; ``SWAPAB_DEP_PREFETCH``,
default 1) GEMM1 therefore triggers its dependent right after its own
wait -- the routing grid is complete by then -- and GEMM2 executes
``griddepcontrol.wait`` only in the warps that load its row operand,
GEMM1's output (the gather warps, plus the TMA warp when the row tile is
TMA-fed); its scheduler reads the routing tables and its TMA warp streams
the weight stages at once, so GEMM2's tile CTAs take the SMs GEMM1's
tile-less CTAs leave and hold their weights before GEMM1 ends. The
decode-class GEMM2 (short 4-block stages) takes 8-block stages when the
stage ring then covers the whole K (K = 3072: 12 stages of 256), so the
whole tile is resident. Measured on B300 (same GPU, paired, FP64
identical): EP=8 decode rows 1.04-1.05 x (22.4-23.0 -> 21.5-21.9 us),
MoE-TP decode 1.00-1.03 x, EP=8 ``empty`` T=128 1.03 x and T=256..1024
1.01 x; every other row 0.997-1.007 x. Every other input of GEMM2 (tables,
finalize metadata, the routing kernel's output clear) was written by
grids that completed before GEMM1 started, which is what makes the
partial wait sound; the two flags are exclusive with the entry trigger of
the split form's wide launches. On the same chain the finalize GEMM2
splits its K loop on the device (``SWAPAB_GEMM2_SPLIT_K``, default 2; 1
disables): its scheduler warp publishes a K range with every work item
and hands out ``(m_chunk, split)`` halves of the resident tile only while
the launch's valid work fits in half the SMs, otherwise the items map
onto the original raster; the finalize epilogue's ``red.global.add``
makes the partials additive, so no partial buffer or counter is needed.
The shard's K = 384 GEMM2 (one stage) and the deferred form keep one
split. Measured on B300 (same GPU, paired, FP64-checked): EP=8
single-expert decode rows 1.03-1.04 x (21.9-22.2 -> 21.1-21.6 us), EP=8
``empty`` T=128..1024 1.02-1.04 x, every other row 0.99-1.01 x.
On expert-parallel ranks the SiTU GEMM1 of the same chain launches as a
two-CTA cluster (``SWAPAB_GEMM1_CLUSTER_SPLIT``, default on; 0 disables)
when the tile is 16 tokens or narrower: the two CTAs of a pair split one
item's K range and the peer ships its FP32 partial over distributed
shared memory into the leader, which adds it before the activation
epilogue; launches whose work does not fit half the SMs run unsplit and
never touch the peer. Measured on B300 (same GPU, paired, FP64-checked):
EP=8 single-expert decode rows 1.08-1.09 x (20.9-21.7 -> 19.2-20.0 us),
EP=8 ``empty`` T=128 1.10 x, the shard unchanged.

Routing preprocessing (expert histogram, permutation, per-row scale, output
clear) runs as one fused CuTe kernel whenever ``T * top_k <= 8192``
(``FUSED_ROUTE_MAX_ROUTES``, T <= 512 for Kimi K3) or, on a rank with at
most 128 local experts (``MXFP4_FUSED_ROUTE_LARGE_MAX_LOCAL_EXPERTS``),
``T * top_k <= 16384`` (T <= 1024) -- also on the MoE-TP shard where the
split form needs the fused layout -- and as the shared ``moe_sort`` path
above. Up to one route per thread the fused kernel is a single
1024-thread block (plus a grid that clears the output when the swap
epilogue reduce-adds into it); above that it is a thread-block cluster of
``ceil(T x top_k / 1024)`` CTAs, at most eight
(``MXFP4_FUSED_ROUTE_CLUSTER``): each CTA converts, histograms
(shared-memory atomics) and scatters its own chunk of routes, the per-CTA
expert counts are exchanged through the ``out_expert_counts`` scratch
(unused on this path) at one cluster barrier (``barrier.cluster`` with
release / acquire semantics), every CTA derives the same group bases from
the totals, and CTA 0 alone writes the group tables, work lists and totals;
a route's row is its expert's base plus that expert's rows in the
lower-ranked chunks plus the route's rank in its own chunk. The CTAs after
the sort cluster clear the output with a grid-stride loop of 16-byte stores
(``MXFP4_FUSED_ROUTE_CLEAR_CTAS`` = 120 of them; clusters are scheduled as
units, so the one-word-per-thread grid of the single block cost 4-5 us in
cluster waves, and 4-byte stores from a 4-byte-aligned pointer type another
6 us). The kernel replaces the unpack/conversion kernels, ``moe_sort`` and
the memset (about 10.6 us at T=128) with roughly 3 us at decode; the
generic clear kernel uses 16-byte stores.
In the single-block form its cost grows with the route count: the
histogram issues eight routes per thread before ranking any of them (one
block's 32 warps are latency-bound otherwise), and above 8192 rows the
inverse permutation (row -> route) is scattered in shared memory (int16)
and copied out coalesced, because scattered 4-byte global stores cost one
32-byte L1 sector each: 16384 of them took 13.6 us of a 19.9 us kernel at
T=1024 on the shard, 8.7 us of 15.0 us staged (per-phase globaltimer
stamps, B300; below 8192 rows the direct stores are as fast, 1.2 against
1.5 us at 2048 rows, and the 8192-route variant keeps them). Shard,
balanced routing, T=128 / 512 / 1024: histogram 2.1 / 2.9 / 5.4 us, scan
1.8 / 1.8 / 0.5 us, scatter 1.2 / 6.3 / 8.7 us, whole kernel 5.4 / 11.3 /
15.0 us; EP8 rank (112 local experts): 3.6 / 4.9 / 8.3 us. On the EP8 rank
the larger bound wins every row (T=512 ``empty`` 38.0 -> 32.5 us, T=1024
39.6 -> 37.8 us, ``balanced``/``hot`` 0.993-0.999 x). Up to one route per
thread the histogram keeps its plain loop (the clamped duplicate loads cost
0.1 us at T=1); the decode rows of the EP8 rank read 0.2-0.3 us (1 %) above
the revision before the fused sort in a same-GPU pair (T=1: 24.75 -> 25.0,
22.65 -> 22.85, 22.9 -> 23.15 us for balanced / hot / remote_dominated), a
residual of the kernel's larger parameter block and int16 scratch that is
not attributed further; they stay 4-5 % under the phase-2 baseline.
The cluster form (B300, same GPU, paired, FP64-checked) takes 4.5-4.8 /
4.8-5.0 / 5.3-5.4 / 6.3-6.8 us at EP8 T=128 / 256 / 512 / 1024 against
4.6-4.9 / 4.9-5.4 / 7.0 / 12.8 us for the single block, and 6.1-6.6 /
6.5-6.8 / 6.8-7.5 / 8.4-9.2 us on the shard against 6.6-7.1 / 7.6-8.5 /
9.9-12.3 / 16.1-17.6 us. Its phases at EP8 T=1024 (globaltimer stamps):
histogram 1.8 us, count exchange 1.4, group bases 0.55, scatter 0.3, sort
end 4.6-4.9 us, output clear end 3.4-4.0 us; on the shard the group-table
phase over 896 experts takes 1.9 us and the sort ends at 6.6-7.4 us. Rows:
``empty`` EP8 T=512 / 1024 1.101 / 1.239 x, shard 1.034 / 1.049 x, every
other T=128..1024 row 0.999-1.015 x, decode rows 0.989-1.002 x (the
kernel's smaller int16 scratch saves 0.1 us).

The finalize (route-weight reduction) has two forms on the swap-AB path.
Up to T=16, and for shards wider than 512 at every T, GEMM2 reduce-adds
``alpha * route_weight * acc`` into the zero-filled output from its epilogue
(``red.global.add.bf16x2``); the long GEMM2 mainloops of a wide shard hide
those reductions. For a shard of at most 512 columns and T > 16 (two-stage
finalize) GEMM2 instead writes ``alpha * acc`` rows in permuted order into
the ``partial_rows`` workspace region and a fifth launch,
``mxfp4_finalize.plan_finalize_rows``, gathers each token's local slots
with FP32 accumulation and one BF16 rounding, overwriting every output row
(the route preprocess then skips the output clear). On the 384-wide TP8
shard the epilogue reductions cost about 380 us at T=1024 (8192 local rows
of 7168) against a 244 us GEMM2; the two-stage form replaces them with a
46 us gather. ``get_workspace_size`` includes the region whenever the
two-stage form applies; the hybrid form above T=1024 holds the dispatch
work lists instead and reduces into the output from the dense GEMM2; the
deferred form (``do_finalize=False``) uses the caller's row buffer
instead.
``mxfp4_moe_capability`` checks architecture, quantization, activation,
dimensions, top-k and the explicit parallel metadata before a plan is
constructed. Its result reports ``supported``, ``reason``, ``cuda_graph``
(true for every supported configuration in both parallel modes),
``layout``, the resolved ``Mxfp4MoERankLayout`` with the mode, local expert
interval and intermediate shard, and ``deferred_output`` (the
``do_finalize=False`` form, available for SiTU).

At H=7168, I=3072 and 112 local experts, one prepared weight bank occupies
3,930,587,136 bytes. With top-k=16 and the conservative tactic, workspace is
6,499,072 bytes at T=1, 45,885,440 bytes at T=16, and 253,748,224 bytes at
T=4096. A MoE-tensor-parallel rank (896 local experts, shard 384) holds a
prepared bank of the same size and needs 828,160 bytes at T=1, 13,120,000
at T=16, 48,907,264 at T=512 and 72,543,744 at T=4096: the GEMM1
intermediate region shrinks with the shard while the per-expert regions grow
with 896 local experts. These totals exclude caller activations/output and
temporary load-time preparation storage. Evaluation retains CuTe- and TRT-prepared banks, with
the CuTe revisions sharing prepared tensors; serving with the CuTe runner
needs only its own bank.

Under the default M128/N128 tactic, GEMM1 allocates 228,352 shared bytes
per CTA, six operand stages, two accumulator stages and 512 TMEM columns.
GEMM2 allocates 207,872 shared bytes, five operand stages, two accumulator
stages and 512 TMEM columns. The narrow A transfer reduces transferred
bytes, not these allocated capacities. Workspace layout and size match the
previous implementation. Other tactics have separate resource requirements.

.. list-table:: Measured workspace and separate output sizes
   :header-rows: 1

   * - T
     - Workspace bytes
     - BF16 output bytes
   * - 1
     - 6,499,072
     - 14,336
   * - 16
     - 45,885,440
     - 229,376
   * - 128
     - 51,591,168
     - 1,835,008
   * - 512
     - 71,154,176
     - 7,340,032
   * - 2048
     - 149,412,864
     - 29,360,128
   * - 4096
     - 253,748,224
     - 58,720,256

Evaluation status
-----------------

Validation uses NVIDIA B300 (SM103), CUDA 13.1, PyTorch 2.10.0a0 and CuTe DSL
4.8.0.dev0. Full geometry is H=7168, I=3072, E=896, top-k=16, 112 local
experts and offset 336. All inputs are synthetic; matching real-checkpoint
evaluation has not been performed.

The current implementation passed 96 numerical cases, 14 dispatch/cache
boundary cases and 188 established regression tests. Three additional public
regressions pass for both T16/T17 compilation orders and routing opt-in/fallback. These include 72 full-dimension
configurations: 69 shape/routing combinations plus three parameter cases.
Full dimensions cover every integer T1..17 and T128/256/512/1024/2048/4096
with balanced, empty-expert and hot routing. Tests cover poisoned workspace,
mutable graphs for every integer T1..16, optional/scalar/per-expert SiTU
parameters, T16/T17 cache order, PDL and alternate tactics. Independent
concurrent streams complete 480 plan replays across 30 cases. The 344 stage
observations include repeated eager/graph executions, not independent inputs.

Three CUDA API trace audits cover T1 packed, T16 separate BF16 routing and
T17 packed, with five forwards each. All 55 correlated kernel activities
execute on the calibrated caller stream. Audited forward ranges contain no
allocation/free, host synchronization, device-to-host transfer, synchronous
copy, module loading or compilation. Preparation and the final stream drain
are outside those ranges. The audit job took 153.894 seconds of physical
turnaround; its three workload drivers took 46.369, 11.159 and 10.436 seconds.

The default decode forward has three launches. The traced packed T17 eager
path has five; measured required prefill has five eager kernel activities
and six Graph activities, including the graph-lowered output clear. These
are measured per-forward counts, not a total of compiled variants. The
compiled inventory depends on routing mode, tactic and specialization scope;
no fixed total is asserted here. In the focused public cache-transition and
routing tests, the resident inventory is one gather, two finalize and two
routing callables (five total). This is the inventory for those exercised
small-geometry plans, not every possible configuration.

Targeted Ruff lint and formatting pass for the changed Python files. The
formatted runtime has the same AST as the measured implementation. The three
new public regressions took 11.583 seconds of driver execution (9.079 seconds
in pytest); the packaging job, including queue/setup and histogram rendering,
took 260.092 seconds.

Earlier baseline racecheck passed with zero hazards; its synccheck was
skipped after a hard 20-second timeout. Those results remain source-specific:
ordinary numerical/API checks do not establish a new sanitizer pass. The
current revision was re-run under compute-sanitizer on B300 through the
wrapper (finalized and ``do_finalize=False`` plans in one process; EP=8
rank 3 at T=16/128 hot and T=256 balanced, MoE TP=8 rank 0 at T=16/128 hot
and T=1024 balanced): ``synccheck`` and ``memcheck`` both report zero
errors (``--report-api-errors no``; the runtime's own ``cuGetProcAddress``
capability probes are the only reports otherwise).

Numerical semantics and limits
------------------------------

Both FP64 references use the same stored quantized operands. The ideal
reference evaluates FP64 GEMMs, SiTU and routing-weight accumulation without
intermediate quantization. The explicit reference rounds the FP64 activation
to FP32, applies the unchanged group-32 MXFP8 quantizer, then evaluates GEMM2
and routing in FP64. Reports include relative L2, cosine, absolute-error
percentiles/max, worst-token error and finite checks. All full-dimension
outputs are finite.

The following are independent worst values across the required full-size
qualification observations; columns need not identify the same case. Previous/current each have 216
observations; TRT has 210 because its adapter omits absent-linear-beta cases.
The initial and remaining numerical batches use fixture banks generated at
T128 and T4096, respectively, with seed 123. Empty outputs are included.

.. list-table:: Full-dimension numerical extrema
   :header-rows: 1

   * - Reference / implementation
     - Max relative L2
     - Min cosine
     - Max p95 abs
     - Max p99 abs
     - Max abs
   * - Ideal FP64 / Previous
     - 0.02742379
     - 0.99962416
     - 0.00377511
     - 0.00717171
     - 0.02299803
   * - Ideal FP64 / Current
     - 0.02742379
     - 0.99962416
     - 0.00377577
     - 0.00717597
     - 0.02299803
   * - Ideal FP64 / TRT
     - 0.02748178
     - 0.99962251
     - 0.00373778
     - 0.00706444
     - 0.02154345
   * - Explicit MXFP8 / Previous
     - 0.00459769
     - 0.99998943
     - 0.00055471
     - 0.00138613
     - 0.01067550
   * - Explicit MXFP8 / Current
     - 0.00462994
     - 0.99998928
     - 0.00055403
     - 0.00139324
     - 0.01143374
   * - Explicit MXFP8 / TRT
     - 0.00270023
     - 0.99999635
     - 0.00034163
     - 0.00065426
     - 0.00331082

Accuracy remains a tradeoff. At T16 balanced, explicit-reference relative L2
increases from 0.00242913 to 0.00276122. The largest full-size ideal-reference
increase over the previous implementation is 0.00001034. At T512, a graph
replay changed from empty-expert to hot routing has current/TRT explicit
relative L2 of 0.00462994/0.00241501. At T4096 hot routing, worst-token
explicit relative L2 rises from 0.00804290 to 0.01578682. The ideal-reference
aggregate alone does not resolve these differences.

The combined implementation matches its native-arithmetic-only control's
valid intermediate FP8/UE8M0 bytes in all 344 stage comparisons. This does
not imply bitwise output identity: BF16 contribution rounding and atomic
accumulation remain part of finalization. A separate T4 diagnostic on that
control, using a different T2048-generated fixture bank, reproduced one FP8
value changing from 1.125 to 1.0 with its scale unchanged. FP64 GEMM2/routing
of each actual intermediate, followed by BF16 rounding, exactly reproduced
both outputs. That result accounts for the downstream difference in that
fixture; it did not observe pre-quantization FP32 registers and is not a
fresh combined-implementation T4 result.

The four-backend evolution comparison requires current ideal-FP64 error no
greater than the previous CuTe implementation's error plus the reference's
BF16 representation floor. The public tests retain their existing gate:
current ideal-FP64 error no greater than TRT's error plus that floor. Neither
gate requires explicit-reference nonregression. The parallel-layout tests sum
the BF16 rank outputs in FP64 and allow one BF16 representation floor per
rounded partial beyond the unsharded comparison's error; they also check that
the FP64 partial references sum to the unsharded reference, which validates
the slicing independently of the kernels.

TRT-LLM Gen baseline comparisons use ``trtllm_fp4_block_scale_routed_moe``
with the same shard geometry (for MoE TP, ``intermediate_size=384`` with 896
local experts). Its SiTU argument names are inverted relative to this
document: ``gemm1_alpha`` receives the gate bound ``beta`` and ``gemm1_beta``
receives the up bound ``linear_beta`` (its reference is
``situ_activation_reference`` with linear ``x0`` and gate ``x1``). The historical
``atol=0.1, rtol=0.15``, greater-than-95% precedent remains diagnostic.
Final customer numerical acceptance is unresolved.

Comparative performance
-----------------------

Swap-AB path results (current revision)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The swap-AB grouped GEMMs (weights as the MMA-M operand, ``n_tile``-row
groups of routed rows as MMA-N; ``swapab_moe.py`` and
``blackwell/blockscaled_swapab_grouped_gemm.py``) serve T <= 1024 in both
layouts, with the two-stage finalize on the 384-wide MoE-TP shard for
T > 16 and the dense grouped GEMMs above T=1024. Method:
``benchmarks/bench_mxfp4_situ_moe.py --full --accuracy`` per layout, same
process, CUPTI activity tracing with an L2 flush before every sample,
alternating arm order per row, 10 warmup and 30 timed iterations, medians;
kernel sum is the sum of correlated kernel durations, e2e the synchronized
runner latency. Ratios are TRT-LLM Gen divided by candidate. The TRT-LLM
Gen arm runs once under ``flashinfer.autotuner.autotune`` before timing, as
a serving stack does: its tuned tactics are 25-40% faster than the default
tactic for T >= 512 (TP8 T=512: 710 us tuned vs 993 us untuned), while
decode and T <= 256 are unaffected; earlier revisions of these tables
compared against the untuned default (``--untuned-baseline`` reproduces
that). The autotuner's cache key does not include the routing, so the
tuned tactic is the one that won on the routing it was shown: every table
here tunes on the balanced routing of each token count (the benchmark
lists ``balanced`` first), the input a serving stack would tune with.
Tuning on the ``empty`` routing instead (T=128, MoE-TP shard) picks a
tactic that runs that row in 87 us (against 195 us for the
balanced-tuned tactic and 144 us for the candidate) but costs the same
shard's balanced row 5% (709 us against 676 us); no single tactic wins
both, and the per-row-oracle baseline is not one a caller can deploy. Both arms record their error against the FP64 oracles on every row
(relative L2 0.026-0.027 for both).

Note on the kernel-sum rows: from the dependent-side prefetch (round 12)
the swap GEMM2 of the plain chain starts right after GEMM1's own
dependency wait, so on the decode-class rows its kernel duration counts
8-11 us of waiting for GEMM1; the summed kernel durations of those rows
exceed the row's GPU span and are no longer a measure of work. Every
graph-span and end-to-end ratio is above one; the kernel-sum rows are
kept for continuity with the earlier revisions only.

.. list-table:: Paired ratios (TRT / candidate), all required rows
   :header-rows: 1

   * - Layout
     - Mode / metric
     - Rows
     - Geometric mean
     - Minimum
     - Rows <= 1.0
   * - EP=8 (rank 3, 112 local experts)
     - Graph, GPU span
     - 44
     - 1.373
     - 1.031 (T128 balanced)
     - 0
   * - EP=8
     - Graph, kernel sum (includes GEMM2's dependency wait, see note)
     - 44
     - 1.096
     - 0.836 (T2 remote-dominated)
     - 14 (swap-chain rows whose GEMM2 waits inside the kernel)
   * - EP=8
     - eager, kernel sum (same note)
     - 44
     - 1.283
     - 0.981 (T128 balanced)
     - 3 (T128 balanced / hot / remote-dominated; eager e2e min 1.085, 0 rows)
   * - MoE TP=8 (rank 0, shard 384)
     - Graph, GPU span
     - 33
     - 1.397
     - 1.074 (T2048 hot)
     - 0
   * - MoE TP=8
     - Graph, kernel sum (same note)
     - 33
     - 1.172
     - 0.856 (T1 empty)
     - 5 (decode rows of the swap chain)
   * - MoE TP=8
     - eager, kernel sum
     - 33
     - 1.320
     - 1.037 (T2048 hot)
     - 0

Per-``T`` Graph-mode rows, EP=8 (kernel-sum milliseconds for balanced
routing, kernel-sum ratios per routing distribution, synchronized
graph-replay e2e ratio for balanced routing):

.. table::
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  remote-dom. ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   1     0.0394            0.0321           0.814           0.883      0.883        0.864              1.215
   2     0.0482            0.0491           1.019           0.894      0.886        0.814              1.258
   4     0.0693            0.0752           1.086           0.872      0.860        1.012              1.272
   8     0.1130            0.1220           1.080           1.066      1.057        1.150              1.203
   16    0.1945            0.1975           1.015           0.935      1.085        1.067              1.106
   128   0.6200            0.5986           0.965           0.983      1.013        0.963              1.021
   256   0.6245            0.6224           0.997           0.996      0.883        0.994              1.040
   512   0.6253            0.6837           1.093           1.031      1.040        1.118              1.134
   1024  0.6331            0.7481           1.182           1.070      1.149        1.198              1.219
   2048  0.7759            1.2406           1.599           1.524      1.707        1.625              1.587
   4096  0.8061            1.2402           1.539           1.456      2.085        1.614              1.525
   ====  ================  ===============  ==============  =========  ===========  =================  ==================

Per-``T`` Graph-mode rows, MoE TP=8:

.. table::
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  ==================
   1     0.0389            0.0336           0.863           0.882      0.873        1.289
   2     0.0495            0.0524           1.057           1.057      0.884        1.295
   4     0.0711            0.0749           1.053           1.088      0.878        1.226
   8     0.1201            0.1310           1.091           1.106      1.021        1.217
   16    0.2005            0.2198           1.096           1.113      1.255        1.193
   128   0.6245            0.6816           1.091           1.095      2.553        1.101
   256   0.6443            0.6966           1.081           1.084      2.149        1.091
   512   0.6694            0.7204           1.076           1.086      2.755        1.087
   1024  0.7262            0.7851           1.081           1.070      2.107        1.093
   2048  0.8184            0.8816           1.077           1.031      1.347        1.092
   4096  0.9127            1.0984           1.203           1.183      1.235        1.200
   ====  ================  ===============  ==============  =========  ===========  ==================

Eager-mode ratios match the Graph-mode ones within 1% on every row, and
the synchronized graph-replay e2e ratio is above one on every row as well
(minimum 1.010 EP=8, 1.064 MoE TP=8). The thinnest
margin is the EP=8 T128 balanced row (0.8%; 0.9% eager), where both arms
stream the rank's 3.9 GB of expert weights at about 6.4 TB/s and the row
moves 2-3% between runs on the same node type; next is the MoE-TP T2048
hot row (6.8%) of the hybrid form. The MoE-TP T1024 empty row, the
previous minimum at 3.2%, sits at 10.7% with the grouped GEMM2.

The deferred form (``plan(..., do_finalize=False)`` against
``trtllm_fp4_block_scale_routed_moe(do_finalize=False)``, both arms without
the route-weight reduction) measured with ``--deferred`` on the same rows
(Graph mode, same method and tuned baseline, same node and revision as
the tables above; the grouped GEMM2 serves the deferred MoE-TP rows from
T=16 up as well): EP=8 geometric mean 1.094 over 44 rows, MoE TP=8 1.102
over 33 rows. Every T <= 1024 row is above one on the MoE-TP shard
(1.07-1.71) and on EP=8 except six rows that tie within 2% (T1 hot, empty
and remote-dominated at 0.98-0.99, about 27 us both arms; T128 balanced, hot
and remote-dominated at 0.994-0.996). The deferred form is always served by
the swap-AB path, so at T=2048/4096 its swap GEMMs lose to trtllm-gen's
dense GEMM2 (EP=8 T2048 hot 0.988, T4096 balanced 0.791, hot 0.669; MoE
TP=8 T2048 empty 0.84, T4096 balanced 0.706; unchanged at this revision,
whose deferred tables report GPU-span ratios).
Serving the deferred output from the dense path is the open follow-up for
those token counts.

.. table:: Deferred finalize, Graph mode, EP=8 (trtllm-gen do_finalize=False / candidate do_finalize=False)
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  remote-dom. ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   1     0.0245            0.0302           1.233           1.272      1.253        1.259              1.161
   2     0.0381            0.0475           1.246           1.268      1.267        1.230              1.201
   4     0.0575            0.0708           1.233           1.275      1.276        1.291              1.186
   8     0.1013            0.1210           1.195           1.584      1.599        1.360              1.183
   16    0.1813            0.1985           1.095           1.390      1.530        1.181              1.090
   128   0.5957            0.6077           1.020           1.012      1.358        1.020              1.021
   256   0.6095            0.6341           1.040           1.032      1.200        1.039              1.040
   512   0.6132            0.6856           1.118           1.052      1.330        1.139              1.117
   1024  0.6213            0.7412           1.193           1.077      1.279        1.206              1.189
   2048  1.0081            1.1966           1.187           0.982      1.500        1.889              1.185
   4096  1.4797            1.1632           0.786           0.653      1.301        1.175              0.789
   ====  ================  ===============  ==============  =========  ===========  =================  ==================

.. table:: Deferred finalize, Graph mode, MoE TP=8
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  ==================
   1     0.0241            0.0302           1.252           1.311      1.269        1.176
   2     0.0374            0.0483           1.291           1.285      1.298        1.221
   4     0.0573            0.0714           1.245           1.255      1.277        1.200
   8     0.1035            0.1273           1.230           1.308      1.581        1.207
   16    0.1844            0.2168           1.175           1.196      1.589        1.168
   128   0.6055            0.6673           1.102           1.111      1.327        1.100
   256   0.6154            0.6807           1.106           1.113      1.188        1.103
   512   0.6256            0.7020           1.122           1.127      1.822        1.122
   1024  0.6619            0.7410           1.120           1.112      1.137        1.116
   2048  0.7452            0.8021           1.076           1.047      0.827        1.073
   4096  1.3373            0.9367           0.700           0.677      0.624        0.705
   ====  ================  ===============  ==============  =========  ===========  ==================

Long prefill (optional rows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional T=8192/16384/32768 rows (finalized path, dense grouped GEMMs,
same paired method, Graph mode) are measured and reported at this revision
(GPU span): EP=8 geometric mean 1.395 over 12 rows with T8192 balanced
0.798 / hot 0.837 and T16384 remote-dominated 0.862 below one (T16384
balanced 1.000 / hot 1.060); MoE TP=8 geometric mean 1.052 over 9 rows
with T8192 balanced 0.974 / hot 0.968, T16384 balanced 0.984 / empty
0.998 and T32768 empty 0.965 below one (T16384 hot 1.035, T32768 balanced
1.122 / hot 1.227). These rows run
on the dense path, which this revision does not change; against the
previous revision they move by -2.7..+6.1 % (T16384 hot on the shard),
and every row that reads slower than 1 % against the phase-2 baseline was
re-measured as a same-GPU interleaved pair against that tree in its bench
configuration (T=2048 balanced 0.995 / hot 0.972, T=8192 balanced 0.903,
T=16384 balanced 0.982 / hot 0.985, T=8192 hot 1.001 in a six-arm pair of 60 replays, whose per-arm medians
span 1690-1873 us on both trees, so the +1.5 % the bench read for that
row is inside its run-to-run band). The dense path picks its tactic from the B300 tables while the
trtllm-gen arm autotunes per token count; the shard picks its M tile on
the device from the padded row counts, the expert-parallel rank keeps the
single-tile default until the unchosen launches can be skipped.

Roofline against measured B300 floors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tables above compare against TRT-LLM Gen. This subsection compares every
required Graph-mode row against a per-row hardware floor measured on the
same B300 node, inside the same allocation and with the same CUPTI method
(cold L2, graph replay, medians). No specification numbers enter the floor.

**Measured floors.** Four microbenchmarks fix the constants:

* HBM: a 2 GiB FP32 elementwise add (2 GiB read, 2 GiB written) sustains
  **6.92 TB/s**; a pure 2 GiB read (FP32 sum) 6.25 TB/s, a 2 GiB fill
  3.97 TB/s, a device-to-device ``memcpy`` 2.77 TB/s. The floor uses the
  highest figure, 6.92 TB/s.
* tcgen05 block-scaled MMA: the CUTLASS SM100 grouped MXFP8 x MXFP4 GEMM on
  the GEMM1 shape of the wide shard (M=32768, N=6144, K=7168; two-CTA MMA,
  256-wide N tile) reaches **3618 TFLOPS** (3517 on a 32768 x 8192 x 8192
  square). The floor uses 3618 TFLOPS for every row. The same kernel on the
  shard-specific shapes is slower by construction of the shape, which the
  structural notes below rely on: GEMM2 of the wide shard (K=3072) 3516,
  GEMM1 of the 384-wide MoE-TP shard (N=768) 2878, GEMM2 of that shard
  (K=384) **1427** TFLOPS.
* Launch: 32 back-to-back 1-block kernels replayed from a graph cost
  1.29 us each plus a 0.34 us gap. The candidate's shortest path (fused
  routing, GEMM1, GEMM2) is three kernels, so the fixed-cost floor is
  **4.9 us**.
* Streaming ramp (cold L2, read + write, graph): 2.2 MB 2.75 us, 4.4 MB
  3.5 us, 8.8 MB 4.9 us, 17.6 MB 7.6 us, 35 MB 12.9 us, 70 MB 23.3 us,
  140 MB 43.8 us, 281 MB 84.5 us, 562 MB 165 us, 1.12 GB 327 us
  (a 1 KB kernel: 2.06 us). Below ~70 MB a transfer runs far under
  6.92 TB/s: the curve fits ~2.4 us fixed plus ~5.2 TB/s incremental.
  Restricting the streaming kernel to 48 CTAs (the swap-AB GEMM's grid at
  one expert) costs little: 13.2 MB takes 5.19 us at 48 CTAs and 4.91 us at
  148; 22 MB 7.02 us and 6.59 us.

**Per-row floor.** Bytes are the weights of every local expert with at least
one local route (E2M1 plus UE8M0 scales, ``3 * H * I_shard * (1/2 + 1/32)``
bytes per expert), the activations read once, the GEMM1 output written and
read once, and the BF16 output written once. FLOPs are
``local_assignments * 6 * H * I_shard``. The floor of a row is the largest
of bytes / 6.92 TB/s, FLOPs / 3618 TFLOPS and 4.9 us. Rows are accepted at
or below 1.10 x floor; the rest carry a structural note or stay open.

From this revision the candidate and TRT-LLM Gen times in the roofline
tables are the GPU span of a graph replay (first kernel start to last
kernel end): the swap-AB chain is launched with programmatic dependent
launch, so summed kernel durations count the overlapped wait inside the
dependent kernels and would overstate the candidate by 0.4-1.3 us per
row. Span is never smaller than the kernel sum for a plain launch chain,
so the earlier rows are comparable. From the dependent-side prefetch
(round 12) the swap GEMM2 of the plain chain starts right after GEMM1's
own dependency wait and holds its weights while GEMM1 runs, so its kernel
duration includes 8-11 us of waiting on the decode rows: the summed
kernel durations of those rows are no longer a measure of work (their
ratio to TRT-LLM Gen drops below one on the decode rows while the span
and the synchronized replay time improve), and only span and end-to-end
figures are compared from here on.

.. list-table:: Roofline summary, Graph mode (candidate at the two-CTA-GEMM1
   revision, one B300 node, same-node floors)
   :header-rows: 1

   * - Layout
     - Rows
     - Within 1.10 x floor
     - Geomean candidate / floor
     - Worst row
   * - EP=8 rank 3
     - 56
     - 10 (T=128 balanced/hot/remote-dominated; T=256 balanced/hot/remote-dominated; T=512 balanced/remote-dominated; T=1024 balanced/remote-dominated)
     - 1.95
     - 4.68 (T=256 empty)
   * - MoE TP=8 rank 0
     - 42
     - 2 (T=128 balanced/hot)
     - 1.83
     - 4.28 (T=128 empty)

Expert-parallel rank 3 (112 local experts, ``I_shard = 3072``):

   =====  ===========  =======  ================  =======  ==========  ======  =========
   T      routing      experts  floor µs (bound)  cand µs  cand/floor  TRT µs  TRT/floor
   =====  ===========  =======  ================  =======  ==========  ======  =========
   1      balanced     2        10 (mem)          25       2.46        33      3.29
   1      empty        1        5 (mem)           20       3.90        28      5.56
   1      hot          1        5 (mem)           20       3.91        28      5.55
   1      remote-dom.  1        5 (mem)           20       3.89        28      5.44
   2      balanced     4        20 (mem)          38       1.88        50      2.48
   2      empty        1        5 (mem)           20       3.88        29      5.61
   2      hot          1        5 (mem)           20       3.84        28      5.55
   2      remote-dom.  2        10 (mem)          24       2.41        33      3.21
   4      balanced     8        41 (mem)          57       1.41        76      1.87
   4      empty        1        5 (mem)           19       3.83        27      5.35
   4      hot          1        5 (mem)           19       3.82        27      5.38
   4      remote-dom.  4        20 (mem)          38       1.88        50      2.48
   8      balanced     16       81 (mem)          100      1.23        123     1.52
   8      empty        1        5 (mem)           20       3.85        34      6.62
   8      hot          1        5 (mem)           20       3.88        34      6.67
   8      remote-dom.  8        41 (mem)          57       1.40        80      1.96
   16     balanced     32       162 (mem)         179      1.10        198     1.22
   16     empty        1        5 (mem)           20       3.90        35      6.85
   16     hot          1        5 (mem)           23       4.56        36      6.97
   16     remote-dom.  16       81 (mem)          100      1.23        122     1.50
   128    balanced     112      569 (mem)         589      1.03        600     1.05
   128    empty        1        5 (mem)           24       4.30        39      7.04
   128    hot          112      569 (mem)         604      1.06        624     1.10
   128    remote-dom.  112      569 (mem)         585      1.03        596     1.05
   256    balanced     112      570 (mem)         601      1.05        624     1.09
   256    empty        1        6 (mem)           28       4.68        41      7.03
   256    hot          112      570 (mem)         626      1.10        649     1.14
   256    remote-dom.  112      569 (mem)         598      1.05        618     1.09
   512    balanced     112      571 (mem)         604      1.06        685     1.20
   512    empty        1        7 (mem)           28       4.23        49      7.27
   512    hot          112      571 (mem)         667      1.17        714     1.25
   512    remote-dom.  112      570 (mem)         601      1.05        695     1.22
   1024   balanced     112      574 (mem)         613      1.07        749     1.31
   1024   empty        1        8 (mem)           30       3.58        56      6.68
   1024   hot          112      574 (mem)         738      1.29        818     1.42
   1024   remote-dom.  112      573 (mem)         607      1.06        751     1.31
   2048   balanced     112      579 (mem)         777      1.34        1242    2.15
   2048   empty        1        12 (mem)          50       4.34        85      7.34
   2048   hot          112      580 (mem)         847      1.46        1290    2.22
   2048   remote-dom.  112      577 (mem)         772      1.34        1253    2.17
   4096   balanced     112      589 (mem)         808      1.37        1242    2.11
   4096   empty        1        18 (mem)          55       3.07        114     6.33
   4096   hot          112      592 (mem)         961      1.62        1398    2.36
   4096   remote-dom.  112      585 (mem)         787      1.34        1269    2.17
   8192   balanced     112      609 (mem)         1207     1.98        1151    1.89
   8192   empty        1        31 (mem)          70       2.27        173     5.57
   8192   hot          112      830 (mma)         1504     1.81        1528    1.84
   8192   remote-dom.  112      602 (mem)         830      1.38        1279    2.13
   16384  balanced     112      1162 (mma)        2257     1.94        2255    1.94
   16384  empty        1        57 (mem)          90       1.58        293     5.16
   16384  hot          112      1664 (mma)        3097     1.86        3157    1.90
   16384  remote-dom.  112      635 (mem)         1220     1.92        1281    2.02
   32768  balanced     112      2324 (mma)        4082     1.76        4623    1.99
   32768  empty        1        109 (mem)         134      1.23        537     4.94
   32768  hot          112      3323 (mma)        5603     1.69        6272    1.89
   32768  remote-dom.  112      1162 (mma)        2268     1.95        3135    2.70
   =====  ===========  =======  ================  =======  ==========  ======  =========

MoE tensor-parallel rank 0 (896 experts, ``I_shard = 384``):

   =====  ========  =======  ================  =======  ==========  ======  =========
   T      routing   experts  floor µs (bound)  cand µs  cand/floor  TRT µs  TRT/floor
   =====  ========  =======  ================  =======  ==========  ======  =========
   1      balanced  16       10 (mem)          25       2.42        35      3.42
   1      empty     16       10 (mem)          25       2.42        35      3.42
   1      hot       16       10 (mem)          25       2.42        35      3.45
   2      balanced  32       20 (mem)          39       1.94        53      2.62
   2      empty     16       10 (mem)          25       2.45        36      3.52
   2      hot       31       20 (mem)          38       1.96        53      2.68
   4      balanced  64       41 (mem)          60       1.47        76      1.87
   4      empty     16       10 (mem)          25       2.46        35      3.47
   4      hot       61       39 (mem)          58       1.49        75      1.95
   8      balanced  128      81 (mem)          107      1.32        132     1.62
   8      empty     16       10 (mem)          25       2.46        41      4.03
   8      hot       121      77 (mem)          98       1.28        125     1.63
   16     balanced  256      162 (mem)         184      1.13        221     1.36
   16     empty     16       10 (mem)          34       3.36        56      5.49
   16     hot       241      153 (mem)         171      1.12        210     1.37
   128    balanced  896      569 (mem)         618      1.09        682     1.20
   128    empty     16       11 (mem)          46       4.28        184     17.01
   128    hot       896      569 (mem)         620      1.09        687     1.21
   256    balanced  896      570 (mem)         637      1.12        698     1.22
   256    empty     16       18 (mma)          68       3.74        193     10.61
   256    hot       896      570 (mem)         637      1.12        701     1.23
   512    balanced  896      571 (mem)         661      1.16        721     1.26
   512    empty     16       36 (mma)          96       2.63        321     8.85
   512    hot       896      571 (mem)         661      1.16        728     1.27
   1024   balanced  896      574 (mem)         717      1.25        786     1.37
   1024   empty     16       73 (mma)          160      2.20        402     5.53
   1024   hot       896      574 (mem)         726      1.27        788     1.37
   2048   balanced  896      579 (mem)         807      1.39        883     1.53
   2048   empty     16       145 (mma)         293      2.02        429     2.95
   2048   hot       896      579 (mem)         811      1.40        873     1.51
   4096   balanced  896      589 (mem)         914      1.55        1100    1.87
   4096   empty     16       291 (mma)         529      1.82        653     2.25
   4096   hot       896      589 (mem)         918      1.56        1087    1.85
   8192   balanced  896      609 (mem)         1677     2.75        1629    2.67
   8192   empty     16       581 (mma)         1016     1.75        1210    2.08
   8192   hot       896      609 (mem)         1670     2.74        1623    2.66
   16384  balanced  896      1162 (mma)        3087     2.66        3248    2.80
   16384  empty     16       1162 (mma)        2120     1.82        2516    2.16
   16384  hot       896      1162 (mma)        2946     2.53        3117    2.68
   32768  balanced  896      2324 (mma)        5500     2.37        6317    2.72
   32768  empty     16       2324 (mma)        4697     2.02        5154    2.22
   32768  hot       896      2324 (mma)        5237     2.25        6318    2.72
   =====  ========  =======  ================  =======  ==========  ======  =========

**Reachable floor per row.** The floor above charges each row the
bytes it must move at the HBM figure and its FLOPs at the MMA peak. A
second, higher figure follows from the constants below, all measured on
the same node, and every row of the tables below is either at or under
1.10 x of it or named as open:

* the streaming ramp above (2.4 us plus bytes / 5.2 TB/s up to ~70 MB,
  the HBM figure beyond), applied to the bytes a kernel form actually
  moves: the dense form pads each expert's rows to its 128- or 256-row
  tile; the swap-AB form streams every touched expert once from HBM and
  re-reads it from L2 for each further row group (8.9 TB/s: the 16-row
  swap GEMM1 over 16 experts x 16 groups moved 750 MB in 84 us);
* the shape-specific MMA peaks above (GEMM1 of the wide rank 3618, GEMM2
  of the wide rank 3516, GEMM1 of the shard 2878, GEMM2 of the shard 1427
  TFLOPS), applied to the padded FLOPs of the dense form;
* the fused-finalize reduction bound (routed BF16 rows through L2 at
  3.4 TB/s);
* the per-SM MMA issue rate: a ``kind::f8f6f4`` MMA of 128 x N x 32
  occupies the single MMA warp for about 64 issue cycles at N <= 92 and
  runs at the measured peak at N = 256 (``%globaltimer`` stamps: 0.25 us
  per K=256 stage at N <= 32), so a kernel with fewer weight tiles than
  SMs -- one expert holds 48 GEMM1 tiles and 56 GEMM2 tiles on the wide
  rank -- cannot finish before ``prologue 2.2 us + tiles x (K/32 x
  cycles(N) / 2.0 GHz + epilogue) / min(tiles, 148)`` (epilogue 1.0 us
  for GEMM1's activation and quantisation, 0.12 us for GEMM2's 16-row
  partial store, both from stamps).
* the per-SM load ring, re-measured in round 15 with launches of a known
  item count (``probe_tiles.py``: synthetic routings giving ``e`` local
  experts exactly 128 or 32 rows each, per-kernel CUPTI medians, B300
  pool0-0048). The kernels deal their items round-robin to the 148
  persistent CTAs, and an item costs at least the largest of: its operand
  stages through one SM's async-copy path at 120 GB/s (a lone dense
  128 x 256 gather tile -- 56 stages of 50688 B -- takes 23.7 us whether
  the launch has 24 or 144 tiles; a lone 128 x 128 tile 15.8 us), its
  share of the aggregate fill among the CTAs active in its wave (130 GB/s
  per SM with every SM active, 19 TB/s: 27.9 us per 148 dense tiles,
  11.1 us per 148 32-row swap groups), a per-stage latency of 0.16 us for
  the swap kernels' 9-deep ring, and -- for the finalize tiles -- the
  reduce-add of the tile's valid rows at 3.4 TB/s, which does *not*
  overlap the next tile's fill (EP=8 GEMM2 9.3 us per wave = 7.3 fill +
  2.0 reduce; the shard's K=384 tile 1.57 us = 0.86 + 0.82). A CTA's
  first item also pays the form's chain latency measured on single-group
  launches after the routing dependency: 1.0 us (dense GEMM1 epilogue),
  1.6 us (dense GEMM2), 6.0 / 11.0 us for the 16- / 32-row swap GEMM1
  group (+0.5 us for the cluster split-K exchange, +0.18 us per row
  beyond 8 in a 16-row group) and 3.2 / 4.5 us for the swap GEMM2's
  dependent tail. The round-8 ring term let a few tiles share the whole
  chip's fill rate (4 us for the 24-tile launch that takes 26.9 us); the
  ``empty`` rows' distance to that floor was this modelling gap, not a
  kernel gap;
* the route bookkeeping before the first GEMM, measured kernel by kernel
  from T=128 to 32768 (``probe_decode_tl.py``, both layouts, every
  routing): the fused routing kernel of the swap form scans the routes
  in 4.2 us + 2.2 us per 1000 tokens (2.6 us on the T <= 16
  specialisation); the dense form runs the route conversion (1.7-2.9 us),
  the weight copy (3.5-4.3 us) and the sort -- 2.4 us + 0.27 us per 1000
  tokens on the cooperative histogram + offsets path -- plus five launch
  gaps of 0.34 us; on the shard above T=16 the swap form adds the
  two-stage finalize (GEMM2 writes one expanded row per route, a finalize
  kernel streams T x 17 x H x 2 bytes: 7.4 / 12 / 26.5 / 52 us at T=128 /
  256 / 512 / 1024). None of these depend on the routing, so an
  ``empty`` row pays the same bookkeeping as a balanced one;
* the dense form's zero-fill of the T x H BF16 output before the
  reduce-add finalize, T x H x 2 bytes at 6.3 TB/s standalone (5.6 us at
  T=2048, 74.7 us at T=32768). Overlapped with the routing kernels and
  GEMM1 (round 15) it shares HBM with them, so only its excess over
  GEMM1 counts.

The MMA term uses the node's best measured tcgen05 rate (3726 TFLOPS)
rather than the plain-GEMM figures of the floors probe at each GEMM's
shape: the grouped GEMM1 reached 3.33 PFLOPS on the shard at T=8192
(2.53 TFLOP of padded 256-row tiles in 759 us), above the 3012 TFLOPS
the probe measured for a plain M=32768 x N=768 x K=7168 GEMM, so the
shape figures are not bounds. GEMM2's bytes count the routed rows' A
operand and the output once written and once read (the reduce-adds of
a token's routes hit L2); the zero-fill is charged separately.

The reachable floor of a row is the smallest, over the swap-AB form at the
planner's row tile and the dense form at the 128- and 256-row tiles (128 x
128 and 128 x 256 GEMM tiles), of the form's route bookkeeping plus the
two GEMMs, each taking the largest of its terms (bytes at the HBM figure,
padded FLOPs at the MMA peak, the finalize reduction, the per-SM issue
rate and the ring term with its chain latency), plus the zero-fill's
excess over GEMM1 on the dense form and the two-stage finalize on the
shard's swap form above T=16. With the round-16 revision 38 of the 56
expert-parallel rows and 38 of the 42 MoE-TP rows sit within 1.10 x of
their reachable floor (geometric mean candidate / reachable 1.10 and
1.03); the binding term of nearly every row is the ring, i.e. the
kernels are bound by the operand bytes each SM keeps in flight and, on
launches with fewer items than SMs, by one item's chain, not by HBM
bandwidth or MMA rate. It is a lower bound of what any schedule built
from these kernels reaches on this node, not a promise that a schedule
exists; the tables mark rows at or under 1.10 x of it.

Reachable floor, expert-parallel rank 3 (same rows and candidate as above):

   =====  ===========  ========  ===================  =======  ==========  ==============  ======
   T      routing      floor µs  reachable µs (form)  cand µs  cand/floor  cand/reachable  TRT µs
   =====  ===========  ========  ===================  =======  ==========  ==============  ======
   1      balanced     10        24 (swap)            25       2.46        1.04            33
   1      empty        5         22 (swap)            20       3.90        0.92            28
   1      hot          5         22 (swap)            20       3.91        0.92            28
   1      remote-dom.  5         22 (swap)            20       3.89        0.92            28
   2      balanced     20        40 (dense128)        38       1.88        0.94            50
   2      empty        5         22 (swap)            20       3.88        0.91            29
   2      hot          5         22 (swap)            20       3.84        0.91            28
   2      remote-dom.  10        24 (swap)            24       2.41        1.02            33
   4      balanced     41        62 (swap)            57       1.41        0.92            76
   4      empty        5         22 (swap)            19       3.83        0.90            27
   4      hot          5         22 (swap)            19       3.82        0.90            27
   4      remote-dom.  20        40 (dense128)        38       1.88        0.94            50
   8      balanced     81        109 (swap)           100      1.23        0.92            123
   8      empty        5         22 (swap)            20       3.85        0.91            34
   8      hot          5         22 (swap)            20       3.88        0.92            34
   8      remote-dom.  41        62 (swap)            57       1.40        0.92            80
   16     balanced     162       193 (swap)           179      1.10        0.93            198
   16     empty        5         22 (swap)            20       3.90        0.93            35
   16     hot          5         22 (swap)            23       4.56        1.09            36
   16     remote-dom.  81        109 (swap)           100      1.23        0.92            122
   128    balanced     569       621 (swap)           589      1.03        0.95            600
   128    empty        5         22 (swap)            24       4.30        1.09            39
   128    hot          569       657 (swap)           604      1.06        0.92            624
   128    remote-dom.  569       621 (swap)           585      1.03        0.94            596
   256    balanced     570       659 (swap)           601      1.05        0.91            624
   256    empty        6         22 (swap)            28       4.68        1.28            41
   256    hot          570       697 (swap)           626      1.10        0.90            649
   256    remote-dom.  569       659 (swap)           598      1.05        0.91            618
   512    balanced     571       659 (swap)           604      1.06        0.92            685
   512    empty        7         22 (swap)            28       4.23        1.27            49
   512    hot          571       726 (dense128)       667      1.17        0.92            714
   512    remote-dom.  570       659 (swap)           601      1.05        0.91            695
   1024   balanced     574       659 (swap)           613      1.07        0.93            749
   1024   empty        8         24 (swap)            30       3.58        1.26            56
   1024   hot          574       762 (dense128)       738      1.29        0.97            818
   1024   remote-dom.  573       659 (swap)           607      1.06        0.92            751
   2048   balanced     579       726 (dense128)       777      1.34        1.07            1242
   2048   empty        12        26 (swap)            50       4.34        1.89            85
   2048   hot          580       808 (dense128)       847      1.46        1.05            1290
   2048   remote-dom.  577       659 (swap)           772      1.34        1.17            1253
   4096   balanced     589       726 (dense128)       808      1.37        1.11            1242
   4096   empty        18        35 (dense128n128)    55       3.07        1.59            114
   4096   hot          592       916 (dense128)       961      1.62        1.05            1398
   4096   remote-dom.  585       726 (dense128)       787      1.34        1.08            1269
   8192   balanced     609       1406 (dense128)      1207     1.98        0.86            1151
   8192   empty        31        50 (dense128n128)    70       2.27        1.40            173
   8192   hot          830       1787 (dense128)      1504     1.81        0.84            1528
   8192   remote-dom.  602       733 (dense128)       830      1.38        1.13            1279
   16384  balanced     1162      2086 (dense128)      2257     1.94        1.08            2255
   16384  empty        57        70 (dense128)        90       1.58        1.27            293
   16384  hot          1664      2847 (dense128)      3097     1.86        1.09            3157
   16384  remote-dom.  635       1406 (dense128)      1220     1.92        0.87            1281
   32768  balanced     2324      3445 (dense128)      4082     1.76        1.18            4623
   32768  empty        109       110 (dense128)       134      1.23        1.22            537
   32768  hot          3323      4995 (dense128)      5603     1.69        1.12            6272
   32768  remote-dom.  1162      2086 (dense128)      2268     1.95        1.09            3135
   =====  ===========  ========  ===================  =======  ==========  ==============  ======

Reachable floor, MoE tensor-parallel rank 0:

   =====  ========  ========  ===================  =======  ==========  ==============  ======
   T      routing   floor µs  reachable µs (form)  cand µs  cand/floor  cand/reachable  TRT µs
   =====  ========  ========  ===================  =======  ==========  ==============  ======
   1      balanced  10        24 (swap)            25       2.42        1.01            35
   1      empty     10        24 (swap)            25       2.42        1.01            35
   1      hot       10        24 (swap)            25       2.42        1.01            35
   2      balanced  20        40 (swap)            39       1.94        0.98            53
   2      empty     10        24 (swap)            25       2.45        1.02            36
   2      hot       20        40 (swap)            38       1.96        0.96            53
   4      balanced  41        59 (swap)            60       1.47        1.01            76
   4      empty     10        24 (swap)            25       2.46        1.03            35
   4      hot       39        58 (swap)            58       1.49        0.99            75
   8      balanced  81        107 (swap)           107      1.32        1.00            132
   8      empty     10        24 (swap)            25       2.46        1.03            41
   8      hot       77        97 (swap)            98       1.28        1.01            125
   16     balanced  162       193 (swap)           184      1.13        0.95            221
   16     empty     10        35 (dense128n128)    34       3.36        0.97            56
   16     hot       153       181 (swap)           171      1.12        0.94            210
   128    balanced  569       635 (swap)           618      1.09        0.97            682
   128    empty     11        35 (dense128n128)    46       4.28        1.31            184
   128    hot       569       638 (swap)           620      1.09        0.97            687
   256    balanced  570       654 (swap)           637      1.12        0.97            698
   256    empty     18        46 (dense128)        68       3.74        1.47            193
   256    hot       570       657 (swap)           637      1.12        0.97            701
   512    balanced  571       692 (swap)           661      1.16        0.95            721
   512    empty     36        96 (dense128)        96       2.63        0.99            321
   512    hot       571       696 (swap)           661      1.16        0.95            728
   1024   balanced  574       692 (swap)           717      1.25        1.04            786
   1024   empty     73        157 (dense128)       160      2.20        1.02            402
   1024   hot       574       712 (swap)           726      1.27        1.02            788
   2048   balanced  579       770 (swap)           807      1.39        1.05            883
   2048   empty     145       307 (dense128)       293      2.02        0.96            429
   2048   hot       579       791 (swap)           811      1.40        1.03            873
   4096   balanced  589       959 (dense128)       914      1.55        0.95            1100
   4096   empty     291       579 (dense128)       529      1.82        0.91            653
   4096   hot       589       974 (dense128)       918      1.56        0.94            1087
   8192   balanced  609       1883 (dense128)      1677     2.75        0.89            1629
   8192   empty     581       1124 (dense128)      1016     1.75        0.90            1210
   8192   hot       609       1940 (dense128)      1670     2.74        0.86            1623
   16384  balanced  1162      2806 (dense128)      3087     2.66        1.10            3248
   16384  empty     1162      2239 (dense128)      2120     1.82        0.95            2516
   16384  hot       1162      2948 (dense128)      2946     2.53        1.00            3117
   32768  balanced  2324      4658 (dense128)      5500     2.37        1.18            6317
   32768  empty     2324      4470 (dense128)      4697     2.02        1.05            5154
   32768  hot       2324      4912 (dense128)      5237     2.25        1.07            6318
   =====  ========  ========  ===================  =======  ==========  ==============  ======

**Where the gaps come from (measured on the same node).**

*Decode, T <= 16.* The floor is the touched weight bytes at 6.92 TB/s
(5 us for one expert on the wide shard, 10 us for 16 shard experts). At
this revision a T=1 hot row costs 22.5 us of GPU span: fused
routing 2.6 us, swap-AB GEMM1 (22 MB of weights) and GEMM2 (13 MB)
launched as programmatic dependents of their predecessor. Per-CTA
``%globaltimer`` stamps inside the two GEMMs (148 CTAs, one B300 node,
graph replay) locate the time. All CTAs start within 0.2 us of each
other, so the CTA launch ramp is not a term. A CTA without a tile leaves
after 1.2-1.5 us (TMEM allocation, barrier initialisation, descriptor
prefetch). A CTA with a tile spends 0.5 us in that prologue, 0.65 us
issuing its first weight loads, 1.05 us waiting for the first K stage to
land, then 7.1 us in the K loop (28 stages of K=256 at 0.25 us each) and
1.0 us in the epilogue: 10.7 us in total for GEMM1 (6.3 us for GEMM2: 24
stages at 0.14 us). The kernel durations were 13.4-13.9 and 8.3-8.8 us,
so ~3 us of each kernel lay outside any CTA (launch latency and drain).
The programmatic dependent launch removes 2.1-2.6 us of that per row
(EP8 T=1 hot 25.1 -> 23.0 us GPU span, TP8 26.9 -> 24.6; every decode
row of both layouts, same-process A/B, bitwise-identical output); an
earlier trigger (right after the dependency wait instead of at the end of
the epilogue) gains nothing further, because the dependent GEMM's CTAs
cannot become resident before the working CTAs of the primary leave their
SMs.

The K loop itself runs at the MMA warp's issue rate, not at the memory
system's: removing every token-row load, or the TMEM load and store of
the epilogue, or raising the stages in flight from 5 to 10 (200 KB per SM)
changes GEMM1 by less than 1 us, while cutting the stages to 3 costs 3 us.
One 128 x 8 x 32 ``kind::f8f6f4`` MMA occupies the single MMA warp for
about 64 issue cycles, and a K=256 stage issues 8 of them plus two
``tcgen05.cp`` scale-factor copies and a commit: ~500 cycles = 0.25 us
per stage at 2.0 GHz, which is what the stamps measure. The streaming
microbenchmark moves the same 22 MB in 6.7-7.9 us with 48-288 CTAs, so
the loop is within 10 % of the bytes' own ramp; the structural part of
the decode row is therefore routing + two prologues + two first-stage
latencies + the two MMA-issue-bound loops + two epilogues, ~20 us against
the 5 us byte floor, and the row is at 1.1 x of that
reachable figure. Two implementations of split-K (more CTAs per weight
tile, partials reduced by the last CTA) were measured as slower or equal:
with 48 working CTAs the loop is not bandwidth-bound, so spreading it
over 144 CTAs shortens each CTA's loop but not the kernel (13.4 -> 13.4
us at S=3; every configuration that needs a second wave is slower).

*Balanced decode rows, T = 2..16.* Each token routes to 16 experts, so
T=4 balanced touches 8 (16 on the shard) experts: GEMM1 has 4 x 48 = 192
weight tiles for 148 CTAs and its per-CTA lifetimes are bimodal, 22-25 us
for one tile and 30-33 us for two; the second wave streams 44 tiles while
104 SMs idle. During the first wave the 148 CTAs stream at ~3.6 TB/s
(0.7 us per K stage against 0.25 us with one expert), close to the
streaming ramp's 4.6 TB/s for 94 MB. The wave quantisation (1.3 waves ->
~9 us of the 36 us GEMM1, ~5 us of the 22 us GEMM2) is the open lever of
this row class; a remainder-only split (split-K on the tiles past the
first wave only) is the candidate, since the whole-kernel split above was
measured negative.

*Empty routing, T >= 128 (16 hot experts hold every route).* The
swap-AB GEMMs stream each expert once per row group (8 rows on the shard,
32 on the wide rank), so at T=128 on the shard 16 experts x 16 groups
re-read 16 x the touched weights from L2 while the floor charges them once:
142 us against an 11 us floor (13.2 x; TRT-LLM Gen 181 us). The dense
form would pad nothing here (128 rows per expert): forcing it measured 53
us at T=128, 63 us at T=256, 101 us at T=512 and 156 us at T=1024 against
the swap form's 145 / 171 / 212 / 389 us in the same runs (142 / 167 / 200
/ 362 us at this revision; the same forced dense form loses
on every other routing and on every expert-parallel routing, so the choice
must follow the routing). In the hybrid range the mixed GEMM1 tiles above
already take the dense tile for every full group: T=2048 empty 414 -> 295
us in the A/B above, 298 us in the table (1.9 x its floor; TRT-LLM Gen 437
us). Below T=1025 the opt-in mixed form (``SWAPAB_MIXED=1``) takes these
rows to 94 / 103 / 152 / 247 us (0.60-0.72 x) but costs the balanced and
hot rows 1.4-3.1 % through its 128-row groups (measured per kernel: the
GEMM2 over the groups' sub-tiles, not the list construction, which the
routing kernel now performs), so it is not the default. The split form
below takes these rows to 47 / 69 / 100 / 169 us with the default layout
for every other routing; what remains is the fixed cost of the launch
chain and the dense GEMM1 at these tile counts (structural note there).

*Split form (T = 128..1024 on the MoE-TP shard).* The routing kernel lays
the experts that hold more than 64 rows out in 128-row groups behind the
policy-tile groups of the other experts, but only when those experts
together hold at least 25 % of the local rows (``SWAPAB_SPLIT_MIN_ROWS``,
``SWAPAB_SPLIT_MIN_PERMILLE``), so the balanced and hot routings (896
experts of 2-18 rows, one 128-row expert among them) keep the default
layout and only concentrated routings pay for 128-row groups. The wide
groups run first in the chain on the dense gather GEMM1 (the tile the
prefill path uses: 128 x 128 for T <= 512, since at these row counts the
narrower N tile fills more SMs -- T=128 empty 56.9 against 63.6 us with
128 x 256 -- and 128 x 256 at T=1024, where 128 x 128 measured 241.9
against 235.8 us; blocked scale layout) and on the dense
finalize-fusion GEMM2 (below; the previous revision ran them on the swap
GEMM2 at ``n_tile = 128``, whose partial rows shared the two-stage
finalize with the narrow groups); the swap GEMM1 at ``n_tile = 128`` was measured
2.3 x slower than the dense tile on the same groups (65 against 28 us at
T=128: one 128-row activation gather per K stage against the dense
kernel's TMA-fed pipeline) and is not used. The narrow launches are
unchanged and both wide kernels signal their dependents at entry, so on a
routing without wide experts they cost the two dependency hops of the PDL
chain (~2 us each) and nothing else. Same-GPU medians (B300, TP8 rank 0,
graph, PDL): ``empty`` T=128 141.1 -> 56.9 us, T=256 164.2 -> 82.2 us,
T=512 194.6 -> 134.2 us, T=1024 362.7 -> 235.3 us (0.40 / 0.50 / 0.69 /
0.65 x); ``balanced`` +0.3 / 0.0 / +0.2 / +0.4 %, ``hot`` +0.6 / +0.5 /
+0.6 / +0.7 % (the hops; at T=1024 also the 16384-route fused routing
against the conversion kernel plus ``moe_sort``, 17.4 against 16.0 us). Since this revision the wide groups' GEMM2 runs
on the dense finalize-fusion kernel: it reduce-adds the route-weighted 128-row groups
into the zero-filled output (one 128 x N tile per weight tile, TMA-fed,
compacted work list over the wide slots), and the two-stage finalize adds
the narrow groups' rows on top -- it reads the output row only when the
routing produced wide groups, skips the slots whose permuted row lies in
the wide region (both from device-side counts, so a captured graph follows
the routing) and exits at once when no narrow group exists. Same-GPU
interleaved medians (B300, TP8 rank 0, graph, swap wide GEMM2 -> dense):
``empty`` T=128 57.5 -> 47.4 us, T=256 83.1 -> 68.6, T=512 134.8 -> 100.2,
T=1024 236.4 -> 169.1 (0.82 / 0.83 / 0.74 / 0.72 x); ``balanced`` and
``hot`` 0.996-1.003 x (the routing kernel's output clear). The GEMM2 N tile
(128 / 192 / 256) is within 1.6 % on these rows. The 16 wide routes of a
token are summed by the kernel's BF16 reduce-adds (the class the default
fused finalize uses below T=17): relative L2 error of the ``empty`` rows
2.4e-3 -> 5.3e-3 against the FP64 reference, inside the unchanged
tolerances. What remains at T=128 (47 us against an 11 us floor, 29 us
reachable): routing 6.6 us, dense GEMM1 22.9 us (96 items of 128 x 128 x
K=7168 on 96 SMs: 10 us of MMA issue at N=128 plus a 0.9 MB activation
gather and 458 KB of weights per item that the kernel does not fully
overlap), and 17.9 us of span for the dense GEMM2 plus the tail (the two
empty narrow launches and the finalize exit; the GEMM2's own 31.9 us of
duration overlaps the GEMM1 under the early trigger, the narrow GEMM1 /
GEMM2 / finalize durations are 5.7 / 2.9 / 1.9 us, mostly dependency
waits). Running the wide chain on a second stream (opt-in,
``SWAPAB_SPLIT_SIDE_STREAM=1``; fork after the routing kernel, join before
the finalize) removes 3-5 us from these ``empty`` rows but costs 7-9 us on
``balanced`` / ``hot`` at T=512/1024 (the wide kernels' persistent CTAs
contend with the narrow GEMM1's start), so it is not the default. On the
expert-parallel rank the split form itself stays opt-in
(``SWAPAB_SPLIT_EP=1``): with the dense wide GEMM2 and the side stream it
gains 2-11 % on ``hot`` (T=128..1024: 610 -> 597, 628 -> 609, 677 -> 637,
753 -> 673 us) but loses 0.6-1.9 % on ``balanced`` and 13-37 % on
``empty`` (27 -> 37, 30 -> 46, 33 -> 47, 39 -> 53 us), because the empty
wide kernels still pay their launches, prologues and dependency hops.

*Prefill on the dense path, T >= 2048.* The dense GEMM1 runs at about 3.5
PFLOPS with L2-resident weights (95-99% of the peak measured on the two
nodes used), so the gap is not the MMA rate. It is the 128-row M tile: at T=2048 balanced the 112 local
experts hold 36.6 rows each (3.5 x padded compute), at T=4096 73 rows
(1.75 x), at T=8192 146 rows (two tiles, 1.75 x for the second). The swap
form avoids padding but re-streams the weights once per 32-row group,
which at T=2048 already costs more bytes than the dense form's padding
costs FLOPs. Both alternatives are bounded by the same measured constants;
the binding one is the load ring (next paragraph), and the padded-compute
time of the dense form (rows padded to the tile x 6 * H * I_shard / 3618
TFLOPS) is the second term of this row class until a mixed-tile schedule
exists. The device-side tile choice above
takes the rows whose padded totals are equal under both tiles to the
two-CTA M256 tile (GEMM1 and GEMM2 both stream each weight tile once):
on the shard T=8192 balanced/empty/hot fall 8.5/5.7/3.7% and T=16384 and
T=32768 empty 3.7/2.6% (interleaved A/B/A/B, 30 replays); on the wide rank
T=8192 balanced/hot fall 17/14% and T=16384 remote-dominated 16% when the
rule is enabled there. Rows whose padded totals differ keep M128 and its
padding; the wide rank's remote-dominated T=8192 row (73-row experts,
M256 would pad 2 x) stays at the padded-compute floor of M128.

*Dense-path weight stream (round 9, same node, EP=8 rank 3 and the shard).*
The dense gather GEMM1 and finalize GEMM2 stream the weights at 70% and 67%
of the DRAM peak (Nsight Compute, real clocks, T=2048 balanced: 5.39 and
5.17 TB/s, 491 and 272 us), while the 32-row swap GEMM1 / GEMM2 on the same
2.6 / 1.3 GB of weights reach 87% / 81% (6.69 / 6.25 TB/s at T=512, 393 /
213 us); no other unit is saturated (SM 47-75%, L2 44-58%, 2-2.7 active
warps per scheduler, one CTA per SM). The gap is the operand ring: both
dense kernels run 4 A/B stages (GEMM1 50688 B per stage = 16 KB of
LDGSTS-gathered FP8 A, 32 KB of TMA FP4 B in 8-bit containers, scales;
GEMM2 42496 B), and capping the ring at 3 and 2 stages gives GEMM1 509 /
572 / 747 us and GEMM2 274 / 314 / 436 us, i.e. ``a + L / S`` with ``L`` =
0.93 us per K-stage round trip and ``a`` = 0.27 / 0.16 us per stage of
fill. The DRAM-bandwidth time of a stage would be 0.36 us and needs about
ten stages in flight. The following levers were measured on the same GPU
(paired, FP64-checked) and closed: a fifth stage does not fit (a
single-stage epilogue frees 8 KB, GEMM1 stays at 4 stages; the 128 x 128
GEMM1 tile keeps the bytes in flight and is 1-4% slower on every T=2048 /
4096 row, the 192-wide GEMM2 tile is the default); software L2 prefetch of
the operand rows ahead of the ring makes the kernels slower (per-line
``prefetch.global.L2`` from the TMA warp: GEMM1 +6-8%, GEMM2 +11-13%;
bulk ``cp.async.bulk.prefetch.L2`` of eight stages per row: about 2 x
slower, because it moves the same bytes through the same per-SM
async-copy unit that the ``a`` term already measures); the two-stage
finalize is slower on every long row of both layouts (T=8192 / 16384:
EP=8 1487 -> 1653 and 2210 -> 2555 us, shard 1710 -> 1908 and 3075 ->
3496 us, where the plain-store GEMM2 alone costs 765 of the fused 879 us
on the shard, so the reduce-add is 13% of that kernel and the rest is the
per-tile ring of a K=384 GEMM with three K-stages per tile); and the host
tile policy at EP=8 T=2048, where the 64-row hybrid form wins the
balanced / remote-dominated routings by 5 / 7.5% but loses ``empty`` by
15%, and the 32-row swap form wins ``empty`` / remote-dominated by 30 /
21% but loses balanced / hot by 24 / 42%: no per-T policy is admissible
under the per-row rule, a device-side form choice would need the swap
GEMM1 to cost less than 37 us on a nearly empty work list. The reduce path
was measured on its own (16 x T expanded rows into T token rows, 7168 BF16
columns): the epilogue's ``cp.reduce.async.bulk`` of scattered 384 / 512
B row segments runs at 2.6-2.7 TB/s of payload at T=8192 and 2.4 TB/s at
T=16384, within 3-10% of a plain scattered bulk store of the same
segments, so the reduce-add itself is not the cost; a token-major
reduction reads at 4.6 TB/s but needs the expanded rows written first,
which is the two-stage form above. With the round-9 calibration of the
ring term the remaining rows above 1.10 x of the reachable floor were the ``empty`` routings from T=256 (EP=8, 30-132 us
chains bound by the routing kernel and the swap GEMM1's per-tile latency),
the shard's T=1024 / 2048 balanced and hot rows (32-row swap groups
re-read each expert's weights from L2 once per group; the 64-row form
measured slower at T=1024 in an earlier revision), EP=8 T=2048
remote-dominated (1.31 x, the policy conflict above), T=4096 balanced
(1.13), T=8192 remote-dominated (1.14) and the T=32768 balanced rows of
both layouts (1.15 / 1.23); each is listed with its binding term in the
tables.

*Shared-memory fill rate and the chains of the open rows (round 10, same
node).* The swap kernels' rings were printed and swept as the ring bullet
above records; the shard's dense finalize GEMM2 (K = 384, 6 stages of 34304
B) is insensitive to its depth (320 / 320 / 386 us at 6 / 3 / 2 stages,
T=2048 balanced) because it is bound per tile: 97 tiles per CTA at 3.3 us
each, 0.94 us of fill (three stages at 110 GB/s) plus about 2.4 us for the
scattered 32 KB reduce epilogue (470 MB of payload at the reduce
microbenchmark's 2.7 TB/s is 174 of the 320 us). With the fill-rate ring
term the rows above 1.10 x of the reachable floor and their measured
composition are: on the expert-parallel rank every ``empty`` routing from
T=128 (1.16-2.06 x) plus ``hot`` T=16 (1.10), the T=2048 / 4096 balanced
and remote-dominated rows (1.11-1.22), T=8192 remote-dominated (1.17) and
the T=32768 rows (1.11-1.20); on the shard ``empty`` T=128 / 256 (1.35 /
1.48) and the T=16384 / 32768 balanced and ``empty`` rows (1.12-1.25).
Their chains were measured kernel by kernel (standalone launches on the
same GPU, FP64-checked). EP=8 T=2048 ``empty`` (one local expert, 42 rows)
runs 54.4 us on the dense form as routing 5.3 + ``moe_sort`` 10.4 + GEMM1
25.9 + GEMM2 11.6 us plus launch hops, against 70.1 us on the 64-row
hybrid form (5.4 + 11.5 + dispatch 4.2 + swap GEMM1 23.5 + the wide GEMM1's
no-work exit 2.9 + GEMM2 13.4): the swap GEMM1 is no slower than the dense
one on a single 48-tile group (a 56-stage chain either way), the hybrid
costs its two extra launches and hops, so no form choice brings this row
within 1% of both its ``empty`` and its balanced routing; the same
per-row rule keeps EP=8 T=2048 / 4096 remote-dominated on the dense form
although the hybrid form measured 741 against 798 and 765 against 816 us
there. Shard T=256 ``empty`` runs 67.6 us as routing 7.5 + dense wide
GEMM1 33.6 (32 wide groups x 6 N tiles = 192 items on 148 SMs: two
56-stage chains) + dense wide GEMM2 34.9 (1792 items, 12 per CTA x 3
stages plus the reduce epilogue) + the narrow kernels' exits 6.7 + 2.9 +
2.2; T=512 ``empty`` 98.8 = 9.8 + 43.2 + 51.6 + 7.0 + 2.9 + 3.2. On the
long rows the first kernel is the route conversion with the output
zero-fill: 33.6-38.3 us at T=16384 and 64.9-86.8 us at T=32768 on both
layouts, i.e. ``T x 7168 x 2 B`` at 6.9-7.2 TB/s on the ``empty`` rows
and 5.4-6.2 TB/s on the others, against the 6.91 TB/s measured fill
bandwidth, so where it matters it is at the DRAM bound; the fused
finalize's reduce-add needs the zero base, and the token-major alternative
that would write every row once was measured slower above. It is 41-49% of
the EP=8 ``empty`` rows at T=16384 / 32768 (82 / 132 us) and 1.3-3.2% of
the other routings. On the decode-class rows the fused routing kernel is a
single CTA: 4.6 / 5.2 / 7.0 / 12.7 us at EP=8 T=128 / 256 / 512 / 1024
(1024 to 8192 routes) and 6.5-7.0 / 7.6-8.4 / 9.9-12.9 / 16.0-17.6 us on
the shard, 17-33% of the EP=8 ``empty`` rows (27.6-38.6 us) and 9-14% of
the shard's (47.7-169.5 us), at most 2% elsewhere; its ``%globaltimer``
phases at T=1024 put 5.3 us in the histogram (one CTA's shared-memory
atomics over 8192 routes after the eight-route unrolled loads) and the
rest in the scan, the staged inverse permutation and the launch. A
multi-CTA routing kernel (per-CTA histograms, a cluster or grid prefix,
parallel scatter) is the open lever for these rows; every other term of
the ``empty`` chains is a launch, a dependency hop or a single 56-stage
GEMM1 chain on one expert.

*Decode-class rows, cluster routing kernel (round 11, same node).* The
single-CTA routing kernel named as the open lever above is replaced by the
cluster form described under "Kernel selection": at EP=8 T=1024 the kernel
takes 6.3-6.8 us instead of 12.8 (T=512: 5.3 instead of 7.0), on the shard
8.4-9.2 instead of 16.1-17.6 us (T=512: 6.8-7.5 instead of 9.9-12.3), and
its phases end at 4.6-4.9 us on the rank (histogram 1.8, count exchange
1.4, group bases 0.55, scatter 0.3) and 6.6-7.4 us on the shard (896-expert
group tables 1.9 us), with the output clear done at 3.4-4.0 us. The
``empty`` rows move accordingly (EP=8 T=512 / 1024 32.7 -> 29.6 and 38.8
-> 31.4 us, shard 99.2 -> 96.0 and 169.3 -> 161.4 us) while every other
T=128..1024 row stays within 0.999-1.015 x and the decode rows within
0.989-1.002 x. What remains of the EP=8 ``empty`` chains at T=128..1024
(27-31 us) is three launches with their dependency hops (the routing
kernel's own launch plus 4.5-6.8 us of work, the swap GEMM1's 56-stage
chain on one expert, 17-18 us, and the swap GEMM2, 11-12 us), which the
loop term of the reachable floor already charges; at that revision 39 of
the 56 rank rows and 36 of the 42 shard rows sat within 1.10 x of the
reachable floor (geometric mean 1.08 / 1.02).

*Decode rows, dependent-side weight prefetch (round 12, same node).* Per-
kernel CUPTI start / end offsets of one graph replay (EP=8 T=1 hot):
before, routing 0-2.4 us, GEMM1 1.0-14.9, GEMM2 13.5-22.3, i.e. GEMM2
starts 1.4 us before GEMM1 ends and its tail after GEMM1 is 7.4 us. With
the partial wait GEMM2 starts at 3.8 us (right after GEMM1's own wait),
streams its 13 MB while GEMM1's loop runs (GEMM1 ends 0.2 us later, 15.1),
and its tail after GEMM1 is 6.8 us with half the tile in the 12 x 128-wide
stages and ~6 us with the whole tile resident (12 x 256-wide stages; row
22.4 -> 21.5 us). An 11 us head start on the weight stream therefore
removes 1-1.4 us: what remains after GEMM1 is the row gather from GEMM1's
output, the MMA-issue-bound loop (96 ``kind::f8f6f4`` MMAs of K = 32 at
~64 issue cycles, ~3.1 us; N = 8 leaves the tensor pipe idle), the
epilogue and the drain. The GEMM2 half of the decode chain is thus
independent of the weight bytes once they are resident; the remaining
lever there is spreading the issue-bound loop over more CTAs with the
resident tile (split-K with a cluster reduction), not the memory system.
GEMM1's stream cannot start before the routing kernel publishes the
expert ids (2.4 us), so the same prefetch does not apply to it. The tables
below are regenerated on this revision: 38 of the 56 rank rows and 36 of
the 42 shard rows within 1.10 x of the reachable floor (geometric mean
1.07 / 1.02; the rank loses one long dense row to the node's run-to-run
band, its decode rows move from 1.07-1.09 x to 1.01-1.06 x of the
reachable figure), and against the bytes / FLOPs / launch floor 9 of 56
and 2 of 42 rows as before (geometric mean 2.04 / 1.86).

*Decode rows, adaptive split-K over the resident GEMM2 tile (round 13,
same node).* The finalize epilogue of the swap GEMM2 accumulates into the
output with ``red.global.add``, so partial K sums are additive and need no
cross-CTA reduction, partial buffer or counter (the earlier split-K
above kept partials in a buffer behind a counter and reduced them in the
last CTA, and measured slower or equal, 8.8 -> 10.0 us at S = 2, to that
traffic and its second prologue). The scheduler warp now
publishes a K range with every work item and, only when the launch's
valid work fits in half the SMs (``num_groups x m_chunks <= 74`` items on
this node), hands out ``(m_chunk, split)`` items that each run half of the
12 K stages of the resident tile; otherwise the items are remapped onto
the original raster exactly, so no CTA idles (a first form that skipped
the unsplit items idled the odd CTAs and cost the multi-expert decode rows
9-12 %). Measured on the same GPU against the round-12 revision (20
repeats, FP64-checked): the single-expert decode rows (hot / empty /
remote-dominated ``T = 1..8``, empty ``T = 16``) 1.028-1.038 x (21.9-22.2
-> 21.1-21.6 us), the ``empty`` rows at T = 128..1024 1.023-1.036 x (26.1
-> 25.6, 28.8 -> 27.8, 28.9 -> 28.0, 31.0 -> 30.1 us), the multi-expert
decode rows 0.997-1.004 x (unsplit path), the shard's decode rows
0.992-1.008 x (K = 384 is one stage, S = 1), every other row 0.996-1.001 x.
S = 3 loses 2 % and S = 4 11 % on the same rows (each split re-runs the
2.6 us prologue and halves the stage depth per CTA), so S = 2 is the only
value used. Per-kernel CUPTI offsets of one replay (EP=8 T=1 hot, both
trees in one run): routing 0-2.8 us and GEMM1 1.1-15.5 us unchanged,
GEMM2 4.1-21.9 -> 4.1-21.2 us, i.e. the tail after GEMM1 is 6.3 -> 5.5
us. Splitting the issue-bound loop in two removed 0.8 us of a ~3.1 us
loop, not 1.5: the row gather from GEMM1's output, the six remaining
stages per CTA (~1.6 us of issue), the finalize epilogue and the drain
are serial per CTA and the two CTAs of a tile end together. The rows'
relative L2 error against the FP64 reference moves from 0.00167 to
0.00247 because the two partials each round through the BF16 ``red.add``
(1e-2 class, tolerances unchanged). The single-expert decode rows now
stand at 21.1-21.6 us against a 5 us bytes floor and a 9-10 us
three-launch floor: the remaining 11-12 us is the three dependent
kernels' own fill and drain (routing 2.8 us, GEMM1's 56-stage chain on
one expert ~14 us of which ~11 overlap GEMM2's weight stream, GEMM2's
5.5 us tail), which only a fusion of the routing kernel into GEMM1 or a
shorter GEMM1 chain would remove. The tables below are regenerated on
this revision: 38 of the 56 rank rows and 36 of the 42
shard rows within 1.10 x of the reachable floor (geometric mean
1.06 / 1.02), and against the bytes / FLOPs / launch
floor 9 of 56 and 2 of 42 rows (geometric mean
2.03 / 1.85).

*Decode rows, 2-CTA cluster split-K over the SiTU GEMM1 (round 14, same
node).* After round 13 the single-expert decode rows spent 12.5 of their
21 us in GEMM1's 56-stage K chain on one expert: 32 work items on 148
SMs, each CTA streaming and issuing the whole K range alone. The swap
GEMM1 now launches as a ``(1, 1, 2)`` cluster along its persistent grid
when the host sees an expert-parallel rank (``num_local_experts <
num_experts``), a tile of ``n_tile <= 16`` tokens, an even K-tile count
and one M chunk per CTA; every TMA and pipeline coordinate keeps the
single-CTA layout, the cluster exists only for the exchange. Both CTAs
of a pair take the same ``(m_chunk, row_group)`` item and split its K
range (the round-13 decision, grid-uniform and device-side: only while
the valid items fit half the CTA budget). The peer scales its 128 x
``n_tile`` FP32 accumulator by alpha and ships it with ``st.async`` into
the leader's shared memory, completing the leader's ``red_full``
mbarrier by transaction count; the leader adds it to its own accumulator
before the SiTU + MXFP8 requant epilogue and frees the slot with a remote
``mbarrier.arrive.release.cluster`` on the peer's ``red_empty``. Unsplit
launches take neither cluster barrier and never touch the peer.
Measured on the same GPU against the round-13 revision (20 repeats,
FP64-checked): the single-expert decode rows (hot / empty /
remote-dominated ``T = 1..16``) 1.081-1.087 x (20.9-21.7 -> 19.2-20.0
us), the ``empty`` row at T = 128 1.104 x and at T = 1024
1.010 x, the multi-expert decode rows 0.988-1.001 x (three-group
launches, 96 items, unsplit) and the shard 0.996-1.006 x (the cluster
form is never launched there). The 1-1.2 % on the unsplit rank rows is
not the cluster launch and not the shared memory: the same kernel
without the cluster launch (``SWAPAB_GEMM1_CS_DEBUG=nocluster``, never
splitting) measures 0.984-0.999 x, and the round-13 kernel with the 8 KB
exchange buffer added to its shared memory and nothing else 0.993-1.001
x; it is the split-capable code path itself (K-range tile info and the
exchange branches in the epilogue), left as the next lever. Against the
phase-2 baseline those rows pair 1.026-1.109 x (``T = 1..16`` balanced /
remote-dominated on the rank) and the single-expert rows 1.28-1.30 x.
Per-kernel CUPTI offsets of one replay (EP=8 ``T = 1`` hot): routing
0-2.6 us unchanged, GEMM1 1.0-15.4 -> 1.1-13.6 us, GEMM2 3.9-20.9
-> 4.3-19.3 us. Splitting the 56-stage chain in two removed ~1.8 us of GEMM1's ~14.4 us: each CTA still runs the 2.6 us prologue, 28 stages, the
exchange round trip and the epilogue, and GEMM2's tail after GEMM1 moves 5.5 -> 5.7 us. The single-expert decode rows now stand at 19.2-20.0 us
against a 5 us bytes floor and a 9-10 us three-launch floor; what remains
is fill and drain of three dependent kernels. Numerics: the peer's FP32
partial is added before the requant, so the rows are bit-identical to
round 13 where the launch does not split and within the same 1e-2 class
where it does (relative L2 error against FP64 0.00247 -> 0.00247 on the
EP=8 decode rows). The tables below are regenerated on this revision:
40 of the 56 rank rows and 37 of the 42 shard rows
within 1.10 x of the reachable floor (geometric mean 1.04 /
1.01), and against the bytes / FLOPs / launch floor 9
of 56 and 3 of 42 rows (geometric mean 2.00 /
1.85).

*Structural terms of the open rows and the host-level levers they exposed
(round 15, same node).* The 21 rows that round 14 left above 1.10 x of
the reachable floor were measured against launches of a known item count
and against per-kernel timelines from T=128 to 32768 (constants in the
ring and bookkeeping bullets above). Three findings are structural to the
kernels and now sit in the floor: a kernel with fewer items than SMs is
bound by its single-item chain (23.7 us for a dense 128 x 256 tile at
K=7168, 11.0 / 16.5 us for a 16- / 32-row swap group after its routing
dependency), not by a share of the chip's fill rate; the finalize tile's
reduce-add does not overlap the next tile's fill; and the route
bookkeeping grows with T independently of the routing. Three others were
schedule choices with no kernel change, adopted in this round:

* the dense form's output zero-fill ran between GEMM1 and GEMM2 in stream
  order, 5.6 / 10.5 / 19.8 / 37.9 / 74.7 us at T=2048 / 4096 / 8192 /
  16384 / 32768 on both layouts (half of the EP=8 T=32768 ``empty`` row).
  On expert-parallel ranks the plan now forks it onto an auxiliary
  stream right after the routing sort and joins before GEMM2 (captured
  graphs get the fork/join as edges; ``MXFP4_DENSE_ASYNC_MEMSET`` =
  ``ep`` by default, ``1`` for every layout, ``0`` for the in-order
  launch everywhere). Hidden behind GEMM1 it shares HBM with it -- the
  fill stretches to 97 us at T=32768 and GEMM1 slows by about the hidden
  bytes on the HBM-bound long rows -- so the rank's long rows gain 1-3 %
  (balanced / remote-dominated 0.970-0.992x of the previous revision's
  time) and its ``empty`` rows 2-13 % (T=2048 / 4096 / 8192 / 16384 /
  32768: 0.913 / 0.892 / 0.868 / 0.907 / 0.986; same GPU, alternating,
  10 repeats, FP64-checked). The MoE-TP shard keeps the in-order launch:
  paired on one GPU (20 repeats) its dense rows read 0.969-1.005x with
  the overlap, because the finalize GEMM2 ran 2-4 % slower once the
  zero-fill no longer sat right before it (the zeroed output left L2
  during GEMM1, so the reduce-adds miss) and GEMM1 paid the hidden
  bytes, more than the 20-120 us the overlap saved (T=8192 hot 0.973,
  T=32768 hot / ``empty`` 0.969 / 0.981). Forking at the start of
  ``run`` instead lets the zero-fill occupy the chip ahead of the small
  routing kernels and costs the rank's ``empty`` rows 14-18 %
  (rejected);
* the routing sort's single-cluster permutation kernel served every
  dense launch up to T=8192 (11.4 / 14.7 / 19.6 us at T=2048 / 4096 /
  8192) where the cooperative histogram + offsets path that serves
  T >= 16384 takes 1.3 + 5.6 us; the dispatcher now keeps the cluster
  kernel up to 1024 tokens (the cluster path's 10.2 us at T=2048
  becomes 1.2 + 4.6 us on the cooperative path; routing output
  identical);
* the dual-tile dense routing (round 4) was on for shards up to 512
  columns only; the expert-parallel rank now uses it as well
  (``MXFP4_DENSE_DUAL_TILE_MAX_SHARD`` default 3072), so its long rows
  take the 256-row tile when the rows per expert pad to the same
  256-row multiple either way: same-GPU pairs against the previous
  revision 0.824 / 0.848 at T=8192 balanced / hot and 0.819 at T=16384
  remote-dominated; the ``empty`` rows at T=8192 / 16384 pay 1.077 /
  1.051 (the 256-row tile's longer single-item chain on their few items)
  and every other row reads 0.994-1.007.

Still open after this round: 19 expert-parallel and 5 MoE-TP rows, in
four groups. The dense long rows (T=8192..32768 balanced / hot /
remote-dominated on the rank, T=8192 balanced and T=32768 ``empty`` on
the shard) stand 1.10-1.39 x above a floor that charges the 256-row tile
at the node's MMA peak and the full wave at the best measured aggregate
fill (130 GB/s per SM, where a full dense wave measures 106 GB/s per SM)
with the reduce-add epilogue of the finalize tiles added, so the gap is
the dense kernels' own efficiency at full occupancy, not a schedule
choice (round 9 closed the tile, N-width and cluster tactics). The
rank's ``empty`` rows at T >= 2048 (1.26-1.84 x) have their floor in a
128 x 128 dense form the dispatcher does not launch (the 128 x 256
tile's 23.7 us single-item chain against the 128 x 128 tile's 15.8 us)
plus the T-proportional bookkeeping and, at T=32768, the zero-fill that
no longer fits behind GEMM1; the T=2048 remote-dominated row (1.32 x)
has its floor in the swap form, which the data-selected form would reach
only on that row and on the ``empty`` rows (closed above). The remaining
rows sit at 1.10-1.14 x (rank T=128 ``empty`` and T=512 hot; shard T=8
balanced and T=128 / 256 ``empty``), within the node's repeat band of
the measured constants. The tables below are regenerated on this
revision: 37 of the 56 rank rows and 37 of the 42 shard rows within 1.10
x of the reachable floor (geometric mean 1.11 / 1.04), and against the
bytes / FLOPs / launch floor 9 of 56 and 2 of 42 rows (geometric mean
1.96 / 1.84).

*MoE-TP shard, GEMM2 with the fused finalize.* Its K is 384, and the
CUTLASS kernel on exactly that shape peaks at 1427 TFLOPS (12 K-steps of
32 cannot hide the tcgen05 pipeline fill), while GEMM1 of the shard peaks
at 2878 TFLOPS. Re-computing the compute floor of the MMA-bound shard rows
with these shape-specific peaks, the 128-row tile padding of the dense form
and the reduction bound below gives 4.41 ms at T=32768 balanced against the
candidate's 5.86 ms (1.33 x) and 2.65 ms at T=16384 against 3.19 ms
(1.20 x); the hot rows sit at 1.04 x (16384) and 1.14 x (32768) of that
reachable floor. The 2.0-2.6 x in the table is the distance to a peak this
GEMM shape cannot reach. The fused finalize adds
its own bound: the top_k=16 reduction moves the routed BF16 rows through
L2 at a measured 2.1-3.7 TB/s (3.66 TB/s with the weights L2-resident); a
two-stage alternative (write the expanded rows, reduce them in a second
kernel) would move twice the rows through HBM, 3.45 TB/s equivalent, so the
fused form is at the structural reduction bound and the GEMM2 floor of the
shard is ``max(W2 / 6.92 TB/s, R / 3.4 TB/s)`` with ``R`` the routed-row
bytes. Two alternatives were measured on the same GPU and closed. A
two-stage finalize (``MXFP4_DENSE_TWO_STAGE=1``: GEMM2 writes the expanded
rows, a second kernel reduces them token-major) is 1.09-1.31 x slower on
every T=4096..32768 row of both layouts: the expanded-row GEMM2 saves up
to 988 us (T=32768 balanced, 3541 -> 2553 us) but the reduce kernel costs
100-1512 us. Nsight Compute on the fused GEMM2 of the shard (T=8192 and
T=16384 balanced, clock control off) reports 30-33% SM throughput, 75-77%
DRAM throughput and a 29-39% L2 hit rate: the reduce-adds miss L2 because
the output (117-235 MB) exceeds it, so each of the top_k=16 adds to a
token row is a DRAM read-modify-write. Walking the tiles M-fastest (groups of ``MXFP4_GEMM2_SWIZZLE`` = 4 N
tiles so each A tile is still reused four times) keeps one 256-column
slab of the output L2-resident while every M tile passes. Measured in
round 16 on one B300 (graph, span against the N-fastest order, 20
repeats, two passes): T=16384 balanced / empty 0.916 / 0.844, T=32768
0.867 / 0.864, but the hot routing (one expert holding every token plus
895 small ones) 1.059 / 1.088 and T=8192 1.01-1.06 on every routing; the
loss is specific to one dominant expert next to a long tail (one giant
expert plus 895 small: +4.5-6 %; 16 giant experts: -20 %; 512 equal
small experts: -8 %). Since round 16 the finalize GEMM2 of shards up to
``MXFP4_GEMM2_RASTER_M_MAX_SHARD`` = 512 columns from
``MXFP4_GEMM2_RASTER_M_MIN_TOKENS`` = 16384 tokens compiles both rasters
(``MXFP4_GEMM2_RASTER_M`` = ``auto``, the default; ``1`` / ``0`` force
M- / N-fastest) and its scheduler warp picks one per launch from the
routing it is given: 64 evenly spaced samples of the expert id of the
valid 128-row groups, two consecutive samples of one expert marking a
dominant expert (more than about 1/64 of the groups, a quarter of the
tokens at top_k=16) and more than 32 distinct samples a long tail; that
combination keeps N-fastest, everything else takes M-fastest, and a
launch without valid tiles keeps N-fastest so its loop exits at the
first padded group. Paired against the round-15 revision on the same GPU
the shard's T=16384 balanced / hot / empty rows read 0.904 / 0.96 / 0.83
and T=32768 0.93-0.95 / 0.99 / 0.88 with the relative L2 against FP64
unchanged to four digits; the wide rank (K=3072, 3-5 % slower with
M-fastest) and T <= 8192 keep the N-fastest raster.

*MoE-TP shard, GEMM2 output traffic (round 16 micro-benchmarks, B300).* The
384-wide shard's finalize GEMM2 streams three K stages per tile and then
reduce-adds a 128 x 256 bf16 tile into the token rows: 16 expanded rows per
token, so the output traffic is 16 x T x 7168 x 2 B (1.9 GB at T = 16384 x
... 7.5 GB at T = 32768) against a 58-470 MB output. Three probe builds of the
kernel (skip the reduce-add; plain bulk store instead of the reduce-add;
no wait for the bulk read before the next tile) put the output traffic at
112-130 us of the 292-355 us GEMM2 at T = 4096 (79-96 us plain write, 33-34
us atomic add, 4.5 % wait) and at 3.5-9 % of the expert-parallel rank's
GEMM2 (wait 0 %). A standalone scatter of the same rows (values generated
in registers, random token per row, graph CUPTI, 20 repeats) fixes the
ceiling of the pattern: plain stores 6.7-8.3 TB/s, ``red.global.add.bf16x2``
3.6-6.3 TB/s, ``cp.reduce.async.bulk`` (the kernel's path) 3.6-6.5 TB/s with
256 B rows 3-5 % faster than 512 B rows, a register-direct reduce in the
TMEM fragment layout (row per lane) 2x slower, a 3.6 MB target no faster than
58 MB, and rows of one token adjacent 2.2x faster for stores but 15-50 %
slower for reduces (same-address serialization). The kernel's reduce path is
at that ceiling (150 us standalone vs 130 us inside the GEMM at 65536 rows),
so the c-stage double buffering, the bulk-read wait and a register-direct
epilogue are closed as levers; the shard's output-traffic floor is 16 x T x
7168 x 2 B / 6.5 TB/s = 290 us at T = 8192 and 1156 us at T = 32768.

What does move the shard is the tile order: walking the M tiles of one N
tile (M-fastest raster, groups of 4 N tiles) keeps every token's 16
reduce-adds of a 256-column slab in L2 and measured, span against the
N-fastest order on the same GPU, T = 16384 balanced 0.916 / empty 0.844 /
hot 1.059 and T = 32768 balanced 0.867 / empty 0.864 / hot 1.088 (T = 8192
1.01-1.06 on every routing). The hot loss is specific to one dominant expert
next to a long tail of small ones (one giant expert plus 895 small: +4.5-6 %;
16 giant experts: -20 %; 512 equal small experts: -8 %), so the kernel's
scheduler warp now picks the raster per launch from the routing (a run of at
least 1/32 of the valid 128-row groups while more than 32 experts are active
keeps N-fastest) for shards up to 512 columns from 16384 tokens.

*Wide rank at T=8192 and T=16384 remote-dominated.* These rows are the
slowest against TRT-LLM Gen (0.79-0.86) and 2.0-2.4 x their floor: the
first from tile padding (146 rows per expert, two M128 tiles), the second
from 73-row experts that M256 would pad 2 x. The device-side tile choice
takes the T=8192 balanced/hot rows down 17/14% and the T=16384
remote-dominated row down 16% (same GPU, A/B/A) but stays opt-in on this
rank because its two unchosen launches cost the rank's 86-160 us empty
routings 1-4%; skipping them on the device (graph conditional nodes) is
the open item, and the rows are reported as open.

Dense-path evaluation history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The remainder of this section records the evaluation of the dense grouped
GEMM path against the previous implementation and TRT-LLM Gen before the
swap-AB path existed; its decode and small-prefill gaps are what the swap-AB
path closes.

All 66 required rows are clock-qualified: T1/2/4/8/16 and
T128/256/512/1024/2048/4096, three routing distributions, eager and Graph.
Every row compares identical operands across the previous implementation,
a native-arithmetic-only control, the current implementation and TRT-LLM Gen
on the same B300. Inputs use seed 123, beta 4, linear_beta 25 and packed BF16
routing. Measurements use cold-L2 CUPTI, alternating backend order, five
warmups and 20 samples, with SM clock at least 97% of reported maximum.
No CUDA-event fallback qualifies. Raw samples and latency/routing histograms
are retained with the measurements.

The 15 Graph-decode rows use a T16-generated fixture bank; the other 51 rows
use a T4096-generated bank. Each comparison is paired within its row; these
batches must not be treated as identical cross-shape inputs. The two timing
jobs took 144.523 and 204.539 seconds of managed physical turnaround,
respectively, with drivers of 21.855801 and 78.972480 seconds: 349.062 seconds
physical turnaround and 100.828281 seconds of driver execution in total.

Balanced medians below are microseconds. Kernel sum covers the complete
runner. Eager synchronized end-to-end time includes host submission; Graph
end-to-end is reported separately. Each current/TRT pair is a same-row
comparison. Graph and eager measurements are not interchangeable.

.. list-table:: Balanced routing; all rows clock-qualified
   :header-rows: 1

   * - T
     - Previous Graph kernel sum
     - Current / TRT Graph kernel sum
     - Current / TRT Graph E2E
     - Current / TRT eager E2E
   * - 1
     - 37.393
     - 36.816 / 32.657
     - 53.849 / 51.614
     - 137.186 / 243.987
   * - 2
     - 58.273
     - 57.409 / 50.144
     - 75.145 / 68.553
     - 150.860 / 259.638
   * - 4
     - 87.392
     - 85.936 / 73.088
     - 102.648 / 91.736
     - 179.223 / 284.914
   * - 8
     - 145.584
     - 143.121 / 122.273
     - 160.927 / 141.992
     - 237.589 / 327.286
   * - 16
     - 250.659
     - 245.874 / 201.969
     - 264.240 / 221.799
     - 341.718 / 406.781
   * - 128
     - 801.992
     - 803.414 / 729.464
     - 827.418 / 750.339
     - 895.606 / 933.914
   * - 256
     - 808.903
     - 806.888 / 763.912
     - 831.788 / 786.077
     - 897.922 / 937.087
   * - 512
     - 816.057
     - 814.664 / 695.671
     - 838.587 / 717.417
     - 908.673 / 891.537
   * - 1024
     - 825.948
     - 822.823 / 1362.717
     - 847.228 / 1387.690
     - 920.750 / 962.227
   * - 2048
     - 840.056
     - 837.656 / 1294.413
     - 863.487 / 1317.369
     - 929.683 / 2059.941
   * - 4096
     - 876.969
     - 875.354 / 1236.013
     - 902.584 / 1260.200
     - 969.024 / 1245.815

Geometric-mean ratios below are baseline latency divided by current latency;
values above one favor the current implementation. Decode means T1..16 and
prefill means T128..4096, each with all three routing distributions.

.. list-table:: Paired performance ratios
   :header-rows: 1

   * - Scope
     - Rows
     - Previous/current kernel
     - TRT/current kernel
     - TRT/current synchronized E2E
   * - Graph decode
     - 15
     - 1.03914
     - 0.94422
     - 0.98184
   * - Graph prefill
     - 18
     - 1.00690
     - 1.24791
     - 1.20115
   * - Eager decode
     - 15
     - 1.04018
     - 0.94463
     - 1.72478
   * - Eager prefill
     - 18
     - 1.00886
     - 1.20491
     - 1.37271
   * - All required
     - 66
     - 1.02220
     - 1.08899
     - 1.29188

The current implementation improves aggregate kernel time over the previous
implementation, but does not achieve performance neutrality against TRT.
Balanced Graph T1 and T16 kernel time remains 12.7% and 21.7% higher than
TRT; Graph end-to-end remains 4.3% and 19.1% higher. Larger-prefill gains do
not remove these decode gaps. Kernel medians improve in 64 of 66 rows versus
the previous implementation; the two regressions are Graph T128 balanced
(0.177%) and Graph T1024 hot (0.053%). Thirty-five rows remain slower than
TRT in kernel sum. Eager host-side gains do not establish Graph GPU parity.
The aspirational 5--10% improvement is not claimed across the required matrix.

Optional T8192/16384/32768 evaluation completed all 18 eager/Graph rows.
The nine shape/routing configurations passed numerical preflight; every row
passed the same ideal-FP64 development gate before and after timing, with
finite outputs and matching routing counts. This separate batch compares
previous/current CuTe and TRT, using a T32768-generated fixture bank with
seed 123. It took 196.536 seconds of physical turnaround and 75.207 seconds
of driver execution.

Only nine rows passed the unchanged clock gate. Their latencies are below;
no excluded-row timing contributes to this table or the required-row results.
Six qualified rows have no local expert assignments, so their large TRT
ratios do not establish balanced or hot-routing performance.

.. list-table:: Clock-qualified optional medians in microseconds
   :header-rows: 1

   * - Mode / T / routing
     - Previous kernel
     - Current / TRT kernel
     - Current / TRT synchronized E2E
   * - Graph / 8192 / balanced
     - 1527.808
     - 1505.904 / 1163.259
     - 1532.277 / 1186.323
   * - Eager / 8192 / empty
     - 57.649
     - 56.562 / 189.443
     - 147.623 / 365.889
   * - Graph / 8192 / empty
     - 76.961
     - 75.394 / 189.218
     - 95.415 / 208.944
   * - Eager / 8192 / hot
     - 1757.058
     - 1774.339 / 1481.568
     - 1883.796 / 1672.830
   * - Graph / 8192 / hot
     - 1797.634
     - 1792.532 / 1479.728
     - 1820.207 / 1504.332
   * - Eager / 16384 / empty
     - 49.632
     - 48.463 / 291.123
     - 179.597 / 521.492
   * - Graph / 16384 / empty
     - 86.450
     - 86.289 / 290.932
     - 108.278 / 312.087
   * - Eager / 32768 / empty
     - 71.857
     - 70.433 / 564.647
     - 232.969 / 758.157
   * - Graph / 32768 / empty
     - 144.546
     - 142.514 / 564.839
     - 165.413 / 587.811

The nine clock-excluded rows are eager T8192 balanced, plus eager and Graph
balanced/hot at T16384 and T32768. They retain numerical evidence but provide
no qualified timing comparison. Among qualified rows, current CuTe is slower
than TRT for T8192 balanced/hot; eager T8192 hot also regresses relative to
the previous CuTe implementation. Older optional measurements are not reused.

Reproduction
------------

Tests also cover native E2M1/UE8M0 decoding, lossless preparation, packed and
separate routing, runtime parameters and preallocated execution:

.. code-block:: bash

   FLASHINFER_KIMI_K3_FULL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py
   FLASHINFER_KIMI_K3_ALL_LOCAL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py -k all_local
   # EP rank-sum and MoE-TP shard-sum identities, TP graph replay and streams
   FLASHINFER_KIMI_K3_FULL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ_layouts.py

The layout tests build the unsharded 896-expert bank once per module and
slice it; the small-geometry cases run on any SM100/SM103 without the
environment flag. Routing histograms (global and per rank) are attached to
the sum tests as ``numerical_report`` properties.

The standalone benchmark reports kernel sum, GPU activity span, host enqueue
and synchronized end-to-end latency, raw samples and paired ratios. GPU span
includes correlated memory operations and gaps. Kernel sum adds kernel
activity durations, excluding separate memcpy/memset records but including
a graph-lowered memset reported as a kernel. Host metrics use CUPTI timestamps.
The command below runs the public two-backend comparison; the reported
four-backend measurements additionally retain previous/control implementations.

.. code-block:: bash

   python benchmarks/bench_mxfp4_situ_moe.py --full --accuracy \
     --tokens 1,2,4,8,16,128,256,512,1024,2048,4096 --distributions balanced,empty,hot \
     --modes eager,graph --routing packed --warmup 5 --repeats 20 \
     --run-id review --output mxfp4-situ-review.json

``--layout ep8 --rank R`` measures expert-parallel rank ``R`` (112 experts at
offset ``112R``) and ``--layout tp8 --rank R`` measures a MoE-tensor-parallel
shard (896 experts, intermediate 384); both require ``--full`` and generate
the rank-local bank directly rather than slicing a full bank. Every row
records the global and per-rank routing histograms; ``--distributions`` also
accepts ``remote_dominated``. The default invocation is unchanged.
