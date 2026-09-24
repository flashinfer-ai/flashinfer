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
clearing and, when PDL is disabled, local expert histogram/prefix/scatter.
The default decode forward then launches GEMM1 and GEMM2 with finalize:
three device-kernel launches in total. Expert IDs must be unique within each
token and lie in ``[0, num_experts)``, as in standard top-k routing.
PDL-enabled decode retains separate
native sorting. The fused sort covers every T with ``T * top_k <= 4096``
(T <= 256 for Kimi K3, see "Kernel selection" below); above that the same
conversion + output-clear kernel runs without the fused sort (it is
grid-strided over any token count), followed by the native ``moe_sort``;
PDL-enabled prefill keeps the torch conversion and separate memset.

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
* The deferred form is provided by the swap-AB path (SiTU, PDL disabled) at
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
not run slower than the single-tile plan. The rule is therefore the
default on shards up to 512 columns (``MXFP4_DENSE_DUAL_TILE_MAX_SHARD``)
and opt-in on the wide rank (``dense_dual_tile=True`` or a larger limit),
where it takes the T=8192 balanced/hot rows down 17/14% and the T=16384
remote-dominated row down 16% (same-GPU A/B/A). Launching the alternate
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
128-row groups. On the fused-routing path (T <= 256) the routing kernel
emits the three lists itself (``SWAPAB_MIXED_FUSED_LISTS``, default 1;
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

Routing preprocessing (expert histogram, permutation, per-row scale, output
clear) runs as one fused CuTe kernel whenever ``T * top_k <= 4096``
(``FUSED_ROUTE_MAX_ROUTES``), i.e. T <= 256 for Kimi K3, and as the shared
``moe_sort`` path above. The fused kernel is a single block (or a grid that
also clears the output when the swap epilogue reduce-adds into it) and
replaces the unpack/conversion kernels, ``moe_sort`` and the memset
(about 10.6 us at T=128) with roughly 3 us; the generic clear kernel uses
16-byte stores.

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

.. list-table:: Paired kernel-sum ratios (TRT / candidate), all required rows
   :header-rows: 1

   * - Layout
     - Mode
     - Rows
     - Geometric mean
     - Minimum
     - Rows <= 1.0
   * - EP=8 (rank 3, 112 local experts)
     - Graph
     - 44
     - 1.228
     - 1.005 (T128 balanced)
     - 0
   * - EP=8
     - eager
     - 44
     - 1.234
     - 1.007 (T128 balanced)
     - 0
   * - MoE TP=8 (rank 0, shard 384)
     - Graph
     - 33
     - 1.222
     - 1.074 (T2048 hot)
     - 0
   * - MoE TP=8
     - eager
     - 33
     - 1.224
     - 1.076 (T2048 hot)
     - 0

Per-``T`` Graph-mode rows, EP=8 (kernel-sum milliseconds for balanced
routing, kernel-sum ratios per routing distribution, synchronized
graph-replay e2e ratio for balanced routing):

.. table::
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  remote-dom. ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   1     0.0266            0.0322           1.209           1.082      1.091        1.097              1.170
   2     0.0408            0.0500           1.224           1.074      1.081        1.185              1.198
   4     0.0600            0.0756           1.258           1.087      1.077        1.187              1.228
   8     0.1048            0.1234           1.178           1.310      1.318        1.318              1.165
   16    0.1848            0.2010           1.088           1.349      1.360        1.179              1.087
   128   0.6097            0.6138           1.007           1.024      1.308        1.010              1.011
   256   0.6048            0.6419           1.061           1.059      1.238        1.058              1.062
   512   0.6199            0.7004           1.130           1.071      1.279        1.143              1.127
   1024  0.6288            0.7635           1.214           1.109      1.378        1.230              1.207
   2048  0.8043            1.2421           1.544           1.486      1.682        1.560              1.531
   4096  0.8368            1.2473           1.490           1.453      1.990        1.556              1.480
   ====  ================  ===============  ==============  =========  ===========  =================  ==================

Per-``T`` Graph-mode rows, MoE TP=8:

.. table::
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  ==================
   1     0.0272            0.0339           1.249           1.269      1.263        1.160
   2     0.0417            0.0517           1.241           1.277      1.284        1.208
   4     0.0625            0.0750           1.199           1.231      1.264        1.171
   8     0.1083            0.1304           1.204           1.258      1.507        1.195
   16    0.1852            0.2189           1.182           1.208      1.483        1.176
   128   0.6067            0.6772           1.116           1.117      1.279        1.116
   256   0.6233            0.6921           1.110           1.108      1.145        1.105
   512   0.6576            0.7171           1.091           1.098      1.599        1.087
   1024  0.7108            0.7831           1.102           1.086      1.099        1.097
   2048  0.8146            0.8892           1.092           1.067      1.463        1.084
   4096  0.9297            1.1154           1.200           1.195      1.239        1.196
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
TP=8 T2048 empty 0.84, T4096 balanced 0.706).
Serving the deferred output from the dense path is the open follow-up for
those token counts.

.. table:: Deferred finalize, Graph mode, EP=8 (trtllm-gen do_finalize=False / candidate do_finalize=False)
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  remote-dom. ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   1     0.0270            0.0296           1.095           1.000      0.984        0.989              1.062
   2     0.0407            0.0468           1.148           1.006      1.008        1.118              1.129
   4     0.0612            0.0707           1.155           1.012      1.010        1.170              1.133
   8     0.1048            0.1199           1.144           1.226      1.251        1.267              1.128
   16    0.1858            0.1989           1.070           1.252      1.255        1.151              1.070
   128   0.6078            0.6069           0.998           0.995      1.122        1.002              0.999
   256   0.6074            0.6343           1.044           1.034      1.104        1.042              1.045
   512   0.6229            0.6859           1.101           1.037      1.011        1.125              1.099
   1024  0.6309            0.7414           1.175           1.058      1.013        1.188              1.170
   2048  1.0128            1.2043           1.189           0.988      1.467        1.905              1.186
   4096  1.4880            1.1740           0.789           0.682      1.266        1.158              0.792
   ====  ================  ===============  ==============  =========  ===========  =================  ==================

.. table:: Deferred finalize, Graph mode, MoE TP=8
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  ==================
   1     0.0265            0.0304           1.145           1.166      1.150        1.095
   2     0.0397            0.0476           1.201           1.200      1.162        1.151
   4     0.0595            0.0718           1.207           1.204      1.151        1.172
   8     0.1054            0.1269           1.204           1.263      1.432        1.178
   16    0.1869            0.2147           1.149           1.173      1.440        1.137
   128   0.6075            0.6658           1.096           1.100      1.383        1.092
   256   0.6245            0.6798           1.089           1.099      1.151        1.087
   512   0.6344            0.7017           1.106           1.108      1.710        1.100
   1024  0.6651            0.7416           1.115           1.107      1.117        1.109
   2048  0.7511            0.8082           1.076           1.046      0.835        1.074
   4096  1.3451            0.9507           0.707           0.684      0.634        0.710
   ====  ================  ===============  ==============  =========  ===========  ==================

Long prefill (optional rows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional T=8192/16384/32768 rows (finalized path, dense grouped GEMMs,
same paired method, Graph mode) are measured and reported: EP=8 geometric
mean 1.392 over 12 rows with T8192 balanced 0.789 / hot 0.824 and T16384
balanced 0.970 / remote-dominated 0.876 below one; MoE TP=8 geometric mean
1.062 over 9 rows with T8192 balanced 0.983 / hot 0.952 and T32768 empty
0.973 below one (the device-side tile choice, on by default for the shard,
moved the T8192 balanced/hot/empty rows from 0.967 / 0.953 / 1.173 to
0.983 / 0.952 / 1.201; T16384 reads 1.009 / 1.131 / 1.000 and T32768
1.127 / 1.225 / 0.973). The dense path picks its tactic from the B300
tables while the trtllm-gen arm autotunes per token count; the shard picks
its M tile on the device from the padded row counts, the expert-parallel
rank keeps the single-tile default until the unchosen launches can be
skipped (graph conditional nodes, the follow-up).

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
     - 9 (T=128 balanced/hot/remote-dominated; T=256 balanced/remote-dominated; T=512 balanced/remote-dominated; T=1024 balanced/remote-dominated)
     - 2.12
     - 5.47 (T=512 empty)
   * - MoE TP=8 rank 0
     - 42
     - 4 (T=128 balanced/hot; T=256 balanced/hot)
     - 2.02
     - 13.15 (T=128 empty)

Expert-parallel rank 3 (112 local experts, ``I_shard = 3072``):

   =====  ===========  =======  ================  =======  ==========  ======  =========
   T      routing      experts  floor µs (bound)  cand µs  cand/floor  TRT µs  TRT/floor
   =====  ===========  =======  ================  =======  ==========  ======  =========
   1      balanced     2        10 (mem)          27       2.62        32      3.17
   1      empty        1        5 (mem)           24       4.78        26      5.22
   1      hot          1        5 (mem)           24       4.78        26      5.18
   1      remote-dom.  1        5 (mem)           25       4.85        27      5.32
   2      balanced     4        20 (mem)          41       2.01        50      2.46
   2      empty        1        5 (mem)           24       4.72        26      5.11
   2      hot          1        5 (mem)           24       4.80        26      5.15
   2      remote-dom.  2        10 (mem)          26       2.61        31      3.09
   4      balanced     8        41 (mem)          60       1.48        76      1.86
   4      empty        1        5 (mem)           24       4.81        26      5.18
   4      hot          1        5 (mem)           24       4.71        26      5.12
   4      remote-dom.  4        20 (mem)          41       2.01        48      2.38
   8      balanced     16       81 (mem)          105      1.29        123     1.52
   8      empty        1        5 (mem)           24       4.78        32      6.30
   8      hot          1        5 (mem)           25       4.83        32      6.32
   8      remote-dom.  8        41 (mem)          60       1.48        79      1.95
   16     balanced     32       162 (mem)         185      1.14        201     1.24
   16     empty        1        5 (mem)           25       4.83        34      6.57
   16     hot          1        5 (mem)           25       4.92        34      6.64
   16     remote-dom.  16       81 (mem)          104      1.28        122     1.51
   128    balanced     112      569 (mem)         610      1.07        614     1.08
   128    empty        1        5 (mem)           28       5.16        37      6.74
   128    hot          112      569 (mem)         622      1.09        636     1.12
   128    remote-dom.  112      568 (mem)         604      1.06        610     1.07
   256    balanced     112      569 (mem)         605      1.06        642     1.13
   256    empty        1        6 (mem)           31       5.24        38      6.48
   256    hot          112      569 (mem)         630      1.11        668     1.17
   256    remote-dom.  112      569 (mem)         603      1.06        637     1.12
   512    balanced     112      570 (mem)         620      1.09        700     1.23
   512    empty        1        7 (mem)           37       5.47        47      6.99
   512    hot          112      571 (mem)         681      1.19        729     1.28
   512    remote-dom.  112      570 (mem)         616      1.08        704     1.23
   1024   balanced     112      573 (mem)         629      1.10        763     1.33
   1024   empty        1        8 (mem)           39       4.65        53      6.40
   1024   hot          112      574 (mem)         753      1.31        835     1.45
   1024   remote-dom.  112      572 (mem)         622      1.09        766     1.34
   2048   balanced     112      578 (mem)         804      1.39        1242    2.15
   2048   empty        1        12 (mem)          53       4.61        90      7.76
   2048   hot          112      580 (mem)         872      1.50        1296    2.24
   2048   remote-dom.  112      576 (mem)         799      1.39        1246    2.16
   4096   balanced     112      588 (mem)         837      1.42        1247    2.12
   4096   empty        1        18 (mem)          62       3.42        123     6.81
   4096   hot          112      592 (mem)         992      1.68        1441    2.44
   4096   remote-dom.  112      585 (mem)         818      1.40        1272    2.18
   8192   balanced     112      614 (mma)         1497     2.44        1181    1.92
   8192   empty        1        31 (mem)          77       2.50        189     6.09
   8192   hot          112      877 (mma)         1848     2.11        1524    1.74
   8192   remote-dom.  112      601 (mem)         859      1.43        1302    2.17
   16384  balanced     112      1229 (mma)        2343     1.91        2273    1.85
   16384  empty        1        57 (mem)          81       1.42        292     5.13
   16384  hot          112      1759 (mma)        3089     1.76        3262    1.85
   16384  remote-dom.  112      634 (mem)         1494     2.36        1309    2.06
   32768  balanced     112      2457 (mma)        4166     1.70        4739    1.93
   32768  empty        1        109 (mem)         130      1.20        536     4.93
   32768  hot          112      3514 (mma)        5966     1.70        6247    1.78
   32768  remote-dom.  112      1229 (mma)        2314     1.88        3211    2.61
   =====  ===========  =======  ================  =======  ==========  ======  =========

MoE tensor-parallel rank 0 (896 experts, ``I_shard = 384``):

   =====  ========  =======  ================  =======  ==========  ======  =========
   T      routing   experts  floor µs (bound)  cand µs  cand/floor  TRT µs  TRT/floor
   =====  ========  =======  ================  =======  ==========  ======  =========
   1      balanced  16       10 (mem)          27       2.68        34      3.34
   1      empty     16       10 (mem)          27       2.64        34      3.33
   1      hot       16       10 (mem)          27       2.64        34      3.35
   2      balanced  32       20 (mem)          42       2.05        52      2.55
   2      empty     16       10 (mem)          27       2.64        34      3.38
   2      hot       31       20 (mem)          41       2.07        52      2.64
   4      balanced  64       41 (mem)          63       1.54        75      1.85
   4      empty     16       10 (mem)          27       2.68        34      3.39
   4      hot       61       39 (mem)          60       1.55        74      1.90
   8      balanced  128      81 (mem)          108      1.33        130     1.61
   8      empty     16       10 (mem)          27       2.69        41      4.05
   8      hot       121      77 (mem)          98       1.28        124     1.61
   16     balanced  256      162 (mem)         185      1.14        219     1.35
   16     empty     16       10 (mem)          37       3.61        55      5.36
   16     hot       241      153 (mem)         172      1.12        207     1.36
   128    balanced  896      569 (mem)         607      1.07        677     1.19
   128    empty     16       11 (mem)          142      13.15       181     16.82
   128    hot       896      569 (mem)         608      1.07        679     1.19
   256    balanced  896      569 (mem)         623      1.10        692     1.22
   256    empty     16       19 (mma)          167      8.70        191     9.96
   256    hot       896      569 (mem)         626      1.10        693     1.22
   512    balanced  896      570 (mem)         658      1.15        717     1.26
   512    empty     16       38 (mma)          200      5.22        320     8.34
   512    hot       896      570 (mem)         657      1.15        722     1.27
   1024   balanced  896      573 (mem)         711      1.24        783     1.37
   1024   empty     16       77 (mma)          363      4.73        399     5.20
   1024   hot       896      573 (mem)         720      1.26        782     1.36
   2048   balanced  896      578 (mem)         815      1.41        889     1.54
   2048   empty     16       154 (mma)         299      1.95        438     2.85
   2048   hot       896      578 (mem)         823      1.42        878     1.52
   4096   balanced  896      588 (mem)         930      1.58        1115    1.90
   4096   empty     16       307 (mma)         543      1.77        673     2.19
   4096   hot       896      588 (mem)         935      1.59        1117    1.90
   8192   balanced  896      614 (mma)         1691     2.75        1662    2.70
   8192   empty     16       614 (mma)         1020     1.66        1225    1.99
   8192   hot       896      614 (mma)         1730     2.82        1646    2.68
   16384  balanced  896      1229 (mma)        3195     2.60        3222    2.62
   16384  empty     16       1229 (mma)        2452     2.00        2452    2.00
   16384  hot       896      1229 (mma)        2844     2.31        3216    2.62
   32768  balanced  896      2457 (mma)        5802     2.36        6542    2.66
   32768  empty     16       2457 (mma)        5178     2.11        5040    2.05
   32768  hot       896      2457 (mma)        5291     2.15        6482    2.64
   =====  ========  =======  ================  =======  ==========  ======  =========

**Where the gaps come from (measured on the same node).**

*Decode, T <= 16.* The floor is the touched weight bytes at 6.92 TB/s
(5 us for one expert on the wide shard, 10 us for 16 shard experts). A
T=1 hot row costs 24 us: fused routing 2.4 us, GEMM1 13.4 us (22 MB of
weights), GEMM2 8.5 us (13 MB), attributed per kernel on a second B300
node of the same type (24.3 us there). The streaming ramp puts the same bytes at
7.0 us and 5.2 us even at the swap GEMM's 48-CTA grid, so about 12 us of
the 24 us is the small-transfer ramp plus the three launches, which no
kernel organisation removes; that part is structural (5 us floor against
~14 us reachable). The remaining ~10 us is the swap-AB kernels streaming at
~37 GB/s per SM: with 48 CTAs they behave like a stream with 32 KB in flight
(12.6 us for 22 MB, 8.7 us for 13 MB). Doubling or halving the K-blocks per
stage and the stage count moved this by 1-3% (the smem budget fixes the
bytes in flight per SM), and a split-K variant that puts more CTAs on each
weight tile was slower in two implementations (its scheduler work exceeded
the streaming gain). More CTAs per tile therefore remains the open lever for
decode; the ramp part is not.

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
routing kernel now performs), so it is not the default; dense tiles over
the default 16-row group layout are the open item for T=128..1024. This
row class is open, not structural.

*Prefill on the dense path, T >= 2048.* The dense GEMM1 runs at about 3.5
PFLOPS with L2-resident weights (95-99% of the peak measured on the two
nodes used), so the gap is not the MMA rate. It is the 128-row M tile: at T=2048 balanced the 112 local
experts hold 36.6 rows each (3.5 x padded compute), at T=4096 73 rows
(1.75 x), at T=8192 146 rows (two tiles, 1.75 x for the second). The swap
form avoids padding but re-streams the weights once per 32-row group,
which at T=2048 already costs more bytes than the dense form's padding
costs FLOPs. Both alternatives are bounded by the same measured constants,
so the padded-compute time of the dense form (rows padded to the tile x
6 * H * I_shard / 3618 TFLOPS) is the reachable floor of this row class
until a mixed-tile schedule exists. The device-side tile choice above
takes the rows whose padded totals are equal under both tiles to the
two-CTA M256 tile (GEMM1 and GEMM2 both stream each weight tile once):
on the shard T=8192 balanced/empty/hot fall 8.5/5.7/3.7% and T=16384 and
T=32768 empty 3.7/2.6% (interleaved A/B/A/B, 30 replays); on the wide rank
T=8192 balanced/hot fall 17/14% and T=16384 remote-dominated 16% when the
rule is enabled there. Rows whose padded totals differ keep M128 and its
padding; the wide rank's remote-dominated T=8192 row (73-row experts,
M256 would pad 2 x) stays at the padded-compute floor of M128.

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
token row is a DRAM read-modify-write. Walking the tiles M-fastest
(``MXFP4_GEMM2_RASTER_M=1``, groups of ``MXFP4_GEMM2_SWIZZLE`` = 4 N
tiles so each A tile is still reused four times) keeps one 256-column
slab of the output L2-resident while every M tile passes and cuts the
shard's GEMM2 by 15-25%: T=8192 balanced -6%, T=16384 balanced/empty
-10/-13%, T=32768 balanced/empty -2/-5% on the row (interleaved A/B/A/B,
30 replays), neutral at T=4096. It costs the hot routing (one expert
holding every token plus 895 small ones) +1.2% at T=32768 and up to +3.4%
at T=16384, and the wide rank (K=3072) 3-5%, so it is opt-in; the raster
has to follow the routing on the device (a second GEMM2 variant chosen by
the routing kernel, affordable on the shard's >= 1 ms rows) and is the
open lever for the shard's balanced and empty rows at T >= 8192.

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
