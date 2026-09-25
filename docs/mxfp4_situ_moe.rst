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
clearing and the local expert histogram/prefix/scatter (on the dense path
only when PDL is disabled).
The default decode forward then launches GEMM1 and GEMM2 with finalize:
three device-kernel launches in total. Expert IDs must be unique within each
token and lie in ``[0, num_experts)``, as in standard top-k routing.
On the dense path, PDL-enabled decode retains separate
native sorting. The fused sort covers every T with ``T * top_k <= 8192``
(``FUSED_ROUTE_MAX_ROUTES``; T <= 512 for Kimi K3) and ``T * top_k <=
16384`` (T <= 1024) on ranks with at most 128 local experts or where the
split form runs (see "Kernel selection" below); above that the same conversion + output-clear kernel
runs without the fused sort (it is grid-strided over any token count),
followed by the native ``moe_sort``; PDL-enabled prefill keeps the torch
conversion and separate memset. The split form of the MoE-TP shard
(T = 128..1024, below) adds two launches to the chain.

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
resident before the working CTAs leave their SMs).

Routing preprocessing (expert histogram, permutation, per-row scale, output
clear) runs as one fused CuTe kernel whenever ``T * top_k <= 8192``
(``FUSED_ROUTE_MAX_ROUTES``, T <= 512 for Kimi K3) or, on a rank with at
most 128 local experts (``MXFP4_FUSED_ROUTE_LARGE_MAX_LOCAL_EXPERTS``),
``T * top_k <= 16384`` (T <= 1024) -- also on the MoE-TP shard where the
split form needs the fused layout -- and as the shared ``moe_sort`` path
above. The fused kernel is a single 1024-thread block (or a grid that also
clears the output when the swap epilogue reduce-adds into it) and replaces
the unpack/conversion kernels, ``moe_sort`` and the memset (about 10.6 us
at T=128) with roughly 3 us; the generic clear kernel uses 16-byte stores.
Its cost grows with the route count: the histogram issues eight routes per
thread before ranking any of them (one block's 32 warps are latency-bound
otherwise), and above 8192 rows the inverse permutation (row -> route) is
scattered in shared memory (int16) and copied out coalesced, because
scattered 4-byte global stores cost one 32-byte L1 sector each: 16384 of
them took 13.6 us of a 19.9 us kernel at T=1024 on the shard, 8.7 us of
15.0 us staged (per-phase globaltimer stamps, B300; below 8192 rows the
direct stores are as fast, 1.2 against 1.5 us at 2048 rows, and the
8192-route variant keeps them). Shard, balanced routing, T=128 / 512 /
1024: histogram 2.1 / 2.9 / 5.4 us, scan 1.8 / 1.8 / 0.5 us, scatter 1.2
/ 6.3 / 8.7 us, whole kernel 5.4 / 11.3 / 15.0 us; EP8 rank (112 local
experts): 3.6 / 4.9 / 8.3 us. On the EP8 rank the larger bound wins every
row (T=512 ``empty`` 38.0 -> 32.5 us, T=1024 39.6 -> 37.8 us,
``balanced``/``hot`` 0.993-0.999 x). Up to one route per thread the
histogram keeps its plain loop (the clamped duplicate loads cost 0.1 us at
T=1); the decode rows of the EP8 rank still read 0.2-0.3 us (1 %) above
the previous revision in a same-GPU pair (T=1: 24.75 -> 25.0, 22.65 ->
22.85, 22.9 -> 23.15 us for balanced / hot / remote_dominated), a residual
of the kernel's larger parameter block and int16 scratch that is not
attributed further; they stay 4-5 % under the phase-2 baseline.

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
     - 1.293
     - 1.036 (T128 remote-dominated)
     - 0
   * - EP=8
     - Graph, kernel sum
     - 44
     - 1.212
     - 1.027 (T128 remote-dominated)
     - 0
   * - EP=8
     - eager, kernel sum
     - 44
     - 1.240
     - 1.032 (T128 remote-dominated)
     - 0
   * - MoE TP=8 (rank 0, shard 384)
     - Graph, GPU span
     - 33
     - 1.249
     - 1.080 (T1024 hot)
     - 0
   * - MoE TP=8
     - Graph, kernel sum
     - 33
     - 1.193
     - 1.037 (T2048 hot)
     - 0
   * - MoE TP=8
     - eager, kernel sum
     - 33
     - 1.211
     - 1.040 (T2048 hot)
     - 0

Per-``T`` Graph-mode rows, EP=8 (kernel-sum milliseconds for balanced
routing, kernel-sum ratios per routing distribution, synchronized
graph-replay e2e ratio for balanced routing):

.. table::
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  remote-dom. ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  =================  ==================
   1     0.0276            0.0325           1.177           1.051      1.066        1.066              1.271
   2     0.0426            0.0504           1.181           1.039      1.047        1.170              1.223
   4     0.0615            0.0732           1.190           1.065      1.073        1.208              1.218
   8     0.1051            0.1231           1.172           1.325      1.326        1.283              1.193
   16    0.1856            0.2031           1.095           1.257      1.284        1.180              1.118
   128   0.5969            0.6144           1.029           1.037      1.129        1.029              1.039
   256   0.6043            0.6425           1.063           1.061      1.208        1.057              1.069
   512   0.6163            0.6976           1.132           1.069      1.303        1.147              1.141
   1024  0.6306            0.7622           1.209           1.112      1.269        1.222              1.215
   2048  0.8075            1.2410           1.537           1.476      1.666        1.544              1.525
   4096  0.8411            1.2664           1.506           1.489      2.016        1.552              1.495
   ====  ================  ===============  ==============  =========  ===========  =================  ==================

Per-``T`` Graph-mode rows, MoE TP=8:

.. table::
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  ==================
   1     0.0275            0.0337           1.228           1.234      1.217        1.275
   2     0.0431            0.0533           1.238           1.261      1.214        1.308
   4     0.0633            0.0764           1.207           1.231      1.234        1.229
   8     0.1093            0.1313           1.202           1.250      1.435        1.225
   16    0.1861            0.2188           1.176           1.198      1.445        1.187
   128   0.6173            0.6758           1.095           1.093      2.696        1.103
   256   0.6360            0.6967           1.096           1.097      2.187        1.101
   512   0.6719            0.7222           1.075           1.082      2.663        1.092
   1024  0.7282            0.7877           1.082           1.068      2.021        1.095
   2048  0.8319            0.8903           1.070           1.027      1.341        1.083
   4096  0.9292            1.1105           1.195           1.199      1.236        1.191
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
   1     0.0254            0.0306           1.203           1.078      1.072        1.091              1.150
   2     0.0393            0.0473           1.204           1.104      1.113        1.206              1.171
   4     0.0589            0.0713           1.211           1.094      1.118        1.224              1.180
   8     0.1026            0.1213           1.182           1.342      1.328        1.332              1.169
   16    0.1840            0.1991           1.082           1.322      1.343        1.171              1.081
   128   0.6090            0.6073           0.997           1.000      1.233        1.002              0.999
   256   0.6042            0.6327           1.047           1.042      1.143        1.047              1.048
   512   0.6165            0.6872           1.115           1.048      1.201        1.132              1.115
   1024  0.6282            0.7436           1.184           1.074      1.022        1.201              1.179
   2048  1.0131            1.2124           1.197           0.994      1.546        1.919              1.195
   4096  1.4882            1.1775           0.791           0.680      1.318        1.163              0.794
   ====  ================  ===============  ==============  =========  ===========  =================  ==================

.. table:: Deferred finalize, Graph mode, MoE TP=8
   :widths: auto

   ====  ================  ===============  ==============  =========  ===========  ==================
   T     balanced cand ms  balanced TRT ms  balanced ratio  hot ratio  empty ratio  balanced e2e ratio
   ====  ================  ===============  ==============  =========  ===========  ==================
   1     0.0246            0.0310           1.258           1.212      1.221        1.176
   2     0.0382            0.0481           1.259           1.234      1.365        1.222
   4     0.0578            0.0714           1.234           1.249      1.240        1.194
   8     0.1037            0.1264           1.219           1.283      1.546        1.201
   16    0.1847            0.2147           1.162           1.184      1.523        1.156
   128   0.6059            0.6570           1.084           1.092      1.300        1.082
   256   0.6222            0.6717           1.080           1.085      1.166        1.081
   512   0.6321            0.6929           1.096           1.105      1.757        1.093
   1024  0.6642            0.7362           1.108           1.095      1.118        1.105
   2048  0.7511            0.8084           1.076           1.047      0.834        1.074
   4096  1.3406            0.9494           0.708           0.688      0.641        0.712
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
so the earlier rows are comparable.

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
     - 2.08
     - 5.07 (T=256 empty)
   * - MoE TP=8 rank 0
     - 42
     - 2 (T=128 balanced/hot)
     - 1.88
     - 4.42 (T=128 empty)

Expert-parallel rank 3 (112 local experts, ``I_shard = 3072``):

   =====  ===========  =======  ================  =======  ==========  ======  =========
   T      routing      experts  floor µs (bound)  cand µs  cand/floor  TRT µs  TRT/floor
   =====  ===========  =======  ================  =======  ==========  ======  =========
   1      balanced     2        10 (mem)          25       2.48        34      3.34
   1      empty        1        5 (mem)           23       4.46        28      5.45
   1      hot          1        5 (mem)           23       4.47        28      5.45
   1      remote-dom.  1        5 (mem)           23       4.50        28      5.51
   2      balanced     4        20 (mem)          40       1.97        52      2.54
   2      empty        1        5 (mem)           23       4.52        28      5.43
   2      hot          1        5 (mem)           23       4.51        28      5.45
   2      remote-dom.  2        10 (mem)          25       2.49        34      3.32
   4      balanced     8        41 (mem)          59       1.46        74      1.82
   4      empty        1        5 (mem)           23       4.52        28      5.53
   4      hot          1        5 (mem)           23       4.56        29      5.63
   4      remote-dom.  4        20 (mem)          39       1.93        51      2.52
   8      balanced     16       81 (mem)          103      1.27        124     1.53
   8      empty        1        5 (mem)           23       4.44        34      6.76
   8      hot          1        5 (mem)           23       4.50        35      6.78
   8      remote-dom.  8        41 (mem)          59       1.46        80      1.96
   16     balanced     32       162 (mem)         183      1.12        204     1.26
   16     empty        1        5 (mem)           23       4.53        35      6.83
   16     hot          1        5 (mem)           24       4.63        35      6.88
   16     remote-dom.  16       81 (mem)          103      1.27        126     1.55
   128    balanced     112      569 (mem)         593      1.04        615     1.08
   128    empty        1        5 (mem)           28       5.04        38      6.96
   128    hot          112      569 (mem)         609      1.07        637     1.12
   128    remote-dom.  112      569 (mem)         589      1.03        610     1.07
   256    balanced     112      570 (mem)         600      1.05        644     1.13
   256    empty        1        6 (mem)           30       5.07        42      7.21
   256    hot          112      570 (mem)         627      1.10        670     1.18
   256    remote-dom.  112      569 (mem)         599      1.05        638     1.12
   512    balanced     112      571 (mem)         613      1.07        699     1.22
   512    empty        1        7 (mem)           33       4.90        49      7.31
   512    hot          112      571 (mem)         676      1.18        728     1.27
   512    remote-dom.  112      570 (mem)         609      1.07        704     1.23
   1024   balanced     112      574 (mem)         627      1.09        763     1.33
   1024   empty        1        8 (mem)           39       4.64        55      6.64
   1024   hot          112      574 (mem)         751      1.31        839     1.46
   1024   remote-dom.  112      573 (mem)         621      1.08        764     1.33
   2048   balanced     112      579 (mem)         809      1.40        1242    2.15
   2048   empty        1        12 (mem)          55       4.73        90      7.81
   2048   hot          112      580 (mem)         878      1.51        1295    2.23
   2048   remote-dom.  112      577 (mem)         803      1.39        1239    2.15
   4096   balanced     112      589 (mem)         842      1.43        1267    2.15
   4096   empty        1        18 (mem)          62       3.46        125     6.91
   4096   hot          112      592 (mem)         1001     1.69        1489    2.52
   4096   remote-dom.  112      585 (mem)         822      1.41        1276    2.18
   8192   balanced     112      609 (mem)         1487     2.44        1186    1.95
   8192   empty        1        31 (mem)          78       2.53        190     6.13
   8192   hot          112      830 (mma)         1789     2.16        1498    1.81
   8192   remote-dom.  112      602 (mem)         857      1.42        1300    2.16
   16384  balanced     112      1162 (mma)        2255     1.94        2255    1.94
   16384  empty        1        57 (mem)          82       1.44        293     5.15
   16384  hot          112      1664 (mma)        2891     1.74        3063    1.84
   16384  remote-dom.  112      635 (mem)         1483     2.34        1277    2.01
   32768  balanced     112      2324 (mma)        4067     1.75        4685    2.02
   32768  empty        1        109 (mem)         132      1.21        537     4.95
   32768  hot          112      3323 (mma)        5527     1.66        6133    1.85
   32768  remote-dom.  112      1162 (mma)        2356     2.03        3080    2.65
   =====  ===========  =======  ================  =======  ==========  ======  =========

MoE tensor-parallel rank 0 (896 experts, ``I_shard = 384``):

   =====  ========  =======  ================  =======  ==========  ======  =========
   T      routing   experts  floor µs (bound)  cand µs  cand/floor  TRT µs  TRT/floor
   =====  ========  =======  ================  =======  ==========  ======  =========
   1      balanced  16       10 (mem)          25       2.49        35      3.45
   1      empty     16       10 (mem)          25       2.48        35      3.42
   1      hot       16       10 (mem)          25       2.48        35      3.46
   2      balanced  32       20 (mem)          40       1.98        54      2.68
   2      empty     16       10 (mem)          26       2.52        35      3.49
   2      hot       31       20 (mem)          39       1.98        54      2.72
   4      balanced  64       41 (mem)          61       1.49        77      1.90
   4      empty     16       10 (mem)          25       2.50        35      3.48
   4      hot       61       39 (mem)          58       1.50        75      1.94
   8      balanced  128      81 (mem)          107      1.32        132     1.63
   8      empty     16       10 (mem)          26       2.53        42      4.11
   8      hot       121      77 (mem)          97       1.26        125     1.63
   16     balanced  256      162 (mem)         183      1.13        220     1.35
   16     empty     16       10 (mem)          35       3.46        56      5.51
   16     hot       241      153 (mem)         171      1.12        209     1.37
   128    balanced  896      569 (mem)         612      1.07        677     1.19
   128    empty     16       11 (mem)          48       4.42        196     18.17
   128    hot       896      569 (mem)         614      1.08        679     1.19
   256    balanced  896      570 (mem)         630      1.11        698     1.22
   256    empty     16       18 (mma)          69       3.77        194     10.67
   256    hot       896      570 (mem)         629      1.10        699     1.23
   512    balanced  896      571 (mem)         663      1.16        723     1.27
   512    empty     16       36 (mma)          100      2.76        320     8.81
   512    hot       896      571 (mem)         662      1.16        726     1.27
   1024   balanced  896      574 (mem)         720      1.25        789     1.38
   1024   empty     16       73 (mma)          170      2.33        403     5.55
   1024   hot       896      574 (mem)         728      1.27        787     1.37
   2048   balanced  896      579 (mem)         818      1.41        891     1.54
   2048   empty     16       145 (mma)         302      2.08        439     3.02
   2048   hot       896      579 (mem)         823      1.42        880     1.52
   4096   balanced  896      589 (mem)         930      1.58        1112    1.89
   4096   empty     16       291 (mma)         545      1.88        674     2.32
   4096   hot       896      589 (mem)         936      1.59        1121    1.90
   8192   balanced  896      609 (mem)         1727     2.83        1681    2.76
   8192   empty     16       581 (mma)         1035     1.78        1277    2.20
   8192   hot       896      609 (mem)         1733     2.84        1678    2.75
   16384  balanced  896      1162 (mma)        3217     2.77        3165    2.72
   16384  empty     16       1162 (mma)        2514     2.16        2509    2.16
   16384  hot       896      1162 (mma)        3038     2.61        3145    2.71
   32768  balanced  896      2324 (mma)        5845     2.52        6557    2.82
   32768  empty     16       2324 (mma)        5382     2.32        5196    2.24
   32768  hot       896      2324 (mma)        5345     2.30        6560    2.82
   =====  ========  =======  ================  =======  ==========  ======  =========

**Reachable floor per row.** The floor above charges each row the
bytes it must move at the HBM figure and its FLOPs at the MMA peak. A
second, higher figure follows from five more constants measured on the
same node, and every row of the tables below is either at or under
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
* the per-SM load ring: each GEMM keeps only ``S`` K-stages of operands
  in flight per SM (4 stages of 50688 B for the dense gather GEMM1 and of
  42496 B for the dense finalize GEMM2, 5 of 36 KB for the swap GEMM1, 11
  for the swap GEMM2; the FP4 operand occupies 8-bit containers in shared
  memory for ``kind::mxf8f6f4``, so a fifth dense stage does not fit the
  227 KB budget), and a K-stage costs the fill of that stage through the
  SM's async-copy unit plus one loaded round trip divided by the depth:
  ``prologue 2.2 us + ceil(tiles / 148) x (K / K_stage) x (fill + 0.93 us
  / S)`` with fill 0.266 / 0.160 / 0.193 / 0.121 us for the four kernels.
  A stage-depth sweep on the dense kernels (capping the ring at 3 and 2
  stages, EP=8 T=2048 balanced, same GPU) gives GEMM1 509 / 572 / 747 us
  and GEMM2 274 / 314 / 436 us at 4 / 3 / 2 stages, i.e. ``a + L / S``
  with ``L`` = 0.93 us per stage in both kernels, and the same ``L`` with
  the kernels' own fill terms reproduces the swap GEMM1 / GEMM2 at T=512
  (393 / 212 us) within 3%.

The reachable floor of a row is the smallest, over the swap-AB form at the
planner's row tile and the dense form at the 128- and 256-row tiles (128 x
128 and 128 x 256 GEMM tiles), of the routing kernel (2.6 us) plus the two
GEMMs, each taking the largest of its five terms, plus one launch gap per
dependent kernel. With the load-ring term 45 of the 56 expert-parallel rows
and 34 of the 42 MoE-TP rows sit within 1.10 x of their reachable floor
(geometric mean candidate / reachable 1.04 on both layouts); the binding
term of nearly every row is the ring, i.e. the kernels are bound by the
operand bytes each SM keeps in flight, not by HBM bandwidth or MMA rate. It is a lower bound of what any schedule built from these
kernels reaches on this node, not a promise that a schedule exists; the
tables mark rows at or under 1.10 x of it.

Reachable floor, expert-parallel rank 3 (same rows and candidate as above):

   =====  ===========  ========  ===================  =======  ==========  ==============  ======
   T      routing      floor µs  reachable µs (form)  cand µs  cand/floor  cand/reachable  TRT µs
   =====  ===========  ========  ===================  =======  ==========  ==============  ======
   1      balanced     10        26 (swap)            25       2.48        0.98            34
   1      empty        5         26 (swap)            23       4.46        0.88            28
   1      hot          5         26 (swap)            23       4.47        0.88            28
   1      remote-dom.  5         26 (swap)            23       4.50        0.89            28
   2      balanced     20        41 (swap)            40       1.97        0.97            52
   2      empty        5         26 (swap)            23       4.52        0.89            28
   2      hot          5         26 (swap)            23       4.51        0.89            28
   2      remote-dom.  10        26 (swap)            25       2.49        0.98            34
   4      balanced     41        62 (swap)            59       1.46        0.96            74
   4      empty        5         26 (swap)            23       4.52        0.89            28
   4      hot          5         26 (swap)            23       4.56        0.90            29
   4      remote-dom.  20        41 (swap)            39       1.93        0.95            51
   8      balanced     81        108 (swap)           103      1.27        0.95            124
   8      empty        5         26 (swap)            23       4.44        0.88            34
   8      hot          5         26 (swap)            23       4.50        0.89            35
   8      remote-dom.  41        62 (swap)            59       1.46        0.96            80
   16     balanced     162       191 (swap)           183      1.12        0.96            204
   16     empty        5         26 (swap)            23       4.53        0.90            35
   16     hot          5         26 (swap)            24       4.63        0.92            35
   16     remote-dom.  81        108 (swap)           103      1.27        0.95            126
   128    balanced     569       615 (swap)           593      1.04        0.96            615
   128    empty        5         26 (swap)            28       5.04        1.07            38
   128    hot          569       651 (swap)           609      1.07        0.94            637
   128    remote-dom.  569       615 (swap)           589      1.03        0.96            610
   256    balanced     570       615 (swap)           600      1.05        0.98            644
   256    empty        6         26 (swap)            30       5.07        1.16            42
   256    hot          570       651 (swap)           627      1.10        0.96            670
   256    remote-dom.  569       615 (swap)           599      1.05        0.97            638
   512    balanced     571       615 (swap)           613      1.07        1.00            699
   512    empty        7         26 (swap)            33       4.90        1.27            49
   512    hot          571       698 (swap)           676      1.18        0.97            728
   512    remote-dom.  570       615 (swap)           609      1.07        0.99            704
   1024   balanced     574       615 (swap)           627      1.09        1.02            763
   1024   empty        8         26 (swap)            39       4.64        1.48            55
   1024   hot          574       780 (swap)           751      1.31        0.96            839
   1024   remote-dom.  573       615 (swap)           621      1.08        1.01            764
   2048   balanced     579       748 (dense128)       809      1.40        1.08            1242
   2048   empty        12        29 (swap)            55       4.73        1.89            90
   2048   hot          580       832 (dense128)       878      1.51        1.05            1295
   2048   remote-dom.  577       615 (swap)           803      1.39        1.31            1239
   4096   balanced     589       748 (dense128)       842      1.43        1.13            1267
   4096   empty        18        35 (swap)            62       3.46        1.81            125
   4096   hot          592       944 (dense128)       1001     1.69        1.06            1489
   4096   remote-dom.  585       748 (dense128)       822      1.41        1.10            1276
   8192   balanced     609       1448 (dense128)      1487     2.44        1.03            1186
   8192   empty        31        54 (swap)            78       2.53        1.46            190
   8192   hot          830       1840 (dense128)      1789     2.16        0.97            1498
   8192   remote-dom.  602       754 (dense128)       857      1.42        1.14            1300
   16384  balanced     1162      2148 (dense128)      2255     1.94        1.05            2255
   16384  empty        57        77 (dense128)        82       1.44        1.06            293
   16384  hot          1664      2933 (dense128)      2891     1.74        0.99            3063
   16384  remote-dom.  635       1448 (dense128)      1483     2.34        1.02            1277
   32768  balanced     2324      3549 (dense128)      4067     1.75        1.15            4685
   32768  empty        109       112 (dense128)       132      1.21        1.18            537
   32768  hot          3323      5146 (dense128)      5527     1.66        1.07            6133
   32768  remote-dom.  1162      2148 (dense128)      2356     2.03        1.10            3080
   =====  ===========  ========  ===================  =======  ==========  ==============  ======

Reachable floor, MoE tensor-parallel rank 0:

   =====  ========  ========  ===================  =======  ==========  ==============  ======
   T      routing   floor µs  reachable µs (form)  cand µs  cand/floor  cand/reachable  TRT µs
   =====  ========  ========  ===================  =======  ==========  ==============  ======
   1      balanced  10        26 (swap)            25       2.49        0.99            35
   1      empty     10        26 (swap)            25       2.48        0.99            35
   1      hot       10        26 (swap)            25       2.48        0.99            35
   2      balanced  20        41 (swap)            40       1.98        0.99            54
   2      empty     10        26 (swap)            26       2.52        1.00            35
   2      hot       20        40 (swap)            39       1.98        0.96            54
   4      balanced  41        59 (swap)            61       1.49        1.03            77
   4      empty     10        26 (swap)            25       2.50        1.00            35
   4      hot       39        58 (swap)            58       1.50        0.99            75
   8      balanced  81        104 (swap)           107      1.32        1.02            132
   8      empty     10        26 (swap)            26       2.53        1.01            42
   8      hot       77        94 (swap)            97       1.26        1.03            125
   16     balanced  162       187 (swap)           183      1.13        0.98            220
   16     empty     10        39 (swap)            35       3.46        0.90            56
   16     hot       153       173 (swap)           171      1.12        0.99            209
   128    balanced  569       612 (swap)           612      1.07        1.00            677
   128    empty     11        45 (dense128)        48       4.42        1.07            196
   128    hot       569       616 (swap)           614      1.08        1.00            679
   256    balanced  570       612 (swap)           630      1.11        1.03            698
   256    empty     18        53 (dense128)        69       3.77        1.29            194
   256    hot       570       616 (swap)           629      1.10        1.02            699
   512    balanced  571       612 (swap)           663      1.16        1.08            723
   512    empty     36        98 (dense128)        100      2.76        1.02            320
   512    hot       571       616 (swap)           662      1.16        1.08            726
   1024   balanced  574       612 (swap)           720      1.25        1.17            789
   1024   empty     73        161 (dense128)       170      2.33        1.05            403
   1024   hot       574       630 (swap)           728      1.27        1.16            787
   2048   balanced  579       612 (swap)           818      1.41        1.34            891
   2048   empty     145       314 (dense128)       302      2.08        0.96            439
   2048   hot       579       630 (swap)           823      1.42        1.31            880
   4096   balanced  589       980 (dense128)       930      1.58        0.95            1112
   4096   empty     291       591 (dense128)       545      1.88        0.92            674
   4096   hot       589       995 (dense128)       936      1.59        0.94            1121
   8192   balanced  609       1795 (swap)          1727     2.83        0.96            1681
   8192   empty     581       1147 (dense128)      1035     1.78        0.90            1277
   8192   hot       609       1887 (swap)          1733     2.84        0.92            1678
   16384  balanced  1162      2868 (dense128)      3217     2.77        1.12            3165
   16384  empty     1162      2286 (dense128)      2514     2.16        1.10            2509
   16384  hot       1162      3013 (dense128)      3038     2.61        1.01            3145
   32768  balanced  2324      4759 (dense128)      5845     2.52        1.23            6557
   32768  empty     2324      4564 (dense128)      5382     2.32        1.18            5196
   32768  hot       2324      5019 (dense128)      5345     2.30        1.06            6560
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
which is the two-stage form above. The remaining rows above 1.10 x of the
reachable floor are the ``empty`` routings from T=256 (EP=8, 30-132 us
chains bound by the routing kernel and the swap GEMM1's per-tile latency),
the shard's T=1024 / 2048 balanced and hot rows (32-row swap groups
re-read each expert's weights from L2 once per group; the 64-row form
measured slower at T=1024 in an earlier revision), EP=8 T=2048
remote-dominated (1.31 x, the policy conflict above), T=4096 balanced
(1.13), T=8192 remote-dominated (1.14) and the T=32768 balanced rows of
both layouts (1.15 / 1.23); each is listed with its binding term in the
tables.

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
