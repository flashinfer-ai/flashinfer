Planned native MXFP4 SiTU MoE
============================

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
native sorting. Prefill conversion, sorting and clearing are unchanged.

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
host shape metadata, with the conservative Blackwell tactic as the fallback.
It never inspects device routing on the host or autotunes during execution.
``mxfp4_moe_capability`` checks architecture, quantization, activation,
dimensions, top-k and the explicit parallel metadata before a plan is
constructed. Its result reports ``supported``, ``reason``, ``cuda_graph``
(true for every supported configuration in both parallel modes) and
``layout``, the resolved ``Mxfp4MoERankLayout`` with the mode, local expert
interval and intermediate shard.

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
ordinary numerical/API checks do not establish a new sanitizer pass.

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
