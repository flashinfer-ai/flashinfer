Planned native MXFP4 SiTU MoE
=============================

The CuTe DSL planned runner targets SM100 and SM103, including NVIDIA B300.
It consumes native packed E2M1 weights and UE8M0 scales with block size 32,
FP8 E4M3 activations and BF16 output. Quantization is explicit metadata.

.. code-block:: python

   import torch

   from flashinfer.fused_moe.prepare import (
       prepare_cute_dsl_mxfp4_decode_weights,
       prepare_cute_dsl_mxfp4_weights,
   )
   from flashinfer.fused_moe.cute_dsl import CuteDslMxfp4MoEWrapper
   from flashinfer.tllm_enums import ActivationType

   # Load time: canonical W1 is [up, gate]; all four tensors contain uint8 bytes.
   weights = prepare_cute_dsl_mxfp4_weights(w1, w1_scale, w2, w2_scale)
   # Optional, load time: paired W1/scale representation for B300 T=2/4/16 decode.
   decode_pair = prepare_cute_dsl_mxfp4_decode_weights(w1, w1_scale)
   runner = CuteDslMxfp4MoEWrapper(
       num_experts=896, top_k=16, hidden_size=7168, intermediate_size=3072,
       num_local_experts=112, local_expert_offset=336,
       quantization="mxfp4_w4a8", activation_type=ActivationType.Situ,
   )
   workspace = torch.empty(runner.get_workspace_size(T), device="cuda", dtype=torch.uint8)
   output = torch.empty((T, 7168), device="cuda", dtype=torch.bfloat16)
   plan = runner.plan(
       x, x_scale, topk_ids, topk_weights, *weights,
       beta=beta, linear_beta=linear_beta,
       paired_decode_weights=decode_pair,  # optional; omit to use the canonical weights only
       workspace=workspace, output=output,
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
expert computation. Output includes only experts in this rank's interval;
cross-rank reduction is external.

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
PDL-enabled decode retains separate native sorting.

On B300 with SiTU, PDL disabled, routing tile size 128 and both GEMMs on the
M128/N128, cluster1, non-raster tactics, T1..16 uses dedicated decode kernels.
Unique IDs bound each expert to at most T valid rows. GEMM1 is a skinny gather
kernel that reads the A operand with aligned 16-byte vector loads over K in
512-element steps (invalid rows read a safe token and are zero-filled) and, at
T16, a constant-shape scheduler with per-slot planes. GEMM2 uses a skinny
finalize kernel at T8 and T16, a compact-output finalize at T4, and otherwise
the 16-row narrow A transfer, which retains full shared stage strides,
scale-factor transfers, MMA shape and valid-row output predicates. Routing,
gather and finalize caches include their specialization modes. SM100, T17+,
other activations, PDL and other tactics retain the existing routing, GEMM
and finalize paths; native SiTU arithmetic still applies to other supported
W4A8 shapes.

Paired decode weights (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prepare_cute_dsl_mxfp4_decode_weights(w1, w1_scale)`` consumes the same
canonical W1 (``[E, 2*I, H/2]`` uint8, up rows then gate rows) and linear
UE8M0 scale (``[E, 2*I, H/32]``) as ``prepare_cute_dsl_mxfp4_weights`` and
returns a W1/scale pair with a different row permutation and the same bytes;
no requantization occurs. Pass exactly the two returned tensor objects as
``plan(paired_decode_weights=...)`` next to the four canonical prepared
weights. Planning selects the pair only for B300 SiTU decode at T=2, 4 and 16
with H=7168, I=3072, 112 local experts, top-k 16, PDL disabled, the default
tactics and a 16-byte-aligned activation pointer; every other plan binds the
canonical representation, and omitting the argument keeps the canonical path
everywhere. Views, clones and serialized copies of the pair are rejected with
``ValueError`` where the pair would be selected; prepare a fresh pair instead.
The pair retains ``E * 2*I * (H/2 + H/32)`` bytes beside the canonical bank
(2,620,391,424 bytes for 112 local experts at H=7168, I=3072); W2 is shared.
Refresh both representations together when weights change, then create new
plans.

Prefill keeps native sorting and clearing except at the full B300 shape
(H=7168, I=3072, 112 local experts, top-k 16) with T=512, where the routing
kernel also sorts assignments into expert tiles, GEMM1 runs as a dense and a
sparse route-tile launch with a sparse-prefill epilogue, GEMM2 adds a single
N16 sparse finalize and a separate combine kernel accumulates the weighted
expanded contributions into the BF16 output. At T=1024 the routing kernel
clears the output with a single-pass vectorized clear. Routing tile limits
select these paths on the device; the host reads no routing data.

Workspace and kernel selection
------------------------------

``get_workspace_size(T)`` needs no GPU allocation. Workspace is a contiguous,
256-byte-aligned CUDA uint8 buffer. Its regions hold worst-case padded route
indices, the FP8 GEMM1 result and scales, routing conversion, unit per-expert
GEMM scales, (for T>1024) expert-count scratch and, at the full B300 shape
with T=512 under the SiTU activation with PDL off and the default tactic (the
only configuration whose plan binds it), a BF16 expanded-contribution buffer
of ``T * top_k * H`` elements (117,440,512 bytes) consumed by the combine
kernel. Output is separate.
The padded-row bound is ``tile_size * max_tiles``, where
``max_tiles = T*top_k`` if ``T*top_k <= local_experts``, otherwise
``(T*top_k + (tile_size-1)*local_experts) // tile_size``.

The runner accepts an offline tactic table mapping token-count upper bounds
to existing W4A8 tactic tuples. It chooses the smallest covering bucket using
host shape metadata, with the conservative Blackwell tactic as the fallback.
It never inspects device routing on the host or autotunes during execution.
``mxfp4_moe_capability`` checks architecture, quantization, activation,
dimensions, top-k, global/local expert metadata and graph support before
constructing a plan.

At H=7168, I=3072 and 112 local experts, one prepared weight bank occupies
3,930,587,136 bytes; the optional paired decode representation adds
2,620,391,424 retained bytes. With top-k=16 and the conservative tactic,
workspace is 6,499,072 bytes at T=1, 45,885,440 bytes at T=16, 188,594,688
bytes at T=512 and 253,748,224 bytes at T=4096 (full table below). These
totals exclude caller activations/output and temporary load-time preparation
storage. Evaluation retains CuTe- and TRT-prepared banks; serving with the
CuTe runner needs only its own bank (plus the pair, if used).

Under the default M128/N128 tactic, GEMM1 allocates 228,352 shared bytes
per CTA, six operand stages, two accumulator stages and 512 TMEM columns.
GEMM2 allocates 207,872 shared bytes, five operand stages, two accumulator
stages and 512 TMEM columns. The narrow A transfer reduces transferred
bytes, not these allocated capacities. Workspace layout matches the previous
implementation except for the T=512 expanded-contribution region. Other
tactics have separate resource requirements; the dedicated decode and T512
kernels have their own shared-memory and TMEM footprints within the same
per-CTA limits.

.. list-table:: Measured workspace, separate output size and kernel activities per forward
   :header-rows: 1

   * - T
     - Workspace bytes
     - BF16 output bytes
     - Kernel activities (this runner / TRT-LLM Gen)
   * - 1
     - 6,499,072
     - 14,336
     - 3 / 4
   * - 2
     - 12,995,328
     - 28,672
     - 3 / 4
   * - 4
     - 25,987,840
     - 57,344
     - 3 / 4
   * - 8
     - 45,477,888
     - 114,688
     - 3 / 4
   * - 16
     - 45,885,440
     - 229,376
     - 3 / 4
   * - 128
     - 51,591,168
     - 1,835,008
     - 4 / 4
   * - 256
     - 58,112,512
     - 3,670,016
     - 4 / 4
   * - 512
     - 188,594,688
     - 7,340,032
     - 6 / 4
   * - 1024
     - 97,238,016
     - 14,680,064
     - 4 / 4
   * - 2048
     - 149,412,864
     - 29,360,128
     - 4 / 4
   * - 4096
     - 253,748,224
     - 58,720,256
     - 4 / 4

Workspace bytes come from ``get_workspace_size(T)`` at the full B300 shape
with the default tactics and are identical for eager and Graph execution and
for balanced, empty-expert and hot routing. Kernel activities are measured
CUPTI ``CONCURRENT_KERNEL`` records per forward at the same shape (again
identical across execution modes and routing distributions): three for
T1..16 (fused route preprocess, GEMM1, GEMM2 with finalize), four for
T128/T256 and T1024/T2048/T4096 (route conversion with output clear, native
sort, GEMM1, GEMM2 with finalize) and six for T512 (fused sort and route
preprocess, dense and sparse GEMM1 launches, GEMM2, the N16 sparse finalize
and the combine). TRT-LLM Gen launches four kernels for every T. These are
per-forward counts, not a total of compiled variants; the compiled inventory
depends on routing mode, tactic and specialization scope.

Evaluation status
-----------------

Evaluated on one NVIDIA B300 (SXM6, compute capability 10.3) with CUDA 13.1
(PyTorch build; cuda-bindings 13.4.1, cupti-python 13.4.0), PyTorch
2.10.0a0+a36e1d39eb.nv26.1.42222806, nvidia-cutlass-dsl 4.8.0.dev0 and
apache-tvm-ffi 0.1.14.post0. ``FLASHINFER_KIMI_K3_FULL=1`` collects 190
tests on the exported tree: 188 passed, 2 skipped, 0 failed;
``FLASHINFER_KIMI_K3_ALL_LOCAL=1 -k all_local`` passes its 2 tests. The
CUDA API trace audit of the planned forward records no allocation, host
synchronization, device-to-host copy or compilation in ``run()`` (21
rejected misuse cases are asserted by the tests). Sanitizer results:
compute-sanitizer synccheck and racecheck on the T1024 route/clear
qualification pass with 0 errors (7.72 s and 12.65 s); on the decode
qualification both tools were stopped at a 20 s hard cap with no reported
errors and were not retried, so no sanitizer verdict is claimed for the
decode path.

Full geometry is H=7168, I=3072, E=896, top-k=16, 112 local experts and
offset 336. All inputs are synthetic; matching real-checkpoint evaluation has
not been performed.

Numerical semantics and limits
------------------------------

Both FP64 references use the same stored quantized operands. The ideal
reference evaluates FP64 GEMMs, SiTU and routing-weight accumulation without
intermediate quantization. The explicit reference rounds the FP64 activation
to FP32, applies the unchanged group-32 MXFP8 quantizer, then evaluates GEMM2
and routing in FP64. Reports include relative L2, cosine, absolute-error
percentiles/max, worst-token error and finite checks. All full-dimension
outputs are finite.

Independent worst values over the 66 required full-size observations that
the export/source paired-comparison harness recorded from outputs produced
after each row's timing pass on the exported tree (all finite; the public
benchmark script computes its accuracy metrics before timing):

.. list-table::
   :header-rows: 1

   * - Reference
     - This runner max relative L2
     - TRT-LLM Gen max relative L2
     - This runner min cosine
     - This runner max p95 / p99 / absolute error
   * - Ideal FP64
     - 0.027003
     - 0.027037
     - 0.999636
     - 0.003738 / 0.007121 / 0.022998
   * - Explicit MXFP8
     - 0.004468
     - 0.002532
     - 0.999990
     - 0.000540 / 0.001380 / 0.010676

Against the ideal reference this runner and TRT-LLM Gen are indistinguishable
at the reported precision (the ideal error is dominated by the shared
quantized operands). Against the explicit MXFP8 reference this runner's worst
relative L2 is 1.8x TRT-LLM Gen's (0.004468 versus 0.002532, both at
T4096/hot); the difference comes from the explicit FP32-to-MXFP8 activation
rounding that this runner performs between GEMM1 and GEMM2 while the reference
rounds once. At T4 and T16 with balanced routing the finalize splits the
last round of work items into K slices and reduce-adds up to six BF16
partials per output element; those outputs are not bit-reproducible run to
run but stay within the tolerances and FP64 gates above. The
``atol=0.1, rtol=0.15`` diagnostic passes for every element in every row.

The public tests gate on ideal-FP64 error no greater than TRT-LLM Gen's error
plus the reference's BF16 representation floor, for the canonical and the
paired decode representation alike. Neither
gate requires explicit-reference nonregression. The historical
``atol=0.1, rtol=0.15``, greater-than-95% precedent remains diagnostic.
Final customer numerical acceptance is unresolved.

Comparative performance
-----------------------

Required rows: T1/2/4/8/16 and T128/256/512/1024/2048/4096, three routing
distributions (balanced, empty-expert, hot), eager and CUDA Graph execution,
66 rows in total. Every row compares identical operands on the same B300 with
packed BF16 routing, seed 123, beta 4 and linear_beta 25. Timing uses CUPTI
activity records with a cold L2 cache, alternating implementation order, five
warmups and 20 samples per measurement. The paired-comparison harness that
produced these tables also sampled the SM clock through NVML before and after
every measurement and required at least 97% of the reported maximum; rows that
failed that check were excluded, never counted as a pass. No CUDA-event
fallback qualifies. ``benchmarks/bench_mxfp4_situ_moe.py`` does not implement
the clock check: lock or verify the SM clock separately when reproducing,
otherwise the rows are not comparable to these tables.

Ratios are TRT-LLM Gen over this runner on the exported tree; values above
one favor this runner. Balanced medians are in microseconds, this runner /
TRT-LLM Gen.

.. list-table::
   :header-rows: 1

   * - T
     - Graph kernel sum
     - Graph synchronized E2E
     - Eager synchronized E2E
   * - 1
     - 31.23 / 32.34
     - 46.88 / 49.15
     - 139.95 / 253.65
   * - 16
     - 198.66 / 201.97
     - 215.96 / 221.27
     - 300.06 / 408.65
   * - 128
     - 708.17 / 726.60
     - 730.39 / 748.04
     - 805.88 / 933.37
   * - 512
     - 665.38 / 697.61
     - 690.45 / 720.71
     - 800.98 / 896.06
   * - 1024
     - 763.03 / 949.82
     - 786.95 / 974.00
     - 859.91 / 1020.23
   * - 2048
     - 796.73 / 1310.02
     - 821.25 / 1335.35
     - 889.40 / 1249.19
   * - 4096
     - 867.48 / 1257.57
     - 892.95 / 1296.53
     - 955.37 / 1271.69

.. list-table::
   :header-rows: 1

   * - Scope
     - Rows
     - Kernel-sum GM
     - Kernel-sum min
     - Synchronized E2E GM
   * - Graph decode (T1..16)
     - 15
     - 1.086
     - 1.0109
     - 1.077
   * - Graph prefill (T128..4096)
     - 18
     - 1.311
     - 1.0261
     - 1.260
   * - Eager decode
     - 15
     - 1.082
     - 1.0076
     - 1.743
   * - Eager prefill
     - 18
     - 1.210
     - 1.0240
     - 1.345
   * - All required
     - 66
     - 1.177
     - 1.0076
     - 1.332

All 66 required rows measured a kernel-sum ratio above one in the paired
ratio and in each execution order (minimum 1.0076 at eager/T2/balanced,
maximum 2.3995 at graph/T4096/empty); the synchronized E2E ratio is above one
in 65 of 66 rows (minimum 0.9822 at graph/T512/empty). Eight decode rows are
within 2% of parity and should be read as parity within run-to-run noise.
Export/source parity: the exported tree's kernel sum is within 1.03 of the
measured source tree in every row (paired maximum 1.0236 at graph/T4/empty,
worst single order 1.0253, geometric mean 0.9991). All 66 rows passed the
clock qualification; no row was excluded. The optional larger token counts
(8192, 16384, 32768) were not measured.

Reproduction
------------

Tests also cover native E2M1/UE8M0 decoding, lossless preparation, packed and
separate routing, runtime parameters, preallocated execution and, under
``FLASHINFER_KIMI_K3_FULL=1``, the paired decode representation at T=2/4/16
(bound-pair check, agreement with the canonical representation, rejection of
anything but the exact prepared pair):

.. code-block:: bash

   FLASHINFER_KIMI_K3_FULL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py
   FLASHINFER_KIMI_K3_ALL_LOCAL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py -k all_local

The standalone benchmark reports kernel sum, GPU activity span, host enqueue
and synchronized end-to-end latency, raw samples and paired ratios. GPU span
includes correlated memory operations and gaps. Kernel sum adds kernel
activity durations, excluding separate memcpy/memset records but including
a graph-lowered memset reported as a kernel. Host metrics use CUPTI timestamps.
The command below runs the public two-backend comparison.
``--paired-decode-weights`` prepares the paired representation once and passes
it to every plan; rows record whether it was selected.

.. code-block:: bash

   python benchmarks/bench_mxfp4_situ_moe.py --full --accuracy \
     --tokens 1,2,4,8,16,128,256,512,1024,2048,4096 --distributions balanced,empty,hot \
     --modes eager,graph --routing packed --warmup 5 --repeats 20 \
     --paired-decode-weights --run-id review --output mxfp4-situ-review.json
