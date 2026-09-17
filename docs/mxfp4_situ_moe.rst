Planned native MXFP4 SiTU MoE
============================

The CuTe DSL planned runner targets SM100 and SM103, including NVIDIA B300.
It consumes native packed E2M1 weights and UE8M0 scales with block size 32,
FP8 E4M3 activations and BF16 output. Quantization is explicit metadata.

.. code-block:: python

   import torch

   from flashinfer.fused_moe.prepare import prepare_cute_dsl_mxfp4_weights
   from flashinfer.fused_moe.cute_dsl import CuteDslMxfp4MoEWrapper
   from flashinfer.tllm_enums import ActivationType

   # Load time: canonical W1 is [up, gate]; all four tensors contain uint8 bytes.
   weights = prepare_cute_dsl_mxfp4_weights(w1, w1_scale, w2, w2_scale)
   runner = CuteDslMxfp4MoEWrapper(
       num_experts=896, top_k=16, hidden_size=7168, intermediate_size=3072,
       num_local_experts=112, local_expert_offset=336,
       quantization="mxfp4_w4a8", activation_type=ActivationType.Situ,
   )
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
expert computation. Output includes only experts in this rank's interval;
cross-rank reduction is external.

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
dimensions, top-k, global/local expert metadata and graph support before
constructing a plan.

At H=7168, I=3072 and 112 local experts, one prepared weight bank occupies
3,930,587,136 bytes. With top-k=16 and the conservative tactic, workspace is
6,499,072 bytes at T=1, 45,885,440 bytes at T=16, and 253,748,224 bytes at
T=4096. These totals exclude caller activations/output and temporary load-time
preparation storage. Paired evaluation holds one bank per backend; serving
with the CuTe runner needs only its own bank.

Evaluation status
-----------------

The integrated fused-sort revision ``r007`` (commit
``86798eb20718f543efd4a7a69dccbb8272050fee``) passed 168 tests with two opt-in
skips on NVIDIA B300 (SM103), CUDA 13.1, PyTorch 2.10.0a0 and CuTe DSL
4.8.0.dev0. Full-dimension coverage uses H=7168, I=3072, E=896, top-k=16,
112 local experts and offset 336. It includes T=1,16,128,512,2048 with
balanced, empty-expert and hot routing; 100 graph
replays for every T=1..16; and concurrent decode/prefill streams with
independent compute/copy traffic. The two additional tests with all 896
experts local also passed on r007. The unchanged GEMMs retain eight earlier
W4A8/W4A4 SwiGLU, two-CTA and partial-tile regression passes from r005.
An additional planned-decode test passed for T16 hot routing with an offline
256-row/two-CTA tactic, changed graph inputs and a paired FP64 comparison
(8.26 seconds; 83.49-second validation job).
R007 full-geometry forward API traces passed at T1 packed, T16 packed and
T16 separate BF16. Each case contains five forwards with exactly three
kernels per forward, all on the calibrated caller stream. Audited ranges
contain no allocation/free, host synchronization, D2H transfers, module/link
operations, synchronous copies or separate memcpy/memset activities, and
compilation guards are clean. The unchanged prefill path retains earlier
trace evidence. The r007 audit job took 113.30 seconds and included passing
Ruff lint and formatting checks for the public benchmark change.

Separate sparse SiTU runs cover the unchanged GEMM barrier repair with zero
synccheck errors and zero racecheck hazards. The fused-routing prototype's
synccheck was skipped after a hard 20-second timeout, with no actual errors
printed and no retry; its racecheck passed with zero hazards. A timeout is
not a pass, and these prototype results are not fresh integrated-r007
sanitizer results. The rejected ``r006`` arithmetic change is absent;
its six retained boundary regressions are part of the newer test coverage.

Tests in ``tests/moe/test_cute_dsl_mxfp4_situ.py`` exercise native-format
decoding, lossless preparation, FP64 references, runtime parameters, packed
routing, streams, allocation behavior and graph replay. Enable the full
geometry with:

.. code-block:: bash

   FLASHINFER_KIMI_K3_FULL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py
   FLASHINFER_KIMI_K3_ALL_LOCAL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py -k all_local

Numerical reports compare both implementations to the same FP64 oracle for
the supplied quantized operands, plus an oracle that explicitly models the
intermediate MXFP8 quantization. They report relative L2, cosine,
absolute-error statistics and the historical ``atol=0.1, rtol=0.15`` passing
fraction. The development gate is candidate relative L2 no greater than the
TRT-LLM Gen error plus the FP64 reference's BF16 representation-error floor.
This gate is provisional; the historical tolerance is diagnostic and is not
an agreed acceptance threshold.

The required benchmark combines 42 new r007 rows at T=1,2,4,8,16,128,2048
with 24 earlier rows for unchanged prefill at T=256,512,1024,4096. These
66 rows cover three routing distributions and eager/graph modes, and all
passed numerical and clock gates. Both recorded backend clocks were
2032 MHz in every new r007 row. Representative balanced
routing medians are below, in microseconds. Kernel sum includes all kernels
launched by the runner; eager end-to-end includes host submission and
synchronization. Ratios elsewhere are TRT-LLM Gen latency / CuTe latency.

.. list-table:: R007 and unchanged prefill evidence, balanced routing
   :header-rows: 1

   * - T
     - Local assignments / empty experts
     - Graph kernel sum: CuTe / TRT
     - Eager end-to-end: CuTe / TRT
   * - 1
     - 2 / 110
     - 37.632 / 32.095
     - 138.619 / 237.640
   * - 2
     - 4 / 108
     - 58.465 / 50.017
     - 153.373 / 266.494
   * - 4
     - 8 / 104
     - 87.105 / 73.216
     - 185.741 / 288.141
   * - 8
     - 16 / 96
     - 145.186 / 122.018
     - 243.978 / 322.313
   * - 16
     - 32 / 80
     - 251.395 / 202.691
     - 349.197 / 447.881
   * - 128
     - 256 / 0
     - 805.322 / 728.359
     - 903.069 / 937.091
   * - 256
     - 512 / 0
     - 809.225 / 758.890
     - 900.420 / 952.858
   * - 512
     - 1024 / 0
     - 814.921 / 695.559
     - 907.639 / 893.588
   * - 1024
     - 2048 / 0
     - 824.072 / 946.761
     - 916.798 / 968.388
   * - 2048
     - 4096 / 0
     - 840.457 / 1318.733
     - 937.714 / 1186.447
   * - 4096
     - 8192 / 0
     - 876.874 / 1239.854
     - 966.730 / 1260.626

Across the three distributions and T=1,2,4,8,16, eager end-to-end ratios are
1.283--1.952x, while graph decode kernel ratios are 0.806--1.084x. Balanced
decode still trails TRT; empty/hot T=8/16 graph cases reach 1.026--1.084x.
Prefill performance is mixed: balanced/hot graph kernel ratios are 0.938/0.988x at
T=256 and 0.854/0.892x at T=512; balanced T=512 eager end-to-end is also below
parity at 0.985x. Graph kernel ratios reach 1.149--1.350x at T=1024,
1.548--1.782x at T=2048 and 1.387--2.053x at T=4096.

These measurements use synthetic inputs, 10 warmups, 15 samples, CUPTI
activity tracing and cold L2; complete assignment histograms are retained
with every result. Each row supplies identical operands to both backends.
The two batches use distinct generated weight banks because input generation
consumes different RNG counts at maximum T=2048 versus T=4096. R007's
42-row benchmark took 41.86 seconds within a 198.87-second validation job;
its 168-test suite took 50.77 seconds and its two all-local tests took
6.55 seconds. The unchanged 24-row prefill continuation took 33.70 seconds
within an earlier 136.57-second job, which also checked rollback boundaries.
The earlier eight shared-GEMM regressions took 81.87 seconds on r005.

The unchanged optional T=8192/16384/32768 matrix adds 18 numerical passes, of which ten
rows pass the clock gate. Overall, 84 rows pass numerical checks and 76 are
clock-qualified, including all 66 required rows. The optional qualification
summary below applies to both eager and graph measurements of each case;
ratios from unqualified rows are excluded from performance conclusions.

.. list-table:: Optional large-prefill clock qualification
   :header-rows: 1

   * - T
     - Routing
     - Clock-qualified
     - Graph kernel ratio
     - Eager end-to-end ratio
   * - 8192
     - balanced
     - yes
     - 0.789x
     - 0.836x
   * - 8192
     - empty
     - yes
     - 2.499x
     - 2.420x
   * - 8192
     - hot
     - yes
     - 0.835x
     - 0.888x
   * - 16384
     - balanced
     - no
     - unqualified
     - unqualified
   * - 16384
     - empty
     - yes
     - 3.362x
     - 2.745x
   * - 16384
     - hot
     - no
     - unqualified
     - unqualified
   * - 32768
     - balanced
     - no
     - unqualified
     - unqualified
   * - 32768
     - empty
     - yes
     - 3.911x
     - 3.225x
   * - 32768
     - hot
     - no
     - unqualified
     - unqualified

At T=16384, the unqualified candidate measurements ran around 1.3--1.4 GHz
versus TRT's 2.032 GHz. At T=32768, unqualified TRT measurements ran around
1.3--1.6 GHz versus the candidate's 2.032 GHz. These eight rows retain raw
results but provide no qualifying performance evidence. The optional batch
took 82.32 seconds within a 167.81-second job. Its generated fixture bank is
separate; each row still supplies identical operands to both backends.

The standalone ``benchmarks/bench_mxfp4_situ_moe.py`` reports kernel sum,
complete-runner GPU activity span, host enqueue time and synchronized
end-to-end latency, including raw samples and paired ratios. GPU span
includes correlated memory operations and gaps. Kernel sum adds
``CONCURRENT_KERNEL`` durations: separate ``MEMCPY``/``MEMSET`` records are
excluded, while a graph-lowered memset reported as a kernel is included.
Host metrics use CUPTI timestamps with tracing enabled. A six-row
full-geometry smoke on r005 covering balanced T=1,16,128 in eager/graph modes passed
with cupti-python 13.4.0 on CUDA 13.1. All 12 backend measurements contained
three samples for every metric and zero dropped records. That validation
job, including two forward API traces, took 136.41 seconds.

.. code-block:: bash

   python benchmarks/bench_mxfp4_situ_moe.py --full --accuracy \
     --tokens 1,2,4,8,16,128,256,512,1024,2048,4096 --distributions balanced,empty,hot \
     --modes eager,graph --routing packed --warmup 10 --repeats 15 \
     --run-id review --output mxfp4-situ-review.json

The fixed r007 packed-routing benchmark has three distinct CuTe compiled
variants: fused routing/preprocessing, gather and finalize. Its decode
timing records contain exactly three kernels per sample. Packed T128/T2048
records contain five in eager mode and six in graph mode, including
graph-lowered memset. The previously inspected, unchanged native module
contains 1,290 unique SM103 entrypoints. R007 bypasses native sort in default
decode; source dispatch selects 512-thread cluster sorting at required
T128 and 1024-thread cluster sorting at required T256..4096. R007 forward
traces independently confirm the three-kernel decode sequence on the caller
stream. Artifact contents, selected variants and
per-forward launch counts are different quantities; CUDA runtime and
PyTorch kernels are outside that native-module inventory.

Required and optional timing evidence is complete with provenance retained
for unchanged prefill. Eight optional rows are excluded from clock-qualified
timing evidence. The separate skinny-N GEMM prototype remains unvalidated.
Balanced decode and some prefill cases still trail the baseline. Final
checks remain, and this report does not establish final acceptance.
