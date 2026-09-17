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

The selected candidate is ``fence-r008``, committed as
``a24cbc26d49b8cc3a34f2bb3eb6dca691b604a01``. The repair adds typed
``tcgen05.fence::after_thread_sync`` after producer acquisition and consumer
waits, and completes TMEM loads with ``tcgen05.wait::ld`` before
``tcgen05.fence::before_thread_sync`` and consumer release. This extends the
earlier unaligned counted-barrier repair in both gather and finalize.
Routing, warp roles, barrier counts and the three-launch decode design are
unchanged from r007.

R008 passed 168 SiTU/routing tests with two opt-in skips (58.67 seconds),
the separate tile256/two-CTA changed-input graph and FP64 test (6.41 seconds),
two all-local-896 tests (5.73 seconds), and eight existing W4A8/W4A4 SwiGLU,
two-CTA and partial-tile tests (12.63 seconds). The full 84-row paired
benchmark passed every numerical check; 76 rows passed clock qualification,
including all 66 required rows. The benchmark took 160.20 seconds within
the 349.84-second managed validation job.

Validation uses NVIDIA B300 (SM103), CUDA 13.1, PyTorch 2.10.0a0 and CuTe DSL
4.8.0.dev0. Full geometry is H=7168, I=3072, E=896, top-k=16, 112 local
experts and offset 336. Numerical tests cover T=1,16,128,512,2048 with
balanced, empty-expert and hot routing. Graph tests replay every T=1..16
100 times; concurrent decode/prefill tests include independent compute and
copy traffic. Per-expert runtime SiTU parameters and their graph mutation
are covered with and without the up-branch tanh.

Follow-up PTX inspection found both fence forms, TMEM loads and load waits
in gather/finalize entries for the default and tile256 configurations.
This is an instruction-presence check with source windows, not proof of
control-flow dominance. R008 synccheck was skipped after its hard
20-second timeout, with no actual errors printed; it is not a pass and
will not be retried automatically. R008 racecheck passed with zero hazards
in 6.81 seconds. All six new full-geometry forward API audit cases passed:
T1 packed, T16 packed/separate BF16, T128 packed, and T1025 packed/separate
BF16. The three decode cases each contain five forwards with exactly three
kernels, all on the calibrated caller stream, with clean compilation and
forbidden-API checks. The follow-up validation job took 194.52 seconds.
Earlier r007 traces remain historical evidence.

Tests in ``tests/moe/test_cute_dsl_mxfp4_situ.py`` also exercise native-format
decoding, lossless preparation, packed/separate routing, streams and
preallocated execution. Enable full geometry with:

.. code-block:: bash

   FLASHINFER_KIMI_K3_FULL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py
   FLASHINFER_KIMI_K3_ALL_LOCAL=1 pytest -q tests/moe/test_cute_dsl_mxfp4_situ.py -k all_local

Numerical reports compare both implementations to the same FP64 oracle for
the supplied quantized operands, plus an oracle that explicitly models
intermediate MXFP8 quantization. They include relative L2, cosine,
absolute-error statistics and the historical ``atol=0.1, rtol=0.15`` passing
fraction. The development gate is candidate relative L2 no greater than
TRT-LLM Gen error plus the FP64 reference's BF16 representation-error floor.
This gate is provisional; the historical tolerance is diagnostic and is not
an agreed acceptance threshold. Across r008's 84 rows, candidate relative
L2 is 0.02485--0.02693 and TRT-LLM Gen is 0.02493--0.02666.

A separate synthetic study with seed 123 and a fixture bank generated at
T=2048 found a larger difference against the explicit intermediate-MXFP8
reference. At T128 with hot routing, selected-implementation relative L2
was 0.00435, versus 0.00239 for TRT-LLM Gen. Targeted stage checks attribute
most of the additional error to BF16 contribution rounding and atomic
accumulation. FP32 accumulation and separate FP32 reduction prototypes
improved hot-routing accuracy but increased full-runner latency, so they
are not included in this implementation. Final numerical acceptance and
this accuracy/performance tradeoff remain unresolved.

All 84 timing rows were measured again on r008 in one full matrix. The 66
required rows cover T=1,2,4,8,16,128,256,512,1024,2048,4096, three routing
distributions and eager/graph modes; every required row is clock-qualified.
The table below uses balanced routing and microseconds; each latency pair
is CuTe / TRT-LLM Gen. Kernel sum includes all kernel activities from the
runner; eager end-to-end includes host submission and synchronization.
Ratios are TRT-LLM Gen latency / CuTe latency, so values above one favor CuTe.

.. list-table:: R008 balanced routing; all rows clock-qualified
   :header-rows: 1

   * - T
     - Local assignments / empty experts
     - Graph kernel sum: CuTe / TRT
     - Eager end-to-end: CuTe / TRT
   * - 1
     - 2 / 110
     - 37.504 / 31.808
     - 139.834 / 242.025
   * - 2
     - 4 / 108
     - 58.528 / 49.473
     - 154.250 / 249.234
   * - 4
     - 8 / 104
     - 87.649 / 73.728
     - 181.496 / 278.828
   * - 8
     - 16 / 96
     - 146.113 / 122.017
     - 237.816 / 320.244
   * - 16
     - 32 / 80
     - 251.523 / 201.954
     - 348.050 / 429.341
   * - 128
     - 256 / 0
     - 804.585 / 729.414
     - 907.161 / 939.663
   * - 256
     - 512 / 0
     - 808.489 / 754.984
     - 915.898 / 962.013
   * - 512
     - 1024 / 0
     - 816.455 / 697.095
     - 925.853 / 899.189
   * - 1024
     - 2048 / 0
     - 827.240 / 952.809
     - 947.461 / 1063.156
   * - 2048
     - 4096 / 0
     - 842.343 / 1308.396
     - 948.743 / 1204.265
   * - 4096
     - 8192 / 0
     - 882.246 / 1232.011
     - 995.418 / 1294.463

Across T=1,2,4,8,16 and the three routing distributions, eager synchronized
end-to-end ratios are 1.234--2.296x, while graph kernel ratios are
0.803--1.047x. Balanced decode remains slower than TRT. Empty/hot T=8/16
graph cases reach 1.015--1.047x. Prefill is mixed: balanced/hot graph kernel
ratios are 0.934/0.985x at T=256 and 0.854/0.886x at T=512; balanced
T=512 eager end-to-end is below parity at 0.971x. Graph kernel ratios span
1.152--1.325x at T=1024, 1.526--1.805x at T=2048 and 1.392--2.025x at
T=4096. Eager host-side gains do not establish graph decode GPU-time parity.

Inputs are synthetic, with seed 123, beta 4, linear_beta 25 and packed BF16
routing. Measurements use 10 warmups, 15 samples, CUPTI activity tracing
and cold L2. Every row supplies identical operands to both backends, and
complete assignment histograms are retained. This full-matrix fixture bank
was generated with maximum T=32768 and differs from earlier revision
batches; cross-revision differences do not isolate the fence change alone.

The optional T=8192/16384/32768 matrix contributes 18 numerical passes and
ten clock-qualified rows. Qualification is specific to each mode: T=8192
hot and T=16384 balanced qualify only in eager mode. At T=16384 hot and
T=32768 balanced/hot, neither mode qualifies. The eight unqualified rows
retain raw measurements but are excluded from performance conclusions.

.. list-table:: Optional r008 prefill; only clock-qualified ratios shown
   :header-rows: 1

   * - T
     - Routing
     - Graph kernel ratio
     - Eager end-to-end ratio
   * - 8192
     - balanced
     - 0.766x
     - 0.833x
   * - 8192
     - empty
     - 2.475x
     - 2.303x
   * - 8192
     - hot
     - unqualified
     - 0.886x
   * - 16384
     - balanced
     - unqualified
     - 0.983x
   * - 16384
     - empty
     - 3.376x
     - 2.652x
   * - 16384
     - hot
     - unqualified
     - unqualified
   * - 32768
     - balanced
     - unqualified
     - unqualified
   * - 32768
     - empty
     - 3.920x
     - 3.045x
   * - 32768
     - hot
     - unqualified
     - unqualified


The standalone ``benchmarks/bench_mxfp4_situ_moe.py`` reports kernel sum,
complete-runner GPU activity span, host enqueue time and synchronized
end-to-end latency, with raw samples and paired ratios. GPU span includes
correlated memory operations and gaps. Kernel sum adds ``CONCURRENT_KERNEL``
durations: separate ``MEMCPY``/``MEMSET`` records are excluded, while a
graph-lowered memset reported as a kernel is included. Host metrics use
CUPTI timestamps with tracing enabled. The helper's historical r005 smoke
covered six full-geometry rows and 12 backend measurements with three
samples per metric and zero dropped records, using cupti-python 13.4.0;
that validation job took 136.41 seconds.

.. code-block:: bash

   python benchmarks/bench_mxfp4_situ_moe.py --full --accuracy \
     --tokens 1,2,4,8,16,128,256,512,1024,2048,4096 --distributions balanced,empty,hot \
     --modes eager,graph --routing packed --warmup 10 --repeats 15 \
     --run-id review --output mxfp4-situ-review.json

The fixed r008 packed-routing matrix has three distinct CuTe compiled
variants: fused routing/preprocessing, gather and finalize. Every decode
timing sample contains three kernel activities. The previously inspected
shared native module has 1,290 unique SM103 entrypoints; that artifact is
unchanged. Default decode bypasses native sorting. Required prefill selects
512-thread cluster sorting at T=128 and 1024-thread cluster sorting at
T=256..4096. Artifact contents, selected variants and per-forward launch
counts are different quantities; CUDA runtime and PyTorch kernels are
outside the native-module inventory.

R005 and r007 timing/trace results are historical. The rejected r006
arithmetic change is absent; its six boundary tests are retained. The
separate skinny-N GEMM prototype has no selected-implementation performance
claim. No real checkpoint distribution is represented by these fixtures.
Balanced decode and some prefill cases still trail TRT-LLM Gen. Performance work and final review checks remain before final acceptance.
