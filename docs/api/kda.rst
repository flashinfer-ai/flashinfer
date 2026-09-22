.. _apikda:

Kimi Delta Attention (KDA)
==========================

Recurrent Kimi Delta Attention, a per-K-dimension gating variant of GDN.
:func:`flashinfer.recurrent_kda` is the canonical entry point for every phase:
it classifies each call and serves prefill or decode accordingly. The two
modules below hold the phase-specific surface it dispatches into, and are
documented here rather than on pages of their own so that one call's whole
route reads in one place.

flashinfer.kda
--------------

Phase-neutral recurrent Kimi Delta Attention (KDA) facade, and the canonical
entry point for every phase. The public ``recurrent_kda`` classifies the call
and serves decode and speculative decode through the same shared decode
dispatcher that :ref:`apikda_decode` uses, while dispatching eligible ordinary
multi-token prefill to the optimized backend described in
:ref:`apikda_prefill`. The automatic prefill selection depends on the device
and workload: eligible SM100a and SM103a calls with small logical
batch-times-head count use the bundled small-BH CuTe-DSL kernel; other calls
follow the standard SM100-family CuTe-DSL/Cake policy. SM120a uses a CuTe-DSL
backend of its own. These prefill routes leave decode and speculative-decode
dispatch unchanged.

.. currentmodule:: flashinfer.kda

.. autosummary::
    :toctree: ../generated

    recurrent_kda

.. _apikda_decode:

flashinfer.kda_decode
---------------------

Key-Driven Attention (KDA) decode API. The CuTe-DSL kernel lives under
``flashinfer.kda_kernels``.

.. deprecated:: 0.8

    ``flashinfer.kda_decode.recurrent_kda`` is deprecated and now a shim over
    the decode dispatcher shared with :func:`flashinfer.recurrent_kda`, which
    is the canonical entry point: it serves this same decode contract, accepts
    a superset of what this one accepts, and defaults to ``backend="auto"``.
    Calling it emits a :class:`DeprecationWarning`. ``fused_kda_decode`` and
    ``packed_kda_decode`` are unaffected and remain here.

The public ``recurrent_kda`` API supports standard decode with one token per
sequence (``T=1``) and packed speculative decode with two or more tokens per
sequence (``T>=2``).

Pass ``backend="cake"`` to select the exported Cake backend. On SM100-family
SM100a (B200/GB200) and SM103a (B300/GB300) devices, its D128 ``T=1..6``
family with in-kernel QK normalization exports 25 frozen CUDA bodies:

* ``T=3`` with raw gates, ``use_gate_in_kernel=True``, a negative
  ``lower_bound``, float32 ``A_log`` and ``dt_bias``, ``H=HV=16``, and
  ``N`` in ``{1, 2, 4, 8, 16}``;
* four value-row splits for each ``T`` in ``{1, 2, 4, 5, 6}`` with
  precomputed gates, ``use_gate_in_kernel=False``, and no ``A_log``,
  ``dt_bias``, or ``lower_bound``;
* two additional one-warp direct-state ``T=1`` schedules with value-row
  splits 16 and 8. ``T=1`` keeps the standard decode API and is normalized
  to the packed frozen ABI with zero-copy views and cached identity metadata;
  explicit ``T=1`` ``cu_seqlens`` metadata is outside the Cake contract.
* two Kimi-Linear ``T=1`` equal-head direct-state schedules for
  ``lower_bound=None``. They accept any positive runtime head count, including
  production per-rank ``H=HV=32/16/8/4`` for TP1/2/4/8, and evaluate
  ``-exp(A_log) * softplus(g + dt_bias)`` and beta sigmoid in-kernel. Q, K,
  and V may be zero-copy views into one padded packed-projection row; raw G
  and beta may have independent positive token-row strides. The single frozen
  kernel therefore consumes SGLang's production views without staging copies.

Let ``W=N*HV`` be the active sequence/value-head work and ``S`` the device SM
count. SM100a retains the B200-measured policy: direct split 16 for T1 when
``W<=2S`` and direct split 8 otherwise; split 4 for T2; split 2 for T4; and
the T5/T6 CTA-wave policy of split 8 for ``W<=3S/8``, split 2 for
``3S/8<W<=S/2``, split 4 for ``S/2<W<=3S/4``, split 2 for
``3S/4<W<=3S/2``, and split 1 above that range.

SM103a uses its separately measured GB300 policy. T1 selects direct split 16
through a conservative ``W<=32S`` extrapolation guard (measured through
``W/S=26.95``), and direct split 8 beyond it. T2 selects split 8 through
``W<=S/2`` and split 4 above it. T4 selects split 8 through ``W<=S/2``, split
4 through ``W<=S``, split 2 through ``W<=3S/2``, split 1 through ``W<=2S``,
and split 2 above it. T5 keeps the SM100a CTA-wave policy except for a measured
split-1 island at ``3S/4<W<=S``. T6 selects split 8 through ``W<=3S/8``, split
2 through ``W<=S/2``, and split 1 above it. T3 uses its sole exact lower-bound
split-4 specialization on both architectures.

With CUDA 12.9 or newer, JIT and AOT compile all 25 checked-in bodies once for
the ``sm_100f`` family target. The family module URI and cubin artifact can run
on both CC 10.0 and CC 10.3; build workspaces may still materialize separate
cache directories for their local architecture context. Runtime split
selection remains device-specific. A cold-L2 CUPTI A/B against exact-target
cubins measured no aggregate change on B200 (``1.0000x`` exact/family) and
``0.9987x`` on GB300. The GB300 direct-T1 path was the repeatable exception
(``0.9790x``), so its two public direct variants retain exact ``sm_103a``
cubins while every other GB300 route uses ``sm_100f``.

CUDA 12.8 cannot compile ``sm_100f``. On B200 it therefore retains exact
``sm_100a`` modules for all 25 bodies. SM103a requires CUDA 12.9 or newer.
Every binding validates its family or exact-device contract before launch, and
the frozen generated body bytes are identical across all physical targets.

Once ``backend="cake"`` is selected, every supported call launches exactly one
exported Cake kernel. An unsupported architecture, shape, gate mode, layout,
aliasing pattern, or optional feature raises an error; it never falls back to
CuTe-DSL. The default ``backend="cute-dsl"`` preserves the existing FlashInfer
implementation. ``backend="auto"`` selects Cake only for the equal-head D128
T1 unbounded-softplus contract and preserves CuTe-DSL for other decode modes.

Serving-native packed Kimi K3 decode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``packed_kda_decode`` is a separate serving adapter for the Kimi K3 ``T=1``,
``H=12``, ``K=V=128`` contract. It consumes post-convolution packed QKV plus
raw gate and beta tensors, then fuses Q/K extraction and L2 normalization, the
fixed ``lower_bound=-5`` gate transform, beta sigmoid, and the recurrent update
into one exported Cake kernel. It is distinct from ``recurrent_kda`` (whose
inputs are already split into Q, K, V, gate, and beta tensors) and
``fused_kda_decode`` (which also performs the convolution and gated RMSNorm).

The operator updates a caller-owned bfloat16 state pool in place. A contiguous
int32 ``state_indices`` tensor selects one unique active slot per batch row;
``-1`` is an inactive CUDA-graph padding row that emits zero without touching
state. By default the operator allocates a contiguous bfloat16 output of shape
``[B, 1, 12, 128]``. Supplying a caller-owned output with that exact layout
makes replay allocation-free. All work runs on the caller's current PyTorch
CUDA stream.

Two frozen schedules are selected from host-visible batch size only: the
eight-row value tile for ``B < 32`` and the sixteen-row value tile for
``B >= 32``. CUDA 12.8 uses a legacy exact ``sm_100a`` module on CC 10.0;
CUDA 12.9 or newer uses one ``sm_100f`` module on CC 10.0 and CC 10.3.
Unsupported devices or contracts raise an error without falling back to
another KDA implementation.

Frozen / speculative-verify mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``recurrent_kda(disable_state_update=True)`` computes the KDA outputs for up
to 16 tokens per sequence from a read-only committed-state pool and never
writes state back — the speculative-decode verify path, mirroring GDN's
``gated_delta_rule_mtp(disable_state_update=True)``. Optional slot-indexed
``correction_cache`` (float32 per-token delta-rule corrections,
``[num_slots, HV, T_max, V]``) and ``kg_cache`` (raw key | raw gate,
``[num_slots, HV, T_max, 2K]``) out-params feed a downstream commit/recovery
kernel, paralleling GDN's slot-indexed ``intermediate_states_buffer``. The
mode accepts the batched ``[B, T, ...]`` form and the packed ``cu_seqlens``
form with ragged per-sequence lengths and null slots.

Two implementations are dispatched internally by problem size: a WY-parallel
tensor-core kernel that replaces the serial recurrence with T x T GEMMs, a
log-depth triangular inverse, and TMA-loaded state GEMMs (per-K-channel decay
folded into ``k * exp(±cumsum g)`` / ``q * exp(cumsum g)`` operand tiles),
and a grouped register recurrence (no per-token state-checkpoint writes) for
small ``B * HV * T``. Requires SM90+ for the WY path and ``K = V = 128``;
``backend="cake"`` raises in this mode.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    fused_kda_decode
    packed_kda_decode
    recurrent_kda

.. _apikda_prefill:

flashinfer.kda_prefill
----------------------

Optimized recurrent Kimi Delta Attention (KDA) prefill support. The
:func:`flashinfer.kda.recurrent_kda` facade exposes frozen Cake and source-level
CuTe DSL implementations for a strict ordinary multi-token prefill subset.

.. currentmodule:: flashinfer.kda_prefill

.. autosummary::
    :toctree: ../generated

    RecurrentKDAPrefillWorkspace

.. note::

    ``flashinfer.RecurrentKDAPrefillWrapper`` is **experimental**, so it has no
    generated reference page here until it graduates. Calling ``plan`` or
    ``run`` is itself the opt-in and needs no environment variable; each warns
    once per process. It is limited to compute capability 10.0 and 10.3, is
    specific to the CuTe DSL backend, and fixes the sequence count and packed
    token extent after the first warmup run. Its planning implementation lives
    in ``flashinfer.experimental.kda_prefill_wrapper``; it contains no kernels
    of its own, and the kernels it dispatches to are the stable AOT-registered
    ones, alongside a package README. ``examples/experimental/kda_prefill_wrapper.py``
    is a runnable plan-and-run example. See
    `#5069 <https://github.com/flashinfer-ai/flashinfer/issues/5069>`_ for the
    graduation plan.

Backend selection
~~~~~~~~~~~~~~~~~

``backend="auto"`` selects the source-level CuTe DSL backend for eligible
ordinary multi-token prefill and falls back to the frozen Cake backend for
unsupported contracts. Decode retains the existing KDA decode routing.
``backend="cake"`` and ``backend="cute-dsl"`` select a backend strictly and
raise when its contract is unsupported.

For multi-token prefill, ``backend="cute-dsl"`` selects a BT=16 CuTe DSL kernel.
It supports contiguous BF16 Q, K, V, G, and beta with one shared head count and
head dimension 128, the in-kernel lower-bound gate, fixed or packed-varlen
layout, BF16 recurrent state, explicit ``seq_order``, and the same checkpoint
contract as Cake. ``checkpoint_cu_starts`` must always be int64. Packed
``cu_seqlens`` must be int64 during CUDA graph capture. The CuTe DSL schedule
is non-persistent.

Optimized Blackwell prefill subset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The strict Cake backend is available only when
every condition below holds:

* the device has compute capability 10.0 (SM100a; B200/GB200) or 10.3
  (SM103a; B300/GB300);
* input is ordinary multi-token prefill: fixed ``T > 1``, or packed input
  whose total token count is greater than its number of sequences;
* Q, K, V, and G are contiguous BF16 ``[B,T,H,128]`` tensors with one shared
  head count; beta is BF16 ``[B,T,H]`` with unit head stride and
  non-overlapping token rows (an aligned slice of a fused projection is
  accepted directly);
* ``A_log`` is contiguous FP32 ``[H]`` and ``dt_bias`` is contiguous FP32
  ``[H,128]`` or flattened ``[H*128]``;
* ``use_qk_l2norm_in_kernel=True``, ``use_gate_in_kernel=True``,
  ``beta_is_logit=True``, and ``lower_bound`` is a finite negative value;
* speculative decode, GQA, committed-state sources, and accepted-token
  features are not enabled. Plain int32 ``ssm_state_indices`` and native
  prefill checkpoints are supported by direct M128.

T=1 decode and speculative decode are not handled by either prefill backend.

CUDA 12.8 predates the family target, so CC 10.0 uses legacy exact
``sm_100a`` modules. With CUDA 12.9 or newer, JIT and AOT compile one
``sm_100f`` module per schedule for both CC 10.0 and CC 10.3. Cache keys also
include the frozen module identity so an older schedule cannot satisfy a
refreshed request. Runtime routing remains device-specific. The legacy
head-grouped and mixed-length persistent M128 routes remain restricted to
measured 148/152-SM CC 10.0 devices. On validated 148/152-SM CC 10.0 or CC 10.3
devices, eligible uniform eager calls with a caller-owned in-place initial and
final state may instead split only the recurrence chains that create a partial
final device wave. An occupancy and roofline model
uses the live SM count plus the frozen schedule's thread, shared-memory,
tensor-memory, BF16, HBM, state-transfer, and refill costs; it selects
recurrence pieces only when their dependency-DAG critical path is shorter than
direct M128. Intermediate BF16 states use device-scope release/acquire
handoffs, and each consumer CTA resets its ready counter before it completes so
subsequent eager launches start from the same state. This route is not selected
when the caller supplies an explicit workspace or ``seq_order``; CUDA Graph
capture therefore continues through the existing non-piece routes.

CC 10.3 also has a tensor-core state-decay specialization for uniform, complete
N32 work when there are at least 64 heads, at least 96 sequence/head tasks,
and the maximum sequence length is a multiple of 32 and at least 256. Mixed or
partial N32 tails retain scalar state decay. On either capability, fixed-layout
calls with at most eight total sequence/head tasks, at most eight heads, and at
least 2,048 tokens per sequence use the small-BH owner/helper schedule when all
eight CTAs per task can reside concurrently. Calls outside that measured region
continue through the existing direct or fallback route; it is not a public-input
allowlist.

The dense beta-TMA BT16 one-wave route uses the S9 chain schedule when both
value-split CTAs for every task fit in one device wave.

At maximum sequence length 16 or below, generic head counts use a one-stage
N16 retrace with one four-warp prepare owner. It preserves the variable-shape
N16 arithmetic while reducing the CTA from 32 to 16 warps. H12 keeps its
dedicated scalar-beta N16 schedule.

The N16 route describes aligned beta storage directly once it contains a full
16-token tile. Calls requiring token or head padding refresh the stable beta-TMA
workspace inside the binding before launching the recurrence kernel.

The frozen H12 N16 schedule's residual recurrence rounds four intermediates
through BF16: the state/K
prediction, the V-minus-prediction delta, sigmoid beta, and the post-beta
update carrier. The final-state contraction starts from a zero accumulator;
the old BF16 state is multiplied by the total decay and explicitly added to
that product in FP32 before the 16-token chunk boundary rounds back to BF16.
The N16 prepare carrier also matches the source-visible BF16 arithmetic graph:
normalized Q/K, positive and inverse prefix decay, and every chained Qd/Kd/Ki/Kr
multiplication round through BF16 at their respective boundaries.
The N32 schedule comes from the same generated export.

Fixed input omits ``cu_seqlens``. Packed input has ``B=1`` and accepts a
contiguous CUDA int32 or int64 ``cu_seqlens``. The frozen binding consumes
int64 offsets; pass int64 directly for CUDA graph capture to avoid an
in-capture conversion allocation. Offset values are a caller contract:
``cu_seqlens[0] == 0``, entries are non-decreasing, and
``cu_seqlens[-1] == total_tokens``. CuTe DSL accepts equal adjacent offsets for
zero-length sequences; Cake requires every sequence to be non-empty.
FlashInfer does not synchronize the device to inspect these values; invalid
offsets may cause out-of-bounds device access.

Packed scheduling
~~~~~~~~~~~~~~~~~

Packed prefill optionally accepts ``seq_order``, a contiguous CUDA int32
tensor with one entry per sequence. It is a caller contract that this tensor
is a permutation of ``[0, N)``. Ordering sequences by decreasing length
reduces the final partial wave. FlashInfer validates dtype, device, rank, and
size without synchronizing the device to inspect permutation values.

For Cake, omitting ``seq_order`` uses its cached eager scheduling metadata. H12
selects the dedicated M128 schedule with a 16-token recurrence chunk for both
fixed and packed layouts. Fixed ``B=1,H=64`` selects the two-CTA M64
value-split kernel; the fixed small-BH region described above selects its
eight-CTA owner/helper schedule. Eligible medium and long shapes instead use a
BT16 prepare/chain route: dense fixed ``B=1,H=60..64`` inputs qualify from
4,096 tokens when two value-split CTAs per head fit on the device; general
M128 shapes qualify from 65,536 tokens for one to eight sequence/head tasks,
or from 4,096 tokens for nine to 32 tasks when two CTAs per task fit. N16
alternatives additionally depend on SM count, chain waves, and sequence
length. Uniform work may use the recurrence-piece route described above when
its modeled critical path wins. Supplying ``seq_order`` disables persistent
host task-bin planning but does not suppress BT16 or otherwise force direct
M128. Remaining eligible inputs select the shape-appropriate non-persistent or
general M128 schedule.
The scalar-prepare/S8 BT16 pair is submitted by one native Cake binding, which
performs both launch plans before enqueueing either kernel so the dependent
launches do not expose a Python/FFI inter-kernel gap. CUDA Graph capture still
records the same two kernels and preserves the workspace contract below.

For packed CuTe DSL engine calls, omitting ``seq_order`` generates a stable
decreasing-length order on the device. CuTe DSL decomp retains the original
sequence order in the eager path because its CTA grid fits in one wave.
``flashinfer.RecurrentKDAPrefillWrapper`` (experimental) provides fixed-address
metadata for CUDA Graph capture: ``plan`` only copies packed offsets into its
device buffer, while a captured GPU prepass generates the order and decomp
``cu_chunks`` prefix before the recurrent kernels run. The decomp prep kernel
binary-searches this compact prefix instead of carrying a dense
chunk-to-sequence tensor. Its workspace and launch use a graph-static chunk
capacity derived from tensor shapes, while the GPU prefix supplies the actual
chunk count on each replay.

State and graph semantics
~~~~~~~~~~~~~~~~~~~~~~~~~

The BF16 state layout remains ``[N,H,V,K]`` and an explicitly supplied
``initial_state`` is still updated in place, even when
``output_final_state=False``. The frozen kernels load each CTA's disjoint
state rows before writing the final rows back to the same storage, so no
separate state scratch or copy-back is required. If no initial state is
supplied, a final state is allocated only when ``output_final_state=True``.

With ``ssm_state_indices``, ``initial_state`` is a caller-owned pool
``[N_pool,H,V,K]``. Sequence ``i`` loads and updates the named pool row in
place. The pool may have padding between first-dimension slots, but each
``[H,V,K]`` slot must be contiguous and both the pool base and slot pitch must
be 16-byte aligned. Slot ids must be unique and in range; this is a caller
contract so the launch path does not synchronize to inspect them.

Native checkpoints use a preallocated BF16
``state_checkpoints[C,H,V,K]``, int64 ``checkpoint_cu_starts[N+1]``, and a
positive ``checkpoint_every_n_tokens`` divisible by 16. KDA checkpoints are
states *before* each interval: every non-empty sequence contributes its
initial state as row zero, followed by states after one, two, ... intervals
that strictly precede its end. Consequently each sequence contributes
``ceil(seq_len / interval)`` rows. The call returns
``(output, final_state, state_checkpoints)`` when enabled. Intervals and
cumulative counts are caller-provided device metadata and are not value-scanned
at launch.

An aligned beta base address and a token pitch divisible by 16 bytes are sent
directly to TMA. Other eligible row-strided beta views remain valid API inputs,
but the binding refreshes reusable padded workspace internally; callers never
need to materialize ``beta.contiguous()``.

The frozen kernel uses restricted output and auxiliary storage. A preallocated
``output`` and checkpoint buffers must not overlap Q, K, V, G, beta, state,
metadata, or descriptor storage.

Eager calls without ``prefill_workspace`` use an internal serialized workspace
for the current CUDA stream. This default workspace is eager-only and cannot
be used during CUDA graph capture.

CUDA graph capture requires a caller-owned
``RecurrentKDAPrefillWorkspace(device)`` and a preallocated ``output``. The
workspace owns optional final-state scratch for calls without an initial
state, beta padding, separate TMA descriptor blocks, and the small-BH compact
packet ring with its generation counters. BT16 schedules additionally own
``cu_chunks`` and chunk-to-sequence metadata, BF16 Qd/Kd/W/QK factors, FP32
diagonal factors, and independent prepare/chain descriptor storage. The
workspace binds to the device and CUDA stream of its first ``recurrent_kda``
call.
Warm it eagerly on the intended capture stream with the exact Q, K, V, G,
beta, and output tensors, then synchronize that stream before capture. Packed
graphs must also pass preallocated int64 ``cu_seqlens`` and int32
``seq_order``. The warm call prepares descriptors; capture accepts only the
exact warmed pointer, shape, stride, and dtype signature and performs no
descriptor preparation. Warm the largest intended small-BH shape before
capturing so its packet-ring storage is already allocated.

The workspace must outlive its graph and every replay. Use one distinct
workspace for each captured ``recurrent_kda`` invocation, including two KDA
invocations in the same graph. Once a workspace participates in capture, any
later Python use through ``recurrent_kda``—eager or another capture—is
rejected. ``graph.replay()`` does not re-enter Python and remains valid.
Sequential replay launches may be issued while a different PyTorch stream is
current, but the caller must provide normal stream ordering. The Python stream
binding applies to eager warmup and capture calls, which must use the same
stream.

When an explicit workspace is used with ``initial_state=None`` and
``output_final_state=True``, the returned final state is workspace-owned
stable scratch. Otherwise an explicitly supplied ``initial_state`` is updated
directly in place by the frozen kernel. Head counts that are not divisible by
eight capture the beta copy into workspace-owned storage padded to the next
eight-head boundary before the frozen launch. The public beta and state shapes
keep the caller's original head count.

SM120a prefill subset
~~~~~~~~~~~~~~~~~~~~~

Compute capability 12.0 devices select their own prefill backend,
``flashinfer.kda_kernels.sm120_prefill``. It is CuTe DSL like the BT=16 backend
above and shares no code and no device with it, nor with Cake: those two are
CC 10.0 and 10.3 and this one is CC 12.0, so at most one of the three can be
eligible for any call, and adding this one cannot change which kernel an SM100
or SM103 call receives. Nothing about the public API changes — the entry point
is still :func:`flashinfer.kda.recurrent_kda`, and no argument names the
architecture.

``backend`` selects an implementation family, not an architecture name. On a
CC 12.0 device both ``"auto"`` and ``"cute-dsl"`` may reach this backend; the
dispatcher tries it before the SM100-family CuTe DSL prefill path. An explicit
``"cake"`` request never probes or runs SM120. If the Cake prefill predicate
does not support that ordinary multi-token prefill call, the request is
refused rather than silently executed by another backend.

``recurrent_kda`` uses it only when every condition below holds:

* the device has compute capability 12.0, *and* the installed CuTe DSL and
  CUDA toolkit can natively target ``sm_120a``. A family-conditional fallback
  target is refused rather than accepted, because the kernels are written
  against architecture-specific instructions;
* input is ordinary multi-token prefill: fixed ``T > 1``, or packed input
  whose total token count exceeds its number of sequences;
* Q, K, V and G are contiguous BF16 ``[B,T,H,128]`` tensors sharing one head
  count, and beta is contiguous BF16 ``[B,T,H]``. GQA and ``V != K`` are not
  supported;
* the output fits an INT32 extent: ``T_total * H * 128 <= 2**31 - 1``, which is
  16383 tokens at H=1024 and no constraint at ordinary head counts. Larger is
  refused with the backend's own error, because the two things that stop there
  — a device index built in INT32, and the DSL packing a memref extent as one —
  otherwise fail as a silent negative offset and as a compile-time overflow
  naming no tensor;
* ``A_log`` is contiguous FP32 ``[H]``, and ``dt_bias`` is contiguous FP32
  ``[H,128]`` or flattened ``[H*128]``;
* ``use_qk_l2norm_in_kernel=True``, ``use_gate_in_kernel=True``,
  ``beta_is_logit=True``, and ``lower_bound`` is in ``[-5.0, 0.0)``. The bound
  exists because the safe gate's worst-case chunk prefix reaches a reciprocal
  approximation's cliff at about ``-5.4585``;
* ``initial_state``, if given, is a contiguous BF16 ``[N,H,128,128]`` tensor.
  A state pool with ``ssm_state_indices`` is not supported;
* ``output``, if given, is contiguous BF16 with V's shape and does not overlap
  any input in GMEM;
* speculative decode, ``seq_order``, prefill checkpoints, committed-state
  sources and FP32 gate or state are not enabled.

Under ``backend="auto"``, calls outside that subset continue through the
existing dispatcher. An explicit ``backend="cute-dsl"`` ordinary multi-token
prefill request is refused when neither CuTe DSL prefill implementation is
eligible. T=1 decode and speculative decode are not rerouted by this backend.

Two variants, chosen per shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The backend has two implementations of the same contract. ``decomp`` runs a
chunk-parallel prepare and a serial recurrence, issued through one compiled
host entry; ``fused`` does both in a single kernel. They agree numerically, so
the choice between them is a performance one and is made from a measured
table keyed on the device's SM count — not on its name, which is not a stable
unique selector.

Thresholds exist for the 110-SM, 156-SM and 188-SM parts, measured on each.
They do not agree: the 110-SM part switches to the fused kernel at CTA 128 and
the other two at 144, because at CTA 128 with short sequences the larger parts
still prefer the decomposed kernel and the smallest one does not. Any other
CC 12.0 device uses the 156-SM thresholds as a labelled fallback. A benchmark
run reports which case applies, so a number taken on an unprofiled card cannot
be read as tuned.

State and graph semantics
^^^^^^^^^^^^^^^^^^^^^^^^^

State semantics match the SM100-family path: a supplied ``initial_state`` is
updated in place whether or not ``output_final_state`` is set, and the second
return value is ``None`` when it is not set. Without an initial state, a BF16
final state is allocated only when ``output_final_state=True``.

CUDA graph capture requires a caller-owned ``RecurrentKDAPrefillWorkspace`` and
a preallocated ``output``, warmed eagerly on the capture stream with the exact
tensors and then synchronized before capture. The warm call is where every
compile, descriptor build, metadata table and allocation happens; capture
performs none of them and a cold capture is refused rather than silently
degraded. Both offsets dtypes are accepted for packed capture; eager warmup
populates a workspace-owned canonical int32 buffer. The offset *values* must
stay fixed for the graph's lifetime. Changing them requires a fresh eager
warmup and capture. Q, K, V, G, beta and state contents may change freely at
unchanged addresses.

The offsets contract is not only a capture one. Validating ``cu_seqlens``
needs a device-to-host read, so what is derived from it — the sequence
lengths, the canonical int32 copy, and the decomposed variant's chunk tables —
is cached against the tensor's address and version counter rather than read
again on every call. Under ``torch.inference_mode`` a tensor has no version
counter, so refilling an offsets buffer in place with a different segmentation
is not detectable and the stale tables are reused, silently computing against
the previous sequence boundaries. Use a different tensor for a different
segmentation, or call
``flashinfer.kda_kernels.sm120_prefill.clear_kda_prefill_sm120_caches()``
after refilling one in place. This applies to eager calls as much as to
captured ones.

A workspace binds to one variant, one stream and one call signature on first
use, and once it has participated in a capture it cannot be used again.

What the caches hold
^^^^^^^^^^^^^^^^^^^^

A warm call is a memo lookup, and the memo addresses the caller's buffers: the
descriptors carry their base addresses and the flat views wrap them. Those
buffers therefore stay allocated for as long as the entry lives, which is what
makes reusing the entry safe — an allocator that had recycled the address would
otherwise hand the kernel someone else's memory.

The retention scales with the number of *distinct buffer sets* a process
rotates through, not with the number of calls. On a 110-SM part at
``[1, 1024, 8, 128]`` one set holds about 14.5 MiB, and eight rotating sets
about 73 MiB. Reuse one set and it stays at one set's worth forever.

The entry ceilings are not a memory budget and lowering them does not trade
speed for memory: below the ceiling the retention is the same whatever the
ceiling is, and above it every call rebuilds its plan — about 7.3 ms against a
100 microsecond hit on that part. A deployment that needs the memory back
should rotate fewer buffer sets, or call
``flashinfer.kda_kernels.sm120_prefill.clear_kda_prefill_sm120_caches()``,
which releases all of it.
