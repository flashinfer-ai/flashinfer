.. _apikda_prefill:

flashinfer.kda_prefill
======================

Optimized recurrent Kimi Delta Attention (KDA) prefill support. The
:func:`flashinfer.kda.recurrent_kda` facade exposes frozen Cake and source-level
CuTe DSL implementations for a strict ordinary multi-token prefill subset.

.. currentmodule:: flashinfer.kda_prefill

.. autosummary::
    :toctree: ../generated

    RecurrentKDAPrefillWorkspace

.. currentmodule:: flashinfer.kda

.. autosummary::
    :toctree: ../generated

    RecurrentKDAPrefillWrapper

Backend selection
-----------------

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
-----------------------------------

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
refreshed request. Runtime routing remains device-specific: persistent M128
is restricted to measured 148/152-SM CC 10.0 devices. CC 10.3 uses the direct
schedules, with a tensor-core state-decay specialization for uniform, complete
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
-----------------

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
length. Supplying ``seq_order`` disables persistent host task-bin planning but
does not suppress BT16 or otherwise force direct M128. Remaining eligible
inputs select the shape-appropriate non-persistent or general M128 schedule.
The scalar-prepare/S8 BT16 pair is submitted by one native Cake binding, which
performs both launch plans before enqueueing either kernel so the dependent
launches do not expose a Python/FFI inter-kernel gap. CUDA Graph capture still
records the same two kernels and preserves the workspace contract below.

For eager packed CuTe DSL engine calls, omitting ``seq_order`` builds and
caches a stable decreasing-length order on the host. CuTe DSL decomp retains
the original sequence order because its CTA grid fits in one wave.
``flashinfer.RecurrentKDAPrefillWrapper`` provides the explicit planned path
needed for packed engine CUDA Graph capture: ``plan`` builds the order and the
decomp ``cu_chunks`` prefix, then ``run`` consumes fixed-address buffers. The
decomp prep kernel binary-searches this compact prefix instead of carrying a
dense chunk-to-sequence tensor. The number of sequences, total tokens, and
total BT=16 chunks are fixed by the first plan so the metadata and launch
geometry remain valid across CUDA Graph replays.

State and graph semantics
-------------------------

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
---------------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

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
