.. _apikda_prefill:

flashinfer.kda_prefill
======================

Optimized recurrent Kimi Delta Attention (KDA) prefill support. The
:func:`flashinfer.kda.recurrent_kda` facade dispatches a strict ordinary
multi-token prefill subset to frozen FlashKDA-compatible SM100-family kernels.

.. currentmodule:: flashinfer.kda_prefill

.. autosummary::
    :toctree: ../generated

    RecurrentKDAPrefillWorkspace

Optimized Blackwell prefill subset
-----------------------------------

``flashinfer.kda.recurrent_kda`` uses the frozen prefill backend only when
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

Calls outside that subset retain the existing CuTe-DSL path. In particular,
T=1 decode and speculative decode are not rerouted.

CUDA 12.8 predates the family target, so CC 10.0 uses legacy exact
``sm_100a`` modules. With CUDA 12.9 or newer, JIT and AOT compile one
``sm_100f`` module per schedule for both CC 10.0 and CC 10.3. Cache keys also
include the frozen module identity so an older schedule cannot satisfy a
refreshed request. Runtime routing remains device-specific: persistent M128
is restricted to measured 148/152-SM CC 10.0 devices, while CC 10.3 uses the
direct schedules. On either capability, fixed-layout calls with at most eight
total sequence/head tasks, at most eight heads, and at least 2,048 tokens per
sequence use the small-BH owner/helper schedule when all eight CTAs per task
can reside concurrently. Calls outside that measured region continue through
the existing direct or fallback route; it is not a public-input allowlist.

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
``cu_seqlens[0] == 0``, entries are strictly increasing (every sequence is
non-empty), and ``cu_seqlens[-1] == total_tokens``. FlashInfer does not
synchronize the device to inspect these values; invalid offsets may cause
out-of-bounds device access.

Packed scheduling
-----------------

Packed prefill optionally accepts ``seq_order``, a contiguous CUDA int32
tensor with one entry per sequence. It is a caller contract that this tensor
is a permutation of ``[0, N)``. Ordering sequences by decreasing length
reduces the final partial wave. FlashInfer validates dtype, device, rank, and
size without synchronizing the device to inspect permutation values.

When ``seq_order=None``, a cached identity order is used. H12 selects the
dedicated M128 schedule with a 16-token recurrence chunk for both fixed and
packed layouts. Fixed ``B=1,H=64`` selects the two-CTA M64 value-split kernel;
the fixed small-BH region described above selects its eight-CTA owner/helper
schedule; all remaining eligible inputs select the general 32-token M128
schedule.

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
positive ``checkpoint_every_n_tokens`` divisible by 32. KDA checkpoints are
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
packet ring with its generation counters. It binds to the device and CUDA
stream of its first ``recurrent_kda`` call.
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
