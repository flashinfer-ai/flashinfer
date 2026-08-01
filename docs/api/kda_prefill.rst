.. _apikda_prefill:

flashinfer.kda_prefill
======================

Optimized recurrent Kimi Delta Attention (KDA) prefill support. The
:func:`flashinfer.kda.recurrent_kda` facade dispatches a strict ordinary
multi-token prefill subset to frozen FlashKDA-compatible SM100a kernels.

.. currentmodule:: flashinfer.kda_prefill

.. autosummary::
    :toctree: ../generated

    RecurrentKDAPrefillWorkspace

Optimized B200 prefill subset
-----------------------------

``flashinfer.kda.recurrent_kda`` uses the frozen prefill backend only when
every condition below holds:

* the device has compute capability 10.0;
* input is ordinary multi-token prefill: fixed ``T > 1``, or packed input
  whose total token count is greater than its number of sequences;
* Q, K, V, and G are contiguous BF16 ``[B,T,H,128]`` tensors with one shared
  head count, and beta is contiguous BF16 ``[B,T,H]``;
* ``A_log`` is contiguous FP32 ``[H]`` and ``dt_bias`` is contiguous FP32
  ``[H,128]`` or flattened ``[H*128]``;
* ``use_qk_l2norm_in_kernel=True``, ``use_gate_in_kernel=True``,
  ``beta_is_logit=True``, and ``lower_bound`` is a finite negative value;
* speculative decode, GQA, state indices, committed-state sources, and
  accepted-token/checkpoint features are not enabled.

Calls outside that subset retain the existing CuTe-DSL path. In particular,
T=1 decode and speculative decode are not rerouted.

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

When ``seq_order=None``, a cached identity order is used. Fixed ``B=1,H=64``
selects the two-CTA M64 value-split kernel; every packed input and every other
head count selects M128.

State and graph semantics
-------------------------

The BF16 state layout remains ``[N,H,V,K]`` and an explicitly supplied
``initial_state`` is still updated in place, even when
``output_final_state=False``. The frozen kernels load each CTA's disjoint
state rows before writing the final rows back to the same storage, so no
separate state scratch or copy-back is required. If no initial state is
supplied, a final state is allocated only when ``output_final_state=True``.

The frozen kernel uses restricted output storage. A preallocated ``output``
must not overlap Q, K, V, G, beta, or ``initial_state``.

Eager calls without ``prefill_workspace`` use an internal serialized workspace
for the current CUDA stream. This default workspace is eager-only and cannot
be used during CUDA graph capture.

CUDA graph capture requires a caller-owned
``RecurrentKDAPrefillWorkspace(device)`` and a preallocated ``output``. The
workspace owns optional final-state scratch for calls without an initial
state, beta padding, and separate 768-byte M64 and M128 TMA descriptor blocks.
It binds to the device and CUDA stream of its first ``recurrent_kda`` call.
Warm it eagerly on the intended capture stream with the exact Q, K, V, G,
beta, and output tensors, then synchronize that stream before capture. Packed
graphs must also pass preallocated int64 ``cu_seqlens`` and int32
``seq_order``. The warm call prepares descriptors; capture accepts only the
exact warmed pointer, shape, stride, and dtype signature and performs no
descriptor preparation.

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
directly in place by the frozen kernel. The small-head ``H < 8`` path captures
the beta copy into workspace-owned padded storage before the frozen launch.
