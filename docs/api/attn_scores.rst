.. _apiattnscores:

flashinfer.attn_scores
======================

Paged MQA logits ("attention scores") kernels for Blackwell (SM100/SM103).

These compute, for every request and KV position, the per-head weighted sum of
rectified query-key scores that a sparse-attention indexer uses to choose which
KV tokens to keep:

.. math::

  \mathrm{logits}[t, p] = \sum_h w_{t,h} \cdot \mathrm{relu}\left( q_{t,h} \cdot k_p \right)

Two things about that expression are easy to get wrong:

* The rectifier is applied **per head, before** the weighted sum -- not to the
  summed score. Because the weights may be negative, the result may be too.
* :func:`fp8_paged_mqa_logits` additionally multiplies by the per-token FP32 KV
  scale carried in the tail of each ``kv_fused`` row, so its result is
  :math:`s_p` times the above. :func:`fp4_paged_mqa_logits` has no such
  per-position factor: MXFP4 block scales are folded into the dequantised
  values themselves.

.. note::

  Within a request's context the kernels write every position unconditionally
  and apply no causal or context mask, so callers must mask positions beyond
  each request's context length themselves. A request with
  ``context_lens[b] == 0`` is skipped entirely and its output row is never
  written -- when passing ``out=``, initialise it if you intend to read those
  rows.

.. currentmodule:: flashinfer

Paged MQA Logits
----------------

.. autosummary::
  :toctree: ../generated

  fp8_paged_mqa_logits
  fp4_paged_mqa_logits

.. autofunction:: fp8_paged_mqa_logits

.. autofunction:: fp4_paged_mqa_logits

Scheduling and Setup
--------------------

The persistent kernels need a per-call CTA work assignment. It is computed on
the GPU by default, so the whole dispatch can be captured in a CUDA graph with
no host round-trip. :func:`compute_paged_mqa_logits_schedule` lets a caller
build it once and pass it back via ``schedule_meta=``.

.. warning::

  A reused schedule is only valid while ``ceil(context_lens / 256)`` is
  unchanged for every request. Recompute it whenever a context length crosses
  a 256-token boundary, including on every CUDA-graph replay.

.. autosummary::
  :toctree: ../generated

  attn_scores.compute_paged_mqa_logits_schedule
  attn_scores.padded_context_len
  attn_scores.precompile_paged_mqa_logits
