.. _apiattnscores:

flashinfer.attn_scores
======================

Paged MQA logits ("attention scores") kernels for Blackwell (SM100/SM103).

These compute, for every request and KV position, the per-head weighted sum of
rectified query-key scores that a sparse-attention indexer uses to choose which
KV tokens to keep:

.. math::

  \mathrm{logits}[t, p] = s_p \cdot \sum_h w_{t,h} \cdot \mathrm{relu}\left( q_{t,h} \cdot k_p \right)

Note that the rectifier is applied **per head**, before the weighted sum.

.. note::

  The kernels write every position unconditionally and do not apply a
  causal or context mask. Callers must mask positions beyond each request's
  context length themselves.

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
