.. _apiattnscores:

flashinfer.attn_scores
======================

Paged MQA logits ("attention scores") kernels for datacentre Blackwell
(SM100/SM103) and Rubin (SM107).

These compute, for every request and KV position, the per-head weighted sum of
rectified query-key scores that a sparse-attention indexer uses to choose which
KV tokens to keep:

.. math::

  \mathrm{logits}[t, p] = \sum_h w_{t,h} \cdot \mathrm{relu}\left( q_{t,h} \cdot k_p \right)

Two things about that expression are easy to get wrong:

* The rectifier is applied **per head, before** the weighted sum -- not to the
  summed score. Because the weights may be negative, the result may be too.
* :func:`fp8_paged_mqa_logits` additionally multiplies by the per-token FP32 KV
  scale carried in the trailing scale region of each ``kv_fused`` block, so its
  result is
  :math:`s_p` times the above. :func:`fp4_paged_mqa_logits` has no such
  per-position factor: MXFP4 block scales are folded into the dequantised
  values themselves. The asymmetry is algebraic -- a scale uniform over a
  token's whole K vector factors out of the dot product (and, being positive,
  commutes with the ReLU), while per-32-element-group scales on both operands
  do not.

The variants also differ beyond the scales. :func:`fp4_paged_mqa_logits` takes
the extra ``q_sf`` scale tensor (plus ``sf_vec_size`` and
``is_kv_sf_interleaved``) and packs two FP4 values per ``q`` byte. Only
:func:`fp8_paged_mqa_logits` exposes ``acc_dtype`` (fp4's MMA accumulator is
fixed float32), and the default ``output_dtype`` is float32 (fp8) versus
bfloat16 (fp4). fp8 accepts parametric shapes while fp4 requires exactly
``num_heads=64``, ``head_dim=128``. Finally, ``next_n=4`` is single-pass on
every supported arch for fp8, but for fp4 only on Rubin (SM107) --
SM100/SM103 run it as two internal KV passes; build any caller
``schedule_meta`` with the schedule helper's matching ``next_n``/``variant``
arguments. Paging arguments, the output contract, and
``out=``/``schedule_meta`` semantics are identical across the pair.

.. note::

  Within each row's scheduled extent the kernels write unconditionally and
  apply no causal or length mask; columns beyond that extent may never be
  written at all.  Output row ``b*next_n + t`` is meaningful only for KV
  positions ``0 .. seq_lens[b] - next_n + t`` inclusive (the newest slot
  ``t = next_n - 1`` sees the whole sequence, each earlier slot one position
  fewer); callers must mask or slice past that per-slot limit themselves.  A
  request with ``seq_lens[b] == 0`` is skipped entirely and its output
  rows are never written -- when passing ``out=``, initialise it if you
  intend to read those rows.

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
build it once and pass it back via ``schedule_meta=``. Pass the helper the
same ``variant`` and ``next_n`` as the main call: it applies the kernel's
internal scheduling policy for that configuration, so the returned tensor
(treat it as opaque) is correct on every architecture.

.. warning::

  A reused schedule is only valid while the contents of ``seq_lens`` are
  unchanged for every request. Recompute it whenever ``seq_lens`` changes,
  including on every CUDA-graph replay.

.. autosummary::
  :toctree: ../generated

  compute_paged_mqa_logits_schedule
  min_block_table_width
  padded_seq_len
  precompile_paged_mqa_logits
