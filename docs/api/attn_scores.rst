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

FP8 selects 64-bit store offsets when the physical output span, including row
padding, reaches ``2**31`` elements. Logical dimensions still fit int32. This
specialization has small-output and lowering coverage. The H64/Q8192/1M
benchmark uses a full-score buffer of ``2**33`` FP32 elements (about 8.6 billion),
selects the Int64 store path, and passes all-row FP32 checks before and after
timing. There is no dedicated unit test at that scale. FP4 continues to reject
spans of ``2**31`` elements or larger.

The FP8 scorer accepts ``math_mode="relu_strict"`` (default) or
``"relu_fast"``. Strict uses NaN-aware ReLU with the full scale. Fast FP32
epilogues use packed ``x + abs(x)`` and one half-scale compensation:
intermediate overflow, tiny-scale underflow and exceptional-value behavior
can differ. FP16 epilogue arithmetic is unchanged; FP16 output storage alone
does not select FP16 arithmetic. The trace reference models FP32 accumulation
and epilogue with FP32 or FP16 output storage, rejecting other traced arithmetic
precisions independently of the public scorer's supported FP16 arithmetic.

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

Selective v2 block-scaled FP8 prefill
-----------------------------------

The continuous-K ``FP8MQATopKWrapper(backend="v2", qkv_dtype="mxfp8")``
accepts E4M3FN ``q[R,H,128]`` and ``k[N,128]`` with one UE8M0 scale per 32
values. Pass ``q_scale`` as ``int32[R,H]`` or ``uint8[R,H,4]`` and ``k_scale``
as ``uint8[N,4]``. Each int32 packs four consecutive scale bytes. Physical K
storage must contain a multiple of 256 rows; ``plan`` describes the live
causal extent. Supported heads are 4, 32 and 64, with D128 and one request.

These buffers can be produced directly by DeepGEMM's
``per_token_cast_to_fp8(..., use_ue8m0=True, gran_k=32,
use_packed_ue8m0=True)``. A comparison should quantize once outside both
measurements and pass the same tensors and scale bytes to both operators.
The legacy ``qkv_dtype="fp8"`` format has an FP32 per-token K scale and no Q
scale; it is a different numerical contract.

MXFP8 supports FP32, FP16 and BF16 logits storage. The default uses FP32
head accumulation. H4, H32 and H64 also accept explicit ``math_mode="native_bf16"``
or ``"native_fp16"`` with the corresponding logits dtype. Only the native
BF16 contract matches the SM100 DeepGEMM BF16 head arithmetic. FP16/BF16
output storage alone does not change accumulation precision. MXFP8 always
uses the general ReLU path; FP4 integer shortcuts do not apply.

The four selective stages retain sampling, threshold filtering, candidate
selection and exact recovery. K2 writes retained candidates, rather than the
full logits matrix. Use ``benchmarks/bench_real_input_data.py`` with
``--qkv fp8 --fp8-quantization deepgemm --scope api --staged-inputs`` and
``--arms v2 deepgemm deepgemm_prepared --gvr-root <GVR-directory>`` to share
quantized inputs. Select matching accumulation and logits dtypes before
reporting a speed comparison.

Backend v2 accepts ``weights_dtype=torch.bfloat16`` in addition to the
default ``torch.float32`` for all supported QK and logits dtypes. Pass
weights with the declared dtype to every run. BF16 weights may use padded
rows with contiguous heads, so DeepGEMM's H4 weights with row stride eight
can be shared directly. Staging retains BF16 storage; native BF16 kernels
load adjacent weights as packed pairs and reuse them in the epilogue.
Weights storage and head accumulation are independent: select
``math_mode="native_bf16"`` explicitly to match DeepGEMM's arithmetic.
Backend v1 still requires FP32 weights.
