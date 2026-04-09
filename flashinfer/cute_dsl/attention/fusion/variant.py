# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AttentionVariant — customization point for attention behavior.

Subclass AttentionVariant to create custom attention behaviors. The hooks
are co-defined on a single object so that coupled invariants are naturally
enforced.

Execution Order
===============

For each query row, the kernel iterates over KV tiles. Within each tile:

  1. ``update_statistics(kv_tile_idx, qo_head_idx, m, d, scale)``
     Modify the online softmax running statistics *before* the tile's QK
     scores are processed. Use this to inject virtual tokens (e.g. attention
     sink) into the softmax denominator.

  2. Masking (causal / sliding window / residual) is applied to QK scores.

  3. ``score_mod(score, batch, qo, kv, qo_head, kv_head)`` — optional
     element-wise modification of QK scores (e.g. ALiBi bias, soft-capping,
     relative positional encoding).  Runs *before* the score-to-weight
     conversion. **Composes with both** standard softmax and custom
     ``transform_logits``.

  4. Score-to-weight conversion:
     - Default (``has_logits_transform = False``):
       Row-max reduction, then ``weights = exp2(score * scale - m * scale)``,
       row-sum accumulation, correction warp rescaling.
     - Custom (``has_logits_transform = True``):
       ``weights = transform_logits(score)``
       Your transform **replaces** the entire softmax machinery (row-max,
       exp2, row-sum, correction).  Must produce non-negative values.

After all KV tiles (final epilogue):

  5. Output normalization:
     - Default (``has_output_transform = False``):
       ``output *= scale_output / d``
     - Custom (``has_output_transform = True``):
       ``output = transform_output(output, batch, qo, qo_head, m, rcp_d, scale)``

Composability
=============

``score_mod`` and ``transform_logits`` are **composable**:

- ``score_mod`` modifies scores (position-dependent bias, capping, etc.)
- ``transform_logits`` replaces the activation function (sigmoid, relu, etc.)

When both are set, scores flow through ``score_mod`` first, then
``transform_logits``.  When only ``score_mod`` is set, scores flow into
the standard softmax (exp2) path.

Variable Domains
================

``m`` (row_max)
    Raw logit domain — the actual maximum QK dot-product value,
    **not** multiplied by ``scale``.

``d`` (row_sum)
    Accumulated sum of ``exp2((score - m) * scale)`` across all tiles.
    When ``update_statistics`` injects virtual tokens, ``d`` includes their
    contributions.

``scale``
    ``log2(e) * sm_scale`` where ``sm_scale = 1 / sqrt(head_dim)``.
    The kernel uses base-2 exponentials for hardware efficiency:
    ``exp(x * sm_scale) == exp2(x * sm_scale * log2(e)) == exp2(x * scale)``.

``rcp_d``
    ``1.0 / d`` — reciprocal of the softmax denominator.

``scale_output``
    Output scaling factor, typically ``1.0``.

Coupling Rules
==============

- If ``update_statistics`` modifies ``d`` (adds virtual tokens), then
  ``transform_output`` **must** account for the modified denominator.
  The default ``output * scale_output / d`` works when ``d`` is unmodified.
  With sink tokens, override ``transform_output`` to use
  ``output * scale * rcp_d`` (where ``rcp_d = 1/d`` already reflects the
  sink contribution).

- If ``transform_logits`` is provided, it replaces the **entire**
  ``exp2(score * scale - m * scale)`` conversion. The correction warp will
  **not** rescale intermediate outputs. Your transform must produce
  non-negative values for correct accumulation.

Runtime Parameters
==================

Variants that need runtime tensor data (e.g. ALiBi slopes, sink values,
RPE tables) expose it via the ``extra_params`` property. The wrapper
converts the tensor to CuTe format and passes it through the kernel; the
variant accesses it as ``self.params`` inside ``@cute.jit`` methods.

``extra_params``
    Python-side property: returns a ``torch.Tensor`` of any shape, or
    ``None``.  Read by the wrapper at ``plan()`` time.

``self.params``
    JIT-side attribute: the kernel binds the CuTe tensor to this name
    before invoking any variant hook.  Access it with natural indexing
    (e.g. ``self.params[head_idx]`` for 1-D, ``self.params[head, offset]``
    for 2-D).

Compile-time scalars (set in ``__init__``, e.g. ``self.cap = 50.0``)
are traced directly by the JIT compiler — no ``extra_params`` needed.

Hardware Primitives
===================

The following hardware-mapped primitives are available for use in
``transform_logits`` and ``score_mod``::

    cute.arch.exp2(x)       # MUFU.EX2  — base-2 exponential (approx)
    cute.arch.rcp_approx(x) # MUFU.RCP  — reciprocal (approx)
    tanh_approx(x)          # MUFU.TANH — hyperbolic tangent (approx)

Each maps to a single-cycle MUFU instruction. Import ``tanh_approx``
from this module.

Examples
========

Sigmoid attention (exp2 + rcp, 2 MUFU ops/element)::

    class SigmoidAttention(AttentionVariant):
        has_logits_transform = True
        def __init__(self, scale=1.0, bias=0.0):
            self.scale = scale * math.log2(math.exp(1.0))
            self.bias = bias * math.log2(math.exp(1.0))
        @cute.jit
        def transform_logits(self, score):
            return cute.arch.rcp_approx(
                1 + cute.arch.exp2(-(score * self.scale + self.bias)))

Sigmoid attention via tanh (1 MUFU op/element)::

    class SigmoidTanhAttention(AttentionVariant):
        has_logits_transform = True
        def __init__(self, scale=1.0, bias=0.0):
            self.half_scale = scale / 2.0
            self.half_bias = bias / 2.0
        @cute.jit
        def transform_logits(self, score):
            return 0.5 + 0.5 * tanh_approx(
                score * self.half_scale + self.half_bias)

ALiBi (score_mod with 1-D per-head slopes)::

    class ALiBiAttention(AttentionVariant):
        has_score_mod = True
        def __init__(self, alibi_slopes):
            self._slopes = alibi_slopes
        @property
        def extra_params(self):
            return self._slopes                      # (H,)
        @cute.jit
        def score_mod(self, score, batch_idx, qo_idx, kv_idx,
                      qo_head_idx, kv_head_idx):
            return score + self.params[qo_head_idx] * (kv_idx - qo_idx)

Soft-capping (compile-time scalars only, no extra_params)::

    class SoftCappingAttention(AttentionVariant):
        has_score_mod = True
        def __init__(self, cap=50.0):
            self.cap = cap
            self.rcp_cap = 1.0 / cap
        @cute.jit
        def score_mod(self, score, ...):
            return self.cap * tanh_approx(score * self.rcp_cap)
"""

import math
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32


@dsl_user_op
def tanh_approx(a, *, loc=None, ip=None):
    """Hardware tanh via MUFU.TANH — single-cycle approximation (SM75+)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class AttentionVariant:
    """Base class for attention variants. Subclass to customize behavior.

    Set the class-level flags to enable compile-time dead code elimination
    via ``cutlass.const_expr``.  Only override the methods corresponding to
    flags you set to ``True``.

    Attributes
    ----------
    has_score_mod : bool
        ``True`` when ``score_mod`` is overridden.  Composes with both
        standard softmax and custom ``transform_logits``.
    has_logits_transform : bool
        ``True`` when ``transform_logits`` is overridden.  Replaces the
        entire softmax machinery (row-max, exp2, row-sum, correction).
    has_vectorized_logits_transform : bool
        ``True`` when ``transform_logits_vec`` is overridden.  Implies
        ``has_logits_transform = True``.  The kernel calls
        ``transform_logits_vec`` instead of the per-element
        ``transform_logits``, enabling stride-2 iteration and packed
        f32x2 operations for higher throughput.
    has_statistics_update : bool
        ``True`` when ``update_statistics`` is overridden.
    has_output_transform : bool
        ``True`` when ``transform_output`` is overridden.
    """

    has_score_mod: bool = False
    has_logits_transform: bool = False
    has_vectorized_logits_transform: bool = False
    has_statistics_update: bool = False
    has_output_transform: bool = False

    params: Any = None

    @property
    def extra_params(self):
        """Return a PyTorch tensor of runtime data for JIT methods, or None.

        The tensor can be any shape. It is converted to a CuTe tensor and
        bound to ``self.params`` before JIT methods are called. Inside
        ``@cute.jit`` methods, access it as ``self.params[...]``.

        Override this in subclasses that need runtime tensor data.
        """
        return None

    @cute.jit
    def score_mod(self, score, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx):
        """Element-wise modification of QK scores.

        Composes with both standard softmax and custom ``transform_logits``.
        The modified score feeds into whichever score-to-weight conversion
        is active.

        Typical uses: ALiBi bias, relative positional encoding, soft-capping.

        Parameters
        ----------
        score : float32
            Raw QK dot product for one (query, key) pair.
        batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx : int32
            Position and head indices.

        Returns
        -------
        float32
            Modified score.
        """
        return score

    @cute.jit
    def transform_logits(self, score):
        """Coordinate-free activation replacing softmax.

        When overridden (with ``has_logits_transform = True``), this replaces
        the standard ``exp2(score * scale - max * scale)`` computation and
        the entire online softmax machinery (row-max, row-sum, correction).

        The kernel calls this in a tight loop over register elements without
        coordinate lookups.  Position-dependent modifications belong in
        ``score_mod``, which runs before this and composes naturally.

        Parameters
        ----------
        score : float32
            QK dot product (possibly modified by ``score_mod``).

        Returns
        -------
        float32
            Transformed score (must be non-negative for correct accumulation).
        """
        return score

    @cute.jit
    def transform_logits_vec(self, scores):
        """Vectorized logits transform over a register fragment.

        Optional performance override.  When provided (with
        ``has_vectorized_logits_transform = True``), the kernel calls this
        instead of the per-element ``transform_logits``, enabling stride-2
        iteration and packed f32x2 operations.

        Like ``transform_logits``, this is coordinate-free: the fragment
        contains raw scores (possibly already modified by ``score_mod``).

        **Why this exists:** The CuTe DSL compiler (as of v4.3.5) does not
        auto-pack adjacent scalar ``fma.rn.f32`` instructions into
        ``fma.rz.ftz.f32x2`` packed ALU ops.  This override lets you
        explicitly use ``cute.arch.fma_packed_f32x2`` for ~2x ALU
        throughput on the arithmetic surrounding MUFU calls.  Once the
        compiler learns to pack scalar FMAs automatically, this override
        will become unnecessary and can be deprecated.

        Parameters
        ----------
        scores : cute.Tensor (register fragment)
            Mutable fragment of scores. Modify elements in-place.
        """
        pass

    @cute.jit
    def update_statistics(self, kv_tile_idx, qo_head_idx, m, d, scale):
        """Modify online softmax running statistics before processing a KV tile.

        Called once per KV tile, before the tile's scores are loaded. Use to
        inject virtual tokens into the softmax computation (e.g. attention
        sink).

        Parameters
        ----------
        kv_tile_idx : int32
            Index of the current KV tile (0-based).
        qo_head_idx : int32
            Query/output head index.
        m : float32
            Current row maximum (raw logit domain, **not** scaled).
        d : float32
            Current row sum of exponentiated scores.
        scale : float32
            ``log2(e) * sm_scale``, the base-2 scaling factor.

        Returns
        -------
        tuple[float32, float32]
            ``(m_new, d_new)`` — updated running statistics.
        """
        return m, d

    @cute.jit
    def transform_output(self, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale):
        """Element-wise transform on final output values.

        Called once per output element after all KV tiles are processed,
        replacing the default ``output *= scale_output / d`` normalization.

        Parameters
        ----------
        output : float32
            Accumulated attention output value (unnormalized).
        batch_idx, qo_idx, qo_head_idx : int32
            Batch, position, and head indices.
        m : float32
            Final row maximum (raw logit domain).
        rcp_d : float32
            ``1.0 / row_sum`` (reciprocal of softmax denominator).
        scale : float32
            Output scaling factor (``scale_output``, typically ``1.0``).

        Returns
        -------
        float32
            Transformed output value.
        """
        return output * rcp_d


class StandardAttention(AttentionVariant):
    """Standard softmax attention — no customization."""

    pass


class AttentionWithSink(AttentionVariant):
    """Attention with a virtual sink token per head.

    Adds a learnable per-head bias to the softmax denominator on the first
    KV tile, preventing attention collapse in long sequences.

    ``update_statistics`` injects the sink into the running ``(m, d)``
    and ``transform_output`` normalises with the modified denominator.

    Parameters
    ----------
    sink : torch.Tensor
        1-D tensor of shape ``(num_qo_heads,)`` with per-head sink values.

    Usage::

        wrapper.plan(..., variant=AttentionWithSink(sink_tensor))
        o = wrapper.run(q, k, v)
    """

    has_statistics_update = True
    has_output_transform = True

    def __init__(self, sink):
        self._sink = sink

    @property
    def extra_params(self):
        return self._sink

    @cute.jit
    def update_statistics(self, kv_tile_idx, qo_head_idx, m, d, scale):
        # Guard: on non-first tiles, return (m, d) unchanged.  Computing
        # with sink_raw = -inf when m is also -inf (initial state on split-KV
        # CTAs that don't own tile 0) would produce -inf - (-inf) = NaN.
        m_new = m
        d_new = d
        if kv_tile_idx == 0:
            log2_e = math.log2(math.exp(1.0))
            sink_raw = self.params[qo_head_idx] * log2_e / scale
            m_new = sink_raw if sink_raw > m else m
            rescale = cute.arch.exp2((m - m_new) * scale)
            d_new = cute.arch.exp2((sink_raw - m_new) * scale) + d * rescale
        return m_new, d_new

    @cute.jit
    def transform_output(self, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale):
        return output * scale * rcp_d


class SigmoidAttention(AttentionVariant):
    """Sigmoid logits transform — replaces softmax with element-wise sigmoid.

    Uses ``rcp_approx(1 + exp2(-x))`` (2 MUFU ops per element).
    For a faster variant using tanh (1 MUFU op), see ``SigmoidTanhAttention``.

    Composes with ``score_mod`` — set both ``has_score_mod`` and
    ``has_logits_transform`` on a subclass to combine position-dependent
    score modification with sigmoid activation.

    Parameters
    ----------
    scale : float
        Multiplicative scale applied to QK scores before sigmoid.
        Typically ``1.0`` (or ``sm_scale`` if you want to bake it in).
    bias : float
        Additive bias applied to scaled scores before sigmoid.

    Usage::

        wrapper.plan(..., sm_scale=1.0, variant=SigmoidAttention(scale=1.0))
    """

    has_logits_transform = True
    has_vectorized_logits_transform = True

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale * math.log2(math.exp(1.0))
        self.bias = bias * math.log2(math.exp(1.0))

    @cute.jit
    def transform_logits(self, score):
        return cute.arch.rcp_approx(
            1 + cute.arch.exp2(-(score * self.scale + self.bias))
        )

    @cute.jit
    def transform_logits_vec(self, scores):
        for i in cutlass.range_constexpr(0, cute.size(scores), 2):
            scores[i] = cute.arch.rcp_approx(
                1 + cute.arch.exp2(-(scores[i] * self.scale + self.bias))
            )
            scores[i + 1] = cute.arch.rcp_approx(
                1 + cute.arch.exp2(-(scores[i + 1] * self.scale + self.bias))
            )


class SigmoidTanhAttention(AttentionVariant):
    """Sigmoid via tanh — MUFU-efficient sigmoid that replaces softmax.

    Uses the identity ``sigmoid(x) = 0.5 + 0.5 * tanh(x / 2)`` to replace
    the exp2 + rcp_approx pair (2 MUFU ops/element) with a single tanh_approx
    (1 MUFU op/element), matching softmax's MUFU budget.

    Parameters
    ----------
    scale : float
        Multiplicative scale applied to QK scores before sigmoid.
        Typically ``1.0`` (or ``sm_scale`` if you want to bake it in).
    bias : float
        Additive bias applied to scaled scores before sigmoid.

    Usage::

        wrapper.plan(..., sm_scale=1.0, variant=SigmoidTanhAttention(scale=1.0))
    """

    has_logits_transform = True
    has_vectorized_logits_transform = True

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.half_scale = scale / 2.0
        self.half_bias = bias / 2.0

    @cute.jit
    def transform_logits(self, score):
        return 0.5 + 0.5 * tanh_approx(score * self.half_scale + self.half_bias)

    @cute.jit
    def transform_logits_vec(self, scores):
        for i in cutlass.range_constexpr(0, cute.size(scores), 2):
            scores[i], scores[i + 1] = cute.arch.fma_packed_f32x2(
                (scores[i], scores[i + 1]),
                (self.half_scale, self.half_scale),
                (self.half_bias, self.half_bias),
            )
            scores[i] = tanh_approx(scores[i])
            scores[i + 1] = tanh_approx(scores[i + 1])
            scores[i], scores[i + 1] = cute.arch.fma_packed_f32x2(
                (scores[i], scores[i + 1]),
                (0.5, 0.5),
                (0.5, 0.5),
            )


class ALiBiAttention(AttentionVariant):
    """ALiBi (Attention with Linear Biases) — adds position-dependent bias.

    Adds ``slope * (kv_pos - qo_pos)`` to each QK score before softmax.
    This composes with the standard exp2 softmax path via ``score_mod``,
    so the kernel's online softmax and correction logic remain unchanged.

    Parameters
    ----------
    alibi_slopes : torch.Tensor
        1-D tensor of shape ``(num_qo_heads,)`` with per-head slopes.

    Usage::

        slopes = ALiBiAttention.get_slopes(num_heads).cuda()
        wrapper.plan(..., variant=ALiBiAttention(slopes))
        o = wrapper.run(q, k, v)

    Reference: https://arxiv.org/abs/2108.12409
    """

    has_score_mod = True

    def __init__(self, alibi_slopes):
        self._slopes = alibi_slopes

    @property
    def extra_params(self):
        return self._slopes

    @cute.jit
    def score_mod(self, score, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx):
        return score + self.params[qo_head_idx] * (kv_idx - qo_idx)

    @staticmethod
    def get_slopes(num_heads: int):
        """Return the standard ALiBi slope schedule for ``num_heads`` heads.

        When ``num_heads`` is a power of 2, slopes are
        ``2^{-8/n}, 2^{-16/n}, ..., 2^{-8}`` where ``n = num_heads``.
        Otherwise, the nearest larger power-of-2 slopes are interpolated.
        """
        import torch

        def _get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return torch.tensor(_get_slopes_power_of_2(num_heads), dtype=torch.float32)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = _get_slopes_power_of_2(closest_power_of_2)
            extra = _get_slopes_power_of_2(2 * closest_power_of_2)
            slopes += extra[0::2][: num_heads - closest_power_of_2]
            return torch.tensor(slopes[:num_heads], dtype=torch.float32)


class RPEAttention(AttentionVariant):
    """Relative Positional Encoding via a learned per-head bias table.

    Adds ``rpe_table[head, clamp(kv - qo + max_rel_dist, 0, 2*max_rel_dist)]``
    to each QK score before softmax.  Uses ``score_mod`` so it composes
    with the standard softmax path.

    Parameters
    ----------
    rpe_table : torch.Tensor
        2-D tensor of shape ``(num_qo_heads, 2 * max_rel_dist + 1)``.
    max_rel_dist : int
        Maximum relative distance.  Positions beyond this are clamped.

    Usage::

        wrapper.plan(..., variant=RPEAttention(rpe_table, max_rel_dist=64))
        o = wrapper.run(q, k, v)
    """

    has_score_mod = True

    def __init__(self, rpe_table, max_rel_dist: int):
        self._rpe_table = rpe_table
        self._offset = max_rel_dist
        self._table_size = 2 * max_rel_dist + 1

    @property
    def extra_params(self):
        return self._rpe_table

    @cute.jit
    def score_mod(self, score, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx):
        rel_pos = kv_idx - qo_idx + self._offset
        rel_pos_clamped = rel_pos if rel_pos >= 0 else 0
        rel_pos_clamped = (
            rel_pos_clamped
            if rel_pos_clamped < self._table_size
            else self._table_size - 1
        )
        return score + self.params[qo_head_idx, rel_pos_clamped]


class SoftCappingAttention(AttentionVariant):
    """Soft-capping — prevents logits from growing excessively large.

    Applies ``cap * tanh(score / cap)`` before softmax, bounding scores
    to ``[-cap, +cap]``.  Uses ``score_mod`` so it composes with the
    standard softmax path.

    Parameters
    ----------
    cap : float
        Soft-capping value (e.g. 50.0 for Gemma-2).

    Usage::

        wrapper.plan(..., variant=SoftCappingAttention(cap=50.0))
        o = wrapper.run(q, k, v)

    Reference: Gemma-2 (https://arxiv.org/abs/2408.00118)
    """

    has_score_mod = True

    def __init__(self, cap: float = 50.0):
        self.cap = cap
        self.rcp_cap = 1.0 / cap

    @cute.jit
    def score_mod(self, score, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx):
        return self.cap * tanh_approx(score * self.rcp_cap)
