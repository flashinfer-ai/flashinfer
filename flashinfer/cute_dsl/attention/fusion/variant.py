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
     relative positional encoding).  Runs *before* row-max and softmax.
     **Composes with** standard softmax — the modified scores feed into the
     normal exp2 path.

  4. Row-max reduction: ``m = max(m, max(scores))``.

  5. Score-to-weight conversion:
     - Default (``has_logits_transform = False``):
       ``weights = exp2(score * scale - m * scale)``
     - Custom (``has_logits_transform = True``):
       ``weights = transform_logits(score, batch, qo, kv, qo_head, kv_head)``
       Your transform **replaces** the entire exp2 computation.

  6. Row-sum accumulation: ``d += sum(weights)``.

  7. Correction warp: rescale partial output by ``exp2((old_m - new_m) * scale)``.
     **Skipped** when ``has_logits_transform = True`` (the transform is assumed
     to handle its own normalization).

After all KV tiles (final epilogue):

  8. Output normalization:
     - Default (``has_output_transform = False``):
       ``output *= scale_output / d``
     - Custom (``has_output_transform = True``):
       ``output = transform_output(output, batch, qo, qo_head, m, rcp_d, scale)``

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

- ``score_mod`` and ``transform_logits`` are **mutually exclusive**.
  ``score_mod`` adds a bias/transform *before* standard softmax;
  ``transform_logits`` *replaces* softmax entirely.

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

Examples
========

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

Attention sink (update_statistics + transform_output)::

    class AttentionWithSink(AttentionVariant):
        has_statistics_update = True
        has_output_transform = True
        def __init__(self, sink):
            self._sink = sink
        @property
        def extra_params(self):
            return self._sink                        # (H,)
        @cute.jit
        def update_statistics(self, kv_tile_idx, qo_head_idx, m, d, scale):
            sink_val = self.params[qo_head_idx]
            ...

RPE (score_mod with 2-D per-head lookup table)::

    class RPEAttention(AttentionVariant):
        has_score_mod = True
        def __init__(self, rpe_table, max_rel_dist):
            self._rpe = rpe_table
            self._offset = max_rel_dist
        @property
        def extra_params(self):
            return self._rpe                         # (H, 2W+1)
        @cute.jit
        def score_mod(self, score, batch_idx, qo_idx, kv_idx,
                      qo_head_idx, kv_head_idx):
            return score + self.params[qo_head_idx, kv_idx - qo_idx + self._offset]

Soft-capping (compile-time scalars only, no extra_params)::

    class SoftCappingAttention(AttentionVariant):
        has_score_mod = True
        def __init__(self, cap=50.0):
            self.cap = cap
            self.rcp_cap = 1.0 / cap
        @cute.jit
        def score_mod(self, score, ...):
            return self.cap * cute.arch.tanh(score * self.rcp_cap)
"""

import math

import cutlass.cute as cute


class AttentionVariant:
    """Base class for attention variants. Subclass to customize behavior.

    Set the class-level flags to enable compile-time dead code elimination
    via ``cutlass.const_expr``.  Only override the methods corresponding to
    flags you set to ``True``.

    Attributes
    ----------
    has_score_mod : bool
        ``True`` when ``score_mod`` is overridden. Mutually exclusive with
        ``has_logits_transform``.
    has_logits_transform : bool
        ``True`` when ``transform_logits`` is overridden. Mutually exclusive
        with ``has_score_mod``.
    has_statistics_update : bool
        ``True`` when ``update_statistics`` is overridden.
    has_output_transform : bool
        ``True`` when ``transform_output`` is overridden.
    """

    has_score_mod: bool = False
    has_logits_transform: bool = False
    has_statistics_update: bool = False
    has_output_transform: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.has_score_mod and cls.has_logits_transform:
            raise TypeError(
                f"{cls.__name__} sets both has_score_mod and has_logits_transform. "
                "These are mutually exclusive: score_mod composes with standard "
                "softmax, while transform_logits replaces it entirely."
            )

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
    def score_mod(self, score, batch_idx, qo_idx, kv_idx,
                  qo_head_idx, kv_head_idx):
        """Element-wise modification of QK scores before softmax.

        Unlike ``transform_logits``, this **composes with** the standard
        softmax path.  The modified score feeds into the normal
        ``exp2(score * scale - m * scale)`` computation.

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
            Modified score (will flow into standard exp2 softmax).
        """
        return score

    @cute.jit
    def transform_logits(self, score, batch_idx, qo_idx, kv_idx,
                         qo_head_idx, kv_head_idx):
        """Element-wise transform on raw QK dot-product scores.

        When overridden (with ``has_logits_transform = True``), this replaces
        the standard ``exp2(score * scale - max * scale)`` computation. Your
        function **is** the complete score-to-weight conversion.

        Mutually exclusive with ``score_mod``.

        Parameters
        ----------
        score : float32
            Raw QK dot product for one (query, key) pair.
        batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx : int32
            Position and head indices.

        Returns
        -------
        float32
            Transformed score (must be non-negative for correct accumulation).
        """
        return score

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
    def transform_output(self, output, batch_idx, qo_idx, qo_head_idx,
                         m, rcp_d, scale):
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
        log2_e = math.log2(math.exp(1.0))
        sink_raw = (
            self.params[qo_head_idx] * log2_e / scale
            if kv_tile_idx == 0
            else -math.inf
        )
        m_new = sink_raw if sink_raw > m else m
        rescale = cute.arch.exp2((m - m_new) * scale)
        d_new = cute.arch.exp2((sink_raw - m_new) * scale) + d * rescale
        return m_new, d_new

    @cute.jit
    def transform_output(self, output, batch_idx, qo_idx, qo_head_idx,
                         m, rcp_d, scale):
        return output * scale * rcp_d


class SigmoidAttention(AttentionVariant):
    """Sigmoid logits transform — replaces softmax with element-wise sigmoid.

    Useful for quiet-attention / sigmoid-attention variants where each
    position attends independently rather than competing via softmax.

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

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale * math.log2(math.exp(1.0))
        self.bias = bias

    @cute.jit
    def transform_logits(self, x, batch_idx, qo_idx, kv_idx,
                         qo_head_idx, kv_head_idx):
        return cute.arch.rcp_approx(
            1 + cute.arch.exp2(-(x * self.scale + self.bias))
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
    def score_mod(self, score, batch_idx, qo_idx, kv_idx,
                  qo_head_idx, kv_head_idx):
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
            return torch.tensor(
                _get_slopes_power_of_2(num_heads), dtype=torch.float32
            )
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
    def score_mod(self, score, batch_idx, qo_idx, kv_idx,
                  qo_head_idx, kv_head_idx):
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
    def score_mod(self, score, batch_idx, qo_idx, kv_idx,
                  qo_head_idx, kv_head_idx):
        return self.cap * cute.arch.tanh(score * self.rcp_cap)
