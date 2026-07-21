"""Canonical MoE weight container for all moe_ep paths.

Users supply a single :class:`MoEWeightPack` as the ``weights`` argument at
layer construction (:func:`~flashinfer.moe_ep.layer.MoEEpLayer`).
Split and mega kernel plugins materialize backend-specific layouts in
:meth:`~flashinfer.moe_ep.core.kernel.base.SplitKernelBackend.preprocess_weights`
or the mega equivalent — callers never touch per-backend native views directly.

Quantized-or-not is a type, not a pair of optional fields:
:class:`UnquantizedMoEWeights` carries canonical bf16/fp32 tensors,
:class:`PrequantizedMoEWeights` carries packed data plus BOTH block-scale
planes. ``MoEWeightPack(...)`` keeps the historical constructor signature and
returns the matching variant — a pack with exactly one scale set (the old
silent-requantize footgun) now raises at construction instead of falling into
the "unquantized" path downstream. Backends discriminate with
``isinstance(weights, PrequantizedMoEWeights)``; recipe-specific expectations
(DeepGEMM ue8m0-32 vs NVFP4 e4m3-16 vs MXFP8) stay validated per backend in
``preprocess_weights``, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

import torch


class MoEWeightPack:
    """Per-rank expert weights in canonical logical layout (factory base).

    Instantiating this class returns the matching variant:

    * ``MoEWeightPack(w13, w2)`` -> :class:`UnquantizedMoEWeights` —
      ``w13`` ``[local_experts, 2*intermediate, hidden]`` and ``w2``
      ``[local_experts, hidden, intermediate]`` in bf16/fp32.
    * ``MoEWeightPack(w13, w2, w13_scale, w2_scale)`` ->
      :class:`PrequantizedMoEWeights` — packed quantized data (e.g. fp4
      ``[..., hidden // 2]`` / ``[..., intermediate // 2]``) plus both block
      scale planes. Mega DeepGEMM kernels expect ue8m0-packed ``torch.uint8``
      scales with trailing dims ``hidden // 32`` / ``intermediate // 32``;
      NVFP4 CuTeDSL expects fp8-e4m3 per-16 scales (``hidden // 16`` /
      ``intermediate // 16``).

    Supplying exactly one scale raises: that state used to silently select
    the re-quantize-from-bf16 path in every backend, ignoring the provided
    scale.
    """

    # Annotations only — NO class-level defaults here: a plain ``= None`` on
    # the base is visible to the dataclass machinery of the Prequantized
    # subclass (getattr finds the inherited value), silently turning its
    # required scale fields into optional ones and reopening the exact hole
    # this union closes. The unquantized variant provides its Nones as
    # ClassVars instead.
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: Optional[torch.Tensor]
    w2_scale: Optional[torch.Tensor]

    def __new__(
        cls,
        w13: torch.Tensor = None,  # type: ignore[assignment]
        w2: torch.Tensor = None,  # type: ignore[assignment]
        w13_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
    ):
        if cls is not MoEWeightPack:
            # Variant subclasses construct normally (their dataclass __init__
            # enforces their own field set).
            return object.__new__(cls)
        if w13 is None or w2 is None:
            raise TypeError("MoEWeightPack requires w13 and w2")
        if (w13_scale is None) != (w2_scale is None):
            raise ValueError(
                "MoEWeightPack got exactly one of w13_scale/w2_scale — a "
                "pre-quantized pack requires BOTH scale planes (this state "
                "used to silently re-quantize the packed data as if it were "
                "bf16). Pass both scales, or neither."
            )
        if w13_scale is None:
            return UnquantizedMoEWeights(w13=w13, w2=w2)
        return PrequantizedMoEWeights(
            w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale
        )

    # No __init__ here: after __new__ returns a variant, CPython re-runs the
    # VARIANT's dataclass __init__ with the original arguments (type_call uses
    # type(obj).__init__). The signatures match by construction, so the
    # variant's fields are simply re-set to the same values.


@dataclass(frozen=True)
class UnquantizedMoEWeights(MoEWeightPack):
    """Canonical bf16/fp32 expert weights; backend quantizes at preprocess."""

    w13: torch.Tensor
    w2: torch.Tensor
    # ClassVar: readable as ``pack.w13_scale is None`` but NOT a dataclass
    # field, and invisible to the Prequantized variant's field defaults.
    # mypy flags instance->class overrides; the shadowing is the point here.
    w13_scale: ClassVar[None] = None  # type: ignore[misc]
    w2_scale: ClassVar[None] = None  # type: ignore[misc]


@dataclass(frozen=True)
class PrequantizedMoEWeights(MoEWeightPack):
    """Packed quantized expert weights + both block-scale planes (consumed
    verbatim by the backend; no re-quantization)."""

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor


def dummy_moe_weights(
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int = 1,
    device: Union[torch.device, str] = "cpu",
) -> MoEWeightPack:
    """Placeholder weights for identity / comm-only split paths."""
    w13 = torch.zeros(num_local_experts, 2 * intermediate, hidden, device=device)
    w2 = torch.zeros(num_local_experts, hidden, intermediate, device=device)
    return UnquantizedMoEWeights(w13=w13, w2=w2)
