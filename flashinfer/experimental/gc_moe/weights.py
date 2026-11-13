"""Weight-layout conversion for the experimental GC MoE API."""

from __future__ import annotations

from . import support
from .kernel_src.gc_moe import transform, translated


def transform_weights(
    l1_weights,
    l2_weights,
    *,
    l1_scale_2=None,
    l2_scale_2=None,
    shared_recipe=None,
    activation="swiglu",
    backend=None,
):
    support.check_cutlass_dsl()
    support.validate_activation(activation)
    resolved = support.resolve_backend(backend)
    if resolved != "sm90" and (l1_scale_2 is not None or l2_scale_2 is not None):
        raise ValueError(
            "l1_scale_2/l2_scale_2 are checkpoint secondary scales of the SM90 "
            f"path; the {resolved!r} backend derives its secondary scales itself"
        )
    if resolved == "sm100" and shared_recipe is not None:
        raise ValueError(
            "shared_recipe is not accepted by the SM100 backend; its shared "
            "expert scales come from the packed K128 pairs"
        )
    with translated("converting GC MoE weights"):
        return transform(
            resolved,
            l1_weights,
            l2_weights,
            l1_scale_2=l1_scale_2,
            l2_scale_2=l2_scale_2,
            shared_recipe=shared_recipe,
            activation=activation,
        )
