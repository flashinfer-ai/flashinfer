"""MoEEpLayer — factory for split (dispatch/combine) and mega (fused) EP paths."""

from __future__ import annotations

import warnings
from typing import Sequence, Union

from .algo_knobs import AlgoKnob
from .config import BootstrapConfig, FleetParams
from .core.kernel.registry import is_mega_kernel_config, is_split_kernel_config
from .modes.config import MegaConfig, SplitConfig
from .modes.mega_layer import MoEEpMegaLayer
from .modes.split_layer import MoEEpSplitLayer
from .weights import MoEWeightPack

__all__ = ["MoEEpLayer", "MoEEpMegaLayer", "MoEEpSplitLayer"]


def MoEEpLayer(
    bootstrap: BootstrapConfig,
    fleet_params: FleetParams,
    weights: MoEWeightPack,
    fleet_knobs: Sequence[AlgoKnob] = (),
    backend: Union[str, SplitConfig, MegaConfig, object] = "nccl_ep",
) -> Union[MoEEpSplitLayer, MoEEpMegaLayer]:
    """Construct the appropriate EP layer for ``backend``.

    - ``SplitConfig`` or a comm string/config → :class:`MoEEpSplitLayer`
    - ``MegaConfig`` → :class:`MoEEpMegaLayer`

    ``weights`` is the canonical :class:`~flashinfer.moe_ep.weights.MoEWeightPack`
    holding this rank's expert weights; it is validated and (depending on the
    kernel) preprocessed at construction.
    """
    if isinstance(backend, MegaConfig):
        if fleet_knobs:
            warnings.warn(
                "fleet_knobs are ignored for MegaConfig; the mega path does not "
                "use EP transport backends",
                UserWarning,
                stacklevel=2,
            )
        return MoEEpMegaLayer(bootstrap, fleet_params, weights, backend)

    if is_mega_kernel_config(backend):
        raise TypeError(
            f"Wrap {type(backend).__name__} in MegaConfig(megakernel=...); "
            "raw megakernel configs cannot be passed as backend=."
        )

    if is_split_kernel_config(backend):
        raise TypeError(
            f"Pass split inner kernels via SplitConfig(kernel={type(backend).__name__}(...)); "
            f"got raw {type(backend).__name__} as backend=."
        )

    return MoEEpSplitLayer(
        bootstrap, fleet_params, weights, fleet_knobs, backend=backend
    )
