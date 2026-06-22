"""MoEEpLayer — factory for split (dispatch/combine) and mega (fused) EP paths."""

from __future__ import annotations

import warnings
from typing import Sequence, Union

from .algo_knobs import AlgoKnob
from .config import BootstrapConfig, FleetParams
from .modes.config import MegaConfig, SplitConfig
from .modes.mega_layer import MoEEpMegaLayer
from .modes.split_layer import MoEEpSplitLayer

__all__ = ["MoEEpLayer", "MoEEpMegaLayer", "MoEEpSplitLayer"]


def MoEEpLayer(
    bootstrap: BootstrapConfig,
    fleet_params: FleetParams,
    fleet_knobs: Sequence[AlgoKnob] = (),
    backend: Union[str, SplitConfig, MegaConfig, object] = "nccl_ep",
) -> Union[MoEEpSplitLayer, MoEEpMegaLayer]:
    """Construct the appropriate EP layer for ``backend``.

    - ``SplitConfig`` or a comm string/config → :class:`MoEEpSplitLayer`
    - ``MegaConfig`` → :class:`MoEEpMegaLayer`
    """
    if isinstance(backend, MegaConfig):
        if fleet_knobs:
            warnings.warn(
                "fleet_knobs are ignored for MegaConfig; the mega path does not "
                "use EP transport backends",
                UserWarning,
                stacklevel=2,
            )
        return MoEEpMegaLayer(bootstrap, fleet_params, backend)
    return MoEEpSplitLayer(bootstrap, fleet_params, fleet_knobs, backend=backend)
