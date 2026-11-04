"""PyTorch doors of the unified GEMM API: ``Linear``, ``linear`` and ``gemm``.

Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

``Linear`` *is* a ``torch.nn.Linear``: same constructor, parameters and
state-dict keys, so checkpoints, FSDP and quantization tooling see an ordinary
linear layer. ``prepare_weights()`` (vLLM's ``process_weights_after_loading``)
builds the backend-native views once; ``forward`` keeps ``F.linear`` semantics.
``linear`` mirrors ``torch.nn.functional.linear``'s argument order with the
config keyword-only. Both compose a :class:`~flashinfer.gemm.layer.GemmLayer`,
as ``moe_ep.MoEEpSplitLayer`` composes a ``MoELayer``.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from ..api_logging import flashinfer_api
from ..tllm_enums import SfLayout
from .api import (
    BackendReport,
    GemmActivationPack,
    GemmConfig,
    GemmWeightPack,
    prepare_activations,
    prepare_weight_pack,
)
from .layer import GemmLayer

__all__ = ["Linear", "linear", "gemm"]


class Linear(torch.nn.Linear):
    """``torch.nn.Linear`` whose forward runs a FlashInfer GEMM chosen by ``config``.

    Parameters
    ----------
    in_features, out_features, bias, device, dtype
        As :class:`torch.nn.Linear`. ``weight`` stays the canonical ``[N, K]``
        high-precision parameter; ``bias`` is ``[N]``.
    config : GemmConfig
        Quantization pair, epilogue, backend candidates and execution knobs.
        ``config.epilogue.bias`` is overridden by the ``bias`` argument.

    Before :meth:`prepare_weights` the module computes ``F.linear`` on its
    parameters (so a freshly built or partially loaded model runs). After it,
    ``forward`` quantizes the activation per ``config.quant.activation`` and
    dispatches through a :class:`GemmLayer`; a ``GemmActivationPack`` may also
    be passed directly by callers that quantize activations themselves.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        config: Optional[GemmConfig] = None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        config = config if config is not None else GemmConfig()
        self.config = dataclasses.replace(
            config, epilogue=dataclasses.replace(config.epilogue, bias=bias)
        )
        self._layer: Optional[GemmLayer] = None
        self._weights: Optional[GemmWeightPack] = None
        self._activation_global_scale: Optional[Tensor] = None
        self._sf_layout: SfLayout = SfLayout.layout_128x4

    # ---- preparation ------------------------------------------------------

    @property
    def prepared(self) -> bool:
        return self._layer is not None

    @torch.no_grad()
    def prepare_weights(
        self,
        *,
        for_backends: Optional[Iterable[str]] = None,
        activation_global_scale: Optional[Tensor] = None,
        sf_layout: SfLayout = SfLayout.layout_128x4,
    ) -> None:
        """Build one native weight view per candidate backend from ``self.weight``.

        Call it after the weights are loaded and the module sits on its final
        device; call it again if ``weight`` or ``bias`` are replaced. Idempotent
        otherwise. ``activation_global_scale`` (NVFP4: the static
        ``nvfp4_quantize`` global scale; FP8: the static dequantization scale)
        folds a calibrated activation scale in at load time; without it the
        scale is derived from each activation (one extra reduction per call).
        """
        if self.weight.device.type != "cuda":
            raise RuntimeError(
                "Linear.prepare_weights: move the module to a CUDA device first."
            )
        bias = None if self.bias is None else self.bias.detach()
        self._weights = prepare_weight_pack(
            self.weight.detach(),
            self.config,
            bias=bias,
            for_backends=for_backends,
            activation_global_scale=activation_global_scale,
        )
        self._activation_global_scale = (
            None
            if activation_global_scale is None
            else torch.as_tensor(
                activation_global_scale, dtype=torch.float32, device=self.weight.device
            )
        )
        self._sf_layout = sf_layout
        self._layer = GemmLayer(self.config, device=self.weight.device)
        self._layer.prepare(self._weights)

    @property
    def weight_pack(self) -> Optional[GemmWeightPack]:
        return self._weights

    @property
    def layer(self) -> Optional[GemmLayer]:
        return self._layer

    # ---- forward ----------------------------------------------------------

    def forward(self, input: Union[Tensor, GemmActivationPack]) -> Tensor:  # type: ignore[override]
        if isinstance(input, GemmActivationPack):
            if self._layer is None:
                raise RuntimeError(
                    "Linear: call prepare_weights() before passing a GemmActivationPack."
                )
            return self._layer(input, self._weights)
        if self._layer is None:
            return F.linear(input, self.weight, self.bias)
        lead = input.shape[:-1]
        x2 = input.reshape(-1, self.in_features)
        act = prepare_activations(
            x2,
            quant=self.config.quant,
            global_scale=self._activation_global_scale,
            sf_layout=self._sf_layout,
        )
        out = self._layer(act, self._weights)
        return out.reshape(*lead, self.out_features)

    # ---- introspection ----------------------------------------------------

    @property
    def winner_backend(self) -> Optional[str]:
        return self._layer.winner_backend if self._layer is not None else None

    def winner_report(self) -> Optional[BackendReport]:
        return self._layer.winner_report() if self._layer is not None else None

    def reset_winner(self) -> None:
        if self._layer is not None:
            self._layer.reset_winner()

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, config={self.config!r}, prepared={self.prepared}"
        )


# ---------------------------------------------------------------------------
# Functional doors
# ---------------------------------------------------------------------------

_LAYER_CACHE: "OrderedDict[Tuple[GemmConfig, torch.device], GemmLayer]" = OrderedDict()
_LAYER_CACHE_LIMIT = 64
# id(weight) -> (weakref to the weight, {cache key: pack}). Keyed by identity
# (a Parameter is a stable object) rather than by a WeakKeyDictionary, whose
# lookups compare tensors with ``==``. The weakref callback drops the entry
# when the weight is freed, and the identity check guards id() reuse.
_PACK_CACHE: "OrderedDict[int, Tuple[weakref.ref, Dict[Any, GemmWeightPack]]]" = (
    OrderedDict()
)
_PACK_CACHE_LIMIT = 256


def _layer_for(config: GemmConfig, device: torch.device) -> GemmLayer:
    key = (config, device)
    layer = _LAYER_CACHE.get(key)
    if layer is None:
        layer = GemmLayer(config, device=device)
        _LAYER_CACHE[key] = layer
        if len(_LAYER_CACHE) > _LAYER_CACHE_LIMIT:
            _LAYER_CACHE.popitem(last=False)
    else:
        _LAYER_CACHE.move_to_end(key)
    return layer


def _pack_for(
    weight: Tensor, config: GemmConfig, bias: Optional[Tensor]
) -> GemmWeightPack:
    key = id(weight)
    entry = _PACK_CACHE.get(key)
    if entry is None or entry[0]() is not weight:
        per_weight: Dict[Any, GemmWeightPack] = {}

        def _evict(_ref: Any, key: int = key) -> None:
            _PACK_CACHE.pop(key, None)

        _PACK_CACHE[key] = (weakref.ref(weight, _evict), per_weight)
        if len(_PACK_CACHE) > _PACK_CACHE_LIMIT:
            _PACK_CACHE.popitem(last=False)
    else:
        per_weight = entry[1]
        _PACK_CACHE.move_to_end(key)
    cache_key = (config, None if bias is None else id(bias))
    pack = per_weight.get(cache_key)
    if pack is None:
        pack = prepare_weight_pack(
            weight.detach(), config, bias=None if bias is None else bias.detach()
        )
        per_weight[cache_key] = pack
    return pack


@flashinfer_api
def linear(
    input: Tensor,
    weight: Union[Tensor, GemmWeightPack],
    bias: Optional[Tensor] = None,
    *,
    config: Optional[GemmConfig] = None,
) -> Tensor:
    """``y = input @ weight.T + bias`` through the unified GEMM API.

    Mirrors :func:`torch.nn.functional.linear`. ``weight`` is a ``[N, K]``
    tensor (prepared on first use and cached by tensor identity) or a
    :class:`GemmWeightPack` built with :func:`prepare_weight_pack`. With a plain
    tensor and no ``config`` this is ``F.linear`` (the BF16 pair is not in the
    unified GEMM MVP). Selection is deterministic ``auto``; nothing is
    benchmarked outside ``autotune(True)``.
    """
    if isinstance(weight, GemmWeightPack):
        if config is None:
            raise ValueError(
                "linear: a GemmWeightPack needs the GemmConfig it was prepared for."
            )
        pack = weight
        extra_bias = bias
    else:
        if config is None:
            return F.linear(input, weight, bias)
        pack = _pack_for(weight, config, bias)
        extra_bias = None
    if bias is not None or pack.bias is not None:
        # F.linear semantics: a bias given here is applied, whatever the config says.
        config = dataclasses.replace(
            config, epilogue=dataclasses.replace(config.epilogue, bias=True)
        )
    device = input.device
    layer = _layer_for(config, device)
    lead = input.shape[:-1]
    x2 = input.reshape(-1, pack.in_features)
    act = prepare_activations(x2, quant=config.quant)
    out = layer(act, pack)
    if extra_bias is not None and pack.bias is None:
        out = out + extra_bias.to(out.dtype)
    return out.reshape(*lead, pack.out_features)


@flashinfer_api
def gemm(
    act: GemmActivationPack,
    weights: GemmWeightPack,
    *,
    config: GemmConfig,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Stateless pack-level entry: the layer is cached per ``(config, device)``."""
    layer = _layer_for(config, act.a.device)
    return layer(act, weights, out=out)
