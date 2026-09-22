"""MoELayer — stateful cross-backend MoE dispatcher with autotune.

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

Builds one runner per compatible backend, picks the cross-backend winner
by measuring each runner's best tactic, then dispatches to the winner.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner, TunableRunner, TuningConfig
from ..autotuner.autotuner import _tactic_to_json_hashable
from ..utils import get_compute_capability
from .api import (
    BackendOptions,
    B12xNvfp4Config,
    B12xW4A16Config,
    CakeWarpDecodeConfig,
    CutlassBf16Config,
    CudnnMoeConfig,
    CudnnFp8PerTensorConfig,
    CutlassFp8BlockConfig,
    CutlassFp8PerTensorConfig,
    CutlassHummingConfig,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    CutlassNvfp4Config,
    CutlassW4A16Config,
    CutlassW4A8Config,
    CuTileBf16Config,
    CuTileMxfp4Bf16Config,
    CuTileMxfp4Config,
    CuTileNvfp4Bf16Config,
    CuTileNvfp4Config,
    CuteDslConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
)
from .runners import (
    B12xNvfp4Runner,
    B12xW4A16Runner,
    CakeWarpDecodeRunner,
    CutlassBf16Runner,
    CutlassFp8BlockRunner,
    CutlassFp8PerTensorRunner,
    CutlassHummingRunner,
    CutlassMxfp8Mxfp4Runner,
    CutlassMxfp8Runner,
    CutlassNvfp4Runner,
    CutlassW4A16Runner,
    CutlassW4A8Runner,
    CuTileBf16Runner,
    CuTileMxfp4Bf16Runner,
    CuTileMxfp4Runner,
    CuTileNvfp4Bf16Runner,
    CuTileNvfp4Runner,
    CuteDslRunner,
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorRunner,
    TrtllmMxInt4RoutedRunner,
)
from .cudnn_backend import CudnnMoeRunner
from .cudnn_fp8_backend import CudnnFp8PerTensorRunner
from .utils import map_to_hybrid_bucket

# Union of the concrete runners the layer dispatches to.  All share
# backend_key / tuning_config / pack_inputs as attributes or class members;
# typing the list with this Union gives mypy the visibility it needs.
_RunnerT = Union[
    CudnnMoeRunner,
    CakeWarpDecodeRunner,
    CutlassBf16Runner,
    CutlassFp8BlockRunner,
    CutlassFp8PerTensorRunner,
    CutlassHummingRunner,
    CutlassMxfp8Mxfp4Runner,
    CutlassMxfp8Runner,
    CutlassNvfp4Runner,
    CutlassW4A16Runner,
    CutlassW4A8Runner,
    CuTileBf16Runner,
    CuTileMxfp4Bf16Runner,
    CuTileMxfp4Runner,
    CuTileNvfp4Bf16Runner,
    CuTileNvfp4Runner,
    CuteDslRunner,
    TrtllmFp4RoutedRunner,
    TrtllmBf16RoutedRunner,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorRunner,
    TrtllmMxInt4RoutedRunner,
    B12xNvfp4Runner,
    B12xW4A16Runner,
]

# Map backend-config class -> runner class
_BACKEND_RUNNERS: Dict[type, Type[_RunnerT]] = {
    CudnnMoeConfig: CudnnMoeRunner,
    CudnnFp8PerTensorConfig: CudnnFp8PerTensorRunner,
    CakeWarpDecodeConfig: CakeWarpDecodeRunner,
    CutlassBf16Config: CutlassBf16Runner,
    CutlassFp8BlockConfig: CutlassFp8BlockRunner,
    CutlassFp8PerTensorConfig: CutlassFp8PerTensorRunner,
    CutlassHummingConfig: CutlassHummingRunner,
    CutlassMxfp8Config: CutlassMxfp8Runner,
    CutlassMxfp8Mxfp4Config: CutlassMxfp8Mxfp4Runner,
    CutlassNvfp4Config: CutlassNvfp4Runner,
    CutlassW4A16Config: CutlassW4A16Runner,
    CutlassW4A8Config: CutlassW4A8Runner,
    CuTileBf16Config: CuTileBf16Runner,
    CuTileMxfp4Bf16Config: CuTileMxfp4Bf16Runner,
    CuTileMxfp4Config: CuTileMxfp4Runner,
    CuTileNvfp4Bf16Config: CuTileNvfp4Bf16Runner,
    CuTileNvfp4Config: CuTileNvfp4Runner,
    CuteDslConfig: CuteDslRunner,
    TrtllmFp4Config: TrtllmFp4RoutedRunner,
    TrtllmBf16Config: TrtllmBf16RoutedRunner,
    TrtllmFp8BlockConfig: TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorConfig: TrtllmFp8PerTensorRunner,
    TrtllmMxInt4Config: TrtllmMxInt4RoutedRunner,
    B12xNvfp4Config: B12xNvfp4Runner,
    B12xW4A16Config: B12xW4A16Runner,
}


class _BackendChoiceRunner(TunableRunner):
    """Adapt a selected backend tactic to the common activation contract."""

    def __init__(self, runner, tactic, activation, weights, tensor_names, key):
        self.runner = runner
        self.tactic = tactic
        self.activation = activation
        self.weights = weights
        self.tensor_names = tensor_names
        self.key = key

    def __hash__(self):
        return hash(self.key)

    def get_cache_key_extras(self, inputs):
        return self.key

    def get_valid_tactics(self, inputs, profile):
        return [-1]

    def precompile_tactics(self, inputs, tactics, profile, **kwargs):
        # The underlying runner has already selected and prepared its tactic.
        return True

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        activation = replace(
            self.activation,
            **dict(zip(self.tensor_names, inputs, strict=True)),
        )
        packed = self.runner.pack_inputs(activation, self.weights)
        return self.runner.forward(
            packed, tactic=self.tactic, **self.runner.launch_kwargs_for(packed)
        )


class MoELayer:
    """Stateful MoE layer with cross-backend autotune.

    TRTLLM runners bind their immutable launch metadata to each packed input
    list, so interleaved ``pack_inputs -> forward`` pairs cannot exchange
    weights or routing configuration. Other backend adapters may still retain
    per-call workspace or prepared-weight state; use one ``MoELayer`` per
    thread/stream until those adapters adopt the same convention.

    Example
    -------
    >>> layer = MoELayer(config)
    >>> out = layer(act_pack, weight_pack)
    """

    @flashinfer_api
    def __init__(self, config: MoEConfig, device: Optional[torch.device] = None):
        """Build the layer and the set of backend runners it will autotune over.

        Every backend in ``config.backend`` whose hardware preconditions the
        target device satisfies gets a runner; the rest are skipped here rather
        than at call time, so an unsupported backend costs nothing per call.

        Parameters
        ----------
        config : MoEConfig
            Routing, quantization, expert geometry, activation, backend
            candidates and execution parameters. Cross-field constraints are
            validated by ``MoEConfig`` itself.
        device : torch.device or None
            Device whose compute capability selects the usable backends.
            ``None`` → the current CUDA device.

        Raises
        ------
        RuntimeError
            If no configured backend is usable on this device's architecture.
        """
        self.config = config
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.tuner = AutoTuner.get()

        major, minor = get_compute_capability(self.device)
        arch = major * 10 + minor

        # Build one runner per compatible backend
        self.runners: List[_RunnerT] = []
        for backend_cfg in config.backend:
            if not backend_cfg.supported(arch):
                continue
            runner_cls = _BACKEND_RUNNERS.get(type(backend_cfg))
            if runner_cls is None:
                continue  # MVP scope — skip non-MVP backends silently
            if not runner_cls.supports_quant(config.quant):
                continue
            try:
                # Construction is inside the guard because a runner may reject an
                # unsupported config while binding backend resources; letting that
                # escape would abort selection instead of skipping the backend.
                # Runners select their backend options from config. Bind this
                # candidate so repeated configs of the same type cannot all
                # select the first one (e.g. an FC1/FC2 tactic sweep).
                runner_config = replace(
                    config, backend=BackendOptions(candidates=(backend_cfg,))
                )
                runner = runner_cls(runner_config, device=self.device)
                runner.check_support()
            except (NotImplementedError, ValueError, RuntimeError):
                continue
            runner.build()
            self.runners.append(runner)

        if not self.runners:
            mvp = ", ".join(c.__name__ for c in _BACKEND_RUNNERS)
            # Show all shared-expert runners so a mismatched config or arch
            # does not produce an empty hint.
            hint = ""
            if config.experts.num_fused_shared_experts > 0:
                supporting = ", ".join(
                    r.__name__
                    for r in _BACKEND_RUNNERS.values()
                    if r.supports_fused_shared_experts
                )
                hint = (
                    f" Note num_fused_shared_experts="
                    f"{config.experts.num_fused_shared_experts}: fused shared "
                    f"experts are implemented only by [{supporting}], which must "
                    f"also be configured and supported on this arch."
                )
            local_num_experts = (
                config.experts.local_num_experts or config.routing.num_experts
            )
            if config.experts.local_expert_offset != 0 or (
                local_num_experts != config.routing.num_experts
            ):
                supporting = ", ".join(
                    r.__name__
                    for r in _BACKEND_RUNNERS.values()
                    if r.supports_expert_parallelism
                )
                hint += (
                    f" Note the config is an expert-parallel shard "
                    f"(local_expert_offset={config.experts.local_expert_offset}, "
                    f"local_num_experts={local_num_experts} of "
                    f"{config.routing.num_experts}): expert parallelism is "
                    f"implemented only by [{supporting}], which must also be "
                    f"configured and supported on this arch."
                )
            hint += (
                f" Note quant weight={config.quant.weight.name}, "
                f"activation={config.quant.activation.name}, "
                f"output={config.quant.output.name}."
            )
            raise RuntimeError(
                f"MoELayer: none of the configured backends "
                f"{[type(c).__name__ for c in config.backend]} are usable on "
                f"arch sm{arch} for this configuration. Registered unified "
                f"runners: [{mvp}].{hint}"
            )

        # Dispatch choices are scoped to the autotuner measurement partition.
        # Cross-backend comparisons use actual token counts; a single backend
        # keeps its tuning buckets. The routing-mode qualifier prevents a
        # pre-routed winner from being dispatched incompatible FromLogits data.
        self._winners: Dict[Tuple[int, Any], Tuple[_RunnerT, Any]] = {}
        self._winner_partition = None
        self._winner_context = None
        # Backend key selected on the most recent call (introspection hook).
        self._last_winner_backend: Optional[str] = None

    @flashinfer_api
    def __call__(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run the MoE layer, selecting and caching the fastest usable backend.

        Only runners that support this pack's ``routing_input_mode`` compete —
        not every backend has an in-kernel router — and the winner is cached
        per token count and routing mode under the active measurement policy.
        A single backend retains its per-runner token-count buckets. Profiling
        follows the autotune context; replay and opt-out use cached choices or
        the first usable configured backend without measuring new candidates.

        Parameters
        ----------
        act_pack : MoEActivationPack
            Backend-native activations plus routing inputs for this call.
        weight_pack : MoEWeightPack
            Expert weights, prepared for the selected backend's native layout.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            The layer output. With ``config.finalize.do_finalize=False`` the
            unreduced TRTLLM intermediates are returned instead, as
            ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``,
            leaving the combine to the caller.

        Raises
        ------
        ValueError
            If ``act_pack.num_tokens`` exceeds the
            ``execution.tune_max_num_tokens`` ceiling the layer was built with.
        NotImplementedError
            If no usable backend supports this pack's ``routing_input_mode``.
        """
        ceiling = self.config.execution.tune_max_num_tokens
        if act_pack.num_tokens > ceiling:
            raise ValueError(
                f"num_tokens={act_pack.num_tokens} exceeds "
                f"tune_max_num_tokens={ceiling}. "
                f"Reconstruct MoELayer with a larger ceiling."
            )

        # Only runners that can execute this pack's routing input mode compete.
        # Not every backend has an in-kernel router (CuteDSL is pre-routed-only),
        # so a FromLogits pack must never reach an incapable runner — neither
        # here nor via a winner cached under the other mode, hence the
        # mode-qualified cache key below.
        mode = act_pack.routing_input_mode
        runners = [r for r in self.runners if mode in r.supported_routing_modes]
        if not runners:
            raise NotImplementedError(
                f"MoELayer: none of the usable backends "
                f"{[r.backend_key for r in self.runners]} support "
                f"routing_input_mode={mode!r}."
            )

        partition = self.tuner._winner_cache()
        context = self.tuner._selection_context()
        if self._winner_partition is not partition or self._winner_context != context:
            self._winners.clear()
            self._winner_partition = partition
            self._winner_context = context
        # Compare backends at the actual call shape. Per-runner tactics keep
        # their own tuning buckets; cross-backend inputs retain real routing.
        bucket = (
            act_pack.num_tokens
            if len(runners) > 1
            else map_to_hybrid_bucket(act_pack.num_tokens, ceiling)
        )
        winner = self._winners.get((bucket, mode))
        if winner is None:
            winner = self._select_winner(act_pack, weight_pack, runners)
            self._winners[(bucket, mode)] = winner
            # choose_one may have published new selections during this call.
            self._winner_context = self.tuner._selection_context()
        runner, tactic = winner
        self._last_winner_backend = runner.backend_key

        inputs = runner.pack_inputs(act_pack, weight_pack)
        return runner.forward(
            inputs,
            tactic=tactic,
            **runner.launch_kwargs_for(inputs),
        )

    def _select_winner(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
        runners: List[_RunnerT],
    ) -> Tuple[_RunnerT, Any]:
        """Select tactics and backends through the same autotune policy/cache."""
        choices = []
        identities = []
        for runner in runners:
            inputs = runner.pack_inputs(act_pack, weight_pack)
            _, tactic = self.tuner.choose_one(
                custom_op=f"moe_{runner.backend_key}",
                runners=[runner],
                tuning_config=runner.tuning_config_for(inputs),
                inputs=inputs,
                **runner.launch_kwargs_for(inputs),
            )
            if len(runners) == 1:
                return runner, tactic
            choices.append((runner, tactic))
            identities.append(
                (
                    type(runner).__module__,
                    type(runner).__qualname__,
                    runner.backend_key,
                    _tactic_to_json_hashable(tactic),
                    runner.get_cache_key_extras(inputs),
                )
            )

        tensor_names = tuple(
            field.name
            for field in fields(act_pack)
            if isinstance(getattr(act_pack, field.name), torch.Tensor)
        )
        inputs = [getattr(act_pack, name) for name in tensor_names]
        metadata = tuple(
            (
                (field.name, str(value.dtype), str(value.device), tuple(value.stride()))
                if isinstance(value := getattr(act_pack, field.name), torch.Tensor)
                else (field.name, value)
            )
            for field in fields(act_pack)
        )
        roster = tuple(identities)
        wrapped = [
            _BackendChoiceRunner(
                runner,
                tactic,
                act_pack,
                weight_pack,
                tensor_names,
                (1, roster, identity, metadata),
            )
            for (runner, tactic), identity in zip(choices, identities, strict=True)
        ]
        selected, _ = self.tuner.choose_one(
            custom_op="moe_backend_choice",
            runners=wrapped,
            tuning_config=TuningConfig(use_cuda_graph=True),
            inputs=inputs,
        )
        return selected.runner, selected.tactic

    # ---- Introspection helpers ---------------------------------------------

    @property
    def winner_backend(self) -> Optional[str]:
        """Backend key selected on the most recent call, or None before first call."""
        return self._last_winner_backend

    def reset_winner(self) -> None:
        """Clear layer dispatch choices; the next call consults the autotuner."""
        self._winners.clear()
        self._last_winner_backend = None
