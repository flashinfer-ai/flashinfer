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

from statistics import median
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from ..autotuner import AutoTuner
from ..utils import get_compute_capability
from .api import (
    B12xNvfp4Config,
    B12xW4A16Config,
    CutlassBf16Config,
    CutlassFp8BlockConfig,
    CutlassFp8PerTensorConfig,
    CutlassHummingConfig,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    CutlassNvfp4Config,
    CutlassW4A16Config,
    CutlassW4A8Config,
    CuTileBf16Config,
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
    CuTileNvfp4Runner,
    CuteDslRunner,
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorRunner,
    TrtllmMxInt4RoutedRunner,
)
from .utils import map_to_hybrid_bucket


# Union of the concrete runners the layer dispatches to.  All share
# backend_key / tuning_config / pack_inputs as attributes or class members;
# typing the list with this Union gives mypy the visibility it needs.
_RunnerT = Union[
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

    def __init__(self, config: MoEConfig, device: Optional[torch.device] = None):
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
            if config.quant.variant not in runner_cls.supported_quant_variants:
                continue
            try:
                # Construction is inside the guard because a runner may reject an
                # unsupported config while binding backend resources; letting that
                # escape would abort selection instead of skipping the backend.
                runner = runner_cls(config, device=self.device)
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
            raise RuntimeError(
                f"MoELayer: none of the configured backends "
                f"{[type(c).__name__ for c in config.backend]} are usable on "
                f"arch sm{arch} for this configuration. Registered unified "
                f"runners: [{mvp}].{hint}"
            )

        # Cross-backend winner cache, keyed by (num_tokens tuning bucket,
        # routing input mode).  See the MoELayer reuse contract (CR4): the
        # fastest backend can differ across token-count buckets, so each bucket
        # caches its own winner; the mode qualifier keeps a winner tuned for
        # one routing input style (e.g. pre-routed → CuteDSL) from being
        # dispatched a pack it cannot execute (FromLogits).
        self._winners: Dict[Tuple[int, Any], Tuple[_RunnerT, Any]] = {}
        # Backend key selected on the most recent call (introspection hook).
        self._last_winner_backend: Optional[str] = None

    def __call__(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
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

        bucket = map_to_hybrid_bucket(act_pack.num_tokens, ceiling)
        winner = self._winners.get((bucket, mode))
        if winner is None:
            winner = self._select_winner(act_pack, weight_pack, runners)
            self._winners[(bucket, mode)] = winner
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
        """Run per-runner autotune, then measure each winner-tactic and
        pick cross-backend winner."""
        # Lazy import: keep the library import path (``import flashinfer``) free
        # of a dependency on the testing framework. The GPU timing helper is only
        # needed here, on the autotune path. Relocating it to a non-testing
        # utility module is the cleaner long-term fix (post-MVP).
        from ..testing.utils import bench_gpu_time

        best_time_ms = float("inf")
        best_runner: Optional[_RunnerT] = None
        best_tactic: Any = -1

        for runner in runners:
            inputs = runner.pack_inputs(act_pack, weight_pack)
            launch_kwargs = runner.launch_kwargs_for(inputs)
            # Per-runner tactic selection via autotuner
            _, tactic = self.tuner.choose_one(
                custom_op=f"moe_{runner.backend_key}",
                runners=[runner],
                tuning_config=runner.tuning_config_for(inputs),
                inputs=inputs,
                **launch_kwargs,
            )
            # Measure runner at its winning tactic.  Use CUDA-graph timing so
            # the cross-backend comparison reflects production (graph-captured)
            # latency rather than per-call launch/Python overhead — at low token
            # counts (~tens of us kernels) a no-graph 10-iter median is dominated
            # by that overhead and picks the wrong backend.  Requires a warmed-up
            # layer (the autotune pass above), not a cold capture.
            times = bench_gpu_time(
                lambda r=runner, i=inputs, t=tactic, kw=launch_kwargs: r.forward(
                    i, tactic=t, **kw
                ),
                dry_run_iters=5,
                repeat_iters=30,
                use_cuda_graph=True,
            )
            t_ms = median(times)
            if t_ms < best_time_ms:
                best_time_ms = t_ms
                best_runner = runner
                best_tactic = tactic

        assert best_runner is not None  # runners is non-empty (checked by caller)
        return best_runner, best_tactic

    # ---- Introspection helpers ---------------------------------------------

    @property
    def winner_backend(self) -> Optional[str]:
        """Backend key selected on the most recent call, or None before first call."""
        return self._last_winner_backend

    def reset_winner(self) -> None:
        """Clear all cached per-bucket winners — next call re-tunes."""
        self._winners.clear()
        self._last_winner_backend = None
