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

MVP scope: NVFP4 only, pre-routed path, two backends.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from ..autotuner import AutoTuner
from ..utils import get_compute_capability
from .api import (
    ActivationType,
    CuteDslConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantVariant,
    TrtllmBf16Config,
    TrtllmFp4Config,
)
from .runners import (
    CuteDslNvfp4Runner,
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
)
from .utils import map_to_hybrid_bucket


# Union of the concrete runners the layer dispatches to.  All share
# backend_key / tuning_config / pack_inputs as attributes or class members;
# typing the list with this Union gives mypy the visibility it needs.
_RunnerT = Union[CuteDslNvfp4Runner, TrtllmFp4RoutedRunner, TrtllmBf16RoutedRunner]

# Map backend-config class -> runner class
_BACKEND_RUNNERS: Dict[type, Type[_RunnerT]] = {
    CuteDslConfig: CuteDslNvfp4Runner,
    TrtllmFp4Config: TrtllmFp4RoutedRunner,
    TrtllmBf16Config: TrtllmBf16RoutedRunner,
}

# Quant variants each runner can execute.  Used by _validate_mvp_scope to accept
# a config only when at least one configured backend supports its quant variant.
_RUNNER_QUANTS: Dict[type, Tuple[QuantVariant, ...]] = {
    CuteDslConfig: (QuantVariant.NVFP4,),
    TrtllmFp4Config: (QuantVariant.NVFP4,),
    TrtllmBf16Config: (QuantVariant.BF16,),
}


class MoELayer:
    """Stateful MoE layer with cross-backend autotune.

    Example
    -------
    >>> layer = MoELayer(config)
    >>> out = layer(act_pack, weight_pack)
    """

    def __init__(self, config: MoEConfig, device: Optional[torch.device] = None):
        self.config = config
        self._validate_mvp_scope(config)
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
            # Skip backends that cannot execute the configured quant variant, so a
            # mixed candidate list (e.g. BF16 with (CuteDslConfig, TrtllmBf16Config))
            # never instantiates a runner that would mis-handle the pack contract.
            if config.quant.variant not in _RUNNER_QUANTS.get(type(backend_cfg), ()):
                continue
            self.runners.append(runner_cls(config, device=self.device))

        if not self.runners:
            mvp = ", ".join(c.__name__ for c in _BACKEND_RUNNERS)
            raise RuntimeError(
                f"MoELayer: none of the configured backends "
                f"{[type(c).__name__ for c in config.backend]} are usable on "
                f"arch sm{arch}. The MVP supports only NVFP4 via [{mvp}]."
            )

        # Cross-backend winner cache, keyed by the num_tokens tuning bucket.
        # See the MoELayer reuse contract (CR4): the fastest backend can differ
        # across token-count buckets, so each bucket caches its own winner.
        self._winners: Dict[int, Tuple[_RunnerT, Any]] = {}
        # Backend key selected on the most recent call (introspection hook).
        self._last_winner_backend: Optional[str] = None

    @staticmethod
    def _validate_mvp_scope(config: MoEConfig) -> None:
        """Fail fast on configs no configured backend can execute (CR6).

        Surfacing this at construction time turns a deep C++ crash or silent
        backend skip into a clear, actionable Python error.  NVFP4 (CuteDSL /
        TRTLLM-FP4) and BF16 (TRTLLM-BF16, the EP grouped-GEMM path) are
        supported; FP8 / MXFP4 / MxInt4 remain post-MVP.  Only the Swiglu
        activation is supported.
        """
        variant = config.quant.variant
        supported = {
            q for cfg in config.backend for q in _RUNNER_QUANTS.get(type(cfg), ())
        }
        if variant not in supported:
            raise NotImplementedError(
                f"MoELayer: QuantVariant.{variant.name} is not executable by any "
                f"configured backend (supported here: "
                f"{sorted(q.name for q in supported) or 'none'}). "
                "FP8 / MXFP4 / MxInt4 paths are tracked as post-MVP follow-ups."
            )
        act = config.activation.type
        if act is not ActivationType.Swiglu:
            raise NotImplementedError(
                f"MoELayer MVP supports only the Swiglu activation; got {act!r}."
            )

    def __call__(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
    ) -> torch.Tensor:
        ceiling = self.config.execution.tune_max_num_tokens
        if act_pack.num_tokens > ceiling:
            raise ValueError(
                f"num_tokens={act_pack.num_tokens} exceeds "
                f"tune_max_num_tokens={ceiling}. "
                f"Reconstruct MoELayer with a larger ceiling."
            )

        bucket = map_to_hybrid_bucket(act_pack.num_tokens, ceiling)
        winner = self._winners.get(bucket)
        if winner is None:
            winner = self._select_winner(act_pack, weight_pack)
            self._winners[bucket] = winner
        runner, tactic = winner
        self._last_winner_backend = runner.backend_key

        inputs = runner.pack_inputs(act_pack, weight_pack)
        return runner.forward(inputs, tactic=tactic)

    def _select_winner(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
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

        for runner in self.runners:
            inputs = runner.pack_inputs(act_pack, weight_pack)
            # Per-runner tactic selection via autotuner
            _, tactic = self.tuner.choose_one(
                custom_op=f"moe_{runner.backend_key}",
                runners=[runner],
                tuning_config=runner.tuning_config,
                inputs=inputs,
            )
            # Measure runner at its winning tactic.  Use CUDA-graph timing so
            # the cross-backend comparison reflects production (graph-captured)
            # latency rather than per-call launch/Python overhead — at low token
            # counts (~tens of us kernels) a no-graph 10-iter median is dominated
            # by that overhead and picks the wrong backend.  Requires a warmed-up
            # layer (the autotune pass above), not a cold capture.
            times = bench_gpu_time(
                lambda r=runner, i=inputs, t=tactic: r.forward(i, tactic=t),
                dry_run_iters=5,
                repeat_iters=30,
                use_cuda_graph=True,
            )
            t_ms = median(times)
            if t_ms < best_time_ms:
                best_time_ms = t_ms
                best_runner = runner
                best_tactic = tactic

        assert best_runner is not None  # self.runners is non-empty
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
