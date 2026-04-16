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
from ..testing.utils import bench_gpu_time
from ..utils import get_compute_capability
from .api import (
    CuteDslConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    TrtllmFp4Config,
)
from .runners import CuteDslNvfp4Runner, TrtllmFp4RoutedRunner


# Union of the concrete runners the layer dispatches to.  All share
# backend_key / tuning_config / pack_inputs as attributes or class members;
# typing the list with this Union gives mypy the visibility it needs.
_RunnerT = Union[CuteDslNvfp4Runner, TrtllmFp4RoutedRunner]

# Map backend-config class -> runner class
_BACKEND_RUNNERS: Dict[type, Type[_RunnerT]] = {
    CuteDslConfig: CuteDslNvfp4Runner,
    TrtllmFp4Config: TrtllmFp4RoutedRunner,
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
            self.runners.append(runner_cls(config, device=self.device))

        if not self.runners:
            raise RuntimeError(
                f"MoELayer: no compatible backends for arch sm{arch}. "
                f"Configured: {[type(c).__name__ for c in config.backend]}"
            )

        # Cross-runner winner — populated after first tuning pass
        self._winner: Optional[_RunnerT] = None
        self._winner_tactic: Any = -1

    def __call__(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
    ) -> torch.Tensor:
        if act_pack.num_tokens > self.config.execution.tune_max_num_tokens:
            raise ValueError(
                f"num_tokens={act_pack.num_tokens} exceeds "
                f"tune_max_num_tokens={self.config.execution.tune_max_num_tokens}. "
                f"Reconstruct MoELayer with a larger ceiling."
            )

        if self._winner is None:
            self._winner, self._winner_tactic = self._select_winner(
                act_pack, weight_pack
            )

        inputs = self._winner.pack_inputs(act_pack, weight_pack)
        return self._winner.forward(inputs, tactic=self._winner_tactic)

    def _select_winner(
        self,
        act_pack: MoEActivationPack,
        weight_pack: MoEWeightPack,
    ) -> Tuple[_RunnerT, Any]:
        """Run per-runner autotune, then measure each winner-tactic and
        pick cross-backend winner."""
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
            # Measure runner at its winning tactic
            times = bench_gpu_time(
                lambda r=runner, i=inputs, t=tactic: r.forward(i, tactic=t),
                dry_run_iters=3,
                repeat_iters=10,
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
        """Backend key of the currently selected runner, or None before first call."""
        return self._winner.backend_key if self._winner is not None else None

    def reset_winner(self) -> None:
        """Clear the cached cross-backend winner — next call re-tunes."""
        self._winner = None
        self._winner_tactic = -1
