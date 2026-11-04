"""GemmLayer: stateful cross-backend GEMM dispatcher with autotune.

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

The GEMM counterpart of ``flashinfer.fused_moe.MoELayer`` with two deliberate
differences:

* the cross-backend measurement runs only inside ``autotune(True)``; outside
  tuning mode the first eligible candidate in the deterministic ``auto`` order
  (the ``BackendOptions`` order) is used, so a layer built without an autotune
  context never benchmarks on first use;
* ``supported_backends()`` is public from the first release so a framework can
  ask "what runs here, and why not?" instead of keeping its own table.
"""

from __future__ import annotations

from collections import OrderedDict
from statistics import median
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner
from ..fused_moe.utils import map_to_hybrid_bucket
from ..utils import get_compute_capability
from .api import (
    BackendReport,
    BackendUnsupported,
    GemmActivationPack,
    GemmConfig,
    GemmProblem,
    GemmWeightPack,
    output_torch_dtype,
)
from .runners import GemmRunner, runner_class_for

__all__ = ["GemmLayer", "supported_backends"]

_WINNER_CACHE_LIMIT = 128


def _arch_of(device: torch.device) -> int:
    major, minor = get_compute_capability(device)
    return major * 10 + minor


def _candidate_runners(
    config: GemmConfig, device: torch.device, *, build: bool
) -> Tuple[List[GemmRunner], List[str]]:
    """Construct one runner per usable candidate; collect the reasons for the rest."""
    runners: List[GemmRunner] = []
    rejected: List[str] = []
    for cand in config.backend:
        key = cand.backend_key
        if not cand.supports_quant(config.quant):
            rejected.append(f"{key}: quant pair {config.quant.pair!r} not supported")
            continue
        runner_cls = runner_class_for(cand, config)
        if runner_cls is None:
            rejected.append(f"{key}: no unified runner for {config.quant.pair!r}")
            continue
        try:
            runner = runner_cls(cand, config, device)
            runner.check_support()
        except BackendUnsupported as exc:
            rejected.append(str(exc))
            continue
        if build:
            runner.build()
        runners.append(runner)
    return runners, rejected


class GemmLayer:
    """Stateful GEMM layer: one runner per usable backend, winner cached per M bucket.

    Example
    -------
    >>> layer = GemmLayer(config)
    >>> layer.prepare(weights)              # GemmWeightPack, built once after load
    >>> out = layer(act)                    # GemmActivationPack per call
    >>> with autotune(True):
    ...     out = layer(act)                # measures every candidate for this bucket
    >>> layer.winner_backend
    'cutlass'
    """

    def __init__(self, config: GemmConfig, *, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.tuner = AutoTuner.get()
        self.out_dtype = output_torch_dtype(config.quant)
        self.runners, self.rejected = _candidate_runners(
            config, self.device, build=True
        )
        if not self.runners:
            raise RuntimeError(
                f"GemmLayer: none of the configured backends "
                f"{[type(c).__name__ for c in config.backend]} are usable on "
                f"sm{_arch_of(self.device)} for quant weight={config.quant.weight.name}, "
                f"activation={config.quant.activation.name}, output={config.quant.output.name}."
                + (
                    " Backends rejected this configuration: " + "; ".join(self.rejected)
                    if self.rejected
                    else ""
                )
            )
        self._weights: Optional[GemmWeightPack] = None
        # (M bucket, eligible runner indices, activation dtype, sf layout) -> (runner, tactic)
        self._winners: "OrderedDict[tuple, Tuple[GemmRunner, Any]]" = OrderedDict()
        self._last_winner: Optional[GemmRunner] = None
        self._last_rejections: List[str] = []

    # ---- weights ----------------------------------------------------------

    def prepare(self, weights: GemmWeightPack) -> None:
        """Bind a weight pack so calls can omit it. Clears cached winners."""
        if not any(r.backend_key in weights.native_views for r in self.runners):
            raise BackendUnsupported(
                "GemmLayer.prepare: the weight pack has no view for any usable backend "
                f"{[r.backend_key for r in self.runners]}; prepared: {sorted(weights.native_views)}."
            )
        self._weights = weights
        self.reset_winner()

    @property
    def weights(self) -> Optional[GemmWeightPack]:
        return self._weights

    # ---- introspection ----------------------------------------------------

    @property
    def winner_backend(self) -> Optional[str]:
        return self._last_winner.backend_key if self._last_winner is not None else None

    def winner_report(self) -> Optional[BackendReport]:
        r = self._last_winner
        if r is None:
            return None
        return BackendReport(
            backend_key=r.backend_key,
            ok=True,
            reason=None,
            auto_rank=self.runners.index(r),
            needs_prepare=True,
            fused_bias=r.fused_bias,
        )

    def reset_winner(self) -> None:
        """Clear every cached per-bucket winner; the next call re-selects."""
        self._winners.clear()
        self._last_winner = None

    def supported_backends(
        self, problem: Optional[GemmProblem] = None
    ) -> List[BackendReport]:
        return supported_backends(self.config, problem, device=self.device)

    # ---- dispatch ---------------------------------------------------------

    @flashinfer_api
    def __call__(
        self,
        act: GemmActivationPack,
        weights: Optional[GemmWeightPack] = None,
        *,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the GEMM, selecting and caching the backend for this M bucket.

        Returns ``act @ W^T`` (plus the pack's bias when ``epilogue.bias``) in
        ``config.quant.output``'s dtype. Selection happens on the first call per
        (M bucket, eligible set) and inside ``autotune(True)``; every other call
        is a dictionary lookup and one kernel launch.
        """
        weights = weights if weights is not None else self._weights
        if weights is None:
            raise ValueError(
                "GemmLayer: pass a GemmWeightPack or call prepare(weights) first."
            )
        if act.form != "dense":
            raise NotImplementedError(
                f"GemmLayer: form={act.form!r} is not in the unified GEMM MVP (dense only)."
            )
        m = act.num_rows
        n, k = weights.logical_shape
        if out is None:
            out = torch.empty((m, n), dtype=self.out_dtype, device=act.a.device)
        elif out.shape != (m, n) or out.dtype != self.out_dtype:
            raise ValueError(
                f"GemmLayer: out must be ({m}, {n}) {self.out_dtype}, got {tuple(out.shape)} {out.dtype}."
            )

        views = weights.native_views
        eligible = tuple(
            i
            for i, r in enumerate(self.runners)
            if r.accepts(act, views.get(r.backend_key))
        )
        if not eligible:
            raise NotImplementedError(
                f"GemmLayer: none of the usable backends {[r.backend_key for r in self.runners]} "
                f"accept this activation pack / weight pack: {act!r}, {weights!r}."
            )

        bucket = map_to_hybrid_bucket(m, self.config.execution.tune_max_m)
        key = (bucket, eligible, act.a.dtype, act.sf_layout)
        winner = None if self.tuner.is_tuning_mode else self._winners.get(key)
        if winner is None:
            winner = self._select_winner(
                act, weights, [self.runners[i] for i in eligible], out
            )
            self._winners[key] = winner
            if len(self._winners) > _WINNER_CACHE_LIMIT:
                self._winners.popitem(last=False)
        else:
            self._winners.move_to_end(key)
        runner, tactic = winner
        self._last_winner = runner

        inputs = runner.pack_inputs(act, views[runner.backend_key], out)
        runner.forward(inputs, tactic=tactic)
        if (
            self.config.epilogue.bias
            and weights.bias is not None
            and not runner.fused_bias
        ):
            out.add_(weights.bias.to(out.dtype))
        return out

    def _select_winner(
        self,
        act: GemmActivationPack,
        weights: GemmWeightPack,
        runners: List[GemmRunner],
        out: Tensor,
    ) -> Tuple[GemmRunner, Any]:
        """Admit candidates with the legacy checks, then pick deterministically or by measurement."""
        admitted: List[Tuple[GemmRunner, List[Any]]] = []
        self._last_rejections = []
        for r in runners:
            view = weights.get_view(r.backend_key)
            try:
                r.validate(act, view, out)
            except BackendUnsupported as exc:
                self._last_rejections.append(str(exc))
                continue
            admitted.append((r, r.pack_inputs(act, view, out)))
        if not admitted:
            raise BackendUnsupported(
                "GemmLayer: every eligible backend rejected this problem: "
                + "; ".join(self._last_rejections)
            )

        tuning = self.tuner.is_tuning_mode
        if not tuning or len(admitted) == 1:
            # Deterministic `auto`: the first admitted candidate. choose_one is a
            # cache lookup here (tactic -1 on a miss), exactly as the old names.
            r, inputs = admitted[0]
            _, tactic = self.tuner.choose_one(
                r.op_name, [r.inner], r.tuning_config, inputs
            )
            return r, tactic

        from ..testing.utils import bench_gpu_time

        best: Optional[Tuple[GemmRunner, Any]] = None
        best_ms = float("inf")
        for r, inputs in admitted:
            _, tactic = self.tuner.choose_one(
                r.op_name, [r.inner], r.tuning_config, inputs
            )

            def run_candidate(r=r, inputs=inputs, tactic=tactic):
                r.forward(inputs, tactic=tactic)

            times = bench_gpu_time(
                run_candidate, dry_run_iters=5, repeat_iters=30, use_cuda_graph=True
            )
            t_ms = median(times)
            if t_ms < best_ms:
                best_ms, best = t_ms, (r, tactic)
        assert best is not None
        return best


def supported_backends(
    config: GemmConfig,
    problem: Optional[GemmProblem] = None,
    device: Optional[torch.device] = None,
) -> List[BackendReport]:
    """Which candidates of ``config`` can run ``problem`` on ``device``, and why not.

    One report per candidate, in ``auto`` order; ``ok=False`` rows carry the
    rejection reason. No kernel module is loaded. ``problem=None`` answers the
    config-level question only.
    """
    device = device or torch.device("cuda", torch.cuda.current_device())
    reports: List[BackendReport] = []
    rank = 0
    for cand in config.backend:
        key = cand.backend_key
        reason: Optional[str] = None
        runner: Optional[GemmRunner] = None
        if not cand.supports_quant(config.quant):
            reason = f"quant pair {config.quant.pair!r} not supported"
        else:
            runner_cls = runner_class_for(cand, config)
            if runner_cls is None:
                reason = f"no unified runner for {config.quant.pair!r}"
            else:
                try:
                    runner = runner_cls(cand, config, device)
                    runner.check_support()
                    if problem is not None:
                        runner.check_problem(problem)
                except BackendUnsupported as exc:
                    reason = str(exc)
        ok = reason is None
        reports.append(
            BackendReport(
                backend_key=key,
                ok=ok,
                reason=reason,
                auto_rank=rank if ok else -1,
                needs_prepare=True,
                fused_bias=bool(runner.fused_bias) if runner is not None else False,
            )
        )
        if ok:
            rank += 1
    return reports
