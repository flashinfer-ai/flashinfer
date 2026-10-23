# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General-autotuner adapter for the collective native MXFP8 x MXFP4 kernel."""

from __future__ import annotations

import warnings

from flashinfer.autotuner import (
    AutoTuner,
    TunableRunner,
    TuningConfig,
    get_autotune_process_group,
    set_autotune_process_group,
)

from .integration import _launch_spec, _tactic_key, mxfp8_mxfp4_mega_moe


CUSTOM_OP = "mxfp8_mxfp4_mega_moe"
_TACTIC_VERSION = 1


class Mxfp8Mxfp4MegaMoeRunner(TunableRunner):
    """Profile real staged inputs without replacing symmetric allocations."""

    def __init__(
        self,
        workspace,
        *,
        num_tokens,
        gate_up_clamp=None,
        fast_math=True,
        candidates=None,
    ):
        from ....backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
            candidate_tactics,
        )

        self.workspace = workspace
        self.num_tokens = num_tokens
        self.token_counts = (num_tokens,)
        self.gate_up_clamp = gate_up_clamp
        self.fast_math = fast_math
        self.fallback = _tactic_key(workspace.resolve({}))
        if candidates is None:
            candidates = candidate_tactics(
                hidden=workspace.hidden, intermediate=workspace.intermediate
            )
        kept, dropped = [], []
        for candidate in candidates:
            try:
                tactic = workspace.resolve(candidate)
                if not workspace.fits(tactic):
                    raise ValueError("workspace capacity exceeded")
                key = _tactic_key(tactic)
                if key not in kept:
                    kept.append(key)
            except (TypeError, ValueError) as error:
                dropped.append(f"{candidate}: {error}")
        if dropped:
            warnings.warn(
                f"[mxfp8_mxfp4-autotune] dropped {len(dropped)} candidate(s): {dropped}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not kept:
            raise ValueError("no MXFP8 x MXFP4 autotune candidate fits this workspace")
        self.tactics = tuple(kept)

    def __hash__(self):
        # Neither mutable workspace layout nor pointers belong in this key.
        return hash(
            (
                type(self).__name__,
                self.workspace.x.device,
                self.get_cache_key_extras([]),
            )
        )

    def get_cache_key_extras(self, inputs):
        ws = self.workspace
        return (
            _TACTIC_VERSION,
            ws.world_size,
            ws.num_experts,
            ws.num_topk,
            ws.hidden,
            ws.intermediate,
            ws.max_tokens_per_rank,
            self.token_counts,
            self.gate_up_clamp,
            self.fast_math,
            self.fallback,
        )

    def get_valid_tactics(self, inputs, profile):
        return list(self.tactics)

    def validate_tactic(self, inputs, tactic):
        if tactic == -1:
            return self.workspace.fits(dict(self.fallback))
        return (
            tactic in self.tactics
            and _tactic_key(self.workspace.resolve(dict(tactic))) == tactic
            and self.workspace.fits(dict(tactic))
        )

    def precompile_tactics(self, inputs, tactics, profile, **kwargs):
        # Preparation must occur for each layout immediately before profiling.
        return True

    def prepare_tactic(self, inputs, tactic, **kwargs):
        ws = self.workspace
        ws.apply_knobs(dict(self.fallback if tactic == -1 else tactic))
        destination = ws.y
        frontend, spec = _launch_spec(
            inputs[:2],
            inputs[2:4],
            ws,
            destination,
            self.num_tokens,
            self.gate_up_clamp,
            self.fast_math,
        )
        frontend.warmup(spec)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return mxfp8_mxfp4_mega_moe(
            None,
            inputs[:2],
            inputs[2:4],
            self.workspace,
            num_tokens=self.num_tokens,
            gate_up_clamp=self.gate_up_clamp,
            fast_math=self.fast_math,
        )


def _gather(value, group):
    if group is None:
        return [value]
    import torch.distributed as dist

    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _agree(value, group, description):
    values = _gather(value, group)
    if any(item != values[0] for item in values[1:]):
        raise RuntimeError(f"MXFP8 x MXFP4 autotune ranks disagree on {description}")


def select_tactic(
    fc1,
    fc2,
    workspace,
    *,
    num_tokens,
    gate_up_clamp=None,
    fast_math=True,
    candidates=None,
    warmup_iters=3,
    timed_iters=10,
):
    """Collectively choose/apply a tactic, or only look it up outside autotune.

    Called eagerly once per backend workspace. The compatibility API enters
    ``autotune(True)`` around this call; a normal backend call inherits the
    caller's tuning context. No allocation or collective enters the hot path.
    """
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.comm import (
        ensure_not_capturing,
    )

    ensure_not_capturing("MXFP8 x MXFP4 autotune selection; warm up before capture")
    group = None
    if workspace.world_size > 1:
        import torch.distributed as dist

        # The native workspace currently uses the bootstrapped WORLD domain
        # for allocation, peer offsets, and apply_knobs barriers as well.
        if not dist.is_initialized() or dist.get_world_size() != workspace.world_size:
            raise RuntimeError(
                "MXFP8 x MXFP4 workspace must match the EP process group"
            )
        group = dist.group.WORLD

    # Agree on errors before entering any preparation that contains barriers.
    runner, error = None, None
    try:
        if not 0 <= num_tokens <= workspace.max_tokens_per_rank:
            raise ValueError("num_tokens exceeds the allocated workspace")
        if warmup_iters < 0 or timed_iters < 1:
            raise ValueError("warmup_iters must be >= 0 and timed_iters must be >= 1")
        runner = Mxfp8Mxfp4MegaMoeRunner(
            workspace,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            fast_math=fast_math,
            candidates=candidates,
        )
    except (TypeError, ValueError) as exc:
        error = str(exc)
    reports = _gather((error, num_tokens), group)
    if any(report[0] is not None for report in reports):
        raise ValueError(f"MXFP8 x MXFP4 autotune preparation rejected: {reports}")
    assert runner is not None
    runner.token_counts = tuple(report[1] for report in reports)
    inputs = [
        *fc1,
        *fc2,
        workspace.x,
        workspace.x_sf,
        workspace.topk_idx,
        workspace.topk_weights,
    ]
    config = TuningConfig(
        use_cuda_graph=True,
        preserve_input_storage=True,
        profiling_warmup=warmup_iters,
        profiling_repeat=timed_iters,
        rank_reduction="max",
    )
    tuner = AutoTuner.get()
    # Match choose_one's effective config for the preflight cache decision.
    # Hold the tuner lock through selection and restoring the process group.
    with tuner._lock:
        config = tuner._apply_tuning_overrides(config)
        if tuner._effective_measure_policy is not None:
            config = tuner._apply_measure_policy(
                config, tuner._effective_measure_policy
            )
        previous_group = get_autotune_process_group()
        set_autotune_process_group(group)
        try:
            shapes = tuple(tuple(t.shape) for t in inputs)
            cache_hit = tuner.search_cache(
                CUSTOM_OP, [runner], shapes, config, inputs=inputs
            )
            allowed = tuner._blocklist.filter(CUSTOM_OP, runner, list(runner.tactics))
            skipped = CUSTOM_OP in tuner._effective_skip_ops
            _agree(
                (
                    runner.get_cache_key_extras(inputs),
                    shapes,
                    runner.fallback,
                    _tactic_key(workspace.tactic),
                    tuple(allowed),
                    cache_hit[:3],
                    tuner.is_tuning_mode,
                    skipped,
                    config,
                    tuner._effective_measure_policy,
                    tuner._in_v2_context,
                    workspace.workspace_bytes.numel(),
                    workspace.shared_workspace_bytes.numel(),
                ),
                group,
                "candidates, cache, workspace, or profiling policy",
            )
            workspace._autotune_selected = False
            _, tactic = tuner.choose_one(CUSTOM_OP, [runner], config, inputs)
            hit = tuner.search_cache(
                CUSTOM_OP, [runner], shapes, config, inputs=inputs
            )[0]
            if tuner.is_tuning_mode and not skipped and not hit:
                raise RuntimeError(
                    "MXFP8 x MXFP4 autotune: every candidate failed or preparation ran out of memory"
                )
            if not hit or skipped:
                # With no general result, retain the legacy cache/default
                # resolved before allocation. A measured v2 -1 winner instead
                # means the deterministic default recorded in cache extras.
                tactic = _tactic_key(workspace.tactic)
            _agree(tactic, group, "winning tactic")
            # The last profiled tactic need not be the winner. Rebind even on
            # cache hits, and use the same pre-launch failure handshake.
            tuner._prepare_tactic(runner, inputs, tactic)
            winner = dict(runner.fallback if tactic == -1 else tactic)
            workspace._autotune_selected = True
            if tuner.is_tuning_mode and not skipped and workspace.rank == 0:
                _export_legacy_winner(workspace, winner)
            return winner
        finally:
            set_autotune_process_group(previous_group)


def _export_legacy_winner(workspace, winner):
    """Keep the existing offline CLI/cache workflow without a second tuner."""
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.knob_cache import (
        record_knobs,
    )
    from .integration import KNOB_CACHE_DTYPE

    record_knobs(
        winner,
        dtype=KNOB_CACHE_DTYPE,
        world_size=workspace.world_size,
        hidden=workspace.hidden,
        intermediate=workspace.intermediate,
        num_experts=workspace.num_experts,
        topk=workspace.num_topk,
        max_tokens=workspace.max_tokens_per_rank,
        # The old p50_us was synchronized Python wall time, not GPU time.
        source="flashinfer.autotuner",
    )
