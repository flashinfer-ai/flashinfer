# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General-autotuner adapter for the collective native MXFP8 x MXFP4 kernel."""

from __future__ import annotations

import warnings

from flashinfer.autotuner import (
    AutoTuner,
    TunableRunner,
    TuningConfig,
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
                hidden=workspace.hidden,
                intermediate=workspace.intermediate,
                max_tokens_per_rank=workspace.max_tokens_per_rank,
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


def _collective_call(function, group, description):
    """Agree on host-side failure before any rank advances or rolls back.

    As with tactic preparation, interrupts participate in the handshake before
    propagating. The function must itself preserve collective cardinality.
    """
    result, error = None, None
    try:
        result = function()
    except BaseException as exc:
        error = exc
    report = None
    if error is not None:
        # Bound peer diagnostics while preserving the original local exception.
        report = f"{type(error).__name__}: {error}"
        if len(report) > 512:
            report = report[:509] + "..."
    reports = _gather(report, group)
    if any(report is not None for report in reports):
        if error is not None:
            raise error
        raise RuntimeError(
            f"MXFP8 x MXFP4 autotune {description} failed on a peer: {reports}"
        )
    return result


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
    ``autotune(True)`` around this call. A normal backend call inherits the
    caller's tuning context. No allocation or collective enters the hot path.
    """
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.comm import (
        ensure_not_capturing,
    )

    ensure_not_capturing("MXFP8 x MXFP4 autotune selection; warm up before capture")
    group = None
    if workspace.world_size > 1:
        import torch.distributed as dist

        group = workspace.process_group
        if (
            not dist.is_initialized()
            or group is None
            or dist.get_world_size(group) != workspace.world_size
            or dist.get_rank(group) != workspace.rank
        ):
            raise RuntimeError(
                "MXFP8 x MXFP4 workspace must match the EP process group"
            )

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
    with tuner.collective_session(group):

        def preflight():
            return tuner.selection_preflight(
                CUSTOM_OP, runner, config, inputs, tactics=runner.tactics
            )

        state = _collective_call(preflight, group, "preflight")
        original = dict(workspace.tactic)
        was_autotuned = workspace.is_autotuned
        _agree(
            (
                runner.get_cache_key_extras(inputs),
                tuple(tuple(t.shape) for t in inputs),
                runner.fallback,
                _tactic_key(original),
                was_autotuned,
                state,
                workspace.workspace_bytes.numel(),
                workspace.shared_workspace_bytes.numel(),
            ),
            group,
            "candidates, cache, workspace, or profiling policy",
        )
        # From here all ranks enter the same transaction. Any failure after
        # mutation is agreed before restoring, including cache/export failures.
        workspace.clear_autotune_selection()
        try:
            _, tactic = _collective_call(
                lambda: tuner.choose_one(CUSTOM_OP, [runner], state.config, inputs),
                group,
                "selection",
            )
            selected = _collective_call(preflight, group, "selected cache lookup")
            _agree(selected, group, "selected cache state")
            if state.is_tuning and not state.skipped and not selected.cache_hit:
                raise RuntimeError(
                    "MXFP8 x MXFP4 autotune: every candidate failed or preparation ran out of memory"
                )
            if not selected.cache_hit or state.skipped:
                # Retain the pre-sweep legacy/default tactic on a cache miss.
                tactic = _tactic_key(original)
            _agree(tactic, group, "winning tactic")
            tuner.prepare_selected_tactic(runner, inputs, tactic)
            winner = dict(runner.fallback if tactic == -1 else tactic)
            if state.is_tuning and not state.skipped:
                _collective_call(
                    lambda: _export_legacy_winner(workspace, winner)
                    if workspace.rank == 0
                    else None,
                    group,
                    "compatibility cache export",
                )
            workspace.mark_autotuned()
            return winner
        except BaseException:
            # Force reset even on ranks already at the original tactic: peers
            # may have stopped on another layout and must see equal barriers.
            workspace.restore_tactic(original, autotuned=was_autotuned)
            raise


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
