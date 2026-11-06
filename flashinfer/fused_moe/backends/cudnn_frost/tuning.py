# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Factorized tuning of the bounded Frost candidate pool for every dtype."""

from itertools import product

from ....autotuner import AutoTuner, TunableRunner, TuningConfig
from ...utils import map_to_hybrid_bucket
from .cache import retain_graph_resources
from .heuristics import POLICY_VERSION


def prepared_state(runner, inputs, *, hidden_multiplier=1):
    state = runner.launch_state_for(inputs)
    if state is not None:
        return state
    # The autotuner synthesizes plain tensor lists. Find the same bucket state
    # prepared by pack_inputs; its native executors accept the actual shape.
    t, h = inputs[1].shape
    bucket = map_to_hybrid_bucket(t, runner.config.execution.tune_max_num_tokens)
    return runner._plans.get(
        (
            bucket,
            h * hidden_multiplier,
            runner.config.experts.intermediate_size,
            runner.config.routing.num_experts,
            runner.config.routing.top_k,
        )
    )


class _StageRunner(TunableRunner):
    def __init__(self, state, tag, stage, first, second):
        self.state, self.stage = state, stage
        if stage == 1:
            self.plans = {
                a.tactic: state.plans[(tag, a.tactic, second[0].tactic)] for a in first
            }
        else:
            self.plans = {
                b.tactic: state.plans[(tag, first[0].tactic, b.tactic)] for b in second
            }

    def __hash__(self):
        # Fresh wrappers around the same stage pool must hit the ranking cache.
        return hash((type(self), self.stage, tuple(self.plans)))

    def get_valid_tactics(self, inputs, profile):
        return list(self.plans)

    def get_cache_key_extras(self, inputs):
        kernels = self.state.first if self.stage == 1 else self.state.second
        return (
            POLICY_VERSION,
            self.stage,
            tuple(self.plans),
            tuple(k.source_sha256 for k in kernels),
        )

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if tactic == -1:
            tactic = next(iter(self.plans))
        plan = self.plans[tactic]
        if do_preparation:
            # Native layouts keep routing metadata before variable scratch,
            # so every plan in this mode group shares its routing offsets.
            # Recreate it: FC2's output aliases FC1's grouped input.
            # This runs outside timing/capture, entirely on the device.
            plan["prepare_stages"](*inputs, self.state.workspace)
            return
        retain_graph_resources(self.state, inputs)
        plan[f"run_fc{self.stage}"](*inputs, self.state.workspace)


def ranked_tactics(state, inputs, tag):
    first = tuple(a for a in state.first if not getattr(a, "quantizes_output", False))
    second = state.second
    if len(first) <= 2 and len(second) <= 2:
        # Preserve every offline choice, including its existing fused variants.
        return list(state.launches)
    fused = tuple(a for a in state.first if getattr(a, "quantizes_output", False))
    tuner = AutoTuner.get()
    if tuner.is_tuning_mode:
        config = TuningConfig(
            use_cuda_graph=True,
            cuda_graph_profile_replays=3,
            inputs_pre_hook=lambda tensors: list(inputs),
        )
        ranked = []
        for stage in (1, 2):
            kernels = first if stage == 1 else second
            groups = (kernels,)
            if stage == 1:
                # GATHER borrows original token/SF rows; NONE consumes packed
                # expert rows. Preparing one layout cannot initialize the other.
                grouped = tuple(
                    a for a in kernels if not getattr(a, "gathers_input", False)
                )
                gathered = tuple(
                    a for a in kernels if getattr(a, "gathers_input", False)
                )
                if grouped and gathered:
                    groups = (grouped, gathered)
            if stage == 2:
                # rank_tactics prepares only the default plan once. COMBINE
                # needs inverse routing and weighted scores; ordinary FC2
                # needs forward routing. Profile each layout independently,
                # then compare its best candidate end-to-end.
                grouped = tuple(
                    b for b in kernels if not getattr(b, "combines_output", False)
                )
                combined = tuple(
                    b for b in kernels if getattr(b, "combines_output", False)
                )
                if grouped and combined:
                    groups = (grouped, combined)
            selected = []
            for group in groups:
                runner = _StageRunner(
                    state,
                    tag,
                    stage,
                    group if stage == 1 else first,
                    group if stage == 2 else second,
                )
                name = f"{tag}-fc{stage}-{POLICY_VERSION}"
                if len(groups) > 1:
                    name += (
                        "-gather"
                        if stage == 1 and getattr(group[0], "gathers_input", False)
                        else (
                            "-combine"
                            if getattr(group[0], "combines_output", False)
                            else "-grouped"
                        )
                    )
                count = 2 // len(groups)
                if stage == 1 and not getattr(group[0], "gathers_input", False):
                    # Retain both old FC1 finalists when adding a new input
                    # layout. End-to-end timing decides whether GATHER wins.
                    count = 2
                choices = tuner.rank_tactics(
                    name, [runner], config, list(inputs), k=count
                )
                by_tactic = {a.tactic: a for a in group}
                selected.extend(group[0] if t == -1 else by_tactic[t] for t in choices)
            ranked.append(tuple(selected))
        first, second = ranked
    else:
        grouped = tuple(a for a in first if not getattr(a, "gathers_input", False))
        gathered = tuple(a for a in first if getattr(a, "gathers_input", False))
        first = (grouped[:1] + gathered[:1]) if grouped and gathered else first[:2]
        second = second[:2]
    # Fusion is judged end-to-end, since its benefit includes removing the
    # intermediate quantizer. FMA remains restricted to its measured shapes.
    compound = [(tag, a.tactic, b.tactic) for a, b in product(first + fused, second)]
    # Expanding a stage pool must not drop independent small-token FMA plans.
    return compound + [tactic for tactic in state.launches if tactic[0] != tag]
