# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Factorized tuning of the bounded quantized Frost candidate pool."""

from itertools import product

from ....autotuner import AutoTuner, TunableRunner, TuningConfig
from .cache import retain_graph_resources
from .heuristics import POLICY_VERSION


def prepared_state(runner, inputs, *, hidden_multiplier=1):
    state = runner.launch_state_for(inputs)
    if state is not None:
        return state
    # The autotuner synthesizes plain tensor lists. Exact-shape plans were
    # prepared by pack_inputs before this profile is visited.
    t, h = inputs[1].shape
    for key in reversed(runner._plans):
        if key[:2] == (t, h * hidden_multiplier):
            return runner._plans[key]
    return None


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
        return (POLICY_VERSION, self.stage, tuple(self.plans))

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if tactic == -1:
            tactic = next(iter(self.plans))
        plan = self.plans[tactic]
        if do_preparation:
            # Unfused plans share the grouped tensor layout. Recreate it for
            # each stage: FC2's output aliases FC1's gathered input. This runs
            # outside timing/capture; routing stays entirely on the device.
            plan["prepare_stages"](*inputs, self.state.workspace)
            return
        retain_graph_resources(self.state, inputs)
        plan[f"run_fc{self.stage}"](*inputs, self.state.workspace)


def ranked_tactics(state, inputs, tag):
    first = tuple(a for a in state.first if not a.quantizes_output)
    second = state.second
    if len(first) <= 2 and len(second) <= 2:
        # Preserve every offline choice, including its existing fused variants.
        return list(state.launches)
    fused = tuple(a for a in state.first if a.quantizes_output)
    tuner = AutoTuner.get()
    if tuner.is_tuning_mode:
        config = TuningConfig(
            use_cuda_graph=True,
            cuda_graph_profile_replays=3,
            inputs_pre_hook=lambda tensors: list(inputs),
        )
        ranked = []
        for stage in (1, 2):
            runner = _StageRunner(state, tag, stage, first, second)
            choices = tuner.rank_tactics(
                f"{tag}-fc{stage}-{POLICY_VERSION}", [runner], config, list(inputs), k=2
            )
            kernels = first if stage == 1 else second
            by_tactic = {a.tactic: a for a in kernels}
            ranked.append(
                tuple(kernels[0] if t == -1 else by_tactic[t] for t in choices)
            )
        first, second = ranked
    else:
        first, second = first[:2], second[:2]
    # Fusion is judged end-to-end, since its benefit includes removing the
    # intermediate quantizer. FMA remains restricted to its measured shapes.
    return [(tag, a.tactic, b.tactic) for a, b in product(first + fused, second)]
