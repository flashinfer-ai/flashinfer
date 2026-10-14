from flashinfer.autotuner import AutoTuner, TunableRunner


class DummyRunner(TunableRunner):
    """Minimal no-op runner for autotuner tests."""

    def __init__(self, valid_tactics=(0, 1, 2)):
        self.valid_tactics = valid_tactics

    def get_valid_tactics(self, inputs, profile):
        return list(self.valid_tactics)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


def reset_autotuner() -> AutoTuner:
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()
    tuner.is_tuning_mode = False
    return tuner
