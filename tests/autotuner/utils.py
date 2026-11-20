import torch

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


def make_scoped_tuner(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    monkeypatch.setattr(
        "flashinfer.autotuner._collect_metadata", lambda: {"test": "scope"}
    )
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    calls = []

    def profile(runner, inputs, tactic, config, **kwargs):
        calls.append((config.use_cuda_graph, tactic))
        winner = 2 if config.use_cuda_graph else 1
        return 1.0 if tactic == winner else 2.0 + abs(tactic)

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    return tuner, calls
