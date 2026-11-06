"""CPU regressions for the BF16 x FP4 autotune cache."""

import pytest
import torch

import flashinfer.autotuner.autotuner as autotuner_module
import flashinfer.gemm.gemm_bf16_fp4_cute_dsl as mod
from flashinfer.autotuner import AutoTuner


def _inputs(m):
    n = k = 4096
    return [
        torch.empty((m, k), device="meta", dtype=torch.bfloat16),
        torch.empty((k // 16, n * 2), device="meta", dtype=torch.int32),
        torch.empty((k // 16, n), device="meta", dtype=torch.uint8),
        torch.empty(1, device="meta"),
        torch.bfloat16,
        torch.empty((m, n), device="meta", dtype=torch.bfloat16),
        16,
    ]


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(mod, "get_device_sm_count", lambda device: 84)
    monkeypatch.setattr(mod, "get_compute_capability", lambda device: (12, 0))
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: tuner))
    captured = []

    def capture(op, runners, config, inputs):
        captured.append((op, runners[0], config, inputs))
        return lambda **kwargs: None, -1

    monkeypatch.setattr(tuner, "choose_one", capture)

    def call(m):
        a, b, sf, alpha, dtype, out, block_size = _inputs(m)
        mod._compute_cute_dsl(a, b, sf, alpha, dtype, out, block_size)
        return captured.pop()

    return tuner, call


@pytest.mark.cpu_only
@pytest.mark.parametrize("source", ["memory", "file", "bundled"])
@pytest.mark.parametrize(
    "warmup_m,tuned_m,runtime_m,gemv,hit",
    [(1, 1, 4, True, False), (2, 2, 4, False, True), (128, 1, 1, True, True)],
)
def test_cached_tactics_preserve_warmup_reuse(
    backend, monkeypatch, source, warmup_m, tuned_m, runtime_m, gemv, hit
):
    tuner, call = backend
    op, runner, config, warmup_inputs = call(warmup_m)
    profile = next(
        p
        for p in tuner._generate_optimization_profiles(config, warmup_inputs)
        if p.get_opt_shapes()[0][0] == tuned_m
    )
    profile_inputs = _inputs(tuned_m)
    configs = mod._bf16_fp4_cute_dsl_tactic_configs(4096, 4096, 84)
    candidates = [
        t
        for t in runner.get_valid_tactics(profile_inputs, profile)
        if (configs[t][7] == "gemv") == gemv
    ]
    assert candidates, "warmup must retain this profile's eligible tactics"
    tactic = candidates[0]
    key = tuner._get_cache_key(
        op,
        runner,
        profile.get_opt_shapes(),
        config,
        runner.get_cache_key_extras(profile_inputs),
    )
    if source == "memory":
        tuner.profiling_cache[key] = (tactic, profile)
    elif source == "file":
        tuner._file_configs[key.file_key] = (type(runner).__name__, tactic)
    else:
        # Retain the bucket metadata, but force lookup through to the bundle.
        tuner.profiling_cache[key] = (-2, profile)
        monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
        monkeypatch.setattr(
            autotuner_module,
            "load_from_file",
            lambda file_key: (True, 0, tactic, None)
            if file_key == key.file_key
            else (False, 0, -1, None),
        )

    _, runtime_runner, _, runtime_inputs = call(runtime_m)
    result = tuner.search_cache(
        op,
        [runtime_runner],
        tuple(tuner._get_input_sizes(runtime_inputs)),
        config,
        inputs=runtime_inputs,
    )
    assert result[:3] == ((True, 0, tactic) if hit else (False, 0, -1))


@pytest.mark.cpu_only
@pytest.mark.parametrize("tactic,valid", [(-1, True), (-2, False), (10**6, False)])
def test_tactic_validation_handles_fallback_and_invalid_indices(backend, tactic, valid):
    _, call = backend
    _, runner, _, inputs = call(4)
    assert runner.validate_tactic(inputs, tactic) is valid


@pytest.mark.cpu_only
@pytest.mark.parametrize("source", ["memory", "file"])
def test_repeated_large_warmup_reuses_single_row_profile(backend, monkeypatch, source):
    tuner, call = backend
    op, runner, config, inputs = call(128)
    configs = mod._bf16_fp4_cute_dsl_tactic_configs(4096, 4096, 84)
    winners = {}
    for profile in tuner._generate_optimization_profiles(config, inputs):
        m = profile.get_opt_shapes()[0][0]
        profile_inputs = _inputs(m)
        tactic = next(
            t
            for t in runner.get_valid_tactics(profile_inputs, profile)
            if (configs[t][7] == "gemv") == (m == 1)
        )
        key = tuner._get_cache_key(
            op,
            runner,
            profile.get_opt_shapes(),
            config,
            runner.get_cache_key_extras(profile_inputs),
        )
        if source == "memory":
            tuner.profiling_cache[key] = (tactic, profile)
        else:
            tuner._file_configs[key.file_key] = (type(runner).__name__, tactic)
        winners[m] = tactic

    def unexpected_synthesis(*args, **kwargs):
        pytest.fail("a warm profile must not synthesize inputs or be reprofiled")

    monkeypatch.setattr(tuner, "_prepare_input_tensors", unexpected_synthesis)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    tuner.is_tuning_mode = True
    for _ in range(2):
        # Bypass the fixture's launch stub to exercise the real profiling loop.
        assert (
            AutoTuner.choose_one(tuner, op, [runner], config, inputs)[1] == winners[128]
        )
    tuner.is_tuning_mode = False
    _, decode_runner, _, decode_inputs = call(1)
    assert (
        AutoTuner.choose_one(tuner, op, [decode_runner], config, decode_inputs)[1]
        == winners[1]
    )
