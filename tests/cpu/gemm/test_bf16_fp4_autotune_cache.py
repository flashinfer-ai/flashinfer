"""CPU regressions for the BF16 x FP4 autotune cache."""

import pytest
import torch

import flashinfer.gemm.gemm_bf16_fp4_cute_dsl as mod
from flashinfer.autotuner import AutoTuner


@pytest.mark.cpu_only
@pytest.mark.parametrize("tuned_m", [1, 2])
def test_cute_dsl_cached_tactics_respect_gemv_shape_domain(monkeypatch, tuned_m):
    monkeypatch.setattr(mod, "get_device_sm_count", lambda device: 84)
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    tuner = AutoTuner()
    runner = mod._cute_dsl_bf16_fp4_runner(allow_gemv=tuned_m == 1)
    config = mod._BF16_FP4_CUTE_DSL_TUNING_CONFIG
    n, k = 4096, 4096

    def inputs(m):
        return [
            torch.empty((m, k), device="meta"),
            torch.empty((k // 16, n * 2), device="meta"),
            None,
            None,
            torch.bfloat16,
            torch.empty((m, n), device="meta"),
            16,
        ]

    tuned = inputs(tuned_m)
    tactic = next(
        i
        for i, cfg in enumerate(mod._bf16_fp4_cute_dsl_tactic_configs(n, k, 84))
        if (cfg[7] == "gemv") == (tuned_m == 1)
    )
    key = tuner._get_cache_key(
        "bf16_fp4_gemv_test",
        runner,
        tuple(tuner._get_input_sizes(tuned)),
        config,
        runner.get_cache_key_extras(tuned),
    )
    tuner.profiling_cache[key] = (tactic, None)

    def lookup(m):
        args = inputs(m)
        runtime_runner = mod._cute_dsl_bf16_fp4_runner(allow_gemv=m == 1)
        # Profile synthesis may change M; the runner's key domain must stay fixed.
        assert runtime_runner.get_cache_key_extras(
            args
        ) == runtime_runner.get_cache_key_extras(inputs(1))
        return tuner.search_cache(
            "bf16_fp4_gemv_test",
            [runtime_runner],
            tuple(tuner._get_input_sizes(args)),
            config,
            inputs=args,
        )

    assert lookup(tuned_m)[:3] == (True, 0, tactic)
    expected = (False, 0, -1) if tuned_m == 1 else (True, 0, tactic)
    assert lookup(4)[:3] == expected
