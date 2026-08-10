from unittest.mock import MagicMock, patch

import torch

import flashinfer.grouped_mm._sm120_moe_autotune as moe_autotune
import flashinfer.grouped_mm.cute_sm120_fp8_groupwise.core as fp8_core
from flashinfer.autotuner import AutoTuner


def _inputs(total_rows=10, num_experts=4, n=128, k=128):
    return [
        torch.empty((total_rows, k)),
        torch.empty((num_experts, n, k)),
        torch.empty((1, total_rows + 3 * num_experts)),
        torch.empty((num_experts, 1, 1)),
        torch.tensor([0, 1, 2, 3, total_rows], dtype=torch.int32),
    ]


def test_uniform_m_indptr_is_balanced_without_mutating_caller():
    inputs = _inputs()
    original = inputs[-1].clone()
    prepared = fp8_core._prepare_uniform_m_indptr(inputs)

    assert prepared[-1].tolist() == [0, 3, 6, 8, 10]
    assert torch.equal(inputs[-1], original)


def test_runner_hash_and_tactics_are_mode_specific():
    plain = fp8_core._CuteSm120Fp8MoeRunner(
        torch.empty((10, 128)), False, (1, 128, 128), "MN"
    )
    gated = fp8_core._CuteSm120Fp8MoeRunner(
        torch.empty((10, 64)), True, (1, 128, 128), "MN"
    )

    assert hash(plain) != hash(gated)
    assert tuple(plain.get_valid_tactics(_inputs(), MagicMock())) == (
        fp8_core._FP8_MOE_PLAIN_TACTICS
    )
    assert tuple(gated.get_valid_tactics(_inputs(), MagicMock())) == (
        fp8_core._FP8_MOE_GATED_TACTICS
    )
    swapab = (fp8_core._FP8_MOE_TACTIC_SCHEMA_VERSION, 128, 8)
    assert plain.is_valid_tactic(swapab)
    assert gated.is_valid_tactic(swapab)
    assert not gated.is_valid_tactic(
        (fp8_core._FP8_MOE_TACTIC_SCHEMA_VERSION, 128, 128)
    )
    assert gated.is_valid_tactic((fp8_core._FP8_MOE_TACTIC_SCHEMA_VERSION, 32, 128))
    assert not plain.is_valid_tactic((fp8_core._FP8_MOE_TACTIC_SCHEMA_VERSION, 64, 64))


def test_plain_and_gated_use_separate_cache_keys():
    inputs = _inputs()
    plain = fp8_core._CuteSm120Fp8MoeRunner(
        torch.empty((10, 128)), False, (1, 128, 128), "MN"
    )
    gated = fp8_core._CuteSm120Fp8MoeRunner(
        torch.empty((10, 64)), True, (1, 128, 128), "MN"
    )
    tuner = AutoTuner()
    properties = MagicMock(name="device_properties")
    properties.name = "test-sm120"

    with (
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_properties", return_value=properties),
        patch.object(moe_autotune, "get_compute_capability", return_value=(12, 0)),
    ):
        shapes = tuple(tensor.shape for tensor in inputs)
        plain_key = tuner._get_cache_key(
            "cute_sm120_fp8_groupwise_moe",
            plain,
            shapes,
            fp8_core._FP8_MOE_TUNING_CONFIG,
            plain.get_cache_key_extras(inputs),
        )
        gated_key = tuner._get_cache_key(
            "cute_sm120_fp8_groupwise_moe",
            gated,
            shapes,
            fp8_core._FP8_MOE_TUNING_CONFIG,
            gated.get_cache_key_extras(inputs),
        )
        assert plain_key != gated_key

        tactic = fp8_core._FP8_MOE_PLAIN_TACTICS[0]
        tuner.profiling_cache[plain_key] = (tactic, None)
        assert not tuner.search_cache(
            "cute_sm120_fp8_groupwise_moe",
            [gated],
            shapes,
            fp8_core._FP8_MOE_TUNING_CONFIG,
            inputs,
        )[0]

        tuner.clear_cache()
        gated_tactic = fp8_core._FP8_MOE_GATED_TACTICS[0]
        tuner.profiling_cache[gated_key] = (gated_tactic, None)
        assert not tuner.search_cache(
            "cute_sm120_fp8_groupwise_moe",
            [plain],
            shapes,
            fp8_core._FP8_MOE_TUNING_CONFIG,
            inputs,
        )[0]


def test_runner_rejects_malformed_tactics():
    runner = fp8_core._CuteSm120Fp8MoeRunner(
        torch.empty((10, 128)), False, (1, 128, 128), "MN"
    )
    malformed = (
        (True, 64, 128),
        (1.0, 64, 128),
        torch.tensor([1, 64, 128]),
        [1, 64, 128],
        {"schema": 1, "tile_m": 64, "tile_n": 128},
    )
    assert all(not runner.is_valid_tactic(tactic) for tactic in malformed)


def test_profile_uses_actual_gated_mode_and_output_shape():
    inputs = _inputs(n=128)
    runner = fp8_core._CuteSm120Fp8MoeRunner(
        torch.empty((10, 64)), True, (1, 128, 128), "MN"
    )
    runner(inputs, do_preparation=True)
    module = MagicMock()

    with (
        patch.object(fp8_core, "get_gemm_sm120_module_cute_fp8", return_value=module),
        patch.object(moe_autotune, "is_in_profile_measurement", return_value=True),
    ):
        runner(inputs, tactic=(fp8_core._FP8_MOE_TACTIC_SCHEMA_VERSION, 64, 128))

    args = module.moe_gemm_fp8_nt_groupwise_tuned.call_args.args
    assert args[5].shape == (10, 64)
    assert args[10] is True


def test_unknown_loaded_tactic_falls_back_to_actual_mode():
    inputs = _inputs(n=128)
    out = torch.empty((10, 64))
    runner = fp8_core._CuteSm120Fp8MoeRunner(out, True, (1, 128, 128), "MN")
    module = MagicMock()

    with patch.object(fp8_core, "get_gemm_sm120_module_cute_fp8", return_value=module):
        result = runner(inputs, tactic=(999, 64, 128))

    args = module.moe_gemm_fp8_nt_groupwise.call_args.args
    assert result is out
    assert args[5] is out
    assert args[10] is True
    module.moe_gemm_fp8_nt_groupwise_tuned.assert_not_called()


def test_autotune_eligibility_uses_physical_n_and_uniform_mpe():
    with patch.object(fp8_core, "get_device_sm_count", return_value=110):
        assert fp8_core._should_autotune_fp8_moe(
            torch.empty((64 * 192, 2944), device="meta"),
            torch.empty((64, 5760, 2944), device="meta"),
        )
        for mpe in (1, 8, 12, 15, 16, 32, 64, 376):
            assert fp8_core._should_autotune_fp8_moe(
                torch.empty((64 * mpe, 2944), device="meta"),
                torch.empty((64, 5760, 2944), device="meta"),
            )
        assert not fp8_core._should_autotune_fp8_moe(
            torch.empty((64 * 192, 2048), device="meta"),
            torch.empty((64, 5760, 2048), device="meta"),
        )
        assert not fp8_core._should_autotune_fp8_moe(
            torch.empty((64 * 192, 2944), device="meta"),
            torch.empty((64, 2880, 2944), device="meta"),
        )
        assert not fp8_core._should_autotune_fp8_moe(
            torch.empty((33, 2944), device="meta"),
            torch.empty((1, 128, 2944), device="meta"),
        )
