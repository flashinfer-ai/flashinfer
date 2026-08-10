from unittest.mock import MagicMock, patch

import torch

import flashinfer.grouped_mm._sm120_moe_autotune as moe_autotune
import flashinfer.grouped_mm.cute_sm120_mxfp8_groupwise.core as mxfp8_core
from flashinfer.autotuner import AutoTuner


def _inputs(total_rows=64 * 192, num_experts=64, n=5760, k=2944, gran_k=128):
    packed_k = (k + 4 * gran_k - 1) // (4 * gran_k)
    return [
        torch.empty((total_rows, k), device="meta"),
        torch.empty((num_experts, n, k), device="meta"),
        torch.empty((total_rows, packed_k), dtype=torch.int32, device="meta"),
        torch.empty((num_experts, n, packed_k), dtype=torch.int32, device="meta"),
        torch.arange(num_experts + 1, dtype=torch.int32, device="meta"),
    ]


def test_mxfp8_plain_gated_and_grank_use_distinct_keys():
    inputs = _inputs()
    plain = mxfp8_core._CuteSm120Mxfp8MoeRunner(
        torch.empty((64 * 192, 5760), device="meta"), False, (1, 1, 128), "MN"
    )
    gated = mxfp8_core._CuteSm120Mxfp8MoeRunner(
        torch.empty((64 * 192, 2880), device="meta"), True, (1, 1, 128), "MN"
    )
    grank32 = mxfp8_core._CuteSm120Mxfp8MoeRunner(
        torch.empty((64 * 192, 5760), device="meta"), False, (1, 1, 32), "MN"
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
            "cute_sm120_mxfp8_groupwise_moe",
            plain,
            shapes,
            moe_autotune.SM120_MOE_TUNING_CONFIG,
            plain.get_cache_key_extras(inputs),
        )
        gated_key = tuner._get_cache_key(
            "cute_sm120_mxfp8_groupwise_moe",
            gated,
            shapes,
            moe_autotune.SM120_MOE_TUNING_CONFIG,
            gated.get_cache_key_extras(inputs),
        )
        grank32_key = tuner._get_cache_key(
            "cute_sm120_mxfp8_groupwise_moe",
            grank32,
            shapes,
            moe_autotune.SM120_MOE_TUNING_CONFIG,
            grank32.get_cache_key_extras(inputs),
        )

    assert plain_key != gated_key
    assert plain_key != grank32_key
    assert gated_key != grank32_key


def test_mxfp8_tactics_are_independent_per_grank():
    grank32 = mxfp8_core._CuteSm120Mxfp8MoeRunner(
        torch.empty((64, 128), device="meta"), False, (1, 1, 32), "MN"
    )
    grank128 = mxfp8_core._CuteSm120Mxfp8MoeRunner(
        torch.empty((64, 128), device="meta"), False, (1, 1, 128), "MN"
    )
    assert tuple(grank32.get_valid_tactics(_inputs(gran_k=32), MagicMock())) == (
        mxfp8_core._MXFP8_MOE_TACTICS
    )
    assert tuple(grank128.get_valid_tactics(_inputs(gran_k=128), MagicMock())) == (
        mxfp8_core._MXFP8_MOE_TACTICS_GRANK128
    )
    schema = mxfp8_core._MXFP8_MOE_TACTIC_SCHEMA_VERSION
    for tactic in ((schema, 32, 128), (schema, 128, 8)):
        assert grank32.is_valid_tactic(tactic)
        assert grank128.is_valid_tactic(tactic)
    assert not grank32.is_valid_tactic((schema - 1, 32, 128))


def test_mxfp8_profile_and_actual_launch_use_gated_mode():
    inputs = _inputs(total_rows=64, num_experts=1, n=128, k=128)
    out = torch.empty((64, 64), device="meta")
    runner = mxfp8_core._CuteSm120Mxfp8MoeRunner(out, True, (1, 1, 128), "MN")
    runner(inputs, do_preparation=True)
    module = MagicMock()
    tactic = mxfp8_core._MXFP8_MOE_TACTICS[0]

    with (
        patch.object(
            mxfp8_core, "get_gemm_sm120_module_cute_mxfp8", return_value=module
        ),
        patch.object(moe_autotune, "is_in_profile_measurement", return_value=True),
    ):
        runner(inputs, tactic=tactic)
    profile_args = module.moe_gemm_mxfp8_nt_groupwise_tuned.call_args.args
    assert profile_args[5].shape == (64, 64)
    assert profile_args[10] is True

    module.reset_mock()
    with patch.object(
        mxfp8_core, "get_gemm_sm120_module_cute_mxfp8", return_value=module
    ):
        runner(inputs, tactic=tactic)
    actual_args = module.moe_gemm_mxfp8_nt_groupwise_tuned.call_args.args
    assert actual_args[5] is out
    assert actual_args[10] is True


def test_mxfp8_eligibility_covers_all_positive_mpe():
    assert mxfp8_core._should_autotune_mxfp8_moe(
        torch.empty((64 * 192, 2944), device="meta"),
        torch.empty((64, 5760, 2944), device="meta"),
    )
    for mpe in (1, 8, 12, 15, 16, 32, 64, 376):
        assert mxfp8_core._should_autotune_mxfp8_moe(
            torch.empty((64 * mpe, 2944), device="meta"),
            torch.empty((64, 5760, 2944), device="meta"),
        )
    assert not mxfp8_core._should_autotune_mxfp8_moe(
        torch.empty((64 * 192, 2048), device="meta"),
        torch.empty((64, 5760, 2048), device="meta"),
    )
    assert not mxfp8_core._should_autotune_mxfp8_moe(
        torch.empty((0, 2944), device="meta"),
        torch.empty((64, 5760, 2944), device="meta"),
    )
