from types import SimpleNamespace

import torch

from flashinfer.autotuner import DynamicTensorSpec, DynamicValueSpec, TuningConfig
from flashinfer.fused_moe.dist_aware import da_core


class _RecordingTuner:
    def __init__(self, *, tuning):
        self.is_tuning_mode = tuning
        self.calls = []

    def choose_one(self, custom_op, runners, config, inputs, **kwargs):
        self.calls.append((custom_op, runners, config, inputs, kwargs))
        return runners[0], 7


def _value_aware_config():
    value_spec = DynamicValueSpec(
        input_idx=0,
        gen_value_buckets=(0, 1),
        map_to_value_bucket=lambda _tensor: 0,
        tensor_value_generator=lambda _bucket, tensor: tensor,
    )
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(8,),
                map_to_tuning_buckets=lambda _size: 8,
                value_specs=(value_spec,),
            ),
        ),
        value_sample_count=lambda _buckets, default: default,
        default_value_buckets=da_core.DEFAULT_PROFILE_VALUE_BUCKETS,
    )


def test_da_tuning_uses_one_value_aware_sweep():
    tuner = _RecordingTuner(tuning=True)
    config = _value_aware_config()
    runner = object()
    inputs = [torch.empty(1)]

    result = da_core.choose_one(
        SimpleNamespace(config=SimpleNamespace(enabled=True)),
        tuner,
        custom_op="flashinfer::moe",
        runner=runner,
        tuning_config=config,
        inputs=inputs,
        da_value_specs_active=True,
        marker="unchanged",
    )

    assert result == (runner, 7)
    assert len(tuner.calls) == 1
    assert tuner.calls[0][2] is config
    assert tuner.calls[0][2].default_value_buckets == (
        da_core.DEFAULT_PROFILE_VALUE_BUCKETS
    )
    assert tuner.calls[0][4] == {"marker": "unchanged"}


def test_da_eager_lookup_does_not_start_hidden_tuning():
    tuner = _RecordingTuner(tuning=False)
    config = _value_aware_config()

    da_core.choose_one(
        SimpleNamespace(config=SimpleNamespace(enabled=True)),
        tuner,
        custom_op="flashinfer::moe",
        runner=object(),
        tuning_config=config,
        inputs=[torch.empty(1)],
        da_value_specs_active=True,
    )

    assert len(tuner.calls) == 1
    assert tuner.calls[0][2].dynamic_tensor_specs[0].value_specs == ()
    assert tuner.calls[0][2].value_sample_count is None
    assert tuner.calls[0][2].default_value_buckets is None


def test_verbose_eager_da_warns_without_graph_capture(monkeypatch, capsys):
    execution = SimpleNamespace(
        config=SimpleNamespace(enabled=True, verbose=True),
    )
    tuner = SimpleNamespace(is_tuning_mode=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    tactic = da_core.resolve_static_fallback(
        execution,
        tuner,
        custom_op="flashinfer::moe",
        runner=object(),
        tactic=7,
    )

    assert tactic == 7
    assert "running eagerly outside autotune" in capsys.readouterr().out


def test_verbose_da_does_not_warn_during_graph_capture(monkeypatch, capsys):
    execution = SimpleNamespace(
        config=SimpleNamespace(enabled=True, verbose=True),
    )
    tuner = SimpleNamespace(is_tuning_mode=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    tactic = da_core.resolve_static_fallback(
        execution,
        tuner,
        custom_op="flashinfer::moe",
        runner=object(),
        tactic=7,
    )

    assert tactic == 7
    assert capsys.readouterr().out == ""
