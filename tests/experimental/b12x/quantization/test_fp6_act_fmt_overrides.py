"""Model-agnostic activation-format override resolution (Phase 1.1 escalate).

Pattern matching is architecture-neutral: any dense model can tune
``*.linear_attn.*`` / ``*.mlp.*`` / ``*.self_attn.*`` groups without
hardcoding a model family.
"""
from __future__ import annotations

import pytest

from b12x.quantization.mxfp6.fp6_checkpoint import (
    load_act_fmt_overrides,
    parse_act_fmt_overrides,
    resolve_activation_format,
)


def test_resolve_default_when_no_overrides() -> None:
    assert resolve_activation_format("layers.0.mlp.gate_proj", default_fmt="e4m3") == "e4m3"


def test_resolve_first_matching_pattern_wins() -> None:
    overrides = parse_act_fmt_overrides(
        {"*.linear_attn.*": "e3m2", "*.mlp.*": "e4m3", "*": "e2m3"}
    )
    assert (
        resolve_activation_format(
            "language_model.layers.3.linear_attn.in_proj_qkvz",
            default_fmt="e4m3",
            overrides=overrides,
        )
        == "e3m2"
    )
    assert (
        resolve_activation_format(
            "language_model.layers.3.mlp.down_proj",
            default_fmt="e4m3",
            overrides=overrides,
        )
        == "e4m3"
    )
    assert (
        resolve_activation_format(
            "language_model.layers.3.self_attn.q_proj",
            default_fmt="e4m3",
            overrides=overrides,
        )
        == "e2m3"
    )


def test_resolve_strips_model_prefix() -> None:
    overrides = parse_act_fmt_overrides("*.linear_attn.*=e2m3")
    assert (
        resolve_activation_format(
            "model.language_model.layers.0.linear_attn.out_proj",
            default_fmt="e4m3",
            overrides=overrides,
        )
        == "e2m3"
    )


def test_parse_env_string() -> None:
    got = parse_act_fmt_overrides("*.linear_attn.*=e3m2,*.vision_tower.*=e2m3")
    assert got == [("*.linear_attn.*", "e3m2"), ("*.vision_tower.*", "e2m3")]


def test_parse_rejects_bad_fmt() -> None:
    with pytest.raises(ValueError, match="unsupported activation format"):
        parse_act_fmt_overrides("*.mlp.*=fp16")


def test_load_merges_env_before_config(monkeypatch) -> None:
    monkeypatch.setenv("B12X_FP6_ACT_FMT_OVERRIDES", "*.linear_attn.*=e2m3")
    qcfg = {"activation_format_overrides": {"*.linear_attn.*": "e3m2", "*.mlp.*": "e4m3"}}
    got = load_act_fmt_overrides(qcfg)
    # Env first: first match for linear_attn is e2m3, not the config e3m2.
    assert got[0] == ("*.linear_attn.*", "e2m3")
    assert ("*.mlp.*", "e4m3") in got
    assert (
        resolve_activation_format(
            "layers.0.linear_attn.out_proj", default_fmt="e4m3", overrides=got
        )
        == "e2m3"
    )
