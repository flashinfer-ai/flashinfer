"""The AOT gate for the sparse-score module.

The scorer multiplies with m16n8k16, which every SM8-or-newer device has, so
the build has to register it whenever any target is that new. An earlier gate
keyed on the sm80 capability flag, which is only set when an 8.x target is in
the build, and so skipped the module on an SM90-only or SM120-only build.
"""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("target_archs", "expected"),
    [
        ({(8, "0")}, True),
        ({(9, "0a")}, True),
        ({(10, "0a")}, True),
        ({(12, "0f")}, True),
        ({(9, "0a"), (12, "0f")}, True),
        ({(7, "5")}, False),
    ],
)
def test_the_scorer_is_registered_for_every_sm8_or_newer_target(
    monkeypatch, target_archs, expected
) -> None:
    from flashinfer import aot
    from flashinfer.jit import core as jit_core

    calls = []

    monkeypatch.setattr(
        jit_core,
        "current_compilation_context",
        SimpleNamespace(TARGET_CUDA_ARCHS=target_archs),
    )
    monkeypatch.setattr(
        aot,
        "gen_sparse_scores_module",
        lambda: calls.append("sparse_scores") or SimpleNamespace(name="sparse_scores"),
    )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    # The sm80 flag stays off throughout, so a gate that still keyed on it
    # would register nothing for any of these targets.
    aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {},
        False,  # add_comm
        False,  # add_gemma
        False,  # add_oai_oss
        False,  # add_moe
        False,  # add_act
        True,  # add_misc
        False,  # add_xqa
    )

    assert (calls == ["sparse_scores"]) is expected


def test_the_scorer_is_left_out_when_the_misc_modules_are(monkeypatch) -> None:
    from flashinfer import aot
    from flashinfer.jit import core as jit_core

    calls = []

    monkeypatch.setattr(
        jit_core,
        "current_compilation_context",
        SimpleNamespace(TARGET_CUDA_ARCHS={(8, "0")}),
    )
    monkeypatch.setattr(
        aot,
        "gen_sparse_scores_module",
        lambda: calls.append("sparse_scores") or SimpleNamespace(name="sparse_scores"),
    )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    aot.gen_all_modules(
        [], [], [], [], [], [], {}, False, False, False, False, False, False, False
    )

    assert calls == []
