"""The AOT gate for the QSA modules.

Three modules go in together or not at all -- the scorer, the route it expands
and the gate that closes the step -- because a build carrying some of them
would report a capability it cannot serve. The pre-indexer is JIT-only and is
deliberately not in the set.

The scorer multiplies with m16n8k16, which every SM8-or-newer device has, so
the gate is the target list. An earlier gate keyed on the ``sm80`` capability
flag, which is only set when an 8.x target is in the build, and so skipped the
modules on an SM90-only or SM120-only build.
"""

from types import SimpleNamespace

import pytest

QSA_MODULES = {"qsa_output_gate", "sparse_route", "sparse_scores"}


def _collect(monkeypatch, target_archs, add_misc):
    """Which module names ``gen_all_modules`` asks for, as a set.

    A set rather than a list: what matters is that the three are there or none
    of them is, not the order the builder happens to append them in.
    """
    from flashinfer import aot
    from flashinfer.jit import core as jit_core

    names = set()

    monkeypatch.setattr(
        jit_core,
        "current_compilation_context",
        SimpleNamespace(TARGET_CUDA_ARCHS=target_archs),
    )
    for name in QSA_MODULES:
        monkeypatch.setattr(
            aot,
            f"gen_{name}_module",
            (lambda n: lambda: names.add(n) or SimpleNamespace(name=n))(name),
        )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {},  # sm_capabilities: the sm80 flag stays off throughout, so a gate
        # that still keyed on it would register nothing for any target.
        False,  # add_comm
        False,  # add_gemma
        False,  # add_oai_oss
        False,  # add_moe
        False,  # add_act
        add_misc,
        False,  # add_xqa
    )
    return names


@pytest.mark.parametrize(
    ("target_archs", "add_misc", "expected"),
    [
        ({(7, "5")}, True, False),
        ({(8, "0")}, True, True),
        ({(9, "0a")}, True, True),
        ({(10, "0a")}, True, True),
        ({(12, "0f")}, True, True),
        ({(7, "5"), (9, "0a")}, True, True),
        ({(8, "0")}, False, False),
    ],
    ids=[
        "sm75",
        "sm80",
        "sm90-only",
        "sm100-only",
        "sm120-only",
        "sm75+sm90",
        "sm80-without-misc",
    ],
)
def test_the_qsa_modules_go_in_together_or_not_at_all(
    monkeypatch, target_archs, add_misc, expected
) -> None:
    names = _collect(monkeypatch, target_archs, add_misc)
    assert names == (QSA_MODULES if expected else set())
