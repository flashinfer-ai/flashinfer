"""JIT target-selection tests for the frozen AlphaMoE NVFP4 module."""

from types import SimpleNamespace

import pytest


def test_alphamoe_nvfp4_jit_keeps_only_exact_sm100_sm103(monkeypatch):
    from flashinfer.jit import fused_moe

    monkeypatch.setattr(
        fused_moe.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a"), (10, "3a"), (10, "7a"), (12, "0f")},
    )
    flags = fused_moe._alphamoe_nvfp4_sm100_nvcc_flags()
    gencode = [flag for flag in flags if flag.startswith("-gencode=")]
    assert gencode == [
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
    ]
    assert flags.count("--use_fast_math") == 1


def test_alphamoe_nvfp4_jit_requires_an_exact_target(monkeypatch):
    from flashinfer.jit import fused_moe

    monkeypatch.setattr(
        fused_moe.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "7a"), (12, "0f")},
    )
    with pytest.raises(RuntimeError, match="exact SM100a or SM103a"):
        fused_moe._alphamoe_nvfp4_sm100_nvcc_flags()


@pytest.mark.parametrize(
    ("capabilities", "expected_calls"),
    [
        ({"sm100a_exact": True}, 1),
        ({"sm103a_exact": True}, 1),
        ({"sm103": True}, 0),
        ({"sm100": True}, 0),
        ({"sm100f": True}, 0),
        ({"sm107": True}, 0),
    ],
)
def test_alphamoe_nvfp4_aot_uses_only_exact_arches(
    monkeypatch, capabilities, expected_calls
):
    from flashinfer import aot

    for name in tuple(vars(aot)):
        if name.startswith("gen_") and name != "gen_all_modules":
            monkeypatch.setattr(
                aot,
                name,
                lambda *args, _name=name, **kwargs: SimpleNamespace(name=_name),
            )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    calls = []

    def fake_alphamoe_module():
        calls.append("alphamoe")
        return SimpleNamespace(name="alphamoe_nvfp4_sm100")

    monkeypatch.setattr(aot, "gen_alphamoe_nvfp4_sm100_module", fake_alphamoe_module)
    aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        capabilities,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )
    assert len(calls) == expected_calls
