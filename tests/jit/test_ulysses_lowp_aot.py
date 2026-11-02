"""AOT inventory for the two Ulysses Lowp architecture modules."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version


@pytest.mark.parametrize(
    ("arches", "cuda_version", "expected"),
    [
        ({(8, "0")}, "13.0", set()),
        ({(8, "6")}, "13.0", set()),
        ({(8, "9")}, "12.7", set()),
        ({(8, "9")}, "12.8", {"ulysses_lowp"}),
        ({(9, "0a")}, "13.0", {"ulysses_lowp_sm90"}),
        ({(12, "0f")}, "13.0", {"ulysses_lowp"}),
        ({(8, "9"), (12, "0f")}, "13.0", {"ulysses_lowp"}),
        (
            {(8, "9"), (9, "0a"), (12, "0f")},
            "13.0",
            {"ulysses_lowp", "ulysses_lowp_sm90"},
        ),
    ],
)
@pytest.mark.parametrize("add_comm", [False, True])
def test_ulysses_lowp_aot_inventory(
    monkeypatch, arches, cuda_version, expected, add_comm
):
    from flashinfer import aot
    from flashinfer.jit import comm

    class CompilationContext:
        TARGET_CUDA_ARCHS = arches

        def get_nvcc_flags_list(self, supported_major_versions=None):
            return [
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
                for major, minor in sorted(arches)
            ]

    monkeypatch.setattr(aot, "CompilationContext", CompilationContext)
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda_version))
    capabilities = aot.detect_sm_capabilities()
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    # Isolate inventory selection from unrelated generators as the AOT list evolves.
    for module in (aot, comm):
        for name in dir(module):
            if (
                name.startswith("gen_")
                and "_module" in name
                and name != "gen_all_modules"
            ):
                spec = SimpleNamespace(name=name)
                result = [spec] if name.endswith("_modules") else spec
                monkeypatch.setattr(module, name, Mock(return_value=result))
    generators = {}
    for name in ("ulysses_lowp", "ulysses_lowp_sm90"):
        generators[name] = Mock(return_value=SimpleNamespace(name=name))
        monkeypatch.setattr(comm, f"gen_{name}_module", generators[name])
    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        capabilities,
        add_comm,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    expected = expected if add_comm else set()
    assert {
        spec.name for spec in specs if spec.name.startswith("ulysses_lowp")
    } == expected
    for name, generator in generators.items():
        assert generator.call_count == int(name in expected)
