"""Catalog consistency of the generated MegaMoE v3 family (no GPU required)."""

import json
from pathlib import Path

import pytest

from flashinfer.experimental.mega_moe_v3 import runtime

ROOT = Path(__file__).resolve().parents[2]


def _catalog():
    return runtime.catalog()


def test_catalog_is_the_merged_layout_with_both_architectures():
    catalog = _catalog()
    assert catalog.schema == "mega_moe_v3.v3" and catalog.merged
    assert catalog.arches == ("sm_100a", "sm_103a")
    assert runtime.supported_num_sms("sm_100a") == [148]
    assert runtime.supported_num_sms("sm_103a") == [152]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_every_route_resolves_its_programs_grid_and_sm_count(arch):
    catalog = _catalog()
    programs = catalog.programs(arch)
    for key, route in catalog.routes(arch).items():
        surface, args = json.loads(key)
        assert route["metadata"]["surface"] == surface
        num_sms = catalog.route_num_sms(arch, key, route)
        grid = route["grid"][arch]
        assert len(grid) == 3
        if surface != "grouped_l1":
            # Persistent schedules fill the device: one CTA per physical SM.
            assert grid == [num_sms, 1, 1]
        for stage in route["stages"]:
            name = runtime.stage_program(stage, arch)
            assert name in programs, (key, stage["name"], arch)
            if isinstance(stage["program"], dict):
                # A per-architecture pair names the architecture in both members.
                assert set(stage["program"]) == set(catalog.arches)
                assert name.endswith(arch.replace("_", ""))


def test_programs_sources_symbols_and_compile_line_definitions():
    catalog = _catalog()
    data = json.loads(catalog.path.read_text())
    for name, record in data["programs"].items():
        kernel, binding = (ROOT / src for src in record["sources"])
        assert (
            kernel.name == f"{name}_kernel.cu" and binding.name == f"{name}_binding.cu"
        )
        kernel_text, binding_text = kernel.read_text(), binding.read_text()
        symbol = f"kernel_{name}("
        assert symbol in kernel_text and symbol in binding_text, name
        pipeline = "_pipeline_" in name
        if record.get("definitions"):
            # Architecture-neutral pipeline schedule: the SM count comes from the compile line.
            assert record["definitions"] == ["NUM_CTAS"] and len(record["arches"]) == 2
            assert (
                "#ifndef NUM_CTAS" in kernel_text
                and "#define NUM_CTAS" not in kernel_text
            )
        elif pipeline:
            # A per-architecture pipeline schedule keeps its own SM count.
            (arch,) = record["arches"]
            (num_sms,) = catalog.supported_num_sms(arch)
            assert f"#define NUM_CTAS {num_sms}\n" in kernel_text, name
        else:
            assert "NUM_CTAS" not in kernel_text, name
        assert (
            "ScopedCudaDevice" not in binding_text and "std::mutex" not in binding_text
        )


def test_load_program_supplies_the_route_sm_count_as_num_ctas():
    pytest.importorskip("torch")
    catalog = _catalog()
    for arch in catalog.arches:
        (num_sms,) = catalog.supported_num_sms(arch)
        for name, record in catalog.programs(arch).items():
            spec = catalog.jit_spec(
                arch, name, catalog.definitions(arch, name, {"NUM_CTAS": num_sms})
            )
            flags = spec.extra_cuda_cflags
            if record.get("definitions"):
                assert f"-DNUM_CTAS={num_sms}" in flags
                assert spec.name == f"{name}_{arch}_num_ctas{num_sms}"
            else:
                assert not any(f.startswith("-DNUM_CTAS") for f in flags)
            for flag in catalog.compile_flags(record, arch):
                assert flag in flags
