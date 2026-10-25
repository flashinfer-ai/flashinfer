"""Catalog consistency of the generated MegaMoE v3 family (no GPU required)."""

import hashlib
import json
from pathlib import Path

import pytest

from flashinfer.experimental.deepgemm_common import jit_spec_name
from flashinfer.experimental.mega_moe_v3 import runtime

ROOT = Path(__file__).resolve().parents[2]

# The tensor-parallel and expert-parallel routes of the 384-expert, top-6,
# hidden-5120 model on sm_103a: (experts held, physical intermediate width) at
# every token count, routed FP4, activation clamp 10.
TP_EP_SHAPES = ((384, 640), (384, 384), (96, 2304), (48, 2304))
TP_EP_TOKENS = (1, 4, 16, 32, 128, 512, 1024, 4096)


def expected_block_m(num_tokens):
    """Source tile height of a TP/EP row: 16 rows up to 512 tokens, 32 at 1024, 128 at 4096."""
    return 128 if num_tokens >= 4096 else 32 if num_tokens >= 1024 else 16


def expected_source_form(num_experts, num_tokens):
    """``(readiness, slot_list)`` of a TP/EP row's generated program: per-block
    readiness with the per-expert slot list on every row but the one-token
    rows, whose program folds the dispatch metadata instead, and the two
    384-expert four-token rows, whose program waits for the whole dispatch
    (grid readiness, no slot list)."""
    if num_tokens == 1:
        return "block", False
    if num_experts == 384 and num_tokens == 4:
        return "grid", False
    return "block", True


def _catalog():
    return runtime.catalog()


def test_catalog_is_the_merged_layout_with_both_architectures():
    catalog = _catalog()
    assert catalog.schema == "mega_moe_v3.v3" and catalog.merged
    assert catalog.arches == ("sm_100a", "sm_103a")
    assert runtime.supported_num_sms("sm_100a") == [148]
    assert runtime.supported_num_sms("sm_103a") == [152]


def test_routes_are_keyed_by_architecture():
    """The routes table is keyed by architecture, so a route may ship for one
    architecture only; every route of ``sm_100a`` also ships for ``sm_103a``,
    and the loader hands each architecture its own table."""
    catalog = _catalog()
    data = json.loads(catalog.path.read_text())
    assert set(data["routes"]) == set(catalog.arches)
    for arch in catalog.arches:
        assert catalog.routes(arch) == data["routes"][arch]
        assert catalog.routes(arch), arch
    assert set(catalog.routes("sm_100a")) <= set(catalog.routes("sm_103a"))
    # The JSON spelling the exporter writes.
    assert catalog.path.read_text() == json.dumps(data, indent=1) + "\n"


def test_tp_ep_routes_ship_with_the_local_expert_filter_and_their_source_form():
    """Every TP/EP route of the model is catalogued for sm_103a with the
    local expert filter, its pinned source tile height, its source-schedule
    form and its own program. The GPU tests skip a route the catalog lacks,
    so this is the test that turns a missing or mis-described route into a
    failure instead of a silent skip."""
    catalog = _catalog()
    routes = catalog.routes("sm_103a")
    for num_experts, intermediate in TP_EP_SHAPES:
        for num_tokens in TP_EP_TOKENS:
            key = runtime.key(
                "pipeline",
                dict(
                    num_tokens=num_tokens,
                    num_experts=num_experts,
                    top_k=6,
                    hidden=5120,
                    intermediate=intermediate,
                    routed_weight_dtype="fp4",
                    activation_clamp=10.0,
                ),
            )
            shape = (num_experts, intermediate, num_tokens)
            assert key in routes, shape
            route = routes[key]
            metadata = route["metadata"]
            assert metadata["surface"] == "pipeline", shape
            assert metadata["self_cleaning"] is True, shape
            assert metadata["local_expert_filter"] is True, shape
            assert metadata["kernel_intermediate"] == intermediate, shape
            assert metadata["block_m"] == expected_block_m(num_tokens), shape
            assert (
                metadata["readiness"],
                metadata["slot_list"],
            ) == expected_source_form(num_experts, num_tokens), shape
            assert runtime.route_admitted_num_sms(route, "sm_103a") == (152,), shape
            assert runtime.route_grid(route, "sm_103a", 152) == (152, 1, 1), shape
            (stage,) = route["stages"]
            assert stage["name"] == "pipeline", shape
            assert runtime.stage_program(stage, "sm_103a") == (
                f"deepgemm_mega_moe_v3_pipeline_fp4_e{num_experts}_i{intermediate}"
                f"_t{num_tokens}_sm103a"
            ), shape
            assert key not in catalog.routes("sm_100a"), shape


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_every_route_resolves_its_programs_grid_and_sm_count(arch):
    catalog = _catalog()
    programs = catalog.programs(arch)
    for key, route in catalog.routes(arch).items():
        surface, args = json.loads(key)
        assert route["metadata"]["surface"] == surface
        admitted = runtime.route_admitted_num_sms(route, arch)
        assert admitted and set(admitted) <= set(runtime.supported_num_sms(arch))
        for num_sms in admitted:
            grid = runtime.route_grid(route, arch, num_sms)
            assert len(grid) == 3
            if surface != "grouped_l1":
                # Persistent schedules fill the device: one CTA per physical SM.
                assert grid == (num_sms, 1, 1), (key, arch, num_sms)
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
    programs = data["programs"]
    for name, record in programs.items():
        kernel, binding = (ROOT / src for src in record["sources"])
        assert kernel.name == f"{name}_kernel.cu"
        # A pipeline launcher can be shared: the binding is the one shipped for a
        # catalogued pipeline program and takes this program's kernel symbol from
        # the compile definitions (CAKE_PROGRAM_KERNEL).
        launcher = binding.name.removesuffix("_binding.cu")
        assert launcher in programs and (
            launcher == name or ("_pipeline_" in name and "_pipeline_" in launcher)
        ), (name, binding.name)
        kernel_text, binding_text = kernel.read_text(), binding.read_text()
        symbol = f"kernel_{name}("
        assert symbol in kernel_text, name
        if launcher != name:
            for arch in record["arches"]:
                flags = catalog.compile_flags(record, arch)
                assert f"-DCAKE_PROGRAM_KERNEL=kernel_{name}" in flags, (name, arch)
        pipeline = "_pipeline_" in name
        if record.get("definitions"):
            # Architecture-neutral pipeline schedule: the SM count comes from the
            # compile line, one of the counts the record's architectures route on.
            assert record["definitions"] == ["NUM_CTAS"]
            assert all(runtime.supported_num_sms(arch) for arch in record["arches"]), (
                name
            )
            assert (
                "#ifndef NUM_CTAS" in kernel_text
                and "#define NUM_CTAS" not in kernel_text
            )
        elif pipeline:
            # A per-architecture pipeline schedule keeps its own SM count, one
            # of the counts the architecture's routes run on.
            (arch,) = record["arches"]
            assert any(
                f"#define NUM_CTAS {num_sms}\n" in kernel_text
                for num_sms in runtime.supported_num_sms(arch)
            ), name
        else:
            assert "NUM_CTAS" not in kernel_text, name
        assert (
            "ScopedCudaDevice" not in binding_text and "std::mutex" not in binding_text
        )


def test_closure_sha256_names_the_shipped_sources():
    """Each program's closure_sha256 is the SHA-256 of its kernel then binding source bytes."""
    data = json.loads(_catalog().path.read_text())
    for name, record in data["programs"].items():
        digest = hashlib.sha256(
            b"".join((ROOT / src).read_bytes() for src in record["sources"])
        ).hexdigest()
        assert record["closure_sha256"] == digest, name


def test_load_program_supplies_the_route_sm_count_as_num_ctas():
    pytest.importorskip("torch")
    catalog = _catalog()
    for arch in catalog.arches:
        for num_sms in runtime.supported_num_sms(arch):
            for name, record in catalog.programs(arch).items():
                spec = catalog.jit_spec(
                    arch, name, catalog.definitions(arch, name, {"NUM_CTAS": num_sms})
                )
                flags = spec.extra_cuda_cflags
                if record.get("definitions"):
                    assert f"-DNUM_CTAS={num_sms}" in flags
                    assert spec.name == f"{jit_spec_name(name, arch)}_num_ctas{num_sms}"
                else:
                    assert not any(f.startswith("-DNUM_CTAS") for f in flags)
                for flag in catalog.compile_flags(record, arch):
                    assert flag in flags
