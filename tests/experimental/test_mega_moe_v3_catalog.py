"""Catalog consistency of the generated MegaMoE v3 family (no GPU required)."""

import hashlib
import json
import re
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
# The TP/EP routes whose sm_103a programs may read the FC1 ``A`` map straight
# from the activations (route metadata ``a_source`` "x"): every shape at one
# and four tokens, and the EP shapes (48 and 96 local experts) at 16 tokens.
DIRECT_A_TOKENS = (1, 4)
DIRECT_A_EP_TOKENS = (16,)
DIRECT_A_EP_EXPERTS = (48, 96)
# The TP routes whose sm_103a programs read the down-projection weights as a
# dense image of the logical width (route metadata ``w2_dense_k`` and
# ``w2_logical_k``): both TP shapes at 32 tokens and above, each with its
# (image, logical) widths.
W2_DENSE_TOKENS = (32, 128, 512, 1024, 4096)
W2_DENSE_WIDTHS = {(384, 384): (320, 288), (384, 640): (576, 576)}

# The launch form of the TP/EP programs: the generated binding's Run() launches
# through cudaLaunchKernelExC with three counted attribute slots, the cluster
# shape, the cooperative grid and, last, programmatic stream serialization; the
# kernel runs the programmatic dependent launch protocol, triggering its
# dependents on entry and waiting for its predecessor's completion at the entry
# of every role body, after the role's register budget and before the role's
# first global access, so the setup block (shared memory windows, tensor-map
# prefetches, barrier initialization, tensor-memory allocation and the cluster
# rendezvous) overlaps the predecessor's tail and no global access precedes it.
LAUNCH_ATTRIBUTES = "  cudaLaunchAttribute attrs[3]{};\n"
COOPERATIVE_SLOT = (
    "  attrs[n].id = cudaLaunchAttributeCooperative;\n"
    "  attrs[n].val.cooperative = 1;\n"
    "  ++n;\n"
)
PDL_ATTRIBUTE = "cudaLaunchAttributeProgrammaticStreamSerialization"
PDL_SLOT = (
    f"  attrs[n].id = {PDL_ATTRIBUTE};\n"
    "  attrs[n].val.programmaticStreamSerializationAllowed = 1;\n"
    "  ++n;\n"
)
GRID_DEP_LAUNCH = 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");'
GRID_DEP_WAIT = 'asm volatile("griddepcontrol.wait;" ::: "memory");'
ROLE_MARKER = re.compile(r"// ---- Role: (\w+) ----")
ROLES = ("dispatch", "load_a", "load_b", "mma", "schedule", "epilogue")
SETMAXNREG = "setmaxnreg."


# The TP/EP rows whose sm_103a program runs another source tile height than
# its token count's: the 48-expert shape at 128 and 512 tokens runs the 32-row
# tile of the 1024-token program.
SHAPE_BLOCK_M = {(48, 2304, 128): 32, (48, 2304, 512): 32}


# The TP/EP routes whose sm_103a programs run a persistent grid smaller than
# the 152-SM device (route metadata ``num_ctas`` and grid ``[num_ctas, 1, 1]``):
# the 384-expert I640, 96-expert and 48-expert rank shapes at one token run
# 148 CTAs, the device less their 4 combine CTAs, so a stream's next call is
# admitted while they finish. Every other route, the 384-expert I384 one-token
# route included, runs one CTA per SM.
PERSISTENT_GRID = {
    (384, 640, 1): 148,
    (96, 2304, 1): 148,
    (48, 2304, 1): 148,
}
T1_PROGRAMS = {
    f"deepgemm_mega_moe_v3_pipeline_fp4_e{e}_i{i}_t1_sm103a"
    for e, i, _ in PERSISTENT_GRID
}


def expected_block_m(num_experts, intermediate, num_tokens):
    """Source tile height of a TP/EP row: 16 rows up to 512 tokens, 32 at 1024,
    128 at 4096, unless ``SHAPE_BLOCK_M`` names the row's own."""
    shape = (num_experts, intermediate, num_tokens)
    if shape in SHAPE_BLOCK_M:
        return SHAPE_BLOCK_M[shape]
    return 128 if num_tokens >= 4096 else 32 if num_tokens >= 1024 else 16


def expected_source_form(num_experts, num_tokens):
    """``(readiness, slot_list)`` of a TP/EP row's generated program: per-block
    readiness with the per-expert slot list on the 128-token and larger rows;
    the rows of up to 32 tokens, and the 128-token expert-parallel rows (96 and
    48 local experts), fold the dispatch metadata into every CTA instead (block
    readiness, no slot list)."""
    if num_tokens <= 32 or (num_tokens == 128 and num_experts in (96, 48)):
        return "block", False
    return "block", True


def expected_w1_expert_map(num_experts, intermediate):
    """Whether a TP/EP row's program streams W1 through the per-expert
    three-dimensional map: the tensor-parallel rank shapes (every expert, a
    padded intermediate of 640 or 384), whose last FC1 tile has an N tail the
    map turns into TMA zero fill; the expert-parallel shapes keep the dense
    two-dimensional map."""
    return (num_experts, intermediate) in ((384, 640), (384, 384))


def _catalog():
    return runtime.catalog()


def tp_ep_route_key(num_experts, intermediate, num_tokens):
    """Catalog key of the FP4 pipeline route of a TP/EP shape at ``num_tokens``."""
    return runtime.key(
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


def tp_ep_program_sources():
    """``{program name: (kernel source, binding source)}`` of the TP/EP routes of sm_103a."""
    catalog = _catalog()
    records = json.loads(catalog.path.read_text())["programs"]
    routes = catalog.routes("sm_103a")
    sources = {}
    for num_experts, intermediate in TP_EP_SHAPES:
        for num_tokens in TP_EP_TOKENS:
            key = tp_ep_route_key(num_experts, intermediate, num_tokens)
            assert key in routes, (num_experts, intermediate, num_tokens)
            (stage,) = routes[key]["stages"]
            name = runtime.stage_program(stage, "sm_103a")
            sources[name] = tuple(ROOT / src for src in records[name]["sources"])
    return sources


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
            key = tp_ep_route_key(num_experts, intermediate, num_tokens)
            shape = (num_experts, intermediate, num_tokens)
            assert key in routes, shape
            route = routes[key]
            metadata = route["metadata"]
            assert metadata["surface"] == "pipeline", shape
            assert metadata["self_cleaning"] is True, shape
            assert metadata["local_expert_filter"] is True, shape
            assert metadata["w1_expert_map"] is expected_w1_expert_map(
                num_experts, intermediate
            ), shape
            assert metadata["kernel_intermediate"] == intermediate, shape
            assert metadata["block_m"] == expected_block_m(*shape), shape
            assert (
                metadata["readiness"],
                metadata["slot_list"],
            ) == expected_source_form(num_experts, num_tokens), shape
            assert runtime.route_admitted_num_sms(route, "sm_103a") == (152,), shape
            num_ctas = PERSISTENT_GRID.get(shape, 152)
            assert runtime.route_grid(route, "sm_103a", 152) == (num_ctas, 1, 1), shape
            assert runtime.route_num_ctas(route, "sm_103a", 152) == num_ctas, shape
            assert metadata.get("num_ctas") == PERSISTENT_GRID.get(shape), shape
            (stage,) = route["stages"]
            assert stage["name"] == "pipeline", shape
            assert runtime.stage_program(stage, "sm_103a") == (
                f"deepgemm_mega_moe_v3_pipeline_fp4_e{num_experts}_i{intermediate}"
                f"_t{num_tokens}_sm103a"
            ), shape
            assert key not in catalog.routes("sm_100a"), shape


def test_only_the_short_sm103a_rank_routes_read_a_from_the_activations():
    """A route's ``a_source`` names the storage its program's FC1 ``A`` tensor
    map describes: ``"pool"`` (the dispatch pool rows; also what an absent key
    means) or ``"x"`` (the ``[num_tokens, hidden]`` activation bytes; such a
    program copies no pool row). Only the sm_103a TP/EP routes of one and four
    tokens, and the EP routes of sixteen tokens, may state ``"x"``, and they
    state it together: a catalog rendered before the key existed states it
    nowhere, one rendered at the first key set states it on the eight short
    routes, one rendered at the current set on those and the two sixteen-token
    EP routes. No other route of either architecture states ``"x"``."""
    catalog = _catalog()
    direct_short = {
        tp_ep_route_key(num_experts, intermediate, num_tokens)
        for num_experts, intermediate in TP_EP_SHAPES
        for num_tokens in DIRECT_A_TOKENS
    }
    direct = direct_short | {
        tp_ep_route_key(num_experts, intermediate, num_tokens)
        for num_experts, intermediate in TP_EP_SHAPES
        if num_experts in DIRECT_A_EP_EXPERTS
        for num_tokens in DIRECT_A_EP_TOKENS
    }
    stated = set()
    for arch in catalog.arches:
        for key, route in catalog.routes(arch).items():
            a_source = route["metadata"].get("a_source", "pool")
            assert a_source in ("x", "pool"), (arch, key, a_source)
            if a_source == "x":
                assert arch == "sm_103a" and key in direct, (arch, key)
                stated.add(key)
    assert stated in (set(), direct_short, direct), sorted(direct ^ stated)


def test_only_the_dense_w2_tp_routes_state_their_w2_image_widths():
    """``w2_dense_k`` and ``w2_logical_k`` name a program that reads ``B2`` as
    the dense image of the logical down-projection width when ``K2`` is
    ``w2_dense_k``. Exactly the sm_103a TP routes of ``W2_DENSE_TOKENS`` state
    them, both together, with their shape's widths; no other route of either
    architecture states either key."""
    catalog = _catalog()
    expected = {
        tp_ep_route_key(num_experts, intermediate, num_tokens): widths
        for (num_experts, intermediate), widths in W2_DENSE_WIDTHS.items()
        for num_tokens in W2_DENSE_TOKENS
    }
    assert len(expected) == 10
    stated = {}
    for arch in catalog.arches:
        for key, route in catalog.routes(arch).items():
            metadata = route["metadata"]
            keys = [k for k in ("w2_dense_k", "w2_logical_k") if k in metadata]
            if keys:
                assert arch == "sm_103a" and len(keys) == 2, (arch, key, keys)
                stated[key] = (metadata["w2_dense_k"], metadata["w2_logical_k"])
    assert stated == expected, sorted(set(stated) ^ set(expected))


def test_tp_ep_programs_launch_as_programmatic_dependents():
    """Every TP/EP program launches with programmatic stream serialization
    beside the cluster shape and the cooperative attribute, one counted slot
    each in a three-slot array with the programmatic slot last; its kernel
    runs the programmatic dependent launch protocol: one
    ``griddepcontrol.launch_dependents`` on entry, before the tensor-map
    prefetches, and one ``griddepcontrol.wait`` at the entry of each of the six
    role bodies, after the role's ``setmaxnreg`` and before the role's first
    global access (the dispatch role's ``topk_idx_i32[`` is the kernel's
    first), so the setup block overlaps the predecessor's tail while every
    warp still passes a wait before its first global read or write. Up to its
    wait no role names a pointer parameter: a dependent so launched touches
    nothing its predecessor writes, pool rows, counters, gates or the output,
    before the predecessor has exited and flushed."""
    for name, (kernel, binding) in tp_ep_program_sources().items():
        text = binding.read_text()
        assert text.count("cudaLaunchAttribute attrs[") == 1, name
        assert text.count(LAUNCH_ATTRIBUTES) == 1, name
        assert text.count("cudaLaunchAttributeCooperative") == 1, name
        assert text.count(COOPERATIVE_SLOT) == 1, name
        assert text.count(PDL_ATTRIBUTE) == 1, name
        assert text.count(PDL_SLOT) == 1, name
        assert (
            text.index("cudaLaunchAttributeClusterDimension")
            < text.index(COOPERATIVE_SLOT)
            < text.index(PDL_SLOT)
            < text.index("config.attrs = attrs;")
        ), name
        text = kernel.read_text()
        body = text.index(f"kernel_{name}(")
        brace = text.index("{", body)
        assert text.count("griddepcontrol") == 1 + len(ROLES), name
        assert text.count(GRID_DEP_LAUNCH) == 1, name
        assert text.count(GRID_DEP_WAIT) == len(ROLES), name
        launch_dependents = text.index(GRID_DEP_LAUNCH)
        assert brace < launch_dependents < text.index("prefetch.tensormap", body), name
        pointers = re.findall(r"\*\s*__restrict__\s+(\w+)", text[body:brace])
        assert pointers, name
        markers = list(ROLE_MARKER.finditer(text, body))
        assert sorted(marker.group(1) for marker in markers) == sorted(ROLES), name
        assert all(launch_dependents < marker.start() for marker in markers), name
        for marker, following in zip(markers, [*markers[1:], None], strict=True):
            role = text[marker.end() : following.start() if following else len(text)]
            assert role.count(GRID_DEP_WAIT) == 1, (name, marker.group(1))
            wait = role.index(GRID_DEP_WAIT)
            assert 0 <= role.index(SETMAXNREG) < wait, (name, marker.group(1))
            before_wait = role[:wait]
            touched = [
                pointer
                for pointer in pointers
                if re.search(rf"(?<![.\w]){pointer}\b", before_wait)
            ]
            assert not touched, (name, marker.group(1), touched)
        assert text.index("topk_idx_i32[", body) > text.index(
            GRID_DEP_WAIT, markers[0].start()
        ), name


def test_model_programs_keep_their_launch_form():
    """The programs no TP/EP route uses, the model routes of both
    architectures and the grouped routes, are carried from the base commit and
    keep its launch form: no programmatic stream serialization attribute in
    the binding and no ``griddepcontrol`` instruction in the kernel."""
    records = json.loads(_catalog().path.read_text())["programs"]
    tp_ep = tp_ep_program_sources()
    assert len(tp_ep) == len(TP_EP_SHAPES) * len(TP_EP_TOKENS)
    others = [name for name in records if name not in tp_ep]
    assert others
    for name in others:
        kernel, binding = (ROOT / src for src in records[name]["sources"])
        assert "griddepcontrol" not in kernel.read_text(), name
        assert PDL_ATTRIBUTE not in binding.read_text(), name


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
            num_ctas = runtime.route_num_ctas(route, arch, num_sms)
            assert 0 < num_ctas <= num_sms, (key, arch, num_sms)
            if surface != "grouped_l1":
                # Persistent schedules run one CTA per physical SM, or the
                # smaller persistent grid the route states.
                assert grid == (num_ctas, 1, 1), (key, arch, num_sms)
            if "num_ctas" not in route["metadata"]:
                assert num_ctas == num_sms, (key, arch, num_sms)
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


def test_only_the_one_token_rank_routes_state_a_persistent_grid():
    """Only sm_103a one-token rank routes state ``num_ctas``: exactly those of
    ``PERSISTENT_GRID``, each at 148 CTAs with the grid ``[148, 1, 1]`` and the
    152-SM device it admits. Their programs take ``NUM_CTAS`` from the compile
    line, so the shipped texts run at the stated count."""
    catalog = _catalog()
    stated = [
        (arch, key)
        for arch in catalog.arches
        for key, route in catalog.routes(arch).items()
        if "num_ctas" in route["metadata"]
    ]
    keys = [tp_ep_route_key(*shape) for shape in PERSISTENT_GRID]
    assert sorted(stated) == sorted(("sm_103a", key) for key in keys)
    for key in keys:
        route = catalog.routes("sm_103a")[key]
        assert route["metadata"]["num_ctas"] == 148, key
        assert route["grid"] == {"sm_103a": [148, 1, 1]}, key
        assert runtime.route_admitted_num_sms(route, "sm_103a") == (152,), key
        assert runtime.route_num_ctas(route, "sm_103a", 152) == 148, key
        (stage,) = route["stages"]
        name = runtime.stage_program(stage, "sm_103a")
        assert name in T1_PROGRAMS, key
        assert catalog.programs("sm_103a")[name]["definitions"] == ["NUM_CTAS"]


def _route(grid=None, num_ctas=None, arch="sm_103a"):
    """A pipeline route at 152 SMs with the given stated grid and ``num_ctas``."""
    route = {"num_sms": {arch: 152}, "metadata": {"surface": "pipeline"}}
    if grid is not None:
        route["grid"] = {arch: list(grid)}
    if num_ctas is not None:
        route["metadata"]["num_ctas"] = num_ctas
    return route


def test_route_grid_and_num_ctas_with_and_without_a_stated_grid():
    """A route without a stated grid or ``num_ctas`` runs one CTA per SM; a
    stated grid is the route's launch grid; a stated ``num_ctas`` is the
    persistent grid and must be the stated grid and fit the device."""
    bare = _route()
    assert runtime.route_grid(bare, "sm_103a", 152) == (152, 1, 1)
    assert runtime.route_num_ctas(bare, "sm_103a", 152) == 152
    full = _route(grid=(152, 1, 1))
    assert runtime.route_grid(full, "sm_103a", 152) == (152, 1, 1)
    assert runtime.route_num_ctas(full, "sm_103a", 152) == 152
    smaller = _route(grid=(136, 1, 1), num_ctas=136)
    assert runtime.route_grid(smaller, "sm_103a", 152) == (136, 1, 1)
    assert runtime.route_num_ctas(smaller, "sm_103a", 152) == 136
    for bad in (
        _route(num_ctas=136),
        _route(grid=(152, 1, 1), num_ctas=136),
        _route(grid=(160, 1, 1), num_ctas=160),
        _route(grid=(0, 1, 1), num_ctas=0),
        _route(grid=(136, 2, 1), num_ctas=136),
    ):
        with pytest.raises(ValueError, match="must be the route's grid and fit"):
            runtime.route_num_ctas(bad, "sm_103a", 152)


def test_load_program_defines_num_ctas_as_the_routes_persistent_grid(monkeypatch):
    """``load_program`` passes the route's persistent grid as ``NUM_CTAS``:
    148 for the one-token rank routes of ``PERSISTENT_GRID``, the SM count for
    every other sm_103a pipeline route; the JIT spec of that program names the
    count."""
    pytest.importorskip("torch")
    catalog = _catalog()
    loaded = []

    def load(self, arch, name, definitions):
        loaded.append((arch, name, definitions))
        return None, None

    monkeypatch.setattr(type(catalog), "load_program", load)
    seen = set()
    for key, route in catalog.routes("sm_103a").items():
        if route["metadata"]["surface"] != "pipeline":
            continue
        num_ctas = runtime.route_num_ctas(route, "sm_103a", 152)
        (stage,) = route["stages"]
        name = runtime.stage_program(stage, "sm_103a")
        runtime.load_program("sm_103a", name, num_ctas)
        definitions = catalog.definitions("sm_103a", name, {"NUM_CTAS": num_ctas})
        assert loaded[-1] == ("sm_103a", name, definitions), key
        expected = 148 if name in T1_PROGRAMS else 152
        assert num_ctas == expected, key
        spec = catalog.jit_spec("sm_103a", name, definitions)
        if catalog.programs("sm_103a")[name].get("definitions"):
            assert f"-DNUM_CTAS={expected}" in spec.extra_cuda_cflags, key
            assert spec.name == f"{jit_spec_name(name, 'sm_103a')}_num_ctas{expected}"
        seen.add(name)
    assert seen >= T1_PROGRAMS
