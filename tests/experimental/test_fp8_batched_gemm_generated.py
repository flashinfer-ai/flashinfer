"""Projection routes: exact scale packing, changed inputs, stream replay, token coverage.

The package is named once, through ``tests.test_helpers.cake_package``; the public
entry ``flashinfer.fp8_batched_gemm.prepare_fp8_batched_gemm`` is the second import,
the route it exercises being the package's own.  The host plan of the registry
records and the ``build.json`` contract are checked without a GPU.
"""

import re
from pathlib import Path

import pytest
import torch

from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("deepgemm_batched_gemm")
_runtime = cake_package("deepgemm_batched_gemm", "batched_gemm")
cake_jit = cake_package("deepgemm_batched_gemm", "cake_jit")
cake_build = cake_package("deepgemm_batched_gemm", "cake_build")

# The kernels' parameters in ``run`` order: the host vocabulary of every record.
PARAMETERS = (
    "A",
    "B",
    "SFA",
    "SFB",
    "D",
    "SFD",
    "M",
    "grid_m",
    "sfd_stride",
    "alpha",
)
_KEY = re.compile(
    r"(bf16_t128|bf16_t4|general|n256|swap_ab)_bm(\d+)_bn(\d+)_s(\d+)_(fp8|bf16|alpha)"
)

# The ten originally exported routes.
_CASES = [(t, True, None) for t in (1, 4, 16, 128, 512, 4096)]
_CASES += [(t, False, alpha) for t in (4, 128) for alpha in (None, 0.5)]
# Token counts outside the original route table that select an exported
# schedule (swap_ab BM16 with one and two M tiles, swap_ab BM64, n256 with
# four and eight M tiles, the general alpha schedule).
_CASES += [(t, True, None) for t in (2, 8, 17, 32, 100, 127, 500, 1000)]
_CASES += [(t, False, 0.5) for t in (1, 17, 1000)]
# Token counts whose selected schedule is not exported.
_UNEXPORTED = [(64, True, None), (8, False, None), (1000, False, None)]


def _exported_arch():
    """The device's exported architecture, or the reason the GPU tests skip."""
    if not torch.cuda.is_available():
        return None, "CUDA device required"
    try:
        arch = _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        return None, str(error)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms not in _runtime.supported_num_sms(arch):
        return None, (
            f"The exported {arch} schedules were selected on {_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )
    return arch, None


def _skip_unless_exported():
    _, reason = _exported_arch()
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the device's libraries when they have no prebuilt file before the
    # first test, through the spec objects the package's loader itself builds on
    # first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    arch, _ = _exported_arch()
    if arch is not None:
        for spec in cake_jit.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _operands(tokens, heads=8, inner=4096, width=1024):
    aq = torch.ones((tokens, heads, inner), dtype=torch.float8_e4m3fn, device="cuda")
    bq = torch.full(
        (heads, width, inner), 1 / 64, dtype=torch.float8_e4m3fn, device="cuda"
    )
    asf = torch.ones((tokens, heads, inner // 128), dtype=torch.float32, device="cuda")
    bsf = torch.ones(
        (heads, width // 128, inner // 128), dtype=torch.float32, device="cuda"
    )
    return aq, bq, asf, bsf


@pytest.mark.parametrize("tokens,fp8,alpha", _CASES)
def test_projection_values_scales_current_stream_graph(tokens, fp8, alpha):
    _skip_unless_exported()
    inner = 4096
    aq, bq, asf, bsf = _operands(tokens)
    plan = prepare_fp8_batched_gemm((aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha)
    expected_route = _runtime.route_config(
        tokens,
        8,
        inner,
        1024,
        torch.cuda.get_device_properties(0).multi_processor_count,
        "fp8" if fp8 else "alpha" if alpha is not None else "bf16",
    )
    assert plan.config == expected_route
    assert plan.program == cake_jit.ROUTES[_runtime.schedule_key(expected_route)]
    assert (
        plan.bindings["grid_x"],
        plan.bindings["grid_y"],
        plan.bindings["grid_z"],
    ) == tuple(expected_route["grid"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for sign, bvalue, exponent in (
        (1, 1 / 64, 125),
        (-1, 1 / 32, 126),
        (1, 1 / 64, 125),
    ):
        with torch.cuda.stream(stream):
            aq.fill_(sign)
            bq.fill_(bvalue)
            plan.values.view(torch.uint8).fill_(255)
            if fp8:
                plan.scales.fill_(-1234567)
            graph.replay()
        stream.synchronize()
        if fp8:
            expected = torch.full_like(plan.values, sign * 256)
            torch.testing.assert_close(
                plan.values.view(torch.uint8),
                expected.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            word = exponent * 0x01010101
            torch.testing.assert_close(
                plan.scales, torch.full_like(plan.scales, word), atol=0, rtol=0
            )
        else:
            expected = sign * inner * bvalue * (1 if alpha is None else alpha)
            torch.testing.assert_close(
                plan.values,
                torch.full_like(plan.values, expected),
                atol=0.01,
                rtol=0.01,
            )
    # The alpha value is a launch argument, independent of its compiled route.
    if alpha is not None:
        other = prepare_fp8_batched_gemm(
            (aq, asf), (bq, bsf), output_fp8=False, alpha=-0.25
        )
        other.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            other.values, torch.full_like(other.values, -16), atol=0.01, rtol=0.01
        )


@pytest.mark.parametrize("tokens,fp8,alpha", _UNEXPORTED)
def test_unexported_schedule_raises_at_preparation(tokens, fp8, alpha):
    _skip_unless_exported()
    aq, bq, asf, bsf = _operands(tokens)
    with pytest.raises(
        NotImplementedError, match="general schedule .* is not exported"
    ):
        prepare_fp8_batched_gemm((aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha)


def test_route_table_is_a_function_of_tokens_sms_and_epilogue():
    """Every original route key maps onto the program its token count selects."""
    routes = {
        (1, "fp8"): "swap_ab_bm16_bn128_s12_fp8",
        (4, "fp8"): "swap_ab_bm16_bn128_s12_fp8",
        (16, "fp8"): "swap_ab_bm16_bn128_s12_fp8",
        (128, "fp8"): "swap_ab_bm64_bn128_s10_fp8",
        (512, "fp8"): "n256_bm128_bn256_s5_fp8",
        (4096, "fp8"): "n256_bm128_bn256_s5_fp8",
        (4, "bf16"): "bf16_t4_bm16_bn128_s12_bf16",
        (128, "bf16"): "bf16_t128_bm64_bn128_s10_bf16",
        (4, "alpha"): "general_bm128_bn128_s5_alpha",
        (128, "alpha"): "general_bm128_bn128_s5_alpha",
    }
    for sms in _runtime.PINNED_NUM_SMS:
        for (tokens, epilogue), key in routes.items():
            config = _runtime.route_config(tokens, 8, 4096, 1024, sms, epilogue)
            assert _runtime.schedule_key(config) == key
            assert key in cake_jit.ROUTES
    assert set(cake_jit.ROUTES) == set(routes.values())
    assert set(cake_jit.ROUTES.values()) == set(cake_jit.MODULES)
    assert len(cake_jit.ROUTES) == len(cake_jit.MODULES) == 6


# --- The host plan and the build manifest --------------------------------------


def _schedule(key):
    match = _KEY.fullmatch(key)
    assert match, key
    route, block_m, block_n, stages, epilogue = match.groups()
    return route, int(block_m), int(block_n), int(stages), epilogue


def test_host_plan_of_the_records():
    assert cake_jit.ARCHES == ("sm_100a", "sm_103a")
    assert cake_jit.ARCH_BY_CAPABILITY == {(10, 0): "sm_100a", (10, 3): "sm_103a"}
    assert cake_jit.arch_for_capability((10, 0)) == "sm_100a"
    assert cake_jit.arch_for_capability((10, 3)) == "sm_103a"
    assert cake_jit.arch_for_capability((9, 0)) is None
    assert cake_jit.arch_for_capability((10,)) is None
    assert set(cake_jit.ROUTES.values()) == set(cake_jit.MODULES)
    for key, name in cake_jit.ROUTES.items():
        record = cake_jit.MODULES[name]
        route, block_m, block_n, _stages, _epilogue = _schedule(key)
        rule = record["grid_rule"]
        if route == "swap_ab":
            assert rule == {
                "block_m": block_m,
                "block_n": block_n,
                "kind": cake_jit.WORKING_CTAS_EVEN,
                "num_heads": 8,
                "width": 1024,
            }
        else:
            assert rule == {"block_m": block_m, "kind": cake_jit.ONE_CTA_PER_SM}
        assert cake_jit.library_name(name, "sm_103a") == f"{name}_sm_103a"
        with pytest.raises(RuntimeError, match="not delivered for sm_90a"):
            cake_jit.library_name(name, "sm_90a")
        # The grid rule, evaluated as the dispatcher's own arithmetic spells it.
        for sms in _runtime.PINNED_NUM_SMS:
            for tokens in (1, 2, 4, 8, 16, 17, 32, 37, 100, 127, 128, 300, 1000, 4096):
                if route == "swap_ab":
                    tiles = -(-tokens // block_m) * (1024 // block_n) * 8
                    expected = (min(sms, tiles + tiles % 2), 1, 1)
                else:
                    expected = (sms, 1, 1)
                assert (
                    cake_jit.launch_grid(record, tokens=tokens, sm_count=sms)
                    == expected
                )
    # Every route the dispatcher selects launches the grid route_config reports.
    for sms in _runtime.PINNED_NUM_SMS:
        for tokens in (*range(1, 1025), 4096):
            for epilogue in ("fp8", "bf16", "alpha"):
                config = _runtime.route_config(tokens, 8, 4096, 1024, sms, epilogue)
                name = cake_jit.ROUTES.get(_runtime.schedule_key(config))
                if name is None:
                    continue
                record = cake_jit.MODULES[name]
                assert record["grid_rule"]["block_m"] == config["block_m"]
                assert cake_jit.launch_grid(
                    record, tokens=tokens, sm_count=sms
                ) == tuple(config["grid"])
    record = cake_jit.MODULES[cake_jit.ROUTES["swap_ab_bm16_bn128_s12_fp8"]]
    assert cake_jit.launch_grid(record, tokens=4, sm_count=148) == (64, 1, 1)
    assert cake_jit.launch_grid(record, tokens=17, sm_count=148) == (128, 1, 1)
    assert cake_jit.launch_grid(record, tokens=37, sm_count=148) == (148, 1, 1)
    with pytest.raises(
        ValueError, match="unknown Cake batched FP8 projection grid rule"
    ):
        cake_jit.launch_grid({"grid_rule": {"kind": "linear"}}, tokens=4, sm_count=148)
    with pytest.raises(ValueError, match="positive tokens and SMs"):
        cake_jit.launch_grid(record, tokens=0, sm_count=148)
    with pytest.raises(ValueError, match="positive tokens and SMs"):
        cake_jit.launch_grid(record, tokens=4, sm_count=0)
    # The argument plan: the bindings by logical name, then the grid axes.
    values = {key: object() for kind, key in record["arg_plan"] if kind != "grid"}
    args = cake_jit.launch_args(record, values, (64, 1, 1))
    assert args[-3:] == (64, 1, 1)
    assert args[:-3] == tuple(values[key] for key in PARAMETERS)
    with pytest.raises(ValueError, match="needs logical argument 'alpha'"):
        cake_jit.launch_args(
            record, {k: v for k, v in values.items() if k != "alpha"}, (64, 1, 1)
        )
    with pytest.raises(ValueError, match="has no logical name"):
        cake_jit.launch_args({**record, "logical": {}}, values, (64, 1, 1))


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests, a closure digest and
    an empty nvcc flag list (the programs compile with nvcc's default numerics);
    the registry itself records the host plan only.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(cake_jit.ARCHES) == {"sm_100a", "sm_103a"}
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    build_keys = {
        "deepgemm_fp8_batched_gemm": {
            "epilogue",
            "inner",
            "num_heads",
            "num_stages",
            "width",
        },
        "deepgemm_fp8_batched_gemm_swap_ab": {
            "block_m",
            "inner",
            "num_heads",
            "num_stages",
            "width",
        },
        "deepgemm_fp8_batched_gemm_n256": set(),
        "deepgemm_fp8_batched_gemm_bf16": set(),
        "deepgemm_fp8_batched_gemm_bf16_t4": set(),
    }
    for name, record in cake_jit.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert re.fullmatch(r"cake_deepgemm_batched_gemm_[0-9a-f]{20}", name)
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert list(record["arches"]) == list(cake_jit.ARCHES), name
        module_name = record["module"].rsplit(".", 1)[-1]
        assert set(record["build"]) == build_keys[module_name], name
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False
        assert tuple(launch["block"]) == (256, 1, 1)
        cluster = (1, 1, 1) if module_name == "deepgemm_fp8_batched_gemm" else (2, 1, 1)
        assert tuple(launch["cluster"]) == cluster, name
        assert launch["dynamic_smem_bytes"] > 0
        assert record["grid_rule"]["kind"] in (
            cake_jit.ONE_CTA_PER_SM,
            cake_jit.WORKING_CTAS_EVEN,
        )
        assert record["workspace_rule"] is None
        parameters = [key for kind, key in record["arg_plan"] if kind != "grid"]
        assert tuple(parameters) == PARAMETERS
        assert [kind for kind, _ in record["arg_plan"]] == ["buffer"] * 6 + [
            "parameter"
        ] * 4 + ["grid"] * 3
        assert record["logical"] == {key: key for key in parameters}
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        for arch in record["arches"]:
            library = cake_jit.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert entry["nvcc_flags"] == [], library
            assert re.fullmatch(r"\d+\.\d+", entry["min_cuda"]), library
    assert referenced == set(libraries)
    assert len(libraries) == 12
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
    # The DeepGEMM derivation notice travels at the package root, where every
    # generated source's license comment points, and ships with the package.
    assert (package / "DEEPGEMM_NOTICE.txt").is_file()
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert (
        f'"{pkg.__name__}" = ["csrc/**", "build.json", "DEEPGEMM_NOTICE.txt"]'
        in pyproject
    )
    for source in package.glob("csrc/*_kernel.cu"):
        head = source.read_text(encoding="utf-8").split("*/", 1)[0]
        assert "by FlashInfer team" in head and "DEEPGEMM_NOTICE.txt" in head, source
