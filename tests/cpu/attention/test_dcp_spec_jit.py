"""CPU-side validation for DCP speculative FMHA JIT specialization keys."""

import importlib
import inspect
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_dcp as cake_dcp

from flashinfer.cake_dcp import (
    _select_fp8_num_split,
    _select_num_split,
    get_dcp_spec_counter_bytes,
    get_dcp_spec_workspace_size_bytes,
    run_dcp_spec_decode,
)
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.jit.cake_dcp import (
    DCP_STATIC_FAMILIES,
    get_cake_fmha_csrc_dir,
    get_dcp_spec_registry,
    get_dcp_spec_static_uri,
)
from flashinfer.trace.templates.attention import (
    trtllm_batch_decode_dcp_spec_split_kv_trace,
    trtllm_batch_decode_dcp_spec_trace,
    trtllm_batch_decode_trace_dispatch,
)


@pytest.mark.cpu_only
def test_public_decode_api_adds_optional_dcp_arguments() -> None:
    parameters = inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
    assert parameters["cp_world"].default == 1
    assert parameters["cp_rank"].default == 0
    assert parameters["causal_seqlens_kv_global"].default is None
    assert list(parameters).index("bf16q_fp8kv_transform_mode") > list(
        parameters
    ).index("causal_seqlens_kv_global")


@pytest.mark.cpu_only
def test_public_decode_api_rejects_pdl_for_dcp() -> None:
    with pytest.raises(ValueError, match="does not support enable_pdl=True"):
        trtllm_batch_decode_with_kv_cache(
            query=torch.empty((1, 8, 128), dtype=torch.bfloat16),
            kv_cache=(
                torch.empty((1, 8, 16, 128), dtype=torch.bfloat16),
                torch.empty((1, 8, 16, 128), dtype=torch.bfloat16),
            ),
            workspace_buffer=torch.empty(1, dtype=torch.uint8),
            block_tables=torch.zeros((1, 1), dtype=torch.int32),
            seq_lens=torch.zeros((1,), dtype=torch.int32),
            max_seq_len=0,
            enable_pdl=True,
            causal_seqlens_kv_global=torch.zeros((1,), dtype=torch.int32),
        )


@pytest.mark.cpu_only
def test_dcp_workspace_and_counter_sizes_are_caller_owned_exact_views() -> None:
    # B=8, Q=4, Hq=64, split=6: BF16 O[...128] + FP32 LSE per row.
    rows = 8 * 4 * 64 * 6
    assert get_dcp_spec_workspace_size_bytes(8, 4, 64, 6) == rows * (128 * 2 + 4)
    assert get_dcp_spec_counter_bytes(8, 4, 8) == 8 * 4 * 8 * 4
    assert (
        cake_dcp.get_dcp_spec_workspace_size_bytes is get_dcp_spec_workspace_size_bytes
    )
    assert cake_dcp.get_dcp_spec_counter_bytes is get_dcp_spec_counter_bytes


@pytest.mark.cpu_only
def test_dcp_decode_rejects_an_empty_batch_before_split_selection() -> None:
    with pytest.raises(ValueError, match="non-empty batch"):
        run_dcp_spec_decode(
            query=torch.empty((0, 8, 128), dtype=torch.bfloat16),
            k_cache=torch.empty((1, 8, 16, 128), dtype=torch.bfloat16),
            v_cache=torch.empty((1, 8, 16, 128), dtype=torch.bfloat16),
            workspace_buffer=torch.empty(1, dtype=torch.uint8),
            block_tables=torch.empty((0, 1), dtype=torch.int32),
            seq_lens=torch.empty((0,), dtype=torch.int32),
            causal_seqlens_kv_global=torch.empty((0,), dtype=torch.int32),
            max_local_seq_len=1,
            bmm1_scale=1.0,
            bmm2_scale=1.0,
            cp_world=1,
            cp_rank=0,
            q_len_per_req=1,
            out=torch.empty((0, 8, 128), dtype=torch.bfloat16),
            lse=torch.empty((0, 8), dtype=torch.float32),
            completion_buffer=None,
        )


@pytest.mark.cpu_only
def test_dcp_split_selector_matches_promoted_policy() -> None:
    assert _select_num_split(logical_tiles=32, sm_count=148, local_blocks=9) == 1
    assert _select_num_split(logical_tiles=32, sm_count=148, local_blocks=32) == 4
    assert _select_num_split(logical_tiles=8, sm_count=148, local_blocks=128) == 16
    assert _select_num_split(logical_tiles=64, sm_count=148, local_blocks=128) == 2
    assert (
        _select_fp8_num_split(
            logical_tiles=32, sm_count=148, local_blocks=64, cp_world=1
        )
        == 4
    )
    assert (
        _select_fp8_num_split(
            logical_tiles=32, sm_count=148, local_blocks=64, cp_world=4
        )
        == 3
    )
    assert (
        _select_fp8_num_split(
            logical_tiles=148, sm_count=148, local_blocks=64, cp_world=4
        )
        == 1
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("capability", "target"),
    [
        ((10, 0), "sm100a"),
        ((10, 3), "sm103a"),
        ((10, 7), "sm100f"),
    ],
)
def test_dcp_target_keeps_independent_architecture_baselines(
    monkeypatch, capability, target
) -> None:
    dcp = importlib.import_module("flashinfer.cake_dcp")
    monkeypatch.setattr(dcp, "get_compute_capability", lambda _device: capability)
    monkeypatch.setattr(dcp, "_is_cuda_version_at_least", lambda _version: True)
    assert dcp._select_target(None) == target


@pytest.mark.cpu_only
def test_dcp_trace_dispatch_distinguishes_combined_and_split_kv() -> None:
    marker = object()
    assert (
        trtllm_batch_decode_trace_dispatch(
            causal_seqlens_kv_global=marker, kv_cache=marker
        )
        is trtllm_batch_decode_dcp_spec_trace
    )
    assert (
        trtllm_batch_decode_trace_dispatch(
            causal_seqlens_kv_global=marker, kv_cache=(marker, marker)
        )
        is trtllm_batch_decode_dcp_spec_split_kv_trace
    )


def _empty_rank_inputs(
    seq_lens_dtype=torch.int32,
    *,
    kv_dtype=torch.bfloat16,
    q_len_per_req=1,
    bmm2_scale=1.0,
):
    page_size = 64 if kv_dtype == torch.float8_e4m3fn else 16
    return {
        "query": torch.empty((q_len_per_req, 8, 128), dtype=torch.bfloat16),
        "k_cache": torch.empty((1, 8, page_size, 128), dtype=kv_dtype),
        "v_cache": torch.empty((1, 8, page_size, 128), dtype=kv_dtype),
        "workspace_buffer": torch.empty(1, dtype=torch.uint8),
        "block_tables": torch.zeros((1, 1), dtype=torch.int32),
        "seq_lens": torch.zeros((1,), dtype=seq_lens_dtype),
        "causal_seqlens_kv_global": torch.zeros((1,), dtype=torch.int32),
        "max_local_seq_len": 0,
        "bmm1_scale": 128**-0.5,
        "bmm2_scale": bmm2_scale,
        "cp_world": 8,
        "cp_rank": 7,
        "q_len_per_req": q_len_per_req,
        "out": torch.empty((q_len_per_req, 8, 128), dtype=torch.bfloat16),
        "lse": torch.empty((q_len_per_req, 8), dtype=torch.float32),
        "completion_buffer": None,
    }


@pytest.mark.cpu_only
def test_dcp_all_empty_rank_reaches_native_v1_route(monkeypatch) -> None:
    dcp = importlib.import_module("flashinfer.cake_dcp")
    launches = []
    module = SimpleNamespace(run=lambda *args: launches.append(args))
    jit_dcp = importlib.import_module("flashinfer.jit.cake_dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    monkeypatch.setattr(jit_dcp, "load_dcp_spec_static_module", lambda *args: module)

    run_dcp_spec_decode(**_empty_rank_inputs())

    assert len(launches) == 1


@pytest.mark.cpu_only
def test_fp8_page64_q3_reaches_single_native_launch_with_fused_scales(
    monkeypatch,
) -> None:
    dcp = importlib.import_module("flashinfer.cake_dcp")
    launches = []
    module = SimpleNamespace(run=lambda *args: launches.append(args))
    jit_dcp = importlib.import_module("flashinfer.jit.cake_dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    monkeypatch.setattr(jit_dcp, "load_dcp_spec_static_module", lambda *args: module)

    inputs = _empty_rank_inputs(
        kv_dtype=torch.float8_e4m3fn,
        q_len_per_req=3,
        bmm2_scale=0.25,
    )
    inputs["bmm1_scale"] = 0.125
    run_dcp_spec_decode(**inputs)

    assert len(launches) == 1
    args = launches[0]
    assert args[1].dtype == torch.uint8
    assert args[2].dtype == torch.uint8
    assert args[13] == pytest.approx(0.125 / math.log(2.0))
    assert args[14] == pytest.approx(0.25)


@pytest.mark.cpu_only
def test_fp8_page64_underfill_uses_split3_and_caller_owned_scratch(
    monkeypatch,
) -> None:
    dcp = importlib.import_module("flashinfer.cake_dcp")
    launches = []
    loader_calls = []
    module = SimpleNamespace(run=lambda *args: launches.append(args))
    jit_dcp = importlib.import_module("flashinfer.jit.cake_dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    monkeypatch.setattr(
        jit_dcp,
        "load_dcp_spec_static_module",
        lambda *args: loader_calls.append(args) or module,
    )

    inputs = _empty_rank_inputs(
        kv_dtype=torch.float8_e4m3fn,
        q_len_per_req=4,
        bmm2_scale=0.25,
    )
    inputs["block_tables"] = torch.zeros((1, 128), dtype=torch.int32)
    inputs["seq_lens"] = torch.full((1,), 8192, dtype=torch.int32)
    inputs["max_local_seq_len"] = 8192
    inputs["workspace_buffer"] = torch.empty(
        get_dcp_spec_workspace_size_bytes(1, 4, 8, 3), dtype=torch.uint8
    )
    inputs["completion_buffer"] = torch.zeros(
        get_dcp_spec_counter_bytes(1, 4, 8), dtype=torch.uint8
    )
    run_dcp_spec_decode(**inputs)

    assert len(loader_calls) == 1
    assert loader_calls[0][:2] == ("fp8_d128", "splitn_retain0")
    assert loader_calls[0][3]["NUM_SPLIT"] == 3
    assert len(launches) == 1
    args = launches[0]
    assert args[3].data_ptr() == inputs["workspace_buffer"].data_ptr()
    assert args[7].data_ptr() == inputs["completion_buffer"].data_ptr()


@pytest.mark.cpu_only
def test_bf16_page16_rejects_nonunit_bmm2_scale(monkeypatch) -> None:
    dcp = importlib.import_module("flashinfer.cake_dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    with pytest.raises(ValueError, match="BF16/page16"):
        run_dcp_spec_decode(**_empty_rank_inputs(bmm2_scale=0.5))


@pytest.mark.cpu_only
def test_bf16_page16_q3_reaches_native_v1_route(monkeypatch) -> None:
    dcp = importlib.import_module("flashinfer.cake_dcp")
    launches = []
    module = SimpleNamespace(run=lambda *args: launches.append(args))
    jit_dcp = importlib.import_module("flashinfer.jit.cake_dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    monkeypatch.setattr(jit_dcp, "load_dcp_spec_static_module", lambda *args: module)

    run_dcp_spec_decode(**_empty_rank_inputs(q_len_per_req=3))

    assert len(launches) == 1


@pytest.mark.cpu_only
def test_dcp_rejects_non_int32_local_seq_lens() -> None:
    with pytest.raises(ValueError, match="contiguous int32"):
        run_dcp_spec_decode(**_empty_rank_inputs(torch.int64))


@pytest.mark.cpu_only
def test_dcp_static_registry_lists_each_program_once() -> None:
    registry = get_dcp_spec_registry()
    programs = registry["programs"]
    routes = registry["static_routes"]
    assert set(routes) == set(DCP_STATIC_FAMILIES)
    bound = set()
    for instances in routes.values():
        for by_arch in instances.values():
            assert sorted(by_arch) == ["sm_100a", "sm_103a"]
            for arch, name in by_arch.items():
                assert arch in programs[name]["arches"], (name, arch)
                bound.add(name)
    assert sorted(bound) == sorted(programs)
    for name, program in programs.items():
        assert program["arches"] and set(program["arches"]) <= {"sm_100a", "sm_103a"}
        assert len(program["sources"]) == 2
        assert {"Q_LEN", "CP_WORLD", "NUM_Q_HEADS", "NUM_KV_HEADS"} <= set(
            program["specializations"]
        )
        assert ("NUM_SPLIT" in program["specializations"]) == (
            program["family"] != "bf16_v1"
        )
        for source in program["sources"]:
            assert (Path(get_cake_fmha_csrc_dir()) / source).is_file(), (name, source)


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a", "sm100f"])
def test_dcp_static_uri_names_the_program_target_and_constants(target) -> None:
    constants = {
        "Q_LEN": 4,
        "CP_WORLD": 4,
        "NUM_Q_HEADS": 64,
        "NUM_KV_HEADS": 8,
        "NUM_SPLIT": 4,
    }
    uri = get_dcp_spec_static_uri("bf16_v4", "splitn", target, constants)
    assert uri.startswith(f"cake_fmha_dcp_spec_bf16_v4_splitn_{target}_")
    assert uri.endswith("_CP_WORLD4_NUM_KV_HEADS8_NUM_Q_HEADS64_NUM_SPLIT4_Q_LEN4")
    assert uri == get_dcp_spec_static_uri("bf16_v4", "splitn", target, constants)
    assert uri != get_dcp_spec_static_uri(
        "bf16_v4", "splitn", target, {**constants, "NUM_SPLIT": 8}
    )
    with pytest.raises(ValueError):
        get_dcp_spec_static_uri("bf16_v4", "split7", target, constants)
    with pytest.raises(ValueError):  # a missing or foreign constant
        get_dcp_spec_static_uri(
            "bf16_v4",
            "splitn",
            target,
            {k: v for k, v in constants.items() if k != "NUM_SPLIT"},
        )
    with pytest.raises(ValueError):
        get_dcp_spec_static_uri("bf16_v1", "retain0", target, constants)


# ---------------------------------------------------------------------------
# Folded static program sources
# ---------------------------------------------------------------------------
#
# The programs of a static family compile one kernel source
# (``cuda/dcp_spec/<family>_kernel.cu``; fp8_d256 one per architecture) and one
# binding source (``cuda/dcp_spec/<family>_binding.cu``).  The
# ``-DCAKE_FMHA_DCP_*`` values in a program's ``compile_flags`` select its text:
# the retain flag, the split class (1, 2, or 0 for the NUM_SPLIT-generic text)
# and, for the per-architecture fp8_d256 binding, the architecture.

_DCP_STATIC_SPLIT_CLASS = {"split1": 1, "split2": 2, "splitn": 0}
_DCP_STATIC_NUM_SPLIT = {"split1": 1, "split2": 2, "splitn": 4}


def _dcp_static_program_knobs(program) -> dict[str, int]:
    """The compile-line knobs a static program's family, instance and arches imply."""

    family, instance = program["family"], program["instance"]
    knobs = {}
    if family in ("bf16_v1", "fp8_d128"):
        knobs["CAKE_FMHA_DCP_RETAIN_KV_L2"] = int(instance.rsplit("retain", 1)[1])
    if family != "bf16_v1":
        knobs["CAKE_FMHA_DCP_SPLIT_CLASS"] = _DCP_STATIC_SPLIT_CLASS[
            instance.split("_")[0]
        ]
    if family == "fp8_d256":
        (arch,) = program["arches"]
        knobs["CAKE_FMHA_DCP_ARCH"] = int(arch[3:-1])  # "sm_100a" -> 100
    return knobs


def _dcp_static_sources(program) -> list[str]:
    family = program["family"]
    kernel = f"cuda/dcp_spec/{family}_kernel.cu"
    if family == "fp8_d256":
        (arch,) = program["arches"]
        kernel = f"cuda/dcp_spec/{family}_{arch}_kernel.cu"
    return [kernel, f"cuda/dcp_spec/{family}_binding.cu"]


@pytest.mark.cpu_only
def test_dcp_static_programs_share_one_folded_source_per_family() -> None:
    """Each program compiles its family's folded sources; its flags pin the text."""

    programs = get_dcp_spec_registry()["programs"]
    csrc_dir = Path(get_cake_fmha_csrc_dir())
    texts = {}
    for name, program in programs.items():
        assert program["sources"] == _dcp_static_sources(program), name
        knobs = _dcp_static_program_knobs(program)
        assert program["compile_flags"] == [
            "--use_fast_math",
            *(f"-D{knob}={value}" for knob, value in sorted(knobs.items())),
        ], name
        kernel, binding = program["sources"]
        for source in (kernel, binding):
            texts.setdefault(source, (csrc_dir / source).read_text(encoding="utf-8"))
        assert f"\nkernel_{name}(" in texts[kernel], name
        assert f'extern "C" __global__ void kernel_{name}(' in texts[binding], name
    for source, text in texts.items():
        knobs = {
            knob
            for program in programs.values()
            if source in program["sources"]
            for knob in _dcp_static_program_knobs(program)
        }
        if source.endswith("_kernel.cu"):
            knobs.discard("CAKE_FMHA_DCP_ARCH")  # per-arch kernel files, no guard
        assert knobs, source
        for knob in sorted(knobs):
            macro = knob
            if knob == "CAKE_FMHA_DCP_RETAIN_KV_L2" and source.endswith("_kernel.cu"):
                macro = "RETAIN_KV_L2"
                assert f"#define {macro} {knob}\n" in text, source
                assert not re.search(rf"^#define {macro} \d+$", text, re.M), source
            # the fold tool's guard forms: #if M, #if !M, #if M == v, #if M != v
            guard = rf"^#if !?{macro}(?: (?:==|!=) -?\d+)?$"
            opened = len(re.findall(guard, text, re.M))
            assert opened == text.count(f"#endif  // {macro}\n") > 0, (source, knob)


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a", "sm100f"])
def test_dcp_static_jit_passes_the_program_knobs_and_folded_sources(
    monkeypatch, target
) -> None:
    jit_dcp = importlib.import_module("flashinfer.jit.cake_dcp")
    source_dir = Path(__file__).resolve().parents[3] / "csrc" / "cake_fmha"
    monkeypatch.setattr(jit_dcp, "get_cake_fmha_csrc_dir", lambda: source_dir)
    monkeypatch.setattr(
        jit_dcp, "gen_jit_spec", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    registry = get_dcp_spec_registry()
    arch = jit_dcp._TARGET_MANIFEST_ARCH[target]
    for family, instances in registry["static_routes"].items():
        for instance, by_arch in instances.items():
            name = by_arch[arch]
            program = registry["programs"][name]
            constants = {
                "Q_LEN": 4,
                "CP_WORLD": 4,
                "NUM_Q_HEADS": 64,
                "NUM_KV_HEADS": 8,
            }
            if "NUM_SPLIT" in program["specializations"]:
                constants["NUM_SPLIT"] = _DCP_STATIC_NUM_SPLIT[instance.split("_")[0]]
            spec = jit_dcp.gen_dcp_spec_static_module(
                family, instance, target, constants
            )
            assert spec.name == get_dcp_spec_static_uri(
                family, instance, target, constants
            )
            assert [
                source.relative_to(source_dir).as_posix() for source in spec.sources
            ] == _dcp_static_sources(program)
            assert "--use_fast_math" in spec.extra_cuda_cflags
            for knob, value in _dcp_static_program_knobs(program).items():
                assert f"-D{knob}={value}" in spec.extra_cuda_cflags, (name, knob)
