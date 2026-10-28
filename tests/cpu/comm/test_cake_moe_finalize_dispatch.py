"""CPU-only tests for the Cake MoE finalize backend selector."""

import re

import pytest
import torch

from flashinfer.comm import AllReduceFusionPattern, TRTLLMAllReduceFusionWorkspace
from flashinfer.comm import allreduce as allreduce_module
from flashinfer.comm import trtllm_ar
from flashinfer.comm.workspace_base import AllReduceFusionWorkspace
from flashinfer.jit import cake_moe_finalize_comm


def _arguments() -> dict[str, object]:
    state = torch.empty((1, 7168), dtype=torch.float16)
    return {
        "allreduce_in": torch.empty((8, 7168), dtype=torch.float16),
        "residual_in": state,
        "norm_weight": torch.empty((7168,), dtype=torch.float16),
        "expanded_idx_to_permuted_idx": torch.empty((1, 8), dtype=torch.int32),
        "norm_out": torch.empty_like(state),
        "residual_out": torch.empty_like(state),
        "quant_out": None,
        "scale_out": None,
        "workspace_ptrs": torch.empty((7,), dtype=torch.int64),
        "launch_with_pdl": False,
        "world_rank": 0,
        "world_size": 2,
        "eps": 1e-5,
        "shared_expert_output": None,
        "expert_scale_factor": torch.empty((1, 8), dtype=torch.float16),
        "routed_scaling_factor": None,
    }


def _fake_workspace() -> TRTLLMAllReduceFusionWorkspace:
    workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(workspace, world_size=2, rank=0)
    workspace.mem_handles = []
    workspace.workspace_tensor = torch.empty((7,), dtype=torch.int64)
    workspace.metadata = {}
    workspace._destroyed = True
    return workspace


@pytest.mark.cpu_only
def test_cake_backend_routes_to_the_cake_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cake_moe_finalize_comm, "run_cake_moe_finalize", lambda **kw: calls.append(kw)
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("Cake dispatch loaded the TRT-LLM module"),
    )
    arguments = _arguments()

    trtllm_ar.trtllm_moe_finalize_allreduce_fusion(**arguments, backend="cake")

    assert len(calls) == 1
    assert set(calls[0]) == set(arguments) | {"weight_bias"}
    assert calls[0]["weight_bias"] is None
    for name, expected in arguments.items():
        if isinstance(expected, torch.Tensor):
            assert calls[0][name] is expected
        else:
            assert calls[0][name] == expected


@pytest.mark.cpu_only
def test_unknown_backend_fails_before_loading_a_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("invalid backend loaded the TRT-LLM module"),
    )
    with pytest.raises(ValueError, match="backend must be"):
        trtllm_ar.trtllm_moe_finalize_allreduce_fusion(
            **_arguments(), backend="unknown"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("backend_kwargs", "expected_backend"),
    [({}, "trtllm"), ({"moe_finalize_backend": "cake"}, "cake")],
)
def test_unified_finalize_forwards_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_kwargs: dict[str, str],
    expected_backend: str,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        allreduce_module,
        "trtllm_moe_finalize_allreduce_fusion",
        lambda **kwargs: calls.append(kwargs),
    )
    arguments = _arguments()
    norm_out = arguments["norm_out"]

    result = allreduce_module.allreduce_fusion(
        input=arguments["allreduce_in"],
        workspace=_fake_workspace(),
        pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        residual_in=arguments["residual_in"],
        rms_gamma=arguments["norm_weight"],
        expanded_idx_to_permuted_idx=arguments["expanded_idx_to_permuted_idx"],
        norm_out=norm_out,
        residual_out=arguments["residual_out"],
        expert_scale_factor=arguments["expert_scale_factor"],
        **backend_kwargs,
    )

    assert result is norm_out
    assert len(calls) == 1
    assert calls[0]["backend"] == expected_backend


@pytest.mark.cpu_only
def test_unified_selector_is_not_forwarded_to_other_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        allreduce_module,
        "trtllm_allreduce_fusion",
        lambda **kwargs: calls.append(kwargs),
    )
    input = torch.empty((1, 16), dtype=torch.float16)
    output = torch.empty_like(input)

    result = allreduce_module.allreduce_fusion(
        input=input,
        workspace=_fake_workspace(),
        pattern=AllReduceFusionPattern.kAllReduce,
        output=output,
        moe_finalize_backend="cake",
    )

    assert result is output
    assert len(calls) == 1
    assert "backend" not in calls[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_small_token_max_reaches_the_binding(
    monkeypatch: pytest.MonkeyPatch, world_size: int
) -> None:
    calls: list[tuple[object, ...]] = []
    arches: list[str] = []

    class _Module:
        def cake_moe_finalize_allreduce_fusion(self, *args: object) -> None:
            calls.append(args)

    def _get_module(arch: str) -> _Module:
        arches.append(arch)
        return _Module()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cake_moe_finalize_comm, "target_arch", lambda index: "sm_103a")
    monkeypatch.setattr(
        cake_moe_finalize_comm, "get_cake_moe_finalize_module", _get_module
    )
    arguments = _arguments()
    arguments["world_size"] = world_size

    cake_moe_finalize_comm.run_cake_moe_finalize(**arguments, weight_bias=None)

    assert arches == ["sm_103a"]
    assert len(calls) == 1
    # Positional binding arguments: world_rank, world_size, small_token_max.
    assert calls[0][11:14] == (0, world_size, cake_moe_finalize_comm.SMALL_TOKEN_MAX)


@pytest.mark.cpu_only
def test_module_records_match_the_binding() -> None:
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    declared = set(
        re.findall(r"^CAKE_MOE_FINALIZE_KERNEL\([^,]+, (\w+)\)$", binding, re.M)
    )
    assert cake_moe_finalize_comm.MODULES
    for name, record in cake_moe_finalize_comm.MODULES.items():
        assert record["kernel"] == f"kernel_{name}"
        assert record["kernel"] in declared
        assert set(record["arches"]) <= {"sm_100a", "sm_103a"}
        for source in record["sources"]:
            assert (source_dir / source).is_file(), source


def _program_table(binding: str, name: str) -> list[str | None]:
    """Entries of one [dtype][world size index][quant epilogue] table of the binding, in
    order (float16 then bfloat16; world sizes 2, 4, 8; o110 then o111)."""
    start = binding.index(f"const void* const {name}[2][3][2] = {{")
    body = binding[start : binding.index("};", start)]
    entries = re.findall(r"reinterpret_cast<const void\*>\((\w+)\)|(nullptr)", body)
    assert len(entries) == 12, (name, entries)
    return [kernel or None for kernel, _ in entries]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("table", "suffix", "world_sizes"),
    [
        ("kSmallKernels", "_small", (2, 4, 8)),
        ("kKernels", "", (2, 4)),
        ("kLargeTokenKernels", "_lt", (2,)),
        ("kTwoShotKernels", "_ts", (4, 8)),
    ],
)
def test_program_tables_list_one_program_kind(
    table: str, suffix: str, world_sizes: tuple[int, ...]
) -> None:
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    entries = iter(_program_table(binding, table))
    for dtype in ("float16", "bfloat16"):
        for world_size in (2, 4, 8):
            for output_profile in ("110", "111"):
                expected = (
                    f"kernel_cake_trtllm_moe_finalize_{dtype}_ws{world_size}"
                    f"_o{output_profile}{suffix}"
                    if world_size in world_sizes
                    else None
                )
                assert next(entries) == expected


@pytest.mark.cpu_only
def test_program_tables_cover_the_module_records() -> None:
    source_dir = cake_moe_finalize_comm._source_dir()
    binding = (
        source_dir / f"{cake_moe_finalize_comm.SOURCE_PACKAGE}_binding.cu"
    ).read_text()
    tabled = [
        kernel
        for table in (
            "kSmallKernels",
            "kKernels",
            "kLargeTokenKernels",
            "kTwoShotKernels",
        )
        for kernel in _program_table(binding, table)
        if kernel is not None
    ]
    assert len(tabled) == len(set(tabled))
    assert set(tabled) == {
        record["kernel"] for record in cake_moe_finalize_comm.MODULES.values()
    }
