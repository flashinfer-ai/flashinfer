"""CPU-only dispatch tests for the Cake MoE finalize backend selector."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import AllReduceFusionPattern, TRTLLMAllReduceFusionWorkspace
from flashinfer.comm import trtllm_ar
from flashinfer.comm.workspace_base import AllReduceFusionWorkspace
from flashinfer.jit import cake_moe_finalize_comm

allreduce_module = importlib.import_module("flashinfer.comm.allreduce")


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


def test_default_backend_preserves_trtllm_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    module = SimpleNamespace(
        trtllm_moe_finalize_allreduce_fusion=lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(trtllm_ar, "get_trtllm_comm_module", lambda: module)
    arguments = _arguments()

    trtllm_ar.trtllm_moe_finalize_allreduce_fusion(**arguments)

    assert len(calls) == 1
    assert calls[0]["workspace"] is arguments["workspace_ptrs"]


def test_cake_backend_uses_isolated_source_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cake_moe_finalize_comm,
        "run_cake_moe_finalize",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("Cake dispatch loaded the TRT-LLM module"),
    )
    arguments = _arguments()

    trtllm_ar.trtllm_moe_finalize_allreduce_fusion(
        **arguments,
        backend="cake",
    )

    assert len(calls) == 1
    assert calls[0]["backend"] == "cake"
    assert calls[0]["weight_bias"] is None
    assert set(calls[0]) == set(arguments) | {"backend", "weight_bias"}
    for name, expected in arguments.items():
        actual = calls[0][name]
        if isinstance(expected, torch.Tensor):
            assert actual is expected
        else:
            assert actual == expected


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
            **_arguments(),
            backend="unknown",
        )


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
    workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(workspace, world_size=2, rank=0)
    workspace.mem_handles = []
    workspace.workspace_tensor = torch.empty((7,), dtype=torch.int64)
    workspace._destroyed = True

    arguments = _arguments()
    norm_out = arguments["norm_out"]
    result = allreduce_module.allreduce_fusion(
        input=arguments["allreduce_in"],
        workspace=workspace,
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


def test_unified_selector_is_not_forwarded_to_other_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        allreduce_module,
        "trtllm_allreduce_fusion",
        lambda **kwargs: calls.append(kwargs),
    )
    workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(workspace, world_size=2, rank=0)
    workspace.mem_handles = []
    workspace.workspace_tensor = torch.empty((7,), dtype=torch.int64)
    workspace.metadata = {}
    workspace._destroyed = True
    input = torch.empty((1, 16), dtype=torch.float16)
    output = torch.empty_like(input)

    result = allreduce_module.allreduce_fusion(
        input=input,
        workspace=workspace,
        pattern=AllReduceFusionPattern.kAllReduce,
        output=output,
        moe_finalize_backend="cake",
    )

    assert result is output
    assert len(calls) == 1
    assert "backend" not in calls[0]
