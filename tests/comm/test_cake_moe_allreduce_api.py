"""CPU contract tests for the optional MoE all-reduce backend."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import flashinfer.comm as comm
from flashinfer.comm import trtllm_ar


_PUBLIC_PARAMETER_NAMES = (
    "world_size",
    "world_rank",
    "token_num",
    "hidden_dim",
    "workspace_ptrs",
    "launch_with_pdl",
    "residual_in",
    "rms_gamma",
    "rms_eps",
    "scale_factor",
    "moe_reduction_device_num_experts",
    "moe_reduction_scale_input",
    "moe_reduction_active_experts_token_input",
    "moe_reduction_token_input",
    "layout_code",
    "moe_allreduce_out",
    "residual_out",
    "norm_out",
    "quant_out",
    "scale_out",
    "weight_bias",
    "backend",
)


def _reduction_args() -> dict:
    tokens = 1
    hidden = 4
    experts = 2
    dtype = torch.float16
    return {
        "world_size": 2,
        "world_rank": 0,
        "token_num": tokens,
        "hidden_dim": hidden,
        "workspace_ptrs": torch.zeros(7, dtype=torch.int64),
        "launch_with_pdl": False,
        "residual_in": torch.zeros(tokens * hidden, dtype=dtype),
        "rms_gamma": torch.ones(hidden, dtype=dtype),
        "rms_eps": 1e-6,
        "scale_factor": 1.0,
        "moe_reduction_device_num_experts": experts,
        "moe_reduction_scale_input": torch.ones(
            experts * tokens, dtype=torch.float32
        ),
        "moe_reduction_active_experts_token_input": torch.zeros(
            experts * tokens * hidden, dtype=dtype
        ),
        "moe_reduction_token_input": torch.zeros(tokens * hidden, dtype=dtype),
        "layout_code": None,
        "moe_allreduce_out": None,
        "residual_out": torch.empty(tokens * hidden, dtype=dtype),
        "norm_out": torch.empty(tokens * hidden, dtype=dtype),
        "quant_out": None,
        "scale_out": None,
    }


def test_public_api_has_exact_22_parameter_contract() -> None:
    parameters = inspect.signature(trtllm_ar.trtllm_moe_allreduce_fusion).parameters

    assert tuple(parameters) == _PUBLIC_PARAMETER_NAMES
    assert len(parameters) == 22
    assert parameters["backend"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["backend"].default == "trtllm"
    assert (
        comm.trtllm_moe_allreduce_fusion is trtllm_ar.trtllm_moe_allreduce_fusion
    )


def test_default_backend_keeps_trtllm_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    module = SimpleNamespace(
        trtllm_moe_allreduce_fusion=lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(trtllm_ar, "get_trtllm_comm_module", lambda: module)

    args = _reduction_args()
    trtllm_ar.trtllm_moe_allreduce_fusion(**args)

    assert len(calls) == 1
    assert calls[0]["world_size"] == args["world_size"]
    assert calls[0]["moe_reduction_token_input"] is args[
        "moe_reduction_token_input"
    ]


def test_cake_backend_dispatches_exact_18_argument_ffi_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    module = SimpleNamespace(run_reduction=lambda *args: calls.append(args))
    monkeypatch.setattr(
        trtllm_ar, "_validate_cake_moe_allreduce", lambda **kwargs: 3
    )
    monkeypatch.setattr(
        trtllm_ar, "get_cake_moe_allreduce_module", lambda device_index: module
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("TRT-LLM module must not load for backend='cake'"),
    )

    args = _reduction_args()
    trtllm_ar.trtllm_moe_allreduce_fusion(**args, backend="cake")

    assert len(calls) == 1
    call = calls[0]
    assert len(call) == 18
    assert call[:4] == (2, 0, 1, 4)
    assert call[4] is args["workspace_ptrs"]
    assert call[5] is False
    assert call[6] is args["residual_in"]
    assert call[7] is args["rms_gamma"]
    assert call[8:11] == (1e-6, 1.0, 2)
    assert call[11] is args["moe_reduction_scale_input"]
    assert call[12] is args["moe_reduction_active_experts_token_input"]
    assert call[13] is args["moe_reduction_token_input"]
    assert call[14] is None
    assert call[15] is args["residual_out"]
    assert call[16] is args["norm_out"]
    assert call[17] is None


def test_invalid_backend_fails_before_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("invalid backend must fail before module load"),
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_cake_moe_allreduce_module",
        lambda _device_index: pytest.fail("invalid backend must not load Cake"),
    )

    with pytest.raises(ValueError, match="unsupported MoE all-reduce backend"):
        trtllm_ar.trtllm_moe_allreduce_fusion(
            **_reduction_args(), backend="unknown"
        )
