"""CPU contract tests for the optional MoE all-reduce backend."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import flashinfer.comm as comm
from flashinfer.comm import trtllm_ar
from flashinfer.jit import cake_trtllm_moe_allreduce_union as union


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
        "moe_reduction_scale_input": torch.ones(experts * tokens, dtype=torch.float32),
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
    assert comm.trtllm_moe_allreduce_fusion is trtllm_ar.trtllm_moe_allreduce_fusion


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
    assert calls[0]["moe_reduction_token_input"] is args["moe_reduction_token_input"]


def test_cake_backend_dispatches_to_the_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    union_parameters = set(
        inspect.signature(union.run_cake_moe_allreduce_union).parameters
    )
    monkeypatch.setattr(trtllm_ar, "_validate_cake_moe_allreduce", lambda **kwargs: 3)
    monkeypatch.setattr(
        union, "run_cake_moe_allreduce_union", lambda **kwargs: calls.append(kwargs)
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
    assert call["backend"] == "cake"
    assert (
        call["world_size"],
        call["world_rank"],
        call["token_num"],
        call["hidden_dim"],
    ) == (2, 0, 1, 4)
    assert call["workspace_ptrs"] is args["workspace_ptrs"]
    assert call["launch_with_pdl"] is False
    assert call["residual_in"] is args["residual_in"]
    assert call["rms_gamma"] is args["rms_gamma"]
    assert (call["rms_eps"], call["scale_factor"]) == (1e-6, 1.0)
    assert call["moe_reduction_device_num_experts"] == 2
    assert call["moe_reduction_scale_input"] is args["moe_reduction_scale_input"]
    assert (
        call["moe_reduction_active_experts_token_input"]
        is args["moe_reduction_active_experts_token_input"]
    )
    assert call["moe_reduction_token_input"] is args["moe_reduction_token_input"]
    assert call["moe_allreduce_out"] is None
    assert call["residual_out"] is args["residual_out"]
    assert call["norm_out"] is args["norm_out"]
    assert call["weight_bias"] is None
    assert set(call) == union_parameters


def test_cake_validator_has_no_token_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_num = 2049
    hidden_dim = 7168
    experts = 2
    device = torch.device("cuda:0")

    def fake_tensor(numel: int, dtype: torch.dtype) -> SimpleNamespace:
        return SimpleNamespace(
            device=device,
            dtype=dtype,
            numel=lambda: numel,
            is_contiguous=lambda: True,
        )

    dtype = torch.float16
    token_elements = token_num * hidden_dim
    monkeypatch.setattr(trtllm_ar, "_check_cake_moe_allreduce_arch", lambda _: None)

    device_index = trtllm_ar._validate_cake_moe_allreduce(
        world_size=2,
        world_rank=0,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=fake_tensor(7, torch.int64),
        residual_in=fake_tensor(token_elements, dtype),
        rms_gamma=fake_tensor(hidden_dim, dtype),
        moe_reduction_device_num_experts=experts,
        moe_reduction_scale_input=fake_tensor(experts * token_num, torch.float32),
        moe_reduction_active_experts_token_input=fake_tensor(
            experts * token_elements, dtype
        ),
        moe_reduction_token_input=fake_tensor(token_elements, dtype),
        layout_code=None,
        moe_allreduce_out=None,
        residual_out=fake_tensor(token_elements, dtype),
        norm_out=fake_tensor(token_elements, dtype),
        quant_out=None,
        scale_out=None,
    )

    assert device_index == 0
    assert not hasattr(trtllm_ar, "_CAKE_MOE_ALLREDUCE_MAX_TOKENS")


def test_lamport_byte_limit_rejects_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world_size = 2
    overflow_numel = trtllm_ar.MAX_COMM_SIZE // (2 * world_size) + 1
    args = _reduction_args()
    args["moe_reduction_token_input"] = SimpleNamespace(numel=lambda: overflow_numel)
    monkeypatch.setattr(
        union,
        "run_cake_moe_allreduce_union",
        lambda **_: pytest.fail("oversize payload must fail before the union runs"),
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"required_lamport_comm_size .* is greater than MAX_COMM_SIZE "
            rf"{trtllm_ar.MAX_COMM_SIZE}"
        ),
    ):
        trtllm_ar.trtllm_moe_allreduce_fusion(**args, backend="cake")


def test_invalid_backend_fails_before_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("invalid backend must fail before module load"),
    )
    monkeypatch.setattr(
        union,
        "run_cake_moe_allreduce_union",
        lambda **_: pytest.fail("invalid backend must not reach the union"),
    )

    with pytest.raises(ValueError, match="unsupported MoE all-reduce backend"):
        trtllm_ar.trtllm_moe_allreduce_fusion(**_reduction_args(), backend="unknown")
