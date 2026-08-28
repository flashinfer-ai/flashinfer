"""Host-only API, layout, and validation tests for the rank-major backend."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    EpAlgorithm,
    EpLayout,
    FleetParams,
    MoEWeightPack,
    Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig,
    preprocess_bf16_rank_major_cuda_mega_weights,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_rank_major_cuda.backend import (
    Bf16RankMajorCudaMegaKernelBackend,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_rank_major_cuda.weights import (
    TransformedMegaWeights,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError


_CUDA_DEVICE = torch.device("cuda:0")


def _exact_fleet() -> FleetParams:
    return FleetParams(
        num_experts=256,
        max_tokens_per_rank=128,
        token_hidden_size=7168,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
        layout=EpLayout.RANK_MAJOR,
    )


def _backend() -> Bf16RankMajorCudaMegaKernelBackend:
    backend = create_mega_kernel(Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig())
    assert isinstance(backend, Bf16RankMajorCudaMegaKernelBackend)
    return backend


def test_public_config_is_registered_with_fixed_kernel_identity():
    config = Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig()
    assert config.intermediate_size == 2048
    assert config.top_k == 8
    assert config.kernel_name == "sm100_bf16_bf16_bf16_rank_major_cuda"
    assert isinstance(create_mega_kernel(config), Bf16RankMajorCudaMegaKernelBackend)


def test_init_accepts_only_the_exact_rank_major_coordinate():
    backend = _backend()
    bootstrap = BootstrapConfig(world_size=8, rank=0, auto_bootstrap=False)
    with mock.patch(
        "flashinfer.moe_ep.backends.mega.kernel.sm100."
        "bf16_bf16_bf16_rank_major_cuda.backend.validate_mega_arch"
    ):
        backend.validate_init(bootstrap, _exact_fleet())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_experts", 128),
        ("max_tokens_per_rank", 64),
        ("token_hidden_size", 4096),
        ("dtype_bytes", 1),
        ("layout", EpLayout.EXPERT_MAJOR),
    ),
)
def test_init_rejects_coordinate_drift(field: str, value: object):
    backend = _backend()
    fleet = replace(_exact_fleet(), **{field: value})
    bootstrap = BootstrapConfig(world_size=8, rank=0, auto_bootstrap=False)
    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "bf16_bf16_bf16_rank_major_cuda.backend.validate_mega_arch"
        ),
        pytest.raises(MoEEpConfigError, match=field),
    ):
        backend.validate_init(bootstrap, fleet)


def test_init_rejects_high_throughput_mode():
    backend = _backend()
    fleet = replace(
        _exact_fleet(),
        algorithm=EpAlgorithm.HIGH_THROUGHPUT,
        layout=EpLayout.EXPERT_MAJOR,
    )
    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "bf16_bf16_bf16_rank_major_cuda.backend.validate_mega_arch"
        ),
        pytest.raises(MoEEpConfigError, match="algorithm"),
    ):
        backend.validate_init(
            BootstrapConfig(world_size=8, rank=0, auto_bootstrap=False),
            fleet,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (
            Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig(intermediate_size=1024),
            "intermediate_size=2048",
        ),
        (
            Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig(top_k=4),
            "top_k=8",
        ),
    ),
)
def test_init_rejects_kernel_config_coordinate_drift(config, message: str):
    backend = Bf16RankMajorCudaMegaKernelBackend(config)
    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "bf16_bf16_bf16_rank_major_cuda.backend.validate_mega_arch"
        ),
        pytest.raises(MoEEpConfigError, match=message),
    ):
        backend.validate_init(
            BootstrapConfig(world_size=8, rank=0, auto_bootstrap=False),
            _exact_fleet(),
        )


def test_init_rejects_world_size_and_nondefault_stream():
    backend = _backend()
    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "bf16_bf16_bf16_rank_major_cuda.backend.validate_mega_arch"
        ),
        pytest.raises(MoEEpConfigError, match="world_size=8"),
    ):
        backend.validate_init(
            BootstrapConfig(world_size=4, rank=0, auto_bootstrap=False),
            _exact_fleet(),
        )
    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "bf16_bf16_bf16_rank_major_cuda.backend.validate_mega_arch"
        ),
        pytest.raises(MoEEpConfigError, match="stream must be 0"),
    ):
        backend.validate_init(
            BootstrapConfig(
                world_size=8,
                rank=0,
                stream=1,
                auto_bootstrap=False,
            ),
            _exact_fleet(),
        )


def test_weight_preprocess_swaps_gate_up_before_block_major_shuffle():
    intermediate = hidden = 64
    gate = (
        torch.arange(1, intermediate + 1, dtype=torch.bfloat16)
        .view(1, intermediate, 1)
        .expand(1, intermediate, hidden)
        .contiguous()
    )
    up = (
        torch.arange(101, 101 + intermediate, dtype=torch.bfloat16)
        .view(1, intermediate, 1)
        .expand(1, intermediate, hidden)
        .contiguous()
    )
    w2 = (
        torch.arange(1, hidden + 1, dtype=torch.bfloat16)
        .view(1, hidden, 1)
        .expand(1, hidden, intermediate)
        .contiguous()
    )
    transformed = preprocess_bf16_rank_major_cuda_mega_weights(
        MoEWeightPack(torch.cat((gate, up), dim=1), w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
        num_local_experts=1,
    )
    assert isinstance(transformed, TransformedMegaWeights)
    assert transformed.w13_block_major.shape == (1, 1, 128, 64)
    assert transformed.w2_block_major.shape == (1, 1, 64, 64)
    # Physical rows 0/8/16/24 map logical interleaved rows 0/1/2/3.
    # The required logical order is up0, gate0, up1, gate1.
    physical = transformed.w13_block_major[0, 0, :, 0]
    assert [physical[i].item() for i in (0, 8, 16, 24)] == [101, 1, 102, 2]


def test_weight_preprocess_rejects_non_bf16_canonical_weights():
    weights = MoEWeightPack(
        torch.zeros(1, 128, 64),
        torch.zeros(1, 64, 64),
    )
    with pytest.raises(MoEEpConfigError, match="w13 must be torch.bfloat16"):
        preprocess_bf16_rank_major_cuda_mega_weights(
            weights,
            intermediate_size=64,
            hidden_size=64,
            num_local_experts=1,
        )


class _FakeTensor:
    def __init__(self, shape, dtype, *, device=_CUDA_DEVICE) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.type == "cuda"

    def is_contiguous(self) -> bool:
        return True


def _fake_inputs(*, tokens: int = 128, ids_dtype=torch.int64):
    return SimpleNamespace(
        hidden_states=_FakeTensor((tokens, 7168), torch.bfloat16),
        topk_ids=_FakeTensor((tokens, 8), ids_dtype),
        topk_weights=_FakeTensor((tokens, 8), torch.float32),
        scales=None,
    )


def test_forward_validation_accepts_exact_abi_and_rejects_shape_or_dtype_drift():
    backend = _backend()
    backend.validate_forward(_fake_inputs(), _exact_fleet(), quantize_input=True)
    with pytest.raises(MoEEpConfigError, match=r"shape \(128, 7168\)"):
        backend.validate_forward(
            _fake_inputs(tokens=127),
            _exact_fleet(),
            quantize_input=True,
        )
    with pytest.raises(MoEEpConfigError, match="topk_ids must be torch.int64"):
        backend.validate_forward(
            _fake_inputs(ids_dtype=torch.int32),
            _exact_fleet(),
            quantize_input=True,
        )


def test_stage_and_compute_delegate_to_the_session_in_order():
    backend = _backend()
    tensors = _fake_inputs()
    workspace = mock.Mock()
    backend.stage_inputs(tensors, workspace, quantize_input=True)
    workspace.stage_inputs.assert_called_once_with(
        tensors.hidden_states,
        tensors.topk_ids,
        tensors.topk_weights,
    )

    transformed = SimpleNamespace(
        w13_block_major=object(),
        w2_block_major=object(),
    )
    output = object()
    workspace.run.return_value = output
    parent = mock.Mock()
    parent.attach_mock(workspace.bind_weights, "bind_weights")
    parent.attach_mock(workspace.run, "run")
    assert backend.compute(workspace, transformed, output=output) is output
    assert parent.mock_calls == [
        mock.call.bind_weights(
            transformed.w13_block_major,
            transformed.w2_block_major,
        ),
        mock.call.run(output),
    ]
