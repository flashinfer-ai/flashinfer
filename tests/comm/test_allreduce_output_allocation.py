import socket
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import flashinfer.comm.allreduce as allreduce_module
from flashinfer.comm.allreduce import TRTLLMAllReduceFusionWorkspace
from flashinfer.comm.trtllm_ar import AllReduceFusionPattern, QuantizationSFLayout
from flashinfer.comm.workspace_base import AllReduceFusionWorkspace


STANDARD_PATTERNS = (
    pytest.param(AllReduceFusionPattern.kAllReduce, id="allreduce"),
    pytest.param(AllReduceFusionPattern.kARResidualRMSNorm, id="rmsnorm"),
    pytest.param(AllReduceFusionPattern.kARResidualRMSNormFP8Quant, id="fp8-quant"),
    pytest.param(AllReduceFusionPattern.kARResidualRMSNormFP4Quant, id="fp4-quant"),
    pytest.param(
        AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
        id="rmsnorm-out-fp8-quant",
    ),
    pytest.param(
        AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        id="rmsnorm-out-fp4-quant",
    ),
    pytest.param(
        AllReduceFusionPattern.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
        id="group-fp8-quant",
    ),
    pytest.param(
        AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
        id="rmsnorm-out-group-fp8-quant",
    ),
    pytest.param(
        AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant,
        id="dynamic-fp8-quant",
    ),
    pytest.param(
        AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant,
        id="rmsnorm-out-dynamic-fp8-quant",
    ),
)


@pytest.fixture
def trtllm_workspace() -> TRTLLMAllReduceFusionWorkspace:
    """Build the minimum TRT-LLM workspace needed by the Python dispatcher."""
    workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    AllReduceFusionWorkspace.__init__(workspace, world_size=2, rank=0)
    workspace.workspace_tensor = torch.zeros(1, dtype=torch.int64)
    workspace.mem_handles = []
    workspace.metadata = {}
    # The mock workspace owns no resources and should not warn at interpreter exit.
    workspace._destroyed = True
    return workspace


def _fusion_arguments(pattern: int, input: torch.Tensor) -> dict:
    """Return valid wrapper inputs without invoking a communication kernel."""
    token_num, hidden_dim = input.shape
    arguments = {
        "residual_in": torch.ones_like(input),
        "residual_out": torch.empty_like(input),
        "rms_gamma": torch.ones(hidden_dim, dtype=input.dtype),
    }

    patterns_with_norm_out = {
        AllReduceFusionPattern.kARResidualRMSNorm,
        AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
        AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
        AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant,
    }
    if pattern in patterns_with_norm_out:
        arguments["norm_out"] = torch.empty_like(input)

    fp8_patterns = {
        AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
        AllReduceFusionPattern.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
        AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
        AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant,
        AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant,
    }
    if pattern in fp8_patterns:
        arguments["quant_out"] = torch.empty_like(input, dtype=torch.float8_e4m3fn)

    fp4_patterns = {
        AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
    }
    if pattern in fp4_patterns:
        arguments.update(
            quant_out=torch.empty((token_num, hidden_dim // 2), dtype=torch.uint8),
            layout_code=QuantizationSFLayout.LINEAR,
        )

    group_fp8_patterns = {
        AllReduceFusionPattern.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
        AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
    }
    if pattern in group_fp8_patterns:
        # hidden_dim=8 and group_size=4 produce one packed scale per token.
        # Transposing (1, token_num) gives the TMA layout expected by the wrapper:
        # shape=(token_num, 1), stride=(1, aligned_token_num).
        arguments.update(
            block_quant_group_size=4,
            scale_out=torch.empty((1, token_num), dtype=torch.int32).T,
        )

    dynamic_fp8_patterns = {
        AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant,
        AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant,
    }
    if pattern in dynamic_fp8_patterns:
        arguments["scale_out"] = torch.empty((token_num, 1), dtype=torch.float32)

    return arguments


@pytest.mark.parametrize("pattern", STANDARD_PATTERNS)
@pytest.mark.parametrize("explicit_output", [False, True], ids=["implicit", "explicit"])
def test_trtllm_wrapper_only_allocates_plain_allreduce_output(
    monkeypatch: pytest.MonkeyPatch,
    trtllm_workspace: TRTLLMAllReduceFusionWorkspace,
    pattern: int,
    explicit_output: bool,
) -> None:
    input = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    output = torch.full_like(input, 7) if explicit_output else None
    arguments = _fusion_arguments(pattern, input)

    low_level_call = Mock()
    monkeypatch.setattr(allreduce_module, "trtllm_allreduce_fusion", low_level_call)

    allocations = []
    original_empty_like = torch.empty_like

    def tracked_empty_like(*args, **kwargs):
        allocated = original_empty_like(*args, **kwargs)
        allocations.append(allocated)
        return allocated

    # Auxiliary pattern outputs were constructed above. Any allocation observed
    # from this point must therefore come from allreduce_fusion itself.
    monkeypatch.setattr(allreduce_module.torch, "empty_like", tracked_empty_like)

    allreduce_module.allreduce_fusion(
        input=input,
        workspace=trtllm_workspace,
        pattern=pattern,
        output=output,
        **arguments,
    )

    low_level_call.assert_called_once()
    forwarded_output = low_level_call.call_args.kwargs["allreduce_out"]

    if explicit_output:
        assert allocations == []
        assert forwarded_output is not None
        assert forwarded_output.shape == (input.numel(),)
        assert forwarded_output.data_ptr() == output.data_ptr()
    elif pattern == AllReduceFusionPattern.kAllReduce:
        assert len(allocations) == 1
        assert forwarded_output is not None
        assert forwarded_output.shape == (input.numel(),)
        assert forwarded_output.data_ptr() == allocations[0].data_ptr()
    else:
        assert allocations == []
        assert forwarded_output is None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_fused_rmsnorm_without_allreduce_output(
    rank: int, world_size: int, port: int
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    workspace = None
    try:
        from flashinfer.comm.mnnvl import TorchDistBackend

        input = torch.full((4, 4096), rank + 1, dtype=torch.bfloat16, device="cuda")
        residual_in = torch.full_like(input, 0.25)
        residual_out = torch.empty_like(input)
        norm_out = torch.empty_like(input)
        rms_gamma = torch.ones(4096, dtype=torch.bfloat16, device="cuda")
        rms_eps = 1e-6

        workspace = allreduce_module.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=input.shape[0],
            hidden_dim=input.shape[1],
            dtype=input.dtype,
            comm_backend=TorchDistBackend(),
        )
        dist.barrier()

        result = allreduce_module.allreduce_fusion(
            input=input,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            residual_in=residual_in,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            use_oneshot=True,
        )
        torch.cuda.synchronize()

        expected_residual = torch.full_like(residual_out, 3.25)
        expected_norm = expected_residual.float()
        expected_norm *= torch.rsqrt(
            expected_norm.square().mean(dim=-1, keepdim=True) + rms_eps
        )
        assert result.data_ptr() == norm_out.data_ptr()
        torch.testing.assert_close(residual_out, expected_residual)
        torch.testing.assert_close(
            norm_out.float(), expected_norm, rtol=1e-2, atol=1e-2
        )
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


def _supports_trtllm_two_gpu_smoke() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    try:
        if any(
            torch.cuda.get_device_capability(device)[0] not in (9, 10, 12)
            for device in (0, 1)
        ):
            return False
        return torch.cuda.can_device_access_peer(
            0, 1
        ) and torch.cuda.can_device_access_peer(1, 0)
    except (AssertionError, RuntimeError):
        # Driver/MIG configurations can reject capability or peer-access
        # queries during collection. Such hosts cannot run this VMM smoke test.
        return False


@pytest.mark.skipif(
    not _supports_trtllm_two_gpu_smoke(),
    reason="TRT-LLM fused all-reduce requires two supported peer-accessible GPUs",
)
def test_fused_rmsnorm_without_allreduce_output() -> None:
    """Exercise the null allreduce_out path on a two-GPU CUDA system."""
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    mp.spawn(
        _run_fused_rmsnorm_without_allreduce_output,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )
