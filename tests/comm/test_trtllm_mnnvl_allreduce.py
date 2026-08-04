# Check torch version:
import traceback
from typing import Tuple, Optional

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.fp4_quantization import _compute_swizzled_layout_sf_size
from flashinfer.utils import get_compute_capability

# Use flashinfer.norm.rmsnorm as reference implementation.
from flashinfer.norm import rmsnorm

# Test helpers
from tests.test_helpers.comm import (
    init_torch_distributed_from_mpi,
)
from tests.test_helpers.utils_fp4 import (
    cast_from_fp4,
    recover_swizzled_scales,
    ref_fp4_quant,
)

# Note: torch.distributed cleanup is handled by tests/comm/conftest.py


def fp8_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qinput = (x.float() * scale.reciprocal()).clamp(min=finfo.min, max=finfo.max)
    return qinput.to(torch.float8_e4m3fn)


def dequant(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (x.to(torch.float32) * scale).to(dtype)


def _mnnvl_nvfp4_supported_on_all_ranks(device: torch.device) -> bool:
    local_supported = get_compute_capability(device)[0] >= 10
    if not dist.is_initialized():
        return local_supported

    support_flags = [False] * dist.get_world_size()
    dist.all_gather_object(support_flags, local_supported)
    return all(support_flags)


def test_mnnvl_nvfp4_default_swizzled_scale_out_requires_padded_size():
    token_num = 1
    hidden_dim = 16
    linear_scale_size = token_num * hidden_dim // 16
    swizzled_scale_size = _compute_swizzled_layout_sf_size(token_num, hidden_dim // 16)
    assert linear_scale_size < swizzled_scale_size

    input_tensor = torch.randn((token_num, hidden_dim), dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn((hidden_dim,), dtype=torch.float16)
    scale_out = torch.empty(linear_scale_size, dtype=torch.float8_e4m3fn)

    with pytest.raises(
        ValueError,
        match=(
            f"scale_out is too small for NVFP4: got {linear_scale_size} "
            f"elements, need at least {swizzled_scale_size}"
        ),
    ):
        trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant(
            input=input_tensor,
            residual_in=residual,
            gamma=gamma,
            workspace=object(),  # type: ignore[arg-type]
            scale_out=scale_out,
            quant_type=trtllm_mnnvl_ar.MNNVLQuantType.NVFP4,
        )


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    mapping: Mapping,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
    workspace: trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace,
):
    tensor_parallel_rank = mapping.tp_rank
    dist.barrier()

    def func(
        input,
        residual,
        norm_weight,
        eps,
        enable_fusion,
        workspace,
    ):
        # For both fused and unfused cases:
        shape = input.shape
        input = input.view(-1, shape[-1])
        use_pdl = True

        if enable_fusion:
            dist.barrier()

            output, residual_out = (
                trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_add_rmsnorm(
                    input,
                    residual,
                    norm_weight,
                    workspace,
                    eps,
                    launch_with_pdl=use_pdl,
                    strategy=trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.AUTO,
                )
            )

            return output.view(shape), residual_out.view(shape)

        else:
            output = torch.empty_like(input)

            output = trtllm_mnnvl_ar.trtllm_mnnvl_allreduce(
                input,
                workspace,
                launch_with_pdl=use_pdl,
                strategy=trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.AUTO,
            )
            return (output.view(shape),)

    output = func(x.clone(), residual.clone(), norm_weight, eps, fusion, workspace)

    assert output[0].shape == reference_output[0].shape

    if tensor_parallel_rank == 0:
        print("output[0] (first 10 values):", output[0].flatten()[:10])
        print(
            "reference_output[0] (first 10 values):",
            reference_output[0].flatten()[:10],
        )

        if fusion:
            print("output[1] (first 10 values):", output[1].flatten()[:10])
            print(
                "reference_output[1] (first 10 values):",
                reference_output[1].flatten()[:10],
            )

    torch.testing.assert_close(
        output[0],
        reference_output[0],
        rtol=0.05,
        atol=0.15,
    )

    if fusion:
        torch.testing.assert_close(
            output[1],
            reference_output[1],
            rtol=0.05,
            atol=0.15,
        )


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward_legacy(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hidden_size: int,
    dtype: torch.dtype,
    mapping: Mapping,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
    multicast_ptr: int,
    buffer_ptrs_dev: int,
    unicast_ptr: int,
    max_num_elements_mnnvl: int,
    buffer_flags_mnnvl: torch.Tensor,
):
    tensor_parallel_size = mapping.tp_size
    tensor_parallel_rank = mapping.tp_rank
    dist.barrier()

    def func(
        input,
        residual,
        norm_weight,
        eps,
        enable_fusion,
        multicast_ptr,
        buffer_ptrs_dev,
        unicast_ptr,
        max_num_elements_mnnvl,
    ):
        # For both fused and unfused cases:
        shape = input.shape
        input = input.view(-1, shape[-1])
        buffer_M = max_num_elements_mnnvl // hidden_size

        if enable_fusion:
            use_pdl = True

            prenorm_output = torch.empty_like(residual)
            normed_output = torch.empty_like(residual)

            dist.barrier()

            trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_rmsnorm(
                prenorm_output,
                normed_output,
                input,
                multicast_ptr,
                buffer_ptrs_dev,
                unicast_ptr,
                buffer_M,
                buffer_flags_mnnvl,
                tensor_parallel_size,
                tensor_parallel_rank,
                norm_weight,
                eps,
                residual,
                use_pdl,
            )

            return normed_output.view(shape), prenorm_output.view(shape)

        else:
            output = torch.empty_like(input)

            trtllm_mnnvl_ar.trtllm_mnnvl_all_reduce(
                input,
                multicast_ptr,
                buffer_ptrs_dev,
                buffer_M,
                buffer_flags_mnnvl,
                tensor_parallel_size,
                tensor_parallel_rank,
                True,  # wait_for_results
                False,  # launch_with_pdl
                output,  # Need to provide output tensor since we are writing them out.
            )
            return (output.view(shape),)

    output = func(
        x.clone(),
        residual.clone(),
        norm_weight,
        eps,
        fusion,
        multicast_ptr,
        buffer_ptrs_dev,
        unicast_ptr,
        max_num_elements_mnnvl,
    )

    assert output[0].shape == reference_output[0].shape

    if tensor_parallel_rank == 0:
        print("output[0] (first 10 values):", output[0].flatten()[:10])
        print(
            "reference_output[0] (first 10 values):",
            reference_output[0].flatten()[:10],
        )

        if fusion:
            print("output[1] (first 10 values):", output[1].flatten()[:10])
            print(
                "reference_output[1] (first 10 values):",
                reference_output[1].flatten()[:10],
            )

    torch.testing.assert_close(
        output[0],
        reference_output[0],
        rtol=0.05,
        atol=0.15,
    )

    if fusion:
        torch.testing.assert_close(
            output[1],
            reference_output[1],
            rtol=0.05,
            atol=0.15,
        )


"""Helper function to run the core MNNVL AllReduce test logic"""


def _inject_sentinel_trigger_patterns(x_full: torch.Tensor) -> None:
    """Overlay each rank's input slice with 16-bit patterns that exercise the
    Lamport sentinel polling logic in the MNNVL allreduce kernels.

    Three regions per rank:
      - Region A (alternating 0x0000, 0x8001+i): a 4-byte poll load over each
        pair reinterprets to a fp32 negative-subnormal pattern such as
        0x80010000. A correct bit-exact sentinel compare must reject these as
        ordinary data; a `val == 0.F && signbit(val)` check would FTZ-flush
        them to -0.0 on SM_103+ and falsely match the sentinel 0x80000000,
        deadlocking the poll loop.
      - Region B (bf16/fp16 -0.0): the write-side pre-multicast sanitizer
        must replace these with +0.0 before they reach the polled buffer.
      - Region C (positive subnormals 0x0001..0x007F): control — must never
        match the sentinel under any rule, on either side.
    """
    assert x_full.dtype in (torch.bfloat16, torch.float16)
    world_size = x_full.shape[0]
    flat = x_full.reshape(world_size, -1)
    if flat.shape[1] < 3 * 256:
        return

    def _u16_to_i16(u: int) -> int:
        return u if u < 0x8000 else u - 0x10000

    region_a = []
    for i in range(128):
        region_a.append(_u16_to_i16(0x0000))
        region_a.append(_u16_to_i16(0x8001 + (i % 0x7F)))
    region_b = [_u16_to_i16(0x8000)] * 256
    region_c = [_u16_to_i16(0x0001 + (i % 0x7F)) for i in range(256)]
    pattern_i16 = torch.tensor(region_a + region_b + region_c, dtype=torch.int16)

    for r in range(world_size):
        view16 = flat[r].view(torch.int16)
        view16[: pattern_i16.numel()] = pattern_i16


def prepare_test_data(
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    fusion: bool,
    inject_sentinel_patterns: bool = False,
):
    # Use torch.distributed for communication between ranks
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        x_full = torch.randn((world_size, seq_len, hidden_size), dtype=dtype)
        residual = torch.randn((seq_len, hidden_size), dtype=dtype)
        norm_weight = torch.randn((hidden_size,), dtype=dtype)
        if inject_sentinel_patterns:
            _inject_sentinel_trigger_patterns(x_full)
    else:
        x_full = None
        residual = None
        norm_weight = None

    # Use torch.distributed broadcast_object_list for Python object broadcasting
    data_list = [x_full, residual, norm_weight]
    dist.broadcast_object_list(data_list, src=0)
    x_full, residual, norm_weight = data_list

    x_full = x_full.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()

    x_local = x_full[rank, :, :]
    reference_output: Tuple[torch.Tensor, ...] = None
    if fusion:
        # Fused case: AllReduce + Residual Add + RMS Norm
        allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
        residual_out = allreduce_result + residual  # Add residual
        norm_out = rmsnorm(
            residual_out, norm_weight, torch.finfo(dtype).eps, enable_pdl=False
        )

        reference_output = (norm_out, residual_out)
    else:
        # Non-fused case: Only AllReduce
        allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
        reference_output = (allreduce_result,)
    return (x_local, residual, norm_weight), reference_output


def run_mnnvl_ar_full(
    monkeypatch,
    seq_lens: list[int],
    fusion: bool,
    dtype: torch.dtype,
    hidden_size: int,
    legacy_explicit_workspace_bytes: Optional[int] = None,
    legacy_api: bool = False,
    inject_sentinel_patterns: bool = False,
):
    """Core test logic for MNNVL AllReduce operations.

    Args:
        monkeypatch: pytest monkeypatch fixture
        seq_lens: List of sequence lengths to test
        fusion: Whether to test fused allreduce+rmsnorm or just allreduce
        dtype: Data type for tensors
        hidden_size: Hidden dimension size
        explicit_workspace_bytes: If provided, use this workspace size instead of default
    """

    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("MNNVL allreduce test requires at least one CUDA device per node")

    # Initialize torch.distributed (safe to call if already initialized)
    init_torch_distributed_from_mpi()

    # Get rank info from torch.distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        pytest.skip(f"This test requires at least 2 ranks, got {world_size}")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
        tp_size=world_size,
    )

    # Set CUDA device based on rank
    torch.cuda.set_device(mapping.local_rank)

    # Create TorchDistBackend for workspace creation (non-MPI based)
    comm_backend = TorchDistBackend()

    if mapping.local_rank == 0:
        print(
            f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {world_size} ranks"
        )
        print(
            f"[Node {mapping.node_rank}] Rank {rank} using GPU {torch.cuda.current_device()}"
        )
    eps = 1e-5
    torch.manual_seed(42 + rank)

    # Track if this rank failed
    rank_failed = False
    failure_message = ""

    try:
        if legacy_api:
            legacy_workspace, buffer_flags_mnnvl, max_num_elements_mnnvl = (
                trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(
                    mapping,
                    dtype,
                    comm_backend_for_handle_transfer=comm_backend,
                    buffer_size_in_bytes=legacy_explicit_workspace_bytes,
                )
            )

            multicast_ptr = legacy_workspace.mc_ptr
            buffer_ptrs_dev = legacy_workspace.uc_ptrs_dev
            unicast_ptr = legacy_workspace.uc_ptr_local

        else:
            workspace = trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace(
                mapping,
                max_num_tokens=max(seq_lens),
                hidden_dim=hidden_size,
                dtype=dtype,
                comm_backend=comm_backend,
            )

        test_data = []
        for seq_len in seq_lens:
            (x_local, residual, norm_weight), reference_output = prepare_test_data(
                seq_len,
                hidden_size,
                dtype,
                fusion,
                inject_sentinel_patterns=inject_sentinel_patterns,
            )
            test_data.append(
                (seq_len, x_local, residual, norm_weight, reference_output)
            )

        # Test each sequence length with the same workspace (reusing allocated buffers within this list)
        for seq_len, x, residual, norm_weight, reference_output in test_data:
            if rank == 0:
                print(
                    f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}, dtype={dtype}"
                )
            if legacy_api:
                row_linear_residual_norm_fusion_forward_legacy(
                    x,
                    residual,
                    norm_weight,
                    eps,
                    hidden_size,
                    dtype,
                    mapping,
                    fusion,
                    reference_output,
                    multicast_ptr,
                    buffer_ptrs_dev,
                    unicast_ptr,
                    max_num_elements_mnnvl,
                    buffer_flags_mnnvl,
                )
            else:
                row_linear_residual_norm_fusion_forward(
                    x,
                    residual,
                    norm_weight,
                    eps,
                    mapping,
                    fusion,
                    reference_output,
                    workspace,
                )

            # Synchronize before next test using torch.distributed barrier
            dist.barrier()

            print(
                f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}, dtype={dtype}"
            )

    except Exception as e:
        rank_failed = True
        failure_message = f"FAILED[rank={rank}]: seq_lens={seq_lens}, fusion={fusion}, dtype={dtype} failed: {e}"
        print(failure_message)
        print(traceback.format_exc())

        # Gather failure status from all ranks using torch.distributed
        all_failures = [None] * world_size
        dist.all_gather_object(all_failures, rank_failed)

        if any(all_failures):
            failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
            if rank == 0:
                print(f"Test failed on ranks: {failed_ranks}")

        # Re-raise the original exception so it can be caught by pytest.raises in negative tests
        raise

    finally:
        # Explicitly destroy workspace to avoid __del__ issues during Python shutdown
        if "workspace" in locals() and workspace is not None:
            workspace.destroy()
        if "legacy_workspace" in locals():
            legacy_workspace.destroy()

    # Final synchronization using torch.distributed barrier
    dist.barrier()


def _prepare_quant_test_data(
    seq_len: int, hidden_size: int, dtype: torch.dtype, eps: float
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        x_full = torch.randn((world_size, seq_len, hidden_size), dtype=dtype)
        residual = torch.randn((seq_len, hidden_size), dtype=dtype)
        norm_weight = torch.randn((hidden_size,), dtype=dtype)
    else:
        x_full = None
        residual = None
        norm_weight = None

    data_list = [x_full, residual, norm_weight]
    dist.broadcast_object_list(data_list, src=0)
    x_full, residual, norm_weight = data_list

    x_full = x_full.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()
    x_local = x_full[rank, :, :]

    allreduce_result = torch.sum(x_full, dim=0)
    residual_out = allreduce_result + residual
    norm_out = rmsnorm(residual_out, norm_weight, eps, enable_pdl=False)
    return (x_local, residual, norm_weight), (norm_out, residual_out)


def _assert_quant_close(
    quant_out: torch.Tensor,
    scale_out: Optional[torch.Tensor],
    ref_norm_out: torch.Tensor,
    quant_type: int,
    global_scale: torch.Tensor,
    layout_code: int,
):
    if quant_type == trtllm_mnnvl_ar.MNNVLQuantType.FP8:
        ref_dequant = dequant(
            fp8_quant(ref_norm_out, global_scale), global_scale, torch.float32
        )
        out_dequant = dequant(quant_out, global_scale, torch.float32)
        mismatch_tol = 0.002
    else:
        assert scale_out is not None
        ref_fp4, ref_sf = ref_fp4_quant(ref_norm_out, global_scale, block_size=16)
        if layout_code == comm.QuantizationSFLayout.SWIZZLED_128x4:
            padded_rows = ((ref_norm_out.shape[0] + 127) // 128) * 128
            padded_cols = ((ref_norm_out.shape[1] // 16 + 3) // 4) * 4
            scale_out = recover_swizzled_scales(
                scale_out.reshape(padded_rows, padded_cols),
                ref_norm_out.shape[0],
                ref_norm_out.shape[1],
                16,
            )
        ref_dequant = (
            ref_fp4.to(torch.float32)
            * ref_sf.repeat_interleave(16, dim=-1).to(torch.float32)
            * global_scale.to(torch.float32)
        )
        out_fp4 = cast_from_fp4(quant_out.view(torch.uint8)).to(ref_norm_out.device)
        out_dequant = (
            out_fp4.to(torch.float32)
            * scale_out.to(torch.float32).repeat_interleave(16, dim=-1)
            * global_scale.to(torch.float32)
        )
        mismatch_tol = 0.01

    rtol, atol = 0.05, 0.15
    diff = torch.abs(out_dequant - ref_dequant)
    max_diff = torch.maximum(
        torch.abs(ref_dequant) * rtol, torch.tensor(atol, device=ref_dequant.device)
    )
    mismatch_ratio = (diff > max_diff).sum().item() / out_dequant.numel()
    assert mismatch_ratio <= mismatch_tol, (
        f"Mismatch ratio {mismatch_ratio:.4%} exceeds {mismatch_tol:.4%} threshold"
    )


# Quant fusion reuses the same MNNVL allreduce and RMSNorm paths that the
# exhaustive non-quant tests cover below. Keep this matrix pruned to the axes
# that are quant-specific or easy to regress: FP8/NVFP4, oneshot/twoshot,
# fp16/bf16, norm_out optionality, FP4 scale layout, and representative
# sequence/hidden sizes for small, mixed, wide, and large-token offsets.
MNNVL_QUANT_TEST_CASES = [
    pytest.param(
        [1],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.float16,
        2880,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp8-oneshot-fp16-h2880-s1",
    ),
    pytest.param(
        [4],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.bfloat16,
        5120,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp8-twoshot-bf16-h5120-s4",
    ),
    pytest.param(
        [15],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.float16,
        7168,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp8-twoshot-fp16-h7168-s15",
    ),
    pytest.param(
        [27, 11, 24, 256],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.bfloat16,
        8192,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp8-oneshot-bf16-h8192-mixed",
    ),
    pytest.param(
        [127],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.float16,
        16384,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp8-oneshot-fp16-h16384-s127",
    ),
    pytest.param(
        [998, 2048],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.bfloat16,
        8192,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp8-twoshot-bf16-h8192-large",
    ),
    pytest.param(
        [1],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.float16,
        2880,
        [comm.QuantizationSFLayout.LINEAR, comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp4-oneshot-fp16-h2880-s1",
    ),
    pytest.param(
        [4],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.bfloat16,
        5120,
        [comm.QuantizationSFLayout.LINEAR, comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp4-twoshot-bf16-h5120-s4",
    ),
    pytest.param(
        [15],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.float16,
        7168,
        [comm.QuantizationSFLayout.LINEAR],
        id="fp4-twoshot-fp16-h7168-s15",
    ),
    pytest.param(
        [27, 11, 24, 256],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.bfloat16,
        8192,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp4-oneshot-bf16-h8192-mixed",
    ),
    pytest.param(
        [127],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.float16,
        16384,
        [comm.QuantizationSFLayout.LINEAR],
        id="fp4-oneshot-fp16-h16384-s127",
    ),
    pytest.param(
        [998, 2048],
        comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.bfloat16,
        8192,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="fp4-twoshot-bf16-h8192-large",
    ),
    pytest.param(
        [4],
        comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.bfloat16,
        2880,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="out-fp8-oneshot-bf16-h2880-s4",
    ),
    pytest.param(
        [127],
        comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.float16,
        7168,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="out-fp8-twoshot-fp16-h7168-s127",
    ),
    pytest.param(
        [4],
        comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
        torch.bfloat16,
        2880,
        [comm.QuantizationSFLayout.LINEAR],
        id="out-fp4-twoshot-bf16-h2880-s4",
    ),
    pytest.param(
        [127],
        comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        torch.float16,
        7168,
        [comm.QuantizationSFLayout.SWIZZLED_128x4],
        id="out-fp4-oneshot-fp16-h7168-s127",
    ),
]


@pytest.mark.parametrize(
    "seq_lens,pattern,strategy,dtype,hidden_size,layout_codes",
    MNNVL_QUANT_TEST_CASES,
)
def test_mnnvl_allreduce_quant_unified(
    seq_lens: list[int],
    pattern: int,
    strategy: trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy,
    dtype: torch.dtype,
    hidden_size: int,
    layout_codes: list[int],
):
    if torch.cuda.device_count() == 0:
        pytest.skip("MNNVL quant test requires CUDA")

    init_torch_distributed_from_mpi()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip(f"This test requires at least 2 ranks, got {world_size}")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=world_size,
    )
    torch.cuda.set_device(mapping.local_rank)
    is_fp4 = pattern in (
        comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
    )
    if is_fp4 and not _mnnvl_nvfp4_supported_on_all_ranks(
        torch.device("cuda", torch.cuda.current_device())
    ):
        pytest.skip("MNNVL NVFP4 quantization requires SM100+ on all ranks")

    comm_backend = TorchDistBackend()
    eps = torch.finfo(dtype).eps

    workspace = trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace(
        mapping,
        max_num_tokens=max(seq_lens),
        hidden_dim=hidden_size,
        dtype=dtype,
        comm_backend=comm_backend,
        buffer_size_in_bytes=trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace.get_required_buffer_size_bytes(
            world_size, max(seq_lens), hidden_size, dtype, strategy
        ),
    )
    try:
        for seq_len in seq_lens:
            (x_local, residual, norm_weight), (ref_norm, ref_residual) = (
                _prepare_quant_test_data(seq_len, hidden_size, dtype, eps)
            )
            global_scale = torch.ones(1, dtype=torch.float32, device=x_local.device)
            residual_out = torch.empty_like(x_local)
            has_norm_out = pattern in (
                comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
                comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
            )
            norm_out = torch.empty_like(x_local) if has_norm_out else None

            for layout_code in layout_codes:
                if is_fp4:
                    quant_type = trtllm_mnnvl_ar.MNNVLQuantType.NVFP4
                    quant_out = torch.empty(
                        (seq_len, hidden_size // 2),
                        dtype=torch.uint8,
                        device=x_local.device,
                    )
                    if layout_code == comm.QuantizationSFLayout.LINEAR:
                        scale_out = torch.empty(
                            (seq_len, hidden_size // 16),
                            dtype=torch.float8_e4m3fn,
                            device=x_local.device,
                        )
                    else:
                        scale_out = torch.empty(
                            _compute_swizzled_layout_sf_size(
                                seq_len, hidden_size // 16
                            ),
                            dtype=torch.float8_e4m3fn,
                            device=x_local.device,
                        )
                else:
                    quant_type = trtllm_mnnvl_ar.MNNVLQuantType.FP8
                    quant_out = torch.empty_like(x_local, dtype=torch.float8_e4m3fn)
                    scale_out = None

                result = comm.allreduce_fusion(
                    input=x_local,
                    workspace=workspace,
                    pattern=pattern,
                    residual_in=residual,
                    residual_out=residual_out,
                    norm_out=norm_out,
                    quant_out=quant_out,
                    scale_out=scale_out,
                    rms_gamma=norm_weight,
                    rms_eps=eps,
                    scale_factor=global_scale,
                    layout_code=layout_code,
                    use_oneshot=strategy
                    == trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
                )
                assert result is quant_out
                torch.testing.assert_close(
                    residual_out, ref_residual, rtol=0.05, atol=0.15
                )
                if has_norm_out:
                    assert norm_out is not None
                    torch.testing.assert_close(norm_out, ref_norm, rtol=0.05, atol=0.15)
                _assert_quant_close(
                    quant_out,
                    scale_out,
                    ref_norm,
                    quant_type,
                    global_scale,
                    layout_code,
                )
                dist.barrier()
    finally:
        workspace.destroy()
        dist.barrier()


@pytest.mark.parametrize(
    "strategy",
    [
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
        trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.TWOSHOT,
    ],
)
def test_mnnvl_nvfp4_rejects_fp32(
    strategy: trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy,
):
    if torch.cuda.device_count() == 0:
        pytest.skip("MNNVL quant test requires CUDA")

    init_torch_distributed_from_mpi()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip(f"This test requires at least 2 ranks, got {world_size}")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=world_size,
    )
    torch.cuda.set_device(mapping.local_rank)
    workspace = trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace(
        mapping,
        max_num_tokens=4,
        hidden_dim=2048,
        dtype=torch.float32,
        comm_backend=TorchDistBackend(),
        buffer_size_in_bytes=trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace.get_required_buffer_size_bytes(
            world_size, 4, 2048, torch.float32, strategy
        ),
    )
    try:
        x = torch.randn((4, 2048), dtype=torch.float32, device="cuda")
        residual = torch.randn_like(x)
        gamma = torch.randn((2048,), dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="NVFP4"):
            comm.allreduce_fusion(
                input=x,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
                residual_in=residual,
                rms_gamma=gamma,
                scale_factor=torch.ones(1, dtype=torch.float32, device="cuda"),
                use_oneshot=strategy
                == trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.ONESHOT,
            )
    finally:
        workspace.destroy()
        dist.barrier()


"""Test with default workspace size"""

# Multi-gpu test: mpirun -np 4 pytest tests/comm/test_trtllm_mnnvl_allreduce.py -vv -s
# Multi-node test:srun -A coreai_libraries_cudnn -N4 --container-image=<flashinfer_image> -J --mpi=pmix -- bash -c 'hostname && cd <path_to_flashinfer> && pip install -e . && python -m pytest tests/comm/test_trtllm_mnnvl_allreduce.py'


@pytest.mark.parametrize(
    "seq_lens",
    [[1], [4], [15], [27, 11, 24, 256], [127], [998, 2048]],
)
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2880, 5120, 7168, 8192, 16384])
def test_mnnvl_allreduce_refactored(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test MNNVL AllReduce with refactored API."""
    run_mnnvl_ar_full(
        monkeypatch, seq_lens, fusion, dtype, hidden_size, legacy_api=False
    )


@pytest.mark.parametrize("seq_lens", [[1], [4], [15], [27, 11, 24], [127]])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2048, 4096, 5120, 7168, 8192, 16384])
def test_mnnvl_allreduce_legacy(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test MNNVL AllReduce with legacy API."""
    explicit_workspace_bytes = 3 * 2 * dtype.itemsize * hidden_size * max(seq_lens)
    run_mnnvl_ar_full(
        monkeypatch,
        seq_lens,
        fusion,
        dtype,
        hidden_size,
        legacy_explicit_workspace_bytes=explicit_workspace_bytes,
        legacy_api=True,
    )


# Regression guard for the FTZ-induced Lamport-sentinel hang: the input is
# salted with bf16/fp16 negative subnormals (0x8001-0x807F), -0.0 (0x8000), and
# positive subnormals (0x0001-0x007F). With the bit-exact sentinel compare the
# kernel must complete; with a `val == 0.F && signbit(val)` compare on a
# `-ftz=true` build (the SM_103+ default) the poll loop would deadlock on the
# fp32 negative-subnormal patterns reformed from these bytes.
@pytest.mark.parametrize("seq_lens", [[4], [16, 64]])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2880, 8192])
def test_mnnvl_allreduce_sentinel_patterns(
    monkeypatch,
    seq_lens: list[int],
    fusion: bool,
    dtype: torch.dtype,
    hidden_size: int,
):
    """Regression test: inputs contain subnormals and -0.0 bit patterns that
    would falsely match the Lamport sentinel under an FP-equality compare."""
    run_mnnvl_ar_full(
        monkeypatch,
        seq_lens,
        fusion,
        dtype,
        hidden_size,
        legacy_api=False,
        inject_sentinel_patterns=True,
    )
