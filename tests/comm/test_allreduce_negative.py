# Negative tests for unified AllReduce API
# Run with: mpirun -np <num_gpus> pytest tests/comm/test_allreduce_negative.py -vv -s

import pytest
import torch
from mpi4py import MPI

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar

from flashinfer.comm import (
    create_allreduce_fusion_workspace,
    allreduce_fusion,
    AllReduceFusionPattern,
    QuantizationSFLayout,
)


def setup_mpi_and_cuda():
    """Setup MPI and CUDA device for tests."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("Tests require at least one CUDA device per node")
    if world_size < 2:
        pytest.skip(f"Tests require at least 2 MPI ranks, got {world_size}")

    local_rank = rank % gpus_per_node
    torch.cuda.set_device(local_rank)

    return rank, world_size, gpus_per_node


class TestMNNVLUnsupportedPatterns:
    """Test that MNNVL backend properly rejects unsupported fusion patterns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup workspace for each test."""
        self.rank, self.world_size, self.gpus_per_node = setup_mpi_and_cuda()

        # Create MNNVL workspace
        self.workspace = create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=128,
            hidden_dim=2880,
            dtype=torch.float16,
            topology="single_node",
            gpus_per_node=self.gpus_per_node,
        )

        yield

        # Cleanup
        if self.workspace is not None:
            self.workspace.destroy()
        trtllm_mnnvl_ar.mpi_barrier()

    def _create_test_tensors(self, seq_len: int = 16, hidden_dim: int = 2880):
        """Create test tensors for allreduce operations."""
        input_tensor = torch.randn(
            seq_len, hidden_dim, dtype=torch.float16, device="cuda"
        )
        residual = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device="cuda")
        rms_gamma = torch.randn(hidden_dim, dtype=torch.float16, device="cuda")
        return input_tensor, residual, rms_gamma

    @pytest.mark.parametrize(
        "pattern",
        [
            AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
            AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
            AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
            AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        ],
    )
    def test_unsupported_quantization_patterns(self, pattern):
        """Test that MNNVL rejects quantization fusion patterns."""
        input_tensor, residual, rms_gamma = self._create_test_tensors()

        with pytest.raises(ValueError, match="does not support pattern"):
            allreduce_fusion(
                input=input_tensor,
                workspace=self.workspace,
                pattern=pattern,
                launch_with_pdl=True,
                residual_in=residual,
                rms_gamma=rms_gamma,
            )

    @pytest.mark.parametrize(
        "layout_code",
        [
            QuantizationSFLayout.LINEAR,
            QuantizationSFLayout.SWIZZLED_128x4,
            QuantizationSFLayout.SWIZZLED_8x4,
        ],
    )
    def test_layout_code_not_supported(self, layout_code):
        """Test that MNNVL rejects any layout_code specification."""
        input_tensor, residual, rms_gamma = self._create_test_tensors()

        # Test with kAllReduce pattern
        with pytest.raises(ValueError, match="does not support quantization fusion"):
            allreduce_fusion(
                input=input_tensor,
                workspace=self.workspace,
                pattern=AllReduceFusionPattern.kAllReduce,
                launch_with_pdl=True,
                layout_code=layout_code,
            )

        # Test with kARResidualRMSNorm pattern
        with pytest.raises(ValueError, match="does not support quantization fusion"):
            allreduce_fusion(
                input=input_tensor,
                workspace=self.workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_in=residual,
                rms_gamma=rms_gamma,
                layout_code=layout_code,
            )


class TestMNNVLMissingRequiredParameters:
    """Test that MNNVL backend properly validates required parameters."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup workspace for each test."""
        self.rank, self.world_size, self.gpus_per_node = setup_mpi_and_cuda()

        # Create MNNVL workspace
        self.workspace = create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=128,
            hidden_dim=2880,
            dtype=torch.float16,
            topology="single_node",
            gpus_per_node=self.gpus_per_node,
        )

        yield

        # Cleanup
        if self.workspace is not None:
            self.workspace.destroy()
        trtllm_mnnvl_ar.mpi_barrier()

    def test_rmsnorm_missing_residual_in(self):
        """Test that kARResidualRMSNorm requires residual_in."""
        input_tensor = torch.randn(16, 2880, dtype=torch.float16, device="cuda")
        rms_gamma = torch.randn(2880, dtype=torch.float16, device="cuda")

        with pytest.raises(ValueError, match="requires residual_in"):
            allreduce_fusion(
                input=input_tensor,
                workspace=self.workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                rms_gamma=rms_gamma,
                # residual_in is missing
            )

    def test_rmsnorm_missing_rms_gamma(self):
        """Test that kARResidualRMSNorm requires rms_gamma."""
        input_tensor = torch.randn(16, 2880, dtype=torch.float16, device="cuda")
        residual = torch.randn(16, 2880, dtype=torch.float16, device="cuda")

        with pytest.raises(ValueError, match="requires rms_gamma"):
            allreduce_fusion(
                input=input_tensor,
                workspace=self.workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_in=residual,
                # rms_gamma is missing
            )
