# Negative tests for unified AllReduce API
# Run with: mpirun -np <num_gpus> pytest tests/comm/test_allreduce_negative.py -vv -s

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar

from flashinfer.comm import (
    create_allreduce_fusion_workspace,
    allreduce_fusion,
    AllReduceFusionPattern,
    QuantizationSFLayout,
)

# Test helpers
from tests.test_helpers.comm import (
    setup_mpi_and_cuda,
    init_torch_distributed_from_mpi,
    cleanup_torch_distributed,
)


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


@pytest.mark.parametrize("backend", ["mnnvl", "trtllm"])
class TestBufferSizeSufficient:
    """Test is_buffer_size_sufficient method for different backends."""

    @pytest.fixture(autouse=True)
    def setup(self, backend):
        """Setup workspace with small buffer for testing."""
        self.backend = backend
        self.rank, self.world_size, self.gpus_per_node = setup_mpi_and_cuda()

        # Initialize torch.distributed for trtllm backend
        self.process_group = None
        if backend == "trtllm":
            init_torch_distributed_from_mpi()
            self.process_group = dist.group.WORLD

        # Create workspace with small max_token_num to test buffer limits
        self.max_token_num = 64
        self.hidden_dim = 2880
        self.dtype = torch.float16

        self.workspace = create_allreduce_fusion_workspace(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=self.max_token_num,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            topology="single_node",
            gpus_per_node=self.gpus_per_node,
            process_group=self.process_group,
        )

        yield

        # Cleanup
        if self.workspace is not None:
            self.workspace.destroy()
        if backend == "trtllm":
            cleanup_torch_distributed()
        trtllm_mnnvl_ar.mpi_barrier()

    def test_buffer_sufficient_for_smaller_size(self, backend):
        """Test that is_buffer_size_sufficient returns True for sizes within capacity."""
        # Use smaller size than max_token_num
        result = self.workspace.is_buffer_size_sufficient(
            tp_size=self.world_size,
            num_tokens=self.max_token_num // 2,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )
        assert result is True, (
            f"[{backend}] Buffer should be sufficient for smaller token count"
        )

    def test_buffer_sufficient_for_exact_size(self, backend):
        """Test that is_buffer_size_sufficient returns True for exact capacity."""
        result = self.workspace.is_buffer_size_sufficient(
            tp_size=self.world_size,
            num_tokens=self.max_token_num,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )
        assert result is True, (
            f"[{backend}] Buffer should be sufficient for exact max token count"
        )

    def test_buffer_insufficient_for_larger_size(self, backend):
        """Test that is_buffer_size_sufficient returns False for sizes exceeding capacity."""
        # Calculate the actual buffer capacity and use a size that definitely exceeds it
        elem_size = torch.tensor([], dtype=self.dtype).element_size()

        if backend == "mnnvl":
            # For MNNVL two-shot: buffer_size >= 2 * ceil(num_tokens/tp_size) * tp_size * hidden_dim * elem_size
            max_tokens_in_buffer = self.workspace.buffer_size_bytes // (
                2 * self.hidden_dim * elem_size
            )
        else:
            # For TRTLLM: use metadata to determine max capacity
            max_tokens_in_buffer = (
                self.workspace.metadata["max_token_num"]
                * self.workspace.metadata["hidden_dim"]
            ) // self.hidden_dim

        large_num_tokens = max_tokens_in_buffer * 10  # Use 10x the capacity

        result = self.workspace.is_buffer_size_sufficient(
            tp_size=self.world_size,
            num_tokens=large_num_tokens,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )
        assert result is False, (
            f"[{backend}] Buffer should be insufficient for {large_num_tokens} tokens "
            f"(buffer can hold ~{max_tokens_in_buffer})"
        )
