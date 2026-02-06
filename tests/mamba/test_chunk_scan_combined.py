"""
Test for Mamba2 SSD (Structured State-Space Duality) chunk scan combined kernel.

Compares the FlashInfer SSD combined implementation (CuTe DSL + Triton)
against the production Triton implementation.
"""

import numpy as np
import pytest
import torch

# Import FlashInfer SSD combined API
from flashinfer.mamba import ssd_combined_fwd

# Import Triton reference for comparison
from .triton_reference.ssd_combined import _mamba_chunk_scan_combined_fwd
from .triton_reference.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from .triton_reference.ssd_state_passing import _state_passing_fwd
from .triton_reference.ssd_bmm import _bmm_chunk_fwd
from .triton_reference.ssd_chunk_scan import _chunk_scan_fwd
from einops import rearrange


def is_blackwell_available():
    """Check if Blackwell GPU (SM100) is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10  # SM100 = Blackwell


# Skip all tests if not on Blackwell
pytestmark = pytest.mark.skipif(
    not is_blackwell_available(),
    reason="Blackwell GPU (SM100+) required for CuTe DSL Mamba2 SSD kernel",
)


class TestChunkScanCombined:
    """Test class for chunk scan combined kernel."""

    # Test configuration - slightly relaxed tolerance for bf16 precision
    ATOL = 5e-2
    RTOL = 5e-2
    INPUT_DTYPE = torch.bfloat16

    @pytest.fixture(params=[1, 2])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[8])  # nheads must be divisible by ngroups
    def nheads(self, request):
        return request.param

    @pytest.fixture(params=[64])  # Must match kernel's D
    def headdim(self, request):
        return request.param

    @pytest.fixture(
        params=[128]
    )  # Must match kernel's N (CUTLASS kernel is hardcoded for N=128)
    def dstate(self, request):
        return request.param

    @pytest.fixture(params=[128])  # Must match kernel's L
    def chunk_size(self, request):
        return request.param

    @pytest.fixture(params=[1, 4])  # Number of chunks
    def nchunks(self, request):
        return request.param

    @pytest.fixture(params=[8])  # ngroups divides nheads
    def ngroups(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs."""
        torch.manual_seed(42)

        seqlen = chunk_size * nchunks

        # x: (batch, seqlen, nheads, headdim)
        x = torch.randn(
            batch, seqlen, nheads, headdim, dtype=self.INPUT_DTYPE, device="cuda"
        )

        # dt: (batch, seqlen, nheads)
        dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")

        # A: (nheads,) - should be negative for stability
        A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0

        # B: (batch, seqlen, ngroups, dstate)
        B = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )

        # C: (batch, seqlen, ngroups, dstate)
        C = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )

        # D: (nheads, headdim) or (nheads,)
        D = torch.randn(nheads, dtype=self.INPUT_DTYPE, device="cuda")

        # dt_bias: (nheads,)
        dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

        return {
            "x": x,
            "dt": dt,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "dt_bias": dt_bias,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "nheads": nheads,
            "headdim": headdim,
            "dstate": dstate,
            "ngroups": ngroups,
        }

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output using Triton implementation."""
        out, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=inputs["D"],
            z=None,
            dt_bias=inputs["dt_bias"],
            initial_states=None,
            seq_idx=None,
            dt_softplus=True,
        )
        return out, final_states

    def _print_mismatch_details(self, ref, test, name, atol, rtol):
        """Print detailed mismatch analysis."""
        ref_np = ref.detach().cpu().float().numpy()
        test_np = test.detach().cpu().float().numpy()

        mismatch_mask = ~np.isclose(ref_np, test_np, atol=atol, rtol=rtol)
        num_mismatches = np.sum(mismatch_mask)
        total_elements = ref_np.size

        print(f"\nDetailed {name} mismatch analysis:")
        print(
            f"Number of mismatched elements: {num_mismatches} / {total_elements} "
            f"({100 * num_mismatches / total_elements:.2f}%)"
        )

        if num_mismatches > 0:
            mismatch_indices = np.argwhere(mismatch_mask)
            print(f"First few {name} mismatch locations (up to 10):")
            for idx in mismatch_indices[:10]:
                idx_tuple = tuple(int(i) for i in idx)
                ref_val = ref_np[idx_tuple]
                test_val = test_np[idx_tuple]
                diff = abs(ref_val - test_val)
                rel_diff = diff / (abs(ref_val) + 1e-8)
                print(
                    f"  Index {idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, "
                    f"diff={diff:.6e}, rel_diff={rel_diff:.6e}"
                )

    def test_output_correctness(self, inputs, reference_output):
        """Test that FlashInfer SSD combined kernel output matches Triton reference."""
        out_ref, final_states_ref = reference_output

        # Run FlashInfer SSD combined kernel
        out_test, final_states_test = ssd_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
        )

        # Compare outputs - cast to same dtype for comparison
        out_ref_cmp = out_ref.to(out_test.dtype)
        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print(
                f"✓ Outputs match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print("✗ Outputs do NOT match within tolerance")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        # Compare final states - cast to same dtype for comparison
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )

        if states_match:
            print(
                f"✓ Final states match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print("✗ Final states do NOT match within tolerance")
            self._print_mismatch_details(
                final_states_ref_cmp,
                final_states_test,
                "final_states",
                self.ATOL,
                self.RTOL,
            )

        assert out_match, "Output mismatch between CUTLASS and Triton"
        assert states_match, "Final states mismatch between CUTLASS and Triton"


class TestChunkScanCombinedNoD(TestChunkScanCombined):
    """Test chunk scan without D scaling."""

    @pytest.fixture(params=[1])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[1])
    def nchunks(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs without D."""
        torch.manual_seed(42)

        seqlen = chunk_size * nchunks

        x = torch.randn(
            batch, seqlen, nheads, headdim, dtype=self.INPUT_DTYPE, device="cuda"
        )
        dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
        A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
        B = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )
        C = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )
        dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

        return {
            "x": x,
            "dt": dt,
            "A": A,
            "B": B,
            "C": C,
            "D": None,
            "dt_bias": dt_bias,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "nheads": nheads,
            "headdim": headdim,
            "dstate": dstate,
            "ngroups": ngroups,
        }

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output using Triton implementation without D."""
        out, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=None,
            z=None,
            dt_bias=inputs["dt_bias"],
            initial_states=None,
            seq_idx=None,
            dt_softplus=True,
        )
        return out, final_states

    def test_output_correctness(self, inputs, reference_output):
        """Test without D scaling."""
        out_ref, final_states_ref = reference_output

        # Run FlashInfer SSD combined kernel without D
        out_test, final_states_test = ssd_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=None,
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
        )

        # Cast to same dtype for comparison
        out_ref_cmp = out_ref.to(out_test.dtype)
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)

        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print("✓ [NoD] Outputs match within tolerance")
        else:
            print("✗ [NoD] Outputs do NOT match")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        if states_match:
            print("✓ [NoD] Final states match within tolerance")
        else:
            print("✗ [NoD] Final states do NOT match")
            self._print_mismatch_details(
                final_states_ref_cmp,
                final_states_test,
                "final_states",
                self.ATOL,
                self.RTOL,
            )

        assert out_match
        assert states_match


class TestChunkScanCombinedWithInitialStates(TestChunkScanCombined):
    """Test chunk scan with initial states passed in."""

    @pytest.fixture(params=[1])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[1])
    def nchunks(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs with initial states."""
        torch.manual_seed(42)

        seqlen = chunk_size * nchunks

        x = torch.randn(
            batch, seqlen, nheads, headdim, dtype=self.INPUT_DTYPE, device="cuda"
        )
        dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
        A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
        B = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )
        C = torch.randn(
            batch, seqlen, ngroups, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )
        D = torch.randn(nheads, dtype=self.INPUT_DTYPE, device="cuda")
        dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

        # Initial states: (batch, nheads, headdim, dstate)
        initial_states = torch.randn(
            batch, nheads, headdim, dstate, dtype=self.INPUT_DTYPE, device="cuda"
        )

        return {
            "x": x,
            "dt": dt,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "dt_bias": dt_bias,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "nheads": nheads,
            "headdim": headdim,
            "dstate": dstate,
            "ngroups": ngroups,
            "initial_states": initial_states,
        }

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output using Triton sub-functions with initial states.

        Note: The combined API doesn't support simple batched initial_states without
        seq_idx, so we call the sub-functions directly.
        """
        x = inputs["x"]
        dt = inputs["dt"]
        A = inputs["A"]
        B = inputs["B"]
        C = inputs["C"]
        chunk_size = inputs["chunk_size"]
        D = inputs["D"]
        dt_bias = inputs["dt_bias"]
        initial_states = inputs["initial_states"]
        dstate = inputs["dstate"]

        # 1. Compute chunked cumsum of A * dt
        dA_cumsum, dt_processed = _chunk_cumsum_fwd(
            dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True
        )

        # 2. Compute the state for each intra-chunk
        states = _chunk_state_fwd(B, x, dt_processed, dA_cumsum, seq_idx=None, states_in_fp32=True)

        # 3. Compute the inter-chunk SSM recurrence with initial_states
        states, final_states = _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"),
            dA_cumsum,
            initial_states=(
                rearrange(initial_states, "... p n -> ... (p n)")
                if initial_states is not None
                else None
            ),
            seq_idx=None,
            chunk_size=chunk_size,
            out_dtype=C.dtype,
        )
        states, final_states = (
            rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
        )

        # 4. Compute batched matrix multiply for C_j^T B_i terms
        CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=None, output_dtype=torch.float32)

        # 5. Scan and compute the diagonal blocks (without initial_states param)
        out = _chunk_scan_fwd(
            CB,
            x,
            dt_processed,
            dA_cumsum,
            C,
            states,
            D=D,
            z=None,
            seq_idx=None,
            initial_states=None,  # Already handled by _state_passing_fwd
        )

        return out, final_states

    # @pytest.mark.xfail(reason="initial_states not yet implemented in ssd_combined_fwd")
    def test_output_correctness(self, inputs, reference_output):
        """Test with initial states."""
        out_ref, final_states_ref = reference_output

        # Run FlashInfer SSD combined kernel with initial states
        out_test, final_states_test = ssd_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            initial_states=inputs["initial_states"],
        )

        # Cast to same dtype for comparison
        out_ref_cmp = out_ref.to(out_test.dtype)
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)

        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print("✓ [InitialStates] Outputs match within tolerance")
        else:
            print("✗ [InitialStates] Outputs do NOT match")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        if states_match:
            print("✓ [InitialStates] Final states match within tolerance")
        else:
            print("✗ [InitialStates] Final states do NOT match")
            self._print_mismatch_details(
                final_states_ref_cmp,
                final_states_test,
                "final_states",
                self.ATOL,
                self.RTOL,
            )

        assert out_match, "Output mismatch with initial states"
        assert states_match, "Final states mismatch with initial states"
