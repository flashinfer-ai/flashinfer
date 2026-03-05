"""
Test for Mamba2 SSD (Structured State-Space Duality) chunk scan combined kernel.

Compares the FlashInfer SSD combined implementation (CuTe DSL + Triton)
against the production Triton implementation.
"""

import numpy as np
import pytest
import torch

# Import FlashInfer SSD combined API
from flashinfer.mamba import SSDCombined

# Import Triton reference for comparison
from .triton_reference.ssd_combined import (
    _mamba_chunk_scan_combined_fwd,
    mamba_chunk_scan_combined,
)
from .triton_reference.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from .triton_reference.ssd_state_passing import _state_passing_fwd
from .triton_reference.ssd_bmm import _bmm_chunk_fwd
from .triton_reference.ssd_chunk_scan import _chunk_scan_fwd
from einops import rearrange


def ssd_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    initial_states=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    out=None,
    return_final_states=True,
):
    """Test-local convenience wrapper around SSDCombined."""
    _, _, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    state_dtype = initial_states.dtype if initial_states is not None else x.dtype
    # This constructor is very heavy, so don't you dare use this ssd_combined_fwd wrapper in
    # prod. Do create an SSDCombined instance once and call run() on it multiple times instead.
    ssd = SSDCombined(
        chunk_size=chunk_size,
        nheads=nheads,
        headdim=headdim,
        dstate=dstate,
        ngroups=ngroups,
        has_d=D is not None,
        d_has_hdim=D is not None and D.dim() == 2,
        has_initial_states=initial_states is not None,
        has_varlen=seq_idx is not None,
        has_z=z is not None,
        state_dtype=state_dtype,
        seq_idx_dtype=seq_idx.dtype if seq_idx is not None else torch.int64,
    )
    return ssd.run(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        out=out,
        return_final_states=return_final_states,
    )


def _compute_varlen_metadata(cu_seqlens, chunk_size):
    """Compute seq_idx, chunk_indices, and chunk_offsets for variable-length sequences.

    Given cumulative sequence lengths and chunk_size, this function constructs
    the metadata needed for the varlen chunk scan kernels.

    Args:
        cu_seqlens: 1D tensor of cumulative sequence lengths, e.g. [0, 100, 356, 512]
            for 3 sequences of lengths 100, 256, 156.
        chunk_size: int, the chunk size used by the SSD kernel.

    Returns:
        seq_idx: (1, total_seqlen) tensor mapping each position to its sequence index.
        chunk_indices: 1D tensor mapping logical chunk index -> physical chunk index.
        chunk_offsets: 1D tensor mapping logical chunk index -> offset within that physical chunk.
    """
    total_seqlen = cu_seqlens[-1].item()

    # Build seq_idx: each position gets its sequence index
    seq_idx = torch.zeros(1, total_seqlen, dtype=torch.int32, device=cu_seqlens.device)
    num_seqs = len(cu_seqlens) - 1
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_idx[0, start:end] = i

    # Build chunk_indices and chunk_offsets by scanning physical chunks
    # and splitting at sequence boundaries
    nchunks = (total_seqlen + chunk_size - 1) // chunk_size
    chunk_indices_list = []
    chunk_offsets_list = []

    for phys_chunk in range(nchunks):
        chunk_start = phys_chunk * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_seqlen)

        # Find which sequences are present in this physical chunk
        chunk_seq_vals = seq_idx[0, chunk_start:chunk_end]

        # Detect transitions: first element always starts a segment
        prev_ids = torch.cat(
            [chunk_seq_vals[:1] - 1, chunk_seq_vals[:-1]]
        )  # force first to differ
        transitions = (chunk_seq_vals != prev_ids).nonzero(as_tuple=True)[0]

        for offset in transitions:
            chunk_indices_list.append(phys_chunk)
            chunk_offsets_list.append(offset.item())

    chunk_indices = torch.tensor(
        chunk_indices_list, dtype=torch.int32, device=cu_seqlens.device
    )
    chunk_offsets = torch.tensor(
        chunk_offsets_list, dtype=torch.int32, device=cu_seqlens.device
    )

    return seq_idx, chunk_indices, chunk_offsets


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
    ATOL = 7e-2
    RTOL = 7e-2
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


class TestChunkScanCombinedDHasHdim(TestChunkScanCombined):
    """Test chunk scan with 2D D tensor (nheads, headdim) triggering d_has_hdim=True."""

    @pytest.fixture(params=[1])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[1])
    def nchunks(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs with 2D D tensor."""
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
        # 2D D: (nheads, headdim) -> triggers d_has_hdim=True
        D = torch.randn(nheads, headdim, dtype=self.INPUT_DTYPE, device="cuda")
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

    def test_output_correctness(self, inputs, reference_output):
        """Test with 2D D (d_has_hdim=True)."""
        out_ref, final_states_ref = reference_output

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

        out_ref_cmp = out_ref.to(out_test.dtype)
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)

        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print("✓ [DHasHdim] Outputs match within tolerance")
        else:
            print("✗ [DHasHdim] Outputs do NOT match")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        if states_match:
            print("✓ [DHasHdim] Final states match within tolerance")
        else:
            print("✗ [DHasHdim] Final states do NOT match")
            self._print_mismatch_details(
                final_states_ref_cmp,
                final_states_test,
                "final_states",
                self.ATOL,
                self.RTOL,
            )

        assert out_match, "Output mismatch with d_has_hdim=True"
        assert states_match, "Final states mismatch with d_has_hdim=True"


class TestChunkScanCombinedDHasHdim1D(TestChunkScanCombined):
    """Test chunk scan with 1D D tensor but d_has_hdim=True (broadcast via TMA)."""

    @pytest.fixture(params=[1])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[1])
    def nchunks(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs with 1D D but d_has_hdim=True."""
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
        # 1D D: (nheads,) — will be broadcast to (headdim, nheads) by d_has_hdim path
        D = torch.randn(nheads, dtype=self.INPUT_DTYPE, device="cuda")
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

    def test_output_correctness(self, inputs, reference_output):
        """Test with 1D D and d_has_hdim=True (broadcast)."""
        out_ref, final_states_ref = reference_output

        # Construct SSDCombined with d_has_hdim=True explicitly,
        # even though D is 1D. This exercises the broadcast path.
        ssd = SSDCombined(
            chunk_size=inputs["chunk_size"],
            nheads=inputs["nheads"],
            headdim=inputs["headdim"],
            dstate=inputs["dstate"],
            ngroups=inputs["ngroups"],
            has_d=True,
            d_has_hdim=True,
        )
        out_test, final_states_test = ssd.run(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
        )

        out_ref_cmp = out_ref.to(out_test.dtype)
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)

        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print("✓ [DHasHdim1D] Outputs match within tolerance")
        else:
            print("✗ [DHasHdim1D] Outputs do NOT match")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        if states_match:
            print("✓ [DHasHdim1D] Final states match within tolerance")
        else:
            print("✗ [DHasHdim1D] Final states do NOT match")
            self._print_mismatch_details(
                final_states_ref_cmp,
                final_states_test,
                "final_states",
                self.ATOL,
                self.RTOL,
            )

        assert out_match, "Output mismatch with 1D D + d_has_hdim=True"
        assert states_match, "Final states mismatch with 1D D + d_has_hdim=True"


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
        states = _chunk_state_fwd(
            B, x, dt_processed, dA_cumsum, seq_idx=None, states_in_fp32=True
        )

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
            rearrange(t, "... (p n) -> ... p n", n=dstate)
            for t in [states, final_states]
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


class TestChunkScanCombinedVarlen:
    """Test CuTe DSL kernel with variable-length sequences (continuous batching).

    Compares FlashInfer CuTe DSL kernel (ssd_combined_fwd) against
    per-sequence Triton reference computation.

    Chunk-aligned sequences: all sequence lengths are multiples of chunk_size,
    so no physical chunk is shared by two sequences.

    Non-chunk-aligned sequences (mid-chunk boundaries) are tested separately
    and marked xfail until step 4.2 (chunk_size_limit) is implemented.
    """

    ATOL = 7e-2
    RTOL = 7e-2
    INPUT_DTYPE = torch.bfloat16

    @pytest.fixture(params=[8])
    def nheads(self, request):
        return request.param

    @pytest.fixture(params=[64])
    def headdim(self, request):
        return request.param

    @pytest.fixture(params=[128])
    def dstate(self, request):
        return request.param

    @pytest.fixture(params=[128])
    def chunk_size(self, request):
        return request.param

    @pytest.fixture(params=[8])
    def ngroups(self, request):
        return request.param

    @staticmethod
    def _make_inputs(nheads, headdim, dstate, chunk_size, ngroups, seq_lengths):
        """Create packed varlen inputs (batch=1) with initial states."""
        torch.manual_seed(42)

        num_seqs = len(seq_lengths)
        total_seqlen = sum(seq_lengths)

        # cu_seqlens
        cu_seqlens_list = [0]
        for sl in seq_lengths:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device="cuda")

        # Compute varlen metadata
        seq_idx, chunk_indices, chunk_offsets = _compute_varlen_metadata(
            cu_seqlens, chunk_size
        )

        # Packed tensors (batch=1)
        x = torch.randn(
            1, total_seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda"
        )
        dt = torch.randn(1, total_seqlen, nheads, dtype=torch.float32, device="cuda")
        A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
        B = torch.randn(
            1, total_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda"
        )
        C = torch.randn(
            1, total_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda"
        )
        D = torch.randn(nheads, dtype=torch.bfloat16, device="cuda")
        dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

        # Initial states: one per sequence
        initial_states = torch.randn(
            num_seqs, nheads, headdim, dstate, dtype=torch.bfloat16, device="cuda"
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
            "total_seqlen": total_seqlen,
            "nheads": nheads,
            "headdim": headdim,
            "dstate": dstate,
            "ngroups": ngroups,
            "initial_states": initial_states,
            "seq_idx": seq_idx,
            "chunk_indices": chunk_indices,
            "chunk_offsets": chunk_offsets,
            "cu_seqlens": cu_seqlens,
            "seq_lengths": seq_lengths,
            "num_seqs": num_seqs,
        }

    @staticmethod
    def _compute_per_sequence_reference(inputs):
        """Compute reference by running each sequence independently through Triton."""
        x = inputs["x"]
        dt = inputs["dt"]
        A = inputs["A"]
        B = inputs["B"]
        C = inputs["C"]
        D = inputs["D"]
        dt_bias = inputs["dt_bias"]
        chunk_size = inputs["chunk_size"]
        initial_states = inputs["initial_states"]
        cu_seqlens = inputs["cu_seqlens"]
        num_seqs = inputs["num_seqs"]
        dstate = inputs["dstate"]

        out_parts = []
        final_states_list = []

        for i in range(num_seqs):
            s = cu_seqlens[i].item()
            e = cu_seqlens[i + 1].item()

            x_i = x[:, s:e, :, :]
            dt_i = dt[:, s:e, :]
            B_i = B[:, s:e, :, :]
            C_i = C[:, s:e, :, :]
            init_i = initial_states[i : i + 1]

            dA_cumsum_i, dt_proc_i = _chunk_cumsum_fwd(
                dt_i, A, chunk_size, dt_bias=dt_bias, dt_softplus=True
            )
            states_i = _chunk_state_fwd(
                B_i, x_i, dt_proc_i, dA_cumsum_i, seq_idx=None, states_in_fp32=True
            )
            states_i, fstates_i = _state_passing_fwd(
                rearrange(states_i, "... p n -> ... (p n)"),
                dA_cumsum_i,
                initial_states=rearrange(init_i, "... p n -> ... (p n)"),
                seq_idx=None,
                chunk_size=chunk_size,
                out_dtype=C.dtype,
            )
            states_i, fstates_i = (
                rearrange(t, "... (p n) -> ... p n", n=dstate)
                for t in [states_i, fstates_i]
            )
            CB_i = _bmm_chunk_fwd(
                C_i, B_i, chunk_size, seq_idx=None, output_dtype=torch.float32
            )
            out_i = _chunk_scan_fwd(
                CB_i,
                x_i,
                dt_proc_i,
                dA_cumsum_i,
                C_i,
                states_i,
                D=D,
                z=None,
                seq_idx=None,
                initial_states=None,
            )

            out_parts.append(out_i)
            final_states_list.append(fstates_i)

        out_packed = torch.cat(out_parts, dim=1)
        final_states_packed = torch.cat(final_states_list, dim=0)
        return out_packed, final_states_packed

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

    def _check_outputs(
        self, out_ref, out_test, final_states_ref, final_states_test, tag
    ):
        """Compare output and final states, print details on mismatch."""
        out_ref_cmp = out_ref.to(out_test.dtype)
        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        if out_match:
            print(f"OK [{tag}] Outputs match")
        else:
            print(f"FAIL [{tag}] Outputs do NOT match")
            self._print_mismatch_details(
                out_ref_cmp, out_test, "output", self.ATOL, self.RTOL
            )

        fs_ref_cmp = final_states_ref.to(final_states_test.dtype)
        states_match = torch.allclose(
            fs_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )
        if states_match:
            print(f"OK [{tag}] Final states match")
        else:
            print(f"FAIL [{tag}] Final states do NOT match")
            self._print_mismatch_details(
                fs_ref_cmp, final_states_test, "final_states", self.ATOL, self.RTOL
            )

        assert out_match, f"[{tag}] Output mismatch"
        assert states_match, f"[{tag}] Final states mismatch"

    def _run_and_check(self, inputs, reference_output, tag):
        out_ref, final_states_ref = reference_output

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
            seq_idx=inputs["seq_idx"],
            chunk_indices=inputs["chunk_indices"],
            chunk_offsets=inputs["chunk_offsets"],
        )

        self._check_outputs(out_ref, out_test, final_states_ref, final_states_test, tag)

    def test_constant_seqlen(self, nheads, headdim, dstate, chunk_size, ngroups):
        """Single sequence through varlen code path (constant seq length, chunk-aligned)."""
        inputs = self._make_inputs(
            nheads, headdim, dstate, chunk_size, ngroups, [2 * chunk_size]
        )
        ref = self._compute_per_sequence_reference(inputs)
        self._run_and_check(inputs, ref, "Varlen constant-seqlen")

    def test_variable_seqlen(self, nheads, headdim, dstate, chunk_size, ngroups):
        """Multiple sequences with variable chunk-aligned lengths."""
        inputs = self._make_inputs(
            nheads,
            headdim,
            dstate,
            chunk_size,
            ngroups,
            [1 * chunk_size, 2 * chunk_size, 1 * chunk_size],
        )
        ref = self._compute_per_sequence_reference(inputs)
        self._run_and_check(inputs, ref, "Varlen variable-seqlen")


class TestChunkScanCombinedVarlenNonAligned:
    """Test CuTe DSL kernel with non-chunk-aligned variable-length sequences.

    Sequences don't align to chunk boundaries, so a physical chunk may
    contain data from two sequences. Requires step 4.2 (chunk_size_limit
    masking) — see CHUNK_SCAN_FEATURE_PLAN.md.
    """

    ATOL = 7e-2
    RTOL = 7e-2

    def test_output_correctness(self):
        """Test CuTe kernel with non-chunk-aligned varlen sequences."""
        nheads, headdim, dstate, chunk_size, ngroups = 8, 64, 128, 128, 8
        # Two sequences: 80 + 176 = 256, first chunk has boundary at position 80
        seq_lengths = [80, 176]
        inputs = TestChunkScanCombinedVarlen._make_inputs(
            nheads, headdim, dstate, chunk_size, ngroups, seq_lengths
        )
        ref = TestChunkScanCombinedVarlen._compute_per_sequence_reference(inputs)

        out_ref, final_states_ref = ref
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
            seq_idx=inputs["seq_idx"],
            chunk_indices=inputs["chunk_indices"],
            chunk_offsets=inputs["chunk_offsets"],
        )

        out_ref_cmp = out_ref.to(out_test.dtype)
        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        if not out_match:
            diff = (out_ref_cmp - out_test).abs()
            mm = ~torch.isclose(out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL)
            n = mm.sum().item()
            total = out_ref_cmp.numel()
            print(
                f"\nOutput mismatch: {n}/{total} ({100 * n / total:.2f}%), max_diff={diff.max():.6f}"
            )
            print(
                f"  has_inf={torch.isinf(out_test).any()}, has_nan={torch.isnan(out_test).any()}"
            )
            # Per-sequence breakdown
            cu = [0] + list(np.cumsum(seq_lengths))
            for i in range(len(seq_lengths)):
                s, e = cu[i], cu[i + 1]
                seq_mm = mm[0, s:e]
                seq_n = seq_mm.sum().item()
                seq_tot = seq_mm.numel()
                seq_diff = diff[0, s:e]
                print(
                    f"  Seq {i} [{s}:{e}]: {seq_n}/{seq_tot} ({100 * seq_n / seq_tot:.2f}%), max_diff={seq_diff.max():.6f}"
                )
            idxs = torch.nonzero(mm)
            for idx in idxs[:10]:
                t = tuple(idx.tolist())
                print(
                    f"  {t}: ref={out_ref_cmp[t]:.6f}, test={out_test[t]:.6f}, diff={diff[t]:.6f}"
                )

        fs_ref_cmp = final_states_ref.to(final_states_test.dtype)
        states_match = torch.allclose(
            fs_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )
        # Always print per-seq final states summary
        # fs_diff_all = (fs_ref_cmp - final_states_test).abs()
        # for seq_i in range(final_states_test.shape[0]):
        #     fs_mm_i = ~torch.isclose(
        #         fs_ref_cmp[seq_i],
        #         final_states_test[seq_i],
        #         atol=self.ATOL,
        #         rtol=self.RTOL,
        #     )
        #     fs_n_i = fs_mm_i.sum().item()
        #     fs_tot_i = fs_mm_i.numel()
        #     print(
        #         f"Final states seq {seq_i}: {fs_n_i}/{fs_tot_i} ({100 * fs_n_i / fs_tot_i:.2f}%) mismatch, max_diff={fs_diff_all[seq_i].max():.6f}"
        #     )
        if not states_match:
            fs_diff = (fs_ref_cmp - final_states_test).abs()
            fs_mm = ~torch.isclose(
                fs_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
            )
            fs_n = fs_mm.sum().item()
            fs_total = fs_ref_cmp.numel()
            print(
                f"\nFinal states mismatch: {fs_n}/{fs_total} ({100 * fs_n / fs_total:.2f}%), max_diff={fs_diff.max():.6f}"
            )
            print(
                f"  has_inf={torch.isinf(final_states_test).any()}, has_nan={torch.isnan(final_states_test).any()}"
            )
            for seq_i in range(final_states_test.shape[0]):
                seq_fs_mm = fs_mm[seq_i]
                seq_fs_n = seq_fs_mm.sum().item()
                seq_fs_tot = seq_fs_mm.numel()
                seq_fs_diff = fs_diff[seq_i]
                print(
                    f"  Seq {seq_i}: {seq_fs_n}/{seq_fs_tot} ({100 * seq_fs_n / seq_fs_tot:.2f}%), max_diff={seq_fs_diff.max():.6f}"
                )

        assert states_match, "Varlen non-aligned: final states mismatch"
        assert out_match, "Varlen non-aligned: output mismatch"


class TestChunkScanCombinedWithZ:
    """Test chunk scan with z gating: output *= z * sigmoid(z)."""

    ATOL = 7e-2
    RTOL = 7e-2
    INPUT_DTYPE = torch.bfloat16

    @pytest.fixture(params=[1, 2])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[8])
    def nheads(self, request):
        return request.param

    @pytest.fixture(params=[64])
    def headdim(self, request):
        return request.param

    @pytest.fixture(params=[128])
    def dstate(self, request):
        return request.param

    @pytest.fixture(params=[128])
    def chunk_size(self, request):
        return request.param

    @pytest.fixture(params=[1, 4])
    def nchunks(self, request):
        return request.param

    @pytest.fixture(params=[8])
    def ngroups(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, headdim, dstate, chunk_size, nchunks, ngroups):
        """Create test inputs with z tensor."""
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
        z = torch.randn(
            batch, seqlen, nheads, headdim, dtype=self.INPUT_DTYPE, device="cuda"
        )

        return {
            "x": x,
            "dt": dt,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "z": z,
            "dt_bias": dt_bias,
            "chunk_size": chunk_size,
        }

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output using Triton implementation with z gating.

        The Triton combined reference returns un-gated output when z is provided.
        We compute without z, then apply z gating manually: out *= silu(z).
        """
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
        # Apply z gating: out *= z * sigmoid(z) = out *= silu(z)
        z = inputs["z"].float()
        out = out.float() * (z * torch.sigmoid(z))
        return out, final_states

    def test_output_correctness(self, inputs, reference_output):
        """Test that FlashInfer kernel output matches Triton reference with z gating."""
        out_ref, final_states_ref = reference_output

        out_test, final_states_test = ssd_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=inputs["D"],
            z=inputs["z"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
        )

        out_ref_cmp = out_ref.to(out_test.dtype)
        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )

        if out_match:
            print(
                f"✓ [Z] Outputs match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print("✗ [Z] Outputs do NOT match within tolerance")
            ref_np = out_ref_cmp.detach().cpu().float().numpy()
            test_np = out_test.detach().cpu().float().numpy()
            mismatch_mask = ~np.isclose(ref_np, test_np, atol=self.ATOL, rtol=self.RTOL)
            n = np.sum(mismatch_mask)
            total = ref_np.size
            print(f"  Mismatched: {n}/{total} ({100 * n / total:.2f}%)")
            idxs = np.argwhere(mismatch_mask)
            for idx in idxs[:10]:
                t = tuple(int(i) for i in idx)
                print(
                    f"  {t}: ref={ref_np[t]:.6f}, test={test_np[t]:.6f}, "
                    f"diff={abs(ref_np[t] - test_np[t]):.6e}"
                )

        # Final states should be unaffected by z gating
        final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)
        states_match = torch.allclose(
            final_states_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )
        if states_match:
            print(
                f"✓ [Z] Final states match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print("✗ [Z] Final states do NOT match within tolerance")

        assert out_match, "[Z] Output mismatch"
        assert states_match, "[Z] Final states mismatch"


class TestChunkScanCombinedWithZVarlen:
    """Test z gating combined with variable-length sequences."""

    ATOL = 7e-2
    RTOL = 7e-2

    def test_output_correctness(self):
        """Test z gating with non-aligned varlen sequences."""
        nheads, headdim, dstate, chunk_size, ngroups = 8, 64, 128, 128, 8
        seq_lengths = [80, 176]
        inputs = TestChunkScanCombinedVarlen._make_inputs(
            nheads, headdim, dstate, chunk_size, ngroups, seq_lengths
        )

        # Add z tensor
        torch.manual_seed(123)
        z = torch.randn_like(inputs["x"])
        inputs["z"] = z

        # Compute reference per-sequence with z gating
        ref = self._compute_per_sequence_reference_with_z(inputs)
        out_ref, final_states_ref = ref

        out_test, final_states_test = ssd_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["chunk_size"],
            D=inputs["D"],
            z=inputs["z"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            initial_states=inputs["initial_states"],
            seq_idx=inputs["seq_idx"],
            chunk_indices=inputs["chunk_indices"],
            chunk_offsets=inputs["chunk_offsets"],
        )

        out_ref_cmp = out_ref.to(out_test.dtype)
        out_match = torch.allclose(
            out_ref_cmp, out_test, atol=self.ATOL, rtol=self.RTOL
        )
        if out_match:
            print("OK [Z+Varlen] Outputs match")
        else:
            diff = (out_ref_cmp - out_test).abs()
            print(f"FAIL [Z+Varlen] Outputs mismatch, max_diff={diff.max():.6f}")

        fs_ref_cmp = final_states_ref.to(final_states_test.dtype)
        states_match = torch.allclose(
            fs_ref_cmp, final_states_test, atol=self.ATOL, rtol=self.RTOL
        )
        if states_match:
            print("OK [Z+Varlen] Final states match")
        else:
            diff = (fs_ref_cmp - final_states_test).abs()
            print(f"FAIL [Z+Varlen] Final states mismatch, max_diff={diff.max():.6f}")

        assert out_match, "[Z+Varlen] Output mismatch"
        assert states_match, "[Z+Varlen] Final states mismatch"

    @staticmethod
    def _compute_per_sequence_reference_with_z(inputs):
        """Compute reference per-sequence with z gating."""
        x = inputs["x"]
        dt = inputs["dt"]
        A = inputs["A"]
        B = inputs["B"]
        C = inputs["C"]
        D = inputs["D"]
        z = inputs["z"]
        dt_bias = inputs["dt_bias"]
        chunk_size = inputs["chunk_size"]
        initial_states = inputs["initial_states"]
        cu_seqlens = inputs["cu_seqlens"]
        num_seqs = inputs["num_seqs"]
        dstate = inputs["dstate"]

        out_parts = []
        final_states_list = []

        for i in range(num_seqs):
            s = cu_seqlens[i].item()
            e = cu_seqlens[i + 1].item()

            x_i = x[:, s:e, :, :]
            dt_i = dt[:, s:e, :]
            B_i = B[:, s:e, :, :]
            C_i = C[:, s:e, :, :]
            z_i = z[:, s:e, :, :]
            init_i = initial_states[i : i + 1]

            dA_cumsum_i, dt_proc_i = _chunk_cumsum_fwd(
                dt_i, A, chunk_size, dt_bias=dt_bias, dt_softplus=True
            )
            states_i = _chunk_state_fwd(
                B_i, x_i, dt_proc_i, dA_cumsum_i, seq_idx=None, states_in_fp32=True
            )
            states_i, fstates_i = _state_passing_fwd(
                rearrange(states_i, "... p n -> ... (p n)"),
                dA_cumsum_i,
                initial_states=rearrange(init_i, "... p n -> ... (p n)"),
                seq_idx=None,
                chunk_size=chunk_size,
                out_dtype=C.dtype,
            )
            states_i, fstates_i = (
                rearrange(t, "... (p n) -> ... p n", n=dstate)
                for t in [states_i, fstates_i]
            )
            CB_i = _bmm_chunk_fwd(
                C_i, B_i, chunk_size, seq_idx=None, output_dtype=torch.float32
            )
            out_i = _chunk_scan_fwd(
                CB_i,
                x_i,
                dt_proc_i,
                dA_cumsum_i,
                C_i,
                states_i,
                D=D,
                z=None,
                seq_idx=None,
                initial_states=None,
            )
            # Apply z gating manually: out *= silu(z)
            z_f = z_i.float()
            out_i = out_i.float() * (z_f * torch.sigmoid(z_f))

            out_parts.append(out_i)
            final_states_list.append(fstates_i)

        out_packed = torch.cat(out_parts, dim=1)
        final_states_packed = torch.cat(final_states_list, dim=0)
        return out_packed, final_states_packed


def test_preallocated_output():
    """Test that passing a pre-allocated output tensor works correctly."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 128, 8, 64
    ngroups, dstate, chunk_size = 8, 128, 128
    dtype = torch.bfloat16

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    D = torch.randn(nheads, dtype=dtype, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

    # Run without pre-allocated output
    out_ref, _ = ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    # Run with pre-allocated output in kernel's native layout (B, EH, D, C, L)
    nchunks = seqlen // chunk_size
    out = torch.empty(
        batch, nheads, headdim, nchunks, chunk_size, dtype=dtype, device="cuda"
    )
    out_test, _ = ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
    )

    assert out_test.untyped_storage().data_ptr() == out.untyped_storage().data_ptr(), (
        "Returned tensor should share storage with the pre-allocated output"
    )
    assert torch.allclose(out_ref, out_test, atol=7e-2, rtol=7e-2), (
        "Pre-allocated output values don't match non-pre-allocated run"
    )


def test_return_final_states_flag():
    """Test that return_final_states controls whether final_states is returned."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 128, 8, 64
    ngroups, dstate, chunk_size = 8, 128, 128
    dtype = torch.bfloat16

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    D = torch.randn(nheads, dtype=dtype, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

    # With return_final_states=True (default), should return final_states
    out, final_states = ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        return_final_states=True,
    )
    assert out is not None
    assert final_states is not None
    assert final_states.shape == (batch, nheads, headdim, dstate)

    # With return_final_states=False, final_states should be None
    out2, final_states2 = ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        return_final_states=False,
    )
    assert out2 is not None
    assert final_states2 is None


class TestVarlenEndToEnd:
    """End-to-end test comparing ssd_combined_fwd against the full Triton
    mamba_chunk_scan_combined reference (including chunk_state_varlen).

    The reference decomposes into 5 separate Triton kernels + chunk_state_varlen
    for per-sequence final states. Our fused CuTe kernel computes everything
    inline. This test validates that final_states from our kernel matches the
    reference's varlen_states (the true per-sequence final states).
    """

    ATOL = 7e-2
    RTOL = 7e-2

    @staticmethod
    def _make_inputs(
        seq_lengths, chunk_size=128, nheads=8, headdim=64, dstate=128, ngroups=8
    ):
        torch.manual_seed(42)
        num_seqs = len(seq_lengths)
        total_seqlen = sum(seq_lengths)

        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lengths), dim=0).tolist()),
            dtype=torch.int32,
            device="cuda",
        )
        seq_idx, chunk_indices, chunk_offsets = _compute_varlen_metadata(
            cu_seqlens, chunk_size
        )

        dtype = torch.bfloat16
        x = torch.randn(1, total_seqlen, nheads, headdim, dtype=dtype, device="cuda")
        dt = torch.randn(1, total_seqlen, nheads, dtype=torch.float32, device="cuda")
        A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
        B = torch.randn(1, total_seqlen, ngroups, dstate, dtype=dtype, device="cuda")
        C = torch.randn(1, total_seqlen, ngroups, dstate, dtype=dtype, device="cuda")
        D = torch.randn(nheads, dtype=dtype, device="cuda")
        dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0
        initial_states = torch.randn(
            num_seqs, nheads, headdim, dstate, dtype=dtype, device="cuda"
        )

        return dict(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
        )

    def _run_and_check(self, seq_lengths, tag):
        inp = self._make_inputs(seq_lengths)

        # Reference: full Triton mamba_chunk_scan_combined with return_varlen_states
        # This calls chunk_state_varlen internally to get per-sequence final states.
        # return_varlen_states=True, return_final_states=False returns a single tensor.
        varlen_states_ref = mamba_chunk_scan_combined(
            inp["x"],
            inp["dt"],
            inp["A"],
            inp["B"],
            inp["C"],
            inp["chunk_size"],
            D=inp["D"],
            dt_bias=inp["dt_bias"],
            dt_softplus=True,
            initial_states=inp["initial_states"],
            seq_idx=inp["seq_idx"],
            chunk_indices=inp["chunk_indices"],
            chunk_offsets=inp["chunk_offsets"],
            cu_seqlens=inp["cu_seqlens"],
            return_final_states=False,
            return_varlen_states=True,
        )

        # FlashInfer: our final_states IS the per-sequence final states
        out_test, final_states_test = ssd_combined_fwd(
            inp["x"],
            inp["dt"],
            inp["A"],
            inp["B"],
            inp["C"],
            inp["chunk_size"],
            D=inp["D"],
            dt_bias=inp["dt_bias"],
            dt_softplus=True,
            initial_states=inp["initial_states"],
            seq_idx=inp["seq_idx"],
            chunk_indices=inp["chunk_indices"],
            chunk_offsets=inp["chunk_offsets"],
        )

        assert final_states_test is not None
        assert final_states_test.shape == varlen_states_ref.shape, (
            f"shape mismatch: {final_states_test.shape} vs {varlen_states_ref.shape}"
        )

        ref = varlen_states_ref.to(final_states_test.dtype)
        match = torch.allclose(ref, final_states_test, atol=self.ATOL, rtol=self.RTOL)
        if not match:
            diff = (ref - final_states_test).abs()
            print(f"FAIL [{tag}] max_diff={diff.max().item():.4f}")
        else:
            print(f"OK [{tag}] final_states match varlen_states_ref")

        assert match, f"[{tag}] final_states vs varlen_states mismatch"

    def test_chunk_aligned(self):
        """Chunk-aligned sequences: [128, 256, 128]."""
        self._run_and_check([128, 256, 128], "e2e chunk-aligned")

    def test_non_chunk_aligned(self):
        """Non-chunk-aligned sequences: [80, 176]."""
        self._run_and_check([80, 176], "e2e non-aligned")


def test_seq_idx_int64():
    """Test that seq_idx as int64 works (Python-side cast to int32)."""
    nheads, headdim, dstate, chunk_size, ngroups = 8, 64, 128, 128, 8
    seq_lengths = [1 * chunk_size, 2 * chunk_size, 1 * chunk_size]
    inputs = TestChunkScanCombinedVarlen._make_inputs(
        nheads, headdim, dstate, chunk_size, ngroups, seq_lengths
    )
    ref = TestChunkScanCombinedVarlen._compute_per_sequence_reference(inputs)
    out_ref, final_states_ref = ref

    # Cast seq_idx to int64 before calling the kernel
    seq_idx_int64 = inputs["seq_idx"].to(torch.int64)
    assert seq_idx_int64.dtype == torch.int64

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
        seq_idx=seq_idx_int64,
        chunk_indices=inputs["chunk_indices"],
        chunk_offsets=inputs["chunk_offsets"],
    )

    ATOL, RTOL = 7e-2, 7e-2
    out_ref_cmp = out_ref.to(out_test.dtype)
    assert torch.allclose(out_ref_cmp, out_test, atol=ATOL, rtol=RTOL), (
        "Output mismatch with int64 seq_idx"
    )
    fs_ref_cmp = final_states_ref.to(final_states_test.dtype)
    assert torch.allclose(fs_ref_cmp, final_states_test, atol=ATOL, rtol=RTOL), (
        "Final states mismatch with int64 seq_idx"
    )


def test_fp16_state_dtype():
    """Test that initial_states in fp16 works with bf16 io_dtype."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 128, 8, 64
    ngroups, dstate, chunk_size = 8, 128, 128

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    D = torch.randn(nheads, dtype=torch.bfloat16, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

    # initial_states in fp16 (different from io_dtype=bf16)
    initial_states = torch.randn(
        batch, nheads, headdim, dstate, dtype=torch.float16, device="cuda"
    )

    out_test, final_states_test = ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
    )

    # final_states must be in state_dtype (fp16), not io_dtype (bf16)
    assert final_states_test.dtype == torch.float16, (
        f"final_states dtype {final_states_test.dtype} must be float16"
    )

    # Compute reference with Triton sub-functions
    dA_cumsum, dt_processed = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True
    )
    states = _chunk_state_fwd(
        B, x, dt_processed, dA_cumsum, seq_idx=None, states_in_fp32=True
    )
    states, final_states_ref = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,
        initial_states=rearrange(initial_states, "... p n -> ... (p n)"),
        seq_idx=None,
        chunk_size=chunk_size,
        out_dtype=torch.float16,
    )
    states, final_states_ref = (
        rearrange(t, "... (p n) -> ... p n", n=dstate)
        for t in [states, final_states_ref]
    )
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=None, output_dtype=torch.float32)
    out_ref = _chunk_scan_fwd(
        CB,
        x,
        dt_processed,
        dA_cumsum,
        C,
        states,
        D=D,
        z=None,
        seq_idx=None,
        initial_states=None,
    )

    ATOL, RTOL = 7e-2, 7e-2
    out_ref_cmp = out_ref.to(out_test.dtype)
    assert torch.allclose(out_ref_cmp, out_test, atol=ATOL, rtol=RTOL), (
        "Output mismatch with fp16 state_dtype"
    )
    final_states_ref_cmp = final_states_ref.to(final_states_test.dtype)
    assert torch.allclose(
        final_states_ref_cmp, final_states_test, atol=ATOL, rtol=RTOL
    ), "Final states mismatch with fp16 state_dtype"


@pytest.mark.xfail(
    reason="state_dtype=float32 not yet supported (only float16/bfloat16)",
    strict=True,
)
def test_fp32_state_dtype():
    """Test that initial_states in fp32 is not yet supported."""
    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 1, 128, 8, 64
    ngroups, dstate, chunk_size = 8, 128, 128

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    D = torch.randn(nheads, dtype=torch.bfloat16, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0

    initial_states = torch.randn(
        batch, nheads, headdim, dstate, dtype=torch.float32, device="cuda"
    )

    out_test, final_states_test = ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
    )
