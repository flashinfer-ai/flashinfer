"""Tests for SSM cache checkpointing (xAB fast-forward in STP kernel).

Verifies that the STP kernel's fast-forward loop produces correct output
when given K buffered past tokens. The reference is an fp32 Python simulation
that matches the kernel's semantics: load state once, process all tokens in
fp32 registers, output only for the last token.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.utils import get_compute_capability

from .utils import create_test_inputs


def _get_algorithms():
    major, _ = get_compute_capability(torch.device("cuda"))
    algos = ["simple"]
    if major >= 9:
        algos.extend(["vertical", "horizontal"])
    return algos


NGROUPS = 8
INPUT_DTYPE = torch.bfloat16
WEIGHT_DTYPE = torch.float32
MATRIX_A_DTYPE = torch.float32
STATE_DTYPE = torch.bfloat16

# fmt: off
_CHECKPOINT_PARAMS = [
    # (batch, nheads, dim, dstate, xab_length)
    (  16,    64,     64,  128,    1),   # K=1 (simplest case)
    (  16,    64,     64,  128,    4),   # K=4 (typical)
    (  16,    64,     64,  128,    8),   # K=8
    (   1,    64,     64,  128,    4),   # batch=1
    (  16,     8,     64,  128,    4),   # nheads=8
    (  16,    64,    128,  128,    4),   # dim=128
    (  16,    64,     64,   64,    4),   # dstate=64
]
# fmt: on


def _make_multi_token_inputs(batch, nheads, dim, dstate, total_tokens, seed=0):
    """Create inputs for total_tokens sequential STP calls."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ssm_state_cache_size = max(384, batch * 10)

    state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=STATE_DTYPE, device="cuda"
    )

    all_x = torch.randn(batch, total_tokens, nheads, dim, dtype=INPUT_DTYPE, device="cuda")
    all_dt_base = torch.randn(batch, total_tokens, nheads, dtype=WEIGHT_DTYPE, device="cuda")
    all_B = torch.randn(batch, total_tokens, NGROUPS, dstate, dtype=INPUT_DTYPE, device="cuda")
    all_C = torch.randn(batch, total_tokens, NGROUPS, dstate, dtype=INPUT_DTYPE, device="cuda")

    A_base = -torch.rand(nheads, dtype=MATRIX_A_DTYPE, device="cuda") - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    D_base = torch.randn(nheads, dtype=WEIGHT_DTYPE, device="cuda")
    D = D_base.as_strided((nheads, dim), (1, 0))

    dt_bias_base = torch.rand(nheads, dtype=WEIGHT_DTYPE, device="cuda") - 4.0
    dt_bias = dt_bias_base.as_strided((nheads, dim), (1, 0))

    slot_idx = torch.randperm(ssm_state_cache_size, dtype=torch.int64, device="cuda")[:batch]

    return {
        "state_cache": state_cache,
        "all_x": all_x,
        "all_dt_base": all_dt_base,
        "all_B": all_B,
        "all_C": all_C,
        "A": A,
        "D": D,
        "dt_bias": dt_bias,
        "dt_bias_base": dt_bias_base,
        "D_base": D_base,
        "A_base": A_base,
        "slot_idx": slot_idx,
    }


def _get_single_token_dt(dt_base_slice, nheads, dim):
    """Reshape a (batch, nheads) dt tensor to (batch, nheads, dim) with broadcast."""
    batch = dt_base_slice.shape[0]
    return dt_base_slice.as_strided(
        (batch, nheads, dim), (dt_base_slice.stride(0), 1, 0)
    )


def _reference_fast_forward(inputs, xab_length):
    """Pure fp32 reference matching the kernel's fast-forward semantics.

    Loads state once (cast to fp32), processes all K+1 tokens in fp32
    (no intermediate bf16 round-trip), and computes output only for the
    last token.
    """
    nheads = inputs["A_base"].shape[0]
    nheads_per_group = nheads // NGROUPS
    total_tokens = xab_length + 1

    h = inputs["state_cache"][inputs["slot_idx"]].float()  # (batch, nheads, dim, dstate)
    A_val = inputs["A_base"].float()           # (nheads,)
    dt_bias_val = inputs["dt_bias_base"].float()  # (nheads,)
    D_val = inputs["D_base"].float()           # (nheads,)
    group_idx = torch.arange(nheads, device=h.device) // nheads_per_group

    for t in range(total_tokens):
        x = inputs["all_x"][:, t].float()           # (batch, nheads, dim)
        dt_raw = inputs["all_dt_base"][:, t].float() # (batch, nheads)
        dt = dt_raw + dt_bias_val.unsqueeze(0)
        dt = F.softplus(dt, threshold=20.0)

        dA = torch.exp(A_val.unsqueeze(0) * dt)     # (batch, nheads)
        B = inputs["all_B"][:, t].float()            # (batch, ngroups, dstate)
        B_exp = B[:, group_idx]                      # (batch, nheads, dstate)

        h = h * dA[:, :, None, None] + B_exp[:, :, None, :] * dt[:, :, None, None] * x[:, :, :, None]

    C = inputs["all_C"][:, -1].float()
    C_exp = C[:, group_idx]
    out = (h * C_exp[:, :, None, :]).sum(dim=-1)
    out = out + D_val.unsqueeze(0).unsqueeze(-1) * inputs["all_x"][:, -1].float()

    return out, h


class TestCheckpointingCorrectness:
    """Test that STP with xAB buffer matches fp32 reference simulation."""

    ATOL = 1e-2
    RTOL = 1e-2

    @pytest.mark.parametrize("algorithm", _get_algorithms())
    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,xab_length", _CHECKPOINT_PARAMS
    )
    def test_fast_forward_matches_reference(
        self, batch, nheads, dim, dstate, xab_length, algorithm
    ):
        """STP with xAB buffer must match the fp32 reference simulation."""
        total_tokens = xab_length + 1
        inputs = _make_multi_token_inputs(batch, nheads, dim, dstate, total_tokens)

        # --- fp32 reference ---
        y_ref, state_ref_f32 = _reference_fast_forward(inputs, xab_length)

        # --- Kernel under test ---
        state_test = inputs["state_cache"].clone()

        xab_x = inputs["all_x"][:, :xab_length]
        xab_dt = inputs["all_dt_base"][:, :xab_length]
        xab_B = inputs["all_B"][:, :xab_length]

        x_cur = inputs["all_x"][:, -1]
        dt_cur = _get_single_token_dt(inputs["all_dt_base"][:, -1], nheads, dim)
        B_cur = inputs["all_B"][:, -1]
        C_cur = inputs["all_C"][:, -1]

        y_test = flashinfer.mamba.selective_state_update(
            state_test,
            x_cur, dt_cur, inputs["A"], B_cur, C_cur,
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
            algorithm=algorithm,
            xab_x=xab_x,
            xab_dt=xab_dt,
            xab_B=xab_B,
        )

        # --- Check output ---
        outputs_match = torch.allclose(y_ref.to(y_test.dtype), y_test, atol=self.ATOL, rtol=self.RTOL)
        if not outputs_match:
            diff = (y_ref.float() - y_test.float()).abs()
            print(f"Output max diff: {diff.max().item():.6e}, mean: {diff.mean().item():.6e}")
        assert outputs_match, (
            f"[{algorithm}] Output mismatch with xab_length={xab_length}"
        )

    @pytest.mark.parametrize("algorithm", _get_algorithms())
    def test_xab_length_zero_matches_vanilla(self, algorithm):
        """xab_length=0 (empty buffer) must be identical to vanilla STP."""
        batch, nheads, dim, dstate = 16, 64, 64, 128
        inputs = create_test_inputs(
            batch, nheads, dim, dstate, NGROUPS, INPUT_DTYPE,
            weight_dtype=WEIGHT_DTYPE, matrixA_dtype=MATRIX_A_DTYPE,
            state_dtype=STATE_DTYPE, generate_z=False, seed=0,
        )

        # Vanilla (no xAB)
        state_vanilla = inputs["state_cache"].clone()
        y_vanilla = flashinfer.mamba.selective_state_update(
            state_vanilla, inputs["x"], inputs["dt"], inputs["A"],
            inputs["B"], inputs["C"], D=inputs["D"],
            dt_bias=inputs["dt_bias"], dt_softplus=True,
            state_batch_indices=inputs["slot_idx"], pad_slot_id=-1,
            algorithm=algorithm,
        )

        # With empty xAB buffer (K=0)
        state_xab = inputs["state_cache"].clone()
        xab_x = torch.empty(batch, 0, nheads, dim, dtype=INPUT_DTYPE, device="cuda")
        xab_dt = torch.empty(batch, 0, nheads, dtype=WEIGHT_DTYPE, device="cuda")
        xab_B = torch.empty(batch, 0, NGROUPS, dstate, dtype=INPUT_DTYPE, device="cuda")

        y_xab = flashinfer.mamba.selective_state_update(
            state_xab, inputs["x"], inputs["dt"], inputs["A"],
            inputs["B"], inputs["C"], D=inputs["D"],
            dt_bias=inputs["dt_bias"], dt_softplus=True,
            state_batch_indices=inputs["slot_idx"], pad_slot_id=-1,
            algorithm=algorithm,
            xab_x=xab_x, xab_dt=xab_dt, xab_B=xab_B,
        )

        assert torch.equal(y_vanilla, y_xab), f"[{algorithm}] xab_length=0 differs from vanilla"
        assert torch.equal(
            state_vanilla[inputs["slot_idx"]],
            state_xab[inputs["slot_idx"]],
        ), f"[{algorithm}] state differs with xab_length=0"

    @pytest.mark.parametrize("algorithm", _get_algorithms())
    def test_disable_state_update_with_checkpointing(self, algorithm):
        """State must not change when disable_state_update=True, even with xAB buffer."""
        batch, nheads, dim, dstate, xab_length = 16, 64, 64, 128, 4
        total_tokens = xab_length + 1
        inputs = _make_multi_token_inputs(batch, nheads, dim, dstate, total_tokens)

        state_before = inputs["state_cache"].clone()

        xab_x = inputs["all_x"][:, :xab_length]
        xab_dt = inputs["all_dt_base"][:, :xab_length]
        xab_B = inputs["all_B"][:, :xab_length]

        x_cur = inputs["all_x"][:, -1]
        dt_cur = _get_single_token_dt(inputs["all_dt_base"][:, -1], nheads, dim)
        B_cur = inputs["all_B"][:, -1]
        C_cur = inputs["all_C"][:, -1]

        flashinfer.mamba.selective_state_update(
            inputs["state_cache"],
            x_cur, dt_cur, inputs["A"], B_cur, C_cur,
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
            disable_state_update=True,
            algorithm=algorithm,
            xab_x=xab_x, xab_dt=xab_dt, xab_B=xab_B,
        )

        assert torch.equal(state_before, inputs["state_cache"]), (
            f"[{algorithm}] State was modified despite disable_state_update=True"
        )
