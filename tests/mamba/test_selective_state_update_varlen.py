"""
Varlen / speculative-decoding tests for selective_state_update.

Tests the new features for speculative decoding integration:
- dst_state_batch_indices: separate read/write state cache slots
- cu_seqlens: variable-length sequences (tokens flattened into batch dim)
- num_accepted_tokens: initial state selection for speculative decoding
- 2D state_batch_indices: (N, max_seqlen) index tensors
"""

import numpy as np
import pytest
import torch

import flashinfer

from .triton_reference.selective_state_update_varlen import (
    selective_state_update_varlen_triton,
)


PAD_SLOT_ID = -1


def _make_base_tensors(
    total_tokens,
    nheads,
    dim,
    dstate,
    ngroups,
    state_cache_size,
    input_dtype=torch.bfloat16,
    weight_dtype=torch.float32,
    matrixA_dtype=torch.float32,
    state_dtype=torch.bfloat16,
    device="cuda",
):
    """Create base input tensors for total_tokens in varlen (3D) layout."""
    x = torch.randn(total_tokens, nheads, dim, device=device, dtype=input_dtype)

    dt_base = torch.randn(total_tokens, nheads, device=device, dtype=weight_dtype)
    dt = dt_base.as_strided((total_tokens, nheads, dim), (nheads, 1, 0))

    A_base = -torch.rand(nheads, device=device, dtype=matrixA_dtype) - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    B = torch.randn(total_tokens, ngroups, dstate, device=device, dtype=input_dtype)
    C = torch.randn(total_tokens, ngroups, dstate, device=device, dtype=input_dtype)

    D_base = torch.randn(nheads, device=device, dtype=weight_dtype)
    D = D_base.as_strided((nheads, dim), (1, 0))

    dt_bias_base = torch.rand(nheads, device=device, dtype=weight_dtype) - 4.0
    dt_bias = dt_bias_base.as_strided((nheads, dim), (1, 0))

    state = torch.randn(
        state_cache_size, nheads, dim, dstate, device=device, dtype=state_dtype
    )

    return dict(state=state, x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=dt_bias)


def _assert_match(ref, test, name, atol=1e-3, rtol=1e-2):
    """Assert tensors match with detailed error reporting."""
    match = torch.allclose(ref, test, atol=atol, rtol=rtol)
    if match:
        print(f"  {name}: PASSED")
    else:
        ref_np = ref.detach().cpu().float().numpy()
        test_np = test.detach().cpu().float().numpy()
        mismatch = ~np.isclose(ref_np, test_np, atol=atol, rtol=rtol)
        num_mismatch = np.sum(mismatch)
        print(
            f"  {name}: FAILED ({num_mismatch}/{ref_np.size} elements differ, "
            f"max diff = {(ref - test).abs().max().item():.6e})"
        )
    assert match, f"{name} mismatch: max diff = {(ref - test).abs().max().item()}"


class TestSelectiveStateUpdateDstIndices:
    """Test dst_state_batch_indices in single-token (STP) path."""

    ATOL = 1e-3
    RTOL = 1e-2
    NHEADS = 64
    DIM = 64
    DSTATE = 128
    NGROUPS = 8
    STATE_CACHE_SIZE = 256

    @pytest.mark.parametrize("algorithm", ["simple"])
    @pytest.mark.parametrize("batch", [1, 4, 32, 64])
    def test_dst_different_from_src(self, batch, algorithm):
        """State is read from src slots and written to disjoint dst slots (STP path only)."""
        torch.manual_seed(42)
        tensors = _make_base_tensors(
            batch,
            self.NHEADS,
            self.DIM,
            self.DSTATE,
            self.NGROUPS,
            self.STATE_CACHE_SIZE,
        )
        out = torch.empty(
            batch, self.NHEADS, self.DIM, device="cuda", dtype=torch.bfloat16
        )

        perm = torch.randperm(self.STATE_CACHE_SIZE, device="cuda")
        src_indices = perm[:batch].to(torch.int32)
        dst_indices = perm[batch : 2 * batch].to(torch.int32)

        src_2d = src_indices.unsqueeze(1)
        dst_2d = dst_indices.unsqueeze(1)

        state_ref = tensors["state"].clone()
        out_ref = out.clone()
        selective_state_update_varlen_triton(
            state_ref,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_2d,
            dst_state_batch_indices=dst_2d,
            pad_slot_id=PAD_SLOT_ID,
            out=out_ref,
        )

        state_test = tensors["state"].clone()
        out_test = out.clone()
        flashinfer.mamba.selective_state_update(
            state_test,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_2d,
            dst_state_batch_indices=dst_2d,
            pad_slot_id=PAD_SLOT_ID,
            out=out_test,
            algorithm=algorithm,
        )

        _assert_match(out_ref, out_test, "output", self.ATOL, self.RTOL)
        _assert_match(
            state_ref[dst_indices.long()],
            state_test[dst_indices.long()],
            "dst_state",
            self.ATOL,
            self.RTOL,
        )

        src_orig = tensors["state"][src_indices.long()]
        src_after = state_test[src_indices.long()]
        assert torch.equal(src_orig, src_after), "Source state slots were modified"


class TestSelectiveStateUpdateDstIndices2D:
    """Test 2D state_batch_indices with shape (batch, 1)."""

    ATOL = 1e-3
    RTOL = 1e-2
    NHEADS = 64
    DIM = 64
    DSTATE = 128
    NGROUPS = 8
    STATE_CACHE_SIZE = 256

    @pytest.mark.parametrize("algorithm", ["simple"])
    @pytest.mark.parametrize("batch", [1, 16, 64])
    def test_2d_indices_seqlen1(self, batch, algorithm):
        """2D indices with max_seqlen=1 should behave identically to STP (STP path only)."""
        torch.manual_seed(42)
        tensors = _make_base_tensors(
            batch,
            self.NHEADS,
            self.DIM,
            self.DSTATE,
            self.NGROUPS,
            self.STATE_CACHE_SIZE,
        )
        out = torch.empty(
            batch, self.NHEADS, self.DIM, device="cuda", dtype=torch.bfloat16
        )

        perm = torch.randperm(self.STATE_CACHE_SIZE, device="cuda")
        src_indices = perm[:batch].to(torch.int32).unsqueeze(1)
        dst_indices = perm[batch : 2 * batch].to(torch.int32).unsqueeze(1)

        state_ref = tensors["state"].clone()
        out_ref = out.clone()
        selective_state_update_varlen_triton(
            state_ref,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_ref,
        )

        state_test = tensors["state"].clone()
        out_test = out.clone()
        flashinfer.mamba.selective_state_update(
            state_test,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_test,
            algorithm=algorithm,
        )

        _assert_match(out_ref, out_test, "output", self.ATOL, self.RTOL)

        dst_long = dst_indices.squeeze(1).long()
        _assert_match(
            state_ref[dst_long],
            state_test[dst_long],
            "dst_state",
            self.ATOL,
            self.RTOL,
        )


class TestSelectiveStateUpdateVarlen:
    """Test varlen (cu_seqlens) multi-token support."""

    ATOL = 1e-3
    RTOL = 1e-2
    NHEADS = 64
    DIM = 64
    DSTATE = 128
    NGROUPS = 8
    STATE_CACHE_SIZE = 512

    @pytest.mark.parametrize("algorithm", ["simple", "async_horizontal"])
    @pytest.mark.parametrize(
        "n_seqs,max_seqlen",
        [
            (1, 1),
            (1, 4),
            (4, 1),
            (4, 2),
            (4, 4),
            (16, 2),
            (16, 4),
        ],
    )
    def test_varlen_uniform(self, n_seqs, max_seqlen, algorithm):
        """All sequences have the same length."""
        torch.manual_seed(42)
        total_tokens = n_seqs * max_seqlen
        tensors = _make_base_tensors(
            total_tokens,
            self.NHEADS,
            self.DIM,
            self.DSTATE,
            self.NGROUPS,
            self.STATE_CACHE_SIZE,
        )

        cu_seqlens = torch.arange(
            0, total_tokens + 1, max_seqlen, device="cuda", dtype=torch.int32
        )

        perm = torch.randperm(self.STATE_CACHE_SIZE, device="cuda")
        src_indices = (
            perm[: n_seqs * max_seqlen].reshape(n_seqs, max_seqlen).to(torch.int32)
        )
        dst_indices = (
            perm[n_seqs * max_seqlen : 2 * n_seqs * max_seqlen]
            .reshape(n_seqs, max_seqlen)
            .to(torch.int32)
        )

        num_accepted = torch.ones(n_seqs, device="cuda", dtype=torch.int64)
        out = torch.empty(
            total_tokens, self.NHEADS, self.DIM, device="cuda", dtype=torch.bfloat16
        )

        state_ref = tensors["state"].clone()
        out_ref = out.clone()
        selective_state_update_varlen_triton(
            state_ref,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_ref,
            num_accepted_tokens=num_accepted,
            cu_seqlens=cu_seqlens,
        )

        state_test = tensors["state"].clone()
        out_test = out.clone()
        flashinfer.mamba.selective_state_update(
            state_test,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_test,
            num_accepted_tokens=num_accepted,
            cu_seqlens=cu_seqlens,
            cache_steps=max_seqlen,
            algorithm=algorithm,
        )

        _assert_match(out_ref, out_test, "output", self.ATOL, self.RTOL)

        for s in range(n_seqs):
            for t in range(max_seqlen):
                dst_slot = dst_indices[s, t].long().item()
                _assert_match(
                    state_ref[dst_slot],
                    state_test[dst_slot],
                    f"state[seq={s},token={t},slot={dst_slot}]",
                    self.ATOL,
                    self.RTOL,
                )

    @pytest.mark.parametrize("algorithm", ["simple", "async_horizontal"])
    @pytest.mark.parametrize("n_seqs", [4, 8])
    def test_varlen_variable_lengths(self, n_seqs, algorithm):
        """Sequences have different lengths (padded with PAD_SLOT_ID)."""
        max_seqlen = 6
        torch.manual_seed(42)

        seq_lens = torch.randint(1, max_seqlen + 1, (n_seqs,), device="cuda")
        total_tokens = seq_lens.sum().item()
        cu_seqlens = torch.zeros(n_seqs + 1, device="cuda", dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0).to(torch.int32)

        tensors = _make_base_tensors(
            total_tokens,
            self.NHEADS,
            self.DIM,
            self.DSTATE,
            self.NGROUPS,
            self.STATE_CACHE_SIZE,
        )

        src_indices = torch.full(
            (n_seqs, max_seqlen), PAD_SLOT_ID, device="cuda", dtype=torch.int32
        )
        dst_indices = torch.full(
            (n_seqs, max_seqlen), PAD_SLOT_ID, device="cuda", dtype=torch.int32
        )
        perm = torch.randperm(self.STATE_CACHE_SIZE, device="cuda").to(torch.int32)
        slot_offset = 0
        for s in range(n_seqs):
            sl = seq_lens[s].item()
            src_indices[s, :sl] = perm[slot_offset : slot_offset + sl]
            dst_indices[s, :sl] = perm[
                slot_offset + self.STATE_CACHE_SIZE // 2 : slot_offset
                + self.STATE_CACHE_SIZE // 2
                + sl
            ]
            slot_offset += sl

        num_accepted = torch.ones(n_seqs, device="cuda", dtype=torch.int64)
        out = torch.empty(
            total_tokens, self.NHEADS, self.DIM, device="cuda", dtype=torch.bfloat16
        )

        state_ref = tensors["state"].clone()
        out_ref = out.clone()
        selective_state_update_varlen_triton(
            state_ref,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_ref,
            num_accepted_tokens=num_accepted,
            cu_seqlens=cu_seqlens,
        )

        state_test = tensors["state"].clone()
        out_test = out.clone()
        flashinfer.mamba.selective_state_update(
            state_test,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_test,
            num_accepted_tokens=num_accepted,
            cu_seqlens=cu_seqlens,
            cache_steps=max_seqlen,
            algorithm=algorithm,
        )

        _assert_match(out_ref, out_test, "output", self.ATOL, self.RTOL)


class TestSelectiveStateUpdateNumAcceptedTokens:
    """Test num_accepted_tokens for initial state selection."""

    ATOL = 1e-3
    RTOL = 1e-2
    NHEADS = 64
    DIM = 64
    DSTATE = 128
    NGROUPS = 8
    STATE_CACHE_SIZE = 512

    @pytest.mark.parametrize("algorithm", ["simple", "async_horizontal"])
    @pytest.mark.parametrize("n_seqs", [4, 8, 16])
    @pytest.mark.parametrize("num_accepted_dtype", [torch.int32, torch.int64])
    def test_num_accepted_selects_initial_state(
        self, n_seqs, num_accepted_dtype, algorithm
    ):
        """num_accepted_tokens controls which state slot to read as initial."""
        max_seqlen = 4
        total_tokens = n_seqs * max_seqlen
        torch.manual_seed(42)

        tensors = _make_base_tensors(
            total_tokens,
            self.NHEADS,
            self.DIM,
            self.DSTATE,
            self.NGROUPS,
            self.STATE_CACHE_SIZE,
        )

        cu_seqlens = torch.arange(
            0, total_tokens + 1, max_seqlen, device="cuda", dtype=torch.int32
        )

        num_accepted = torch.randint(
            1, max_seqlen + 1, (n_seqs,), device="cuda", dtype=num_accepted_dtype
        )

        perm = torch.randperm(self.STATE_CACHE_SIZE, device="cuda").to(torch.int32)
        src_indices = perm[: n_seqs * max_seqlen].reshape(n_seqs, max_seqlen)
        dst_indices = perm[n_seqs * max_seqlen : 2 * n_seqs * max_seqlen].reshape(
            n_seqs, max_seqlen
        )

        out = torch.empty(
            total_tokens, self.NHEADS, self.DIM, device="cuda", dtype=torch.bfloat16
        )

        state_ref = tensors["state"].clone()
        out_ref = out.clone()
        selective_state_update_varlen_triton(
            state_ref,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_ref,
            num_accepted_tokens=num_accepted,
            cu_seqlens=cu_seqlens,
        )

        state_test = tensors["state"].clone()
        out_test = out.clone()
        flashinfer.mamba.selective_state_update(
            state_test,
            tensors["x"],
            tensors["dt"],
            tensors["A"],
            tensors["B"],
            tensors["C"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=src_indices,
            dst_state_batch_indices=dst_indices,
            pad_slot_id=PAD_SLOT_ID,
            out=out_test,
            num_accepted_tokens=num_accepted,
            cu_seqlens=cu_seqlens,
            cache_steps=max_seqlen,
            algorithm=algorithm,
        )

        _assert_match(out_ref, out_test, "output", self.ATOL, self.RTOL)

        for s in range(n_seqs):
            for t in range(max_seqlen):
                dst_slot = dst_indices[s, t].long().item()
                if dst_slot == PAD_SLOT_ID:
                    continue
                _assert_match(
                    state_ref[dst_slot],
                    state_test[dst_slot],
                    f"state[seq={s},token={t},num_accepted={num_accepted[s].item()}]",
                    self.ATOL,
                    self.RTOL,
                )
