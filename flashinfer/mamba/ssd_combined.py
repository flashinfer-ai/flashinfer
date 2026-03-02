"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SSD (Structured State-Space Duality) Combined Forward Pass for Mamba2
=====================================================================

This module provides the combined forward pass for Mamba2 SSD, combining:
1. Triton kernel for cumsum computation
2. CuTe DSL kernel for the main SSD computation (Blackwell optimized)
"""

import functools
from typing import Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..triton.kernels.ssd_chunk_state import chunk_cumsum_fwd

from cutlass.base_dsl.compiler import GenerateLineInfo  # profiling


def _import_cutlass_modules():
    """Import CUTLASS modules (only when needed, as they require SM100)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    from .ssd_kernel import SSDKernel

    return {
        "cuda": cuda,
        "cutlass": cutlass,
        "cute": cute,
        "cutlass_torch": cutlass_torch,
        "from_dlpack": from_dlpack,
        "SSDKernel": SSDKernel,
    }


def _create_cutlass_tensor(
    shape, permute_order, dtype, dynamic_modes, cutlass_torch, from_dlpack
):
    """
    Create a tensor using the exact logic from mamba2_ssd.py to ensure compatibility.

    Args:
        shape: Base shape of the tensor (before permutation)
        permute_order: Order to permute dimensions
        dtype: CUTLASS dtype
        dynamic_modes: List of modes to mark as dynamic
        cutlass_torch: cutlass.torch module
        from_dlpack: cutlass.cute.runtime.from_dlpack function

    Returns:
        (cute_tensor, torch_tensor): The CuTe tensor wrapper and the underlying PyTorch tensor on GPU
    """
    # Allocate directly on GPU with the target dtype and apply permutation to
    # establish the stride pattern CUTLASS expects.
    torch_dtype = cutlass_torch.dtype(dtype)
    dst_tensor = torch.empty(*shape, dtype=torch_dtype, device="cuda").permute(
        permute_order
    )

    # Create CuTe tensor
    cute_tensor = from_dlpack(dst_tensor, assumed_align=16)
    for mode in dynamic_modes:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=mode, stride_order=dst_tensor.dim_order()
        )

    return cute_tensor, dst_tensor


class _SSDKernel:
    """
    Internal wrapper around CuTe DSL SSD kernel.

    Handles tensor layout conversions between PyTorch and CUTLASS formats.
    """

    def __init__(
        self,
        chunk_size: int,
        headdim: int,
        dstate: int,
        has_d: bool = True,
        d_has_hdim: bool = False,
        has_init_states: bool = False,
        has_varlen: bool = False,
        has_z: bool = False,
        seq_idx_dtype=None,
        io_dtype=None,
        cumsum_dtype=None,
        acc_dtype=None,
    ):
        """
        Initialize the kernel.

        Args:
            chunk_size: L - size of each chunk
            headdim: D - head dimension
            dstate: N - state dimension
            has_d: Whether to fuse D scaling (Y += X*D)
            d_has_hdim: If True, D is (headdim, nheads), else (1, nheads)
            has_init_states: Whether initial states are provided
            has_z: Whether to apply z gating (y *= z * sigmoid(z))
            seq_idx_dtype: Element type of seq_idx tensor (default: cutlass.Int32)
            io_dtype: Input/output dtype (default: cutlass.BFloat16)
            cumsum_dtype: Cumsum intermediate dtype (default: cutlass.Float32)
            acc_dtype: Accumulator dtype (default: cutlass.Float32)
        """
        self.modules = _import_cutlass_modules()
        cutlass = self.modules["cutlass"]

        self.chunk_size = chunk_size
        self.headdim = headdim
        self.dstate = dstate
        self.has_d = has_d
        self.d_has_hdim = d_has_hdim
        self.has_init_states = has_init_states

        self.io_dtype = io_dtype or cutlass.BFloat16
        self.cumsum_dtype = cumsum_dtype or cutlass.Float32
        self.acc_dtype = acc_dtype or cutlass.Float32
        seq_idx_cutlass_dtype = (
            cutlass.Int64 if seq_idx_dtype == "int64" else cutlass.Int32
        )

        # Create the kernel
        SSDKernel = self.modules["SSDKernel"]
        self.kernel = SSDKernel(
            self.io_dtype,
            self.cumsum_dtype,
            self.acc_dtype,
            chunk_size,
            headdim,
            dstate,
            has_d,
            d_has_hdim,
            has_init_states,
            has_varlen,
            has_z,
            seq_idx_cutlass_dtype,
        )

        self._compiled_kernel = None

    def run(
        self,
        x: torch.Tensor,
        dA_cumsum: torch.Tensor,
        dt_processed: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        init_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the SSD kernel with preprocessed inputs.

        Args:
            x: (batch, seqlen, nheads, headdim)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
            dt_processed: (batch, nheads, nchunks, chunk_size)
            B: (batch, seqlen, ngroups, dstate)
            C: (batch, seqlen, ngroups, dstate)
            out: (batch, nheads, headdim, nchunks, chunk_size) contiguous —
                 kernel writes via permuted (L, D, C, EH, B) view, zero-copy
            D: Optional (nheads, headdim) or (nheads,)
            init_states: Optional (batch, nheads, headdim, dstate)

        Returns:
            final_states: (batch, nheads, headdim, dstate)
        """
        cutlass = self.modules["cutlass"]
        cute = self.modules["cute"]
        cutlass_torch = self.modules["cutlass_torch"]
        from_dlpack = self.modules["from_dlpack"]

        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        chunk_size = self.chunk_size
        nchunks = seqlen // chunk_size

        torch.cuda.nvtx.range_push("run:tensor_prep")

        # Convert tensors to CUTLASS layout
        io_torch_dtype = cutlass_torch.dtype(self.io_dtype)

        # x: zero-copy (D, L, C, EH, B) with D stride 1 (N-major for MMA B operand)
        # Full reversal of (B, C, L, EH, D) gives monotonically increasing strides
        x_reshaped = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
        assert x.dtype == io_torch_dtype, (
            f"x dtype {x.dtype} doesn't match {io_torch_dtype}"
        )
        x_permuted = x_reshaped.permute(4, 2, 1, 3, 0)  # (D, L, C, EH, B)
        x_tensor = from_dlpack(x_permuted, assumed_align=16)
        for mode in [2, 3, 4]:
            x_tensor = x_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=x_permuted.dim_order()
            )

        # z: same layout as x — (D, L, C, EH, B) with D stride 1, zero-copy
        z_tensor = None
        if z is not None:
            z_reshaped = z.reshape(batch, nchunks, chunk_size, nheads, headdim)
            assert z.dtype == io_torch_dtype, (
                f"z dtype {z.dtype} doesn't match {io_torch_dtype}"
            )
            z_permuted = z_reshaped.permute(4, 2, 1, 3, 0)  # (D, L, C, EH, B)
            z_tensor = from_dlpack(z_permuted, assumed_align=16)
            for mode in [2, 3, 4]:
                z_tensor = z_tensor.mark_compact_shape_dynamic(
                    mode=mode, stride_order=z_permuted.dim_order()
                )

        # delta (dt_processed): (batch, nheads, nchunks, chunk_size) -> (chunk_size, nchunks, nheads, batch)
        # Need dtype conversion from float32 to bf16, but keep the non-contiguous permuted layout
        # so that mode 0 (L) has stride 1 as the kernel expects
        # First convert dtype (creates contiguous copy in original layout), then permute (zero-copy view)
        dt_permuted = dt_processed.to(io_torch_dtype).permute(
            3, 2, 1, 0
        )  # (L, C, EH, B)
        delta_tensor = from_dlpack(dt_permuted, assumed_align=16)
        # Mark dynamic modes: C=1, EH=2, B=3 (L=0 is static)
        for mode in [1, 2, 3]:
            delta_tensor = delta_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=dt_permuted.dim_order()
            )

        assert dA_cumsum.dtype == cutlass_torch.dtype(self.cumsum_dtype), (
            f"dA_cumsum dtype {dA_cumsum.dtype} doesn't match cumsum_dtype {self.cumsum_dtype}"
        )
        # cumsum_delta (dA_cumsum): same layout as delta
        # Zero-copy: permute is just a view, from_dlpack sees modes in permuted order
        cumsum_permuted = dA_cumsum.permute(
            3, 2, 1, 0
        )  # (L, C, EH, B) - zero-copy view
        cumsum_delta_tensor = from_dlpack(cumsum_permuted, assumed_align=16)
        # Mark dynamic modes: C=1, EH=2, B=3 (L=0 is static)
        for mode in [1, 2, 3]:
            cumsum_delta_tensor = cumsum_delta_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=cumsum_permuted.dim_order()
            )

        # B: zero-copy (L, N, C, G, B) with N stride 1 (K-major for MMA B operand)
        # PyTorch (B, C, L, G, N) permuted (2, 4, 1, 3, 0) gives (L, N, C, G, B) with N contiguous
        # Modes 0,1 are (N_mma, K_mma) = (L, N) matching make_tiled_tma_atom_B expectations
        B_reshaped = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert B.dtype == io_torch_dtype, (
            f"B dtype {B.dtype} doesn't match {io_torch_dtype}"
        )
        b_permuted = B_reshaped.permute(2, 4, 1, 3, 0)  # (L, N, C, G, B)
        b_tensor = from_dlpack(b_permuted, assumed_align=16)
        for mode in [2, 3, 4]:
            b_tensor = b_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=b_permuted.dim_order()
            )

        # C: zero-copy (L, N, C, G, B) with N stride 1 (K-major for MMA A operand)
        # PyTorch (B, C, L, G, N) permuted (2, 4, 1, 3, 0) gives (L, N, C, G, B) with N contiguous
        # Modes 0,1 are (M, K) = (L, N) matching make_tiled_tma_atom_A expectations
        C_reshaped = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert C.dtype == io_torch_dtype, (
            f"C dtype {C.dtype} doesn't match {io_torch_dtype}"
        )
        c_permuted = C_reshaped.permute(2, 4, 1, 3, 0)  # (L, N, C, G, B)
        c_tensor = from_dlpack(c_permuted, assumed_align=16)
        for mode in [2, 3, 4]:
            c_tensor = c_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=c_permuted.dim_order()
            )

        # D: (nheads,) -> CUTLASS (1, nheads) or (headdim, nheads)
        if self.has_d and D is not None:
            if self.d_has_hdim:
                # D is (nheads, headdim) -> (headdim, nheads)
                if D.dim() == 1:
                    D = D.unsqueeze(1).expand(-1, headdim)
                d_tensor, d_dst = _create_cutlass_tensor(
                    [nheads, headdim],
                    [1, 0],
                    self.io_dtype,
                    [1],
                    cutlass_torch,
                    from_dlpack,
                )
                d_dst.copy_(D.t().to(d_dst.dtype))
            else:
                # D is (nheads,) -> (1, nheads)
                if D.dim() == 2:
                    D = D[:, 0]
                d_tensor, d_dst = _create_cutlass_tensor(
                    [nheads, 1],
                    [1, 0],
                    self.io_dtype,
                    [1],
                    cutlass_torch,
                    from_dlpack,
                )
                d_dst.copy_(D.unsqueeze(0).to(d_dst.dtype))
        else:
            d_tensor = None

        if self.has_init_states and init_states is not None:
            # init_states_dst.copy_(init_states_reshaped.to(init_states_dst.dtype))
            assert init_states.dtype == cutlass_torch.dtype(self.io_dtype), (
                f"init_states dtype {init_states.dtype} doesn't match cumsum_dtype {self.io_dtype}"
            )
            # B, EH, D, N -> D, N, EH, B - zero-copy view
            init_states_reshaped = init_states.permute(3, 2, 1, 0)
            init_states_tensor = from_dlpack(init_states_reshaped, assumed_align=16)
            for mode in [2, 3]:  # EH and B are dynamic
                init_states_tensor = init_states_tensor.mark_compact_shape_dynamic(
                    mode=mode, stride_order=init_states_reshaped.dim_order()
                )
        else:
            init_states_tensor = None

        # Output tensor y: zero-copy from pre-allocated out
        # out is contiguous (B, EH, D, C, L), permute to (L, D, C, EH, B) for kernel
        assert out.dtype == io_torch_dtype, (
            f"out dtype {out.dtype} doesn't match {io_torch_dtype}"
        )
        out_permuted = out.permute(4, 2, 3, 1, 0)  # (L, D, C, EH, B)
        y_tensor = from_dlpack(out_permuted, assumed_align=16)
        for mode in [2, 3, 4]:
            y_tensor = y_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=out_permuted.dim_order()
            )

        # fstate: (headdim, dstate, nheads, batch_or_num_seqs)
        # For varlen, kernel indexes by seq_id so dim 0 must be num_seqs
        fstate_batch = init_states.shape[0] if init_states is not None else batch
        fstate_tensor, fstate_cutlass = _create_cutlass_tensor(
            [fstate_batch, nheads, headdim, dstate],
            [3, 2, 1, 0],
            self.io_dtype,
            [2, 3],
            cutlass_torch,
            from_dlpack,
        )

        # Get max active clusters
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(1)

        stream = cutlass.cuda.default_stream()

        # Varlen metadata: tiny 1D int32 tensors, just wrap via from_dlpack
        seq_idx_tensor = None
        chunk_indices_tensor = None
        chunk_offsets_tensor = None
        seq_chunk_cumsum_tensor = None
        num_seqs = 0
        if seq_idx is not None:
            seq_idx_tensor = from_dlpack(seq_idx, assumed_align=4)
        if chunk_indices is not None:
            chunk_indices_tensor = from_dlpack(chunk_indices, assumed_align=4)
        if chunk_offsets is not None:
            chunk_offsets_tensor = from_dlpack(chunk_offsets, assumed_align=4)

        # Compute per-sequence logical chunk ranges for varlen parallelization
        if (
            seq_idx is not None
            and chunk_indices is not None
            and chunk_offsets is not None
        ):
            # num_seqs from init_states batch dim (already on host, no sync)
            num_seqs = init_states.shape[0] if init_states is not None else 1
            # seq_id for each logical chunk (device ops, no host sync)
            seq_ids = seq_idx[
                0, chunk_indices.long() * chunk_size + chunk_offsets.long()
            ]
            counts = torch.zeros(num_seqs, dtype=torch.int32, device=seq_idx.device)
            counts.scatter_add_(
                0, seq_ids.int(), torch.ones_like(seq_ids, dtype=torch.int32)
            )
            seq_chunk_cumsum = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=seq_idx.device
            )
            torch.cumsum(counts.int(), dim=0, out=seq_chunk_cumsum[1:])
            seq_chunk_cumsum_tensor = from_dlpack(seq_chunk_cumsum, assumed_align=4)

        # Number of logical chunks (may differ from physical chunks for varlen)
        num_logical_chunks = (
            len(chunk_indices) if chunk_indices is not None else nchunks
        )

        torch.cuda.nvtx.range_pop()  # run:tensor_prep

        # Compile kernel if not already done
        if self._compiled_kernel is None:
            torch.cuda.nvtx.range_push("run:compile")
            # self._compiled_kernel = cute.compile(
            self._compiled_kernel = cute.compile[GenerateLineInfo(True)](
                self.kernel,
                x_tensor,
                cumsum_delta_tensor,
                delta_tensor,
                b_tensor,
                c_tensor,
                y_tensor,
                init_states_tensor,
                fstate_tensor,
                d_tensor,
                z_tensor,
                seq_idx_tensor,
                chunk_indices_tensor,
                chunk_offsets_tensor,
                seq_chunk_cumsum_tensor,
                num_logical_chunks,
                num_seqs,
                max_active_clusters,
                stream,
            )
            torch.cuda.nvtx.range_pop()  # run:compile

        # Run kernel
        torch.cuda.nvtx.range_push("run:launch")
        self._compiled_kernel(
            x_tensor,
            cumsum_delta_tensor,
            delta_tensor,
            b_tensor,
            c_tensor,
            y_tensor,
            init_states_tensor,
            fstate_tensor,
            d_tensor,
            z_tensor,
            seq_idx_tensor,
            chunk_indices_tensor,
            chunk_offsets_tensor,
            seq_chunk_cumsum_tensor,
            num_logical_chunks,
            num_seqs,
            stream,
        )
        torch.cuda.nvtx.range_pop()  # run:launch

        torch.cuda.nvtx.range_push("run:output_convert")
        # y was written directly into out (zero-copy)
        fstate_out = fstate_cutlass.permute(3, 2, 1, 0)
        torch.cuda.nvtx.range_pop()  # run:output_convert

        return fstate_out


@functools.cache
def _get_ssd_kernel(
    chunk_size: int,
    headdim: int,
    dstate: int,
    has_d: bool,
    d_has_hdim: bool,
    has_init_states: bool = False,
    has_varlen: bool = False,
    has_z: bool = False,
    seq_idx_dtype=None,
) -> _SSDKernel:
    """Get cached SSD kernel."""
    return _SSDKernel(
        chunk_size=chunk_size,
        headdim=headdim,
        dstate=dstate,
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_init_states=has_init_states,
        has_varlen=has_varlen,
        has_z=has_z,
        seq_idx_dtype=seq_idx_dtype,
    )


@flashinfer_api
def ssd_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_final_states: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    SSD (Structured State-Space Duality) combined forward pass for Mamba2.

    This function combines:
    1. Triton kernel for cumsum computation (_chunk_cumsum_fwd)
    2. CuTe DSL kernel for the main SSD computation (Blackwell optimized)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (batch, seqlen, nheads, headdim)
    dt : torch.Tensor
        Delta time tensor with shape (batch, seqlen, nheads)
    A : torch.Tensor
        A matrix with shape (nheads,) - should be negative for stability
    B : torch.Tensor
        B matrix with shape (batch, seqlen, ngroups, dstate)
    C : torch.Tensor
        C matrix with shape (batch, seqlen, ngroups, dstate)
    chunk_size : int
        Size of each chunk for processing (must divide seqlen)
    D : torch.Tensor, optional
        D vector with shape (nheads,) or (nheads, headdim)
    dt_bias : torch.Tensor, optional
        Optional dt bias with shape (nheads,)
    dt_softplus : bool
        Whether to apply softplus to dt
    dt_limit : tuple
        (min, max) limits for dt values
    initial_states : torch.Tensor, optional
        Optional (batch, nheads, headdim, dstate) or (num_seqs, nheads, headdim, dstate)
        when cu_seqlens is provided
    seq_idx : torch.Tensor, optional
        Sequence index tensor (batch, seqlen) mapping each position to its sequence ID.
        Required for variable-length sequence (continuous batching) support.
    chunk_indices : torch.Tensor, optional
        Int32 tensor mapping logical chunk index to physical chunk index.
        Required together with chunk_offsets when seq_idx and initial_states are provided.
    chunk_offsets : torch.Tensor, optional
        Int32 tensor mapping logical chunk index to offset within physical chunk.
        Required together with chunk_indices when seq_idx and initial_states are provided.
    cu_seqlens : torch.Tensor, optional
        Cumulative sequence lengths (num_seqs + 1,) for variable-length batching.
    out : torch.Tensor, optional
        Pre-allocated output tensor with shape (batch, seqlen, nheads, headdim).
        Must be contiguous and have the same dtype as x. If None, a new tensor
        is allocated internally.
    return_final_states : bool
        Whether to return final_states. If False, the second element of the
        return tuple is None. Default: True.

    Returns
    -------
    out : torch.Tensor
        Output tensor with shape (batch, seqlen, nheads, headdim)
    final_states : torch.Tensor or None
        Final state tensor with shape (batch, nheads, headdim, dstate),
        or (num_seqs, nheads, headdim, dstate) when varlen is active
        (seq_idx provided). None if return_final_states is False.

    Notes
    -----
    - Requires SM100+ (Blackwell) for the CuTe DSL SSD kernel.
    - The kernel is hardcoded for N=128 (dstate), L=128 (chunk_size), D=64 (headdim).
    - Difference from the Triton reference: the reference
      mamba_chunk_scan_combined has a separate return_varlen_states flag
      that triggers chunk_state_varlen — an extra Triton kernel that
      recomputes per-sequence final states from cu_seqlens. This is
      necessary because the reference's _state_passing_fwd only produces
      a single final_states of shape (batch, ...), which for batch=1
      varlen is just the state after the last sequence. Our fused CuTe
      kernel computes per-sequence final states on the fly (stored to
      fstate[seq_id] at every sequence transition), so final_states
      already has shape (num_seqs, nheads, headdim, dstate) when varlen
      is active. No separate varlen_states tensor or flag is needed —
      final_states serves both purposes.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape

    # In varlen (continuous batching) mode, the caller is expected to
    # independently pad each sequence to a chunk boundary so that no
    # physical chunk is shared by two sequences. This is the convention
    # adopted by vLLM (see https://github.com/vllm-project/vllm/pull/24683):
    # each sequence's new tokens are partitioned so that
    # len(previous_state) + len(new_partition) = chunk_size, and the
    # last chunk is zero-padded to a full chunk_size. As a result, the
    # total packed seqlen is always a multiple of chunk_size.
    # Supporting non-aligned total seqlen is not planned.
    assert seqlen % chunk_size == 0, (
        f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
    )

    # Validate varlen arguments
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen), (
            f"seq_idx shape {seq_idx.shape} doesn't match (batch={batch}, seqlen={seqlen})"
        )
        assert seq_idx.dtype in (torch.int32, torch.int64), (
            f"seq_idx must be int32 or int64, got {seq_idx.dtype}"
        )
    if chunk_indices is not None:
        assert chunk_indices.dim() == 1, (
            f"chunk_indices must be 1D, got {chunk_indices.dim()}D"
        )
        assert chunk_indices.dtype == torch.int32, (
            f"chunk_indices must be int32, got {chunk_indices.dtype}"
        )
    if chunk_offsets is not None:
        assert chunk_offsets.dim() == 1, (
            f"chunk_offsets must be 1D, got {chunk_offsets.dim()}D"
        )
        assert chunk_offsets.dtype == torch.int32, (
            f"chunk_offsets must be int32, got {chunk_offsets.dtype}"
        )
    if chunk_indices is not None and chunk_offsets is not None:
        assert chunk_indices.shape == chunk_offsets.shape, (
            f"chunk_indices and chunk_offsets must have the same shape, "
            f"got {chunk_indices.shape} vs {chunk_offsets.shape}"
        )
    # Validate pre-allocated output tensor
    # Kernel's native layout: (B, EH, D, C, L) contiguous
    nchunks = seqlen // chunk_size
    if out is not None:
        assert out.shape == (batch, nheads, headdim, nchunks, chunk_size), (
            f"out shape {out.shape} doesn't match "
            f"expected ({batch}, {nheads}, {headdim}, {nchunks}, {chunk_size})"
        )
        assert out.dtype == x.dtype, (
            f"out dtype {out.dtype} doesn't match x dtype {x.dtype}"
        )
        assert out.is_contiguous(), "out must be contiguous in (B, EH, D, C, L) layout"
        assert out.device == x.device, (
            f"out device {out.device} doesn't match x device {x.device}"
        )

    # Step 1: Compute cumsum using Triton kernel
    # dA_cumsum: (batch, nheads, nchunks, chunk_size)
    # dt_processed: (batch, nheads, nchunks, chunk_size) - after softplus/bias
    torch.cuda.nvtx.range_push("cumsum_fwd")
    dA_cumsum, dt_processed = chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    torch.cuda.nvtx.range_pop()

    # Step 2: Run SSD kernel
    has_d = D is not None
    d_has_hdim = has_d and D.dim() == 2
    has_init_state = True if initial_states is not None else False

    has_varlen = seq_idx is not None
    has_z = z is not None
    # Map torch seq_idx dtype to cutlass type for kernel compilation
    _seq_idx_dtype_map = {torch.int32: "int32", torch.int64: "int64"}
    seq_idx_dtype_key = (
        _seq_idx_dtype_map.get(seq_idx.dtype) if seq_idx is not None else None
    )
    torch.cuda.nvtx.range_push("get_ssd_kernel")
    kernel = _get_ssd_kernel(
        chunk_size=chunk_size,
        headdim=headdim,
        dstate=dstate,
        has_d=has_d,
        d_has_hdim=d_has_hdim,
        has_init_states=has_init_state,
        has_varlen=has_varlen,
        has_z=has_z,
        seq_idx_dtype=seq_idx_dtype_key,
    )
    torch.cuda.nvtx.range_pop()

    # Allocate output in kernel's native layout if not provided
    if out is None:
        out = torch.empty(
            batch,
            nheads,
            headdim,
            nchunks,
            chunk_size,
            dtype=x.dtype,
            device=x.device,
        )

    torch.cuda.nvtx.range_push("kernel.run")
    final_states = kernel.run(
        x=x,
        dA_cumsum=dA_cumsum,
        dt_processed=dt_processed,
        B=B,
        C=C,
        out=out,
        D=D,
        z=z,
        init_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    torch.cuda.nvtx.range_pop()

    # Zero-copy view: (B, EH, D, C, L) -> (B, C, L, EH, D) -> (B, seqlen, EH, D)
    out_view = out.permute(0, 3, 4, 1, 2).reshape(batch, seqlen, nheads, headdim)

    if return_final_states:
        return out_view, final_states
    else:
        return out_view, None
