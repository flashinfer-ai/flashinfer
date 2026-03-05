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

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda_drv
import torch
from cutlass.base_dsl.compiler import GenerateLineInfo  # profiling
from cutlass.cute.runtime import from_dlpack

from ..jit.mamba.seq_chunk_cumsum import gen_seq_chunk_cumsum_module
from ..triton.kernels.ssd_chunk_state import chunk_cumsum_fwd
from .ssd_kernel import SSDKernel


@functools.cache
def _get_seq_chunk_cumsum_module():
    """Get cached seq_chunk_cumsum JIT module."""
    return gen_seq_chunk_cumsum_module().build_and_load()


def _create_cutlass_tensor(shape, permute_order, dtype, dynamic_modes):
    """
    Create a tensor using the exact logic from mamba2_ssd.py to ensure compatibility.

    Args:
        shape: Base shape of the tensor (before permutation)
        permute_order: Order to permute dimensions
        dtype: CUTLASS dtype
        dynamic_modes: List of modes to mark as dynamic

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
    # For a contiguous tensor permuted by p, dim_order == argsort(p)
    cute_tensor = from_dlpack(dst_tensor, assumed_align=16)
    inv = [0] * len(permute_order)
    for i, p in enumerate(permute_order):
        inv[p] = i
    stride_order = tuple(inv)
    for mode in dynamic_modes:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=mode, stride_order=stride_order
        )

    return cute_tensor, dst_tensor


def _wrap_tensor(tensor, dynamic_modes, stride_order, align=16):
    """from_dlpack + mark_compact_shape_dynamic for all dynamic modes.

    stride_order: the dim_order of the tensor — for a contiguous tensor
    permuted by `perm`, this is `argsort(perm)` (the inverse permutation).
    Passing it explicitly avoids torch's expensive
    compute_elementwise_output_logical_to_physical_perm.
    """
    ct = from_dlpack(tensor, assumed_align=align)
    for m in dynamic_modes:
        ct = ct.mark_compact_shape_dynamic(mode=m, stride_order=stride_order)
    return ct


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
        seq_idx_dtype=torch.int64,
        io_dtype=None,
        state_dtype=None,
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
            io_dtype: I/O dtype for x, B, C, z, out (default: cutlass.BFloat16)
            state_dtype: State dtype for initial/final states (default: io_dtype)
            cumsum_dtype: Cumsum intermediate dtype (default: cutlass.Float32)
            acc_dtype: Accumulator dtype (default: cutlass.Float32)
        """
        self.chunk_size = chunk_size
        self.headdim = headdim
        self.dstate = dstate
        self.has_d = has_d
        self.d_has_hdim = d_has_hdim
        self.has_init_states = has_init_states

        self.io_dtype = io_dtype or cutlass.BFloat16
        self.state_dtype = state_dtype or self.io_dtype
        self.cumsum_dtype = cumsum_dtype or cutlass.Float32
        self.acc_dtype = acc_dtype or cutlass.Float32
        _seq_idx_dtype_map = {
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
        }
        seq_idx_cutlass_dtype = _seq_idx_dtype_map.get(
            seq_idx_dtype, cutlass.Int32
        )

        # Create the kernel
        self.kernel = SSDKernel(
            chunk_size,
            headdim,
            dstate,
            has_d,
            d_has_hdim,
            has_init_states,
            has_varlen,
            has_z,
            io_dtype=self.io_dtype,
            state_dtype=self.state_dtype,
            seq_idx_dtype=seq_idx_cutlass_dtype,
            cumsum_delta_dtype=self.cumsum_dtype,
            acc_dtype=self.acc_dtype,
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
        for mode in (2, 3, 4):
            x_tensor = x_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=(4, 2, 1, 3, 0)
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
            for mode in (2, 3, 4):
                z_tensor = z_tensor.mark_compact_shape_dynamic(
                    mode=mode, stride_order=(4, 2, 1, 3, 0)
                )

        # delta (dt_processed): (batch, nheads, nchunks, chunk_size) -> (chunk_size, nchunks, nheads, batch)
        # dt_processed is already in io_dtype from Triton cumsum kernel (dt_out_dtype param)
        assert dt_processed.dtype == io_torch_dtype, (
            f"dt_processed dtype {dt_processed.dtype} doesn't match "
            f"io_dtype {io_torch_dtype}"
        )
        dt_permuted = dt_processed.permute(
            3, 2, 1, 0
        )  # (L, C, EH, B)
        delta_tensor = from_dlpack(dt_permuted, assumed_align=16)
        # Mark dynamic modes: C=1, EH=2, B=3 (L=0 is static)
        for mode in (1, 2, 3):
            delta_tensor = delta_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=(3, 2, 1, 0)
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
        for mode in (1, 2, 3):
            cumsum_delta_tensor = cumsum_delta_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=(3, 2, 1, 0)
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
        for mode in (2, 3, 4):
            b_tensor = b_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=(4, 2, 0, 3, 1)
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
        for mode in (2, 3, 4):
            c_tensor = c_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=(4, 2, 0, 3, 1)
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
                )
                assert D.dtype == d_dst.dtype, (
                    f"D dtype {D.dtype} doesn't match io_dtype {d_dst.dtype}"
                )
                d_dst.copy_(D.t())
            else:
                # D is (nheads,) -> (1, nheads)
                if D.dim() == 2:
                    D = D[:, 0]
                d_tensor, d_dst = _create_cutlass_tensor(
                    [nheads, 1],
                    [1, 0],
                    self.io_dtype,
                    [1],
                )
                assert D.dtype == d_dst.dtype, (
                    f"D dtype {D.dtype} doesn't match io_dtype {d_dst.dtype}"
                )
                d_dst.copy_(D.unsqueeze(0))
        else:
            d_tensor = None

        if self.has_init_states and init_states is not None:
            state_torch_dtype = cutlass_torch.dtype(self.state_dtype)
            assert init_states.dtype == state_torch_dtype, (
                f"init_states dtype {init_states.dtype} doesn't match "
                f"state_dtype {state_torch_dtype}"
            )
            # B, EH, D, N -> D, N, EH, B - zero-copy view
            init_states_reshaped = init_states.permute(3, 2, 1, 0)
            init_states_tensor = from_dlpack(init_states_reshaped, assumed_align=16)
            for mode in (2, 3):  # EH and B are dynamic
                init_states_tensor = init_states_tensor.mark_compact_shape_dynamic(
                    mode=mode, stride_order=(3, 2, 1, 0)
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
        for mode in (2, 3, 4):
            y_tensor = y_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=(4, 3, 1, 2, 0)
            )

        # fstate: (headdim, dstate, nheads, batch_or_num_seqs)
        # For varlen, kernel indexes by seq_id so dim 0 must be num_seqs
        fstate_batch = init_states.shape[0] if init_states is not None else batch
        fstate_tensor, fstate_cutlass = _create_cutlass_tensor(
            [fstate_batch, nheads, headdim, dstate],
            [3, 2, 1, 0],
            self.state_dtype,
            [2, 3],
        )

        # Get max active clusters
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(1)

        # stream = cutlass.cuda.default_stream()
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

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
            num_seqs = init_states.shape[0] if init_states is not None else 1
            num_logical_chunks_local = len(chunk_indices)
            seq_chunk_cumsum = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=seq_idx.device
            )
            _get_seq_chunk_cumsum_module().seq_chunk_cumsum(
                seq_idx,
                chunk_indices,
                chunk_offsets,
                seq_chunk_cumsum,
                chunk_size,
                num_logical_chunks_local,
                num_seqs,
            )
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
    seq_idx_dtype=torch.int64,
    state_dtype=None,
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
        state_dtype=state_dtype,
    )


class SSDCombined:
    """Mamba2 SSD combined forward pass with cached host-side state.

    Caches compiled kernel, hardware info, and pre-allocated buffers
    to minimize host-side overhead on repeated calls.

    Dtype expectations (no runtime conversions — assert on mismatch):
        io_dtype (bf16):  x, B, C, z, D, out, dt_processed
        state_dtype (bf16): initial_states, final_states
        fp32 always:      A, dA_cumsum
        any (bf16/fp32):  dt, dt_bias (consumed by Triton cumsum preprocessor)

    Usage::

        ssd = SSDCombined(chunk_size=128, nheads=8, headdim=64,
                          dstate=128, ngroups=8)
        out, final_states = ssd.run(x, dt, A, B, C, D=D, dt_bias=dt_bias, ...)
    """

    def __init__(
        self,
        chunk_size: int,
        nheads: int,
        headdim: int,
        dstate: int,
        ngroups: int,
        io_dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype = torch.bfloat16,
        has_d: bool = True,
        d_has_hdim: bool = False,
        has_initial_states: bool = False,
        has_varlen: bool = False,
        has_z: bool = False,
        seq_idx_dtype=torch.int64,
    ):
        self.chunk_size = chunk_size
        self.nheads = nheads
        self.headdim = headdim
        self.dstate = dstate
        self.ngroups = ngroups
        self._has_d = has_d
        self._d_has_hdim = d_has_hdim
        self._has_init_states = has_initial_states
        self._has_varlen = has_varlen
        self._has_z = has_z
        self._state_torch_dtype = state_dtype

        # Resolve dtypes
        assert io_dtype == torch.bfloat16, (
            f"io_dtype must be bfloat16, got {io_dtype}"
        )
        self._io_dtype = cutlass.BFloat16
        _state_dtype_map = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
        }
        assert state_dtype in _state_dtype_map, (
            f"state_dtype must be bfloat16 or float16, got {state_dtype}"
        )
        self._state_dtype = _state_dtype_map[state_dtype]
        self._cumsum_dtype = cutlass.Float32
        self._acc_dtype = cutlass.Float32
        self._io_torch_dtype = cutlass_torch.dtype(self._io_dtype)

        # Create kernel object
        _seq_idx_dtype_map = {
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
        }
        seq_idx_cutlass_dtype = _seq_idx_dtype_map.get(
            seq_idx_dtype, cutlass.Int32
        )
        self._kernel_obj = SSDKernel(
            chunk_size,
            headdim,
            dstate,
            has_d,
            d_has_hdim,
            has_initial_states,
            has_varlen,
            has_z,
            io_dtype=self._io_dtype,
            state_dtype=self._state_dtype,
            seq_idx_dtype=seq_idx_cutlass_dtype,
            cumsum_delta_dtype=self._cumsum_dtype,
            acc_dtype=self._acc_dtype,
        )
        self._compiled_kernel = None

        # Lazily cached on first run() (needs active CUDA context)
        self._max_active_clusters = None

        # Pre-allocated buffer caches (lazily initialized on first run)
        self._fstate_shape = None
        self._fstate_cute = None
        self._fstate_torch = None

        self._d_ptr = None
        self._d_shape = None
        self._d_cute = None
        self._d_torch = None

        self._seq_cumsum_size = 0
        self._seq_cumsum_buf = None

    # -- buffer cache helpers --------------------------------------------------

    def _get_or_alloc_fstate(self, fstate_batch):
        shape = (fstate_batch, self.nheads, self.headdim, self.dstate)
        if self._fstate_shape != shape:
            self._fstate_cute, self._fstate_torch = _create_cutlass_tensor(
                list(shape),
                [3, 2, 1, 0],
                self._state_dtype,
                [2, 3],
            )
            self._fstate_shape = shape
        return self._fstate_cute, self._fstate_torch

    def _get_or_wrap_d(self, D):
        if D is None:
            return None
        d_ptr = D.data_ptr()
        d_shape = D.shape
        if self._d_ptr == d_ptr and self._d_shape == d_shape:
            return self._d_cute

        if self._d_has_hdim:
            d_val = D
            if D.dim() == 1:
                d_val = D.unsqueeze(1).expand(-1, self.headdim)
            self._d_cute, self._d_torch = _create_cutlass_tensor(
                [self.nheads, self.headdim],
                [1, 0],
                self._io_dtype,
                [1],
            )
            assert d_val.dtype == self._d_torch.dtype, (
                f"D dtype {d_val.dtype} doesn't match "
                f"io_dtype {self._d_torch.dtype}"
            )
            self._d_torch.copy_(d_val.t())
        else:
            d_val = D
            if D.dim() == 2:
                d_val = D[:, 0]
            self._d_cute, self._d_torch = _create_cutlass_tensor(
                [self.nheads, 1],
                [1, 0],
                self._io_dtype,
                [1],
            )
            assert d_val.dtype == self._d_torch.dtype, (
                f"D dtype {d_val.dtype} doesn't match "
                f"io_dtype {self._d_torch.dtype}"
            )
            self._d_torch.copy_(d_val.unsqueeze(0))

        self._d_ptr = d_ptr
        self._d_shape = d_shape
        return self._d_cute

    def _get_or_alloc_seq_cumsum(self, num_seqs, device):
        size = num_seqs + 1
        if self._seq_cumsum_size != size:
            self._seq_cumsum_buf = torch.zeros(
                size,
                dtype=torch.int32,
                device=device,
            )
            self._seq_cumsum_size = size
        else:
            self._seq_cumsum_buf.zero_()
        return self._seq_cumsum_buf

    # -- main entry point ------------------------------------------------------

    def run(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        dt_softplus: bool = False,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        initial_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_offsets: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run SSD combined forward pass.

        Parameters match ``ssd_combined_fwd`` — see its docstring for details.
        """
        chunk_size = self.chunk_size

        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        nchunks = seqlen // chunk_size

        assert seqlen % chunk_size == 0, (
            f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
        )

        # A is always fp32
        assert A.dtype == torch.float32, f"A must be float32, got {A.dtype}"

        # Validate varlen arguments
        if seq_idx is not None:
            assert seq_idx.shape == (batch, seqlen), (
                f"seq_idx shape {seq_idx.shape} doesn't match "
                f"(batch={batch}, seqlen={seqlen})"
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

        if out is not None:
            assert out.shape == (batch, nheads, headdim, nchunks, chunk_size), (
                f"out shape {out.shape} doesn't match "
                f"expected ({batch}, {nheads}, {headdim}, {nchunks}, {chunk_size})"
            )
            assert out.dtype == x.dtype, (
                f"out dtype {out.dtype} doesn't match x dtype {x.dtype}"
            )
            assert out.is_contiguous(), (
                "out must be contiguous in (B, EH, D, C, L) layout"
            )

        # -- Step 1: Cumsum (Triton) ------------------------------------------
        # Triton kernel outputs dt_processed directly in io_dtype (bf16),
        # avoiding a separate float32->bf16 copy.
        dA_cumsum, dt_processed = chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            dt_out_dtype=self._io_torch_dtype,
        )

        # -- Step 2: Allocate output if needed --------------------------------
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

        # -- Step 3: Wrap input tensors for CuTe DSL --------------------------
        io_torch_dtype = self._io_torch_dtype

        # x: (B, seqlen, EH, D) -> (D, L, C, EH, B)
        x_reshaped = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
        assert x.dtype == io_torch_dtype, (
            f"x dtype {x.dtype} doesn't match {io_torch_dtype}"
        )
        x_tensor = _wrap_tensor(
            x_reshaped.permute(4, 2, 1, 3, 0),
            (2, 3, 4),
            (4, 2, 1, 3, 0),
        )

        # z: same layout as x
        z_tensor = None
        if z is not None:
            z_reshaped = z.reshape(batch, nchunks, chunk_size, nheads, headdim)
            assert z.dtype == io_torch_dtype, (
                f"z dtype {z.dtype} doesn't match {io_torch_dtype}"
            )
            z_tensor = _wrap_tensor(
                z_reshaped.permute(4, 2, 1, 3, 0),
                (2, 3, 4),
                (4, 2, 1, 3, 0),
            )

        # dt: already in io_dtype from cumsum kernel, just permute
        delta_tensor = _wrap_tensor(
            dt_processed.permute(3, 2, 1, 0),
            (1, 2, 3),
            (3, 2, 1, 0),
        )

        # dA_cumsum: (B, EH, C, L) -> (L, C, EH, B)
        assert dA_cumsum.dtype == cutlass_torch.dtype(self._cumsum_dtype), (
            f"dA_cumsum dtype {dA_cumsum.dtype} doesn't match "
            f"cumsum_dtype {self._cumsum_dtype}"
        )
        cumsum_delta_tensor = _wrap_tensor(
            dA_cumsum.permute(3, 2, 1, 0),
            (1, 2, 3),
            (3, 2, 1, 0),
        )

        # B: (B, seqlen, G, N) -> (L, N, C, G, B)
        B_reshaped = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert B.dtype == io_torch_dtype, (
            f"B dtype {B.dtype} doesn't match {io_torch_dtype}"
        )
        b_tensor = _wrap_tensor(
            B_reshaped.permute(2, 4, 1, 3, 0),
            (2, 3, 4),
            (4, 2, 0, 3, 1),
        )

        # C: (B, seqlen, G, N) -> (L, N, C, G, B)
        C_reshaped = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert C.dtype == io_torch_dtype, (
            f"C dtype {C.dtype} doesn't match {io_torch_dtype}"
        )
        c_tensor = _wrap_tensor(
            C_reshaped.permute(2, 4, 1, 3, 0),
            (2, 3, 4),
            (4, 2, 0, 3, 1),
        )

        # D: must be io_dtype (kernel buffer is io_dtype, no conversion)
        if D is not None:
            assert D.dtype == io_torch_dtype, (
                f"D dtype {D.dtype} doesn't match io_dtype {io_torch_dtype}"
            )
        d_tensor = self._get_or_wrap_d(D) if self._has_d else None

        # init_states (state_dtype)
        init_states_tensor = None
        if self._has_init_states and initial_states is not None:
            assert initial_states.dtype == self._state_torch_dtype, (
                f"init_states dtype {initial_states.dtype} doesn't match "
                f"state_dtype {self._state_torch_dtype}"
            )
            init_states_tensor = _wrap_tensor(
                initial_states.permute(3, 2, 1, 0),
                (2, 3),
                (3, 2, 1, 0),
            )

        # out: (B, EH, D, C, L) -> (L, D, C, EH, B)
        assert out.dtype == io_torch_dtype, (
            f"out dtype {out.dtype} doesn't match {io_torch_dtype}"
        )
        y_tensor = _wrap_tensor(
            out.permute(4, 2, 3, 1, 0),
            (2, 3, 4),
            (4, 3, 1, 2, 0),
        )

        # fstate (cached)
        fstate_batch = initial_states.shape[0] if initial_states is not None else batch
        fstate_tensor, fstate_torch = self._get_or_alloc_fstate(fstate_batch)

        # Stream
        stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

        # -- Step 4: Varlen metadata ------------------------------------------
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

        if (
            seq_idx is not None
            and chunk_indices is not None
            and chunk_offsets is not None
        ):
            num_seqs = initial_states.shape[0] if initial_states is not None else 1
            num_logical_chunks_local = len(chunk_indices)
            seq_chunk_cumsum = self._get_or_alloc_seq_cumsum(
                num_seqs,
                seq_idx.device,
            )
            _get_seq_chunk_cumsum_module().seq_chunk_cumsum(
                seq_idx,
                chunk_indices,
                chunk_offsets,
                seq_chunk_cumsum,
                chunk_size,
                num_logical_chunks_local,
                num_seqs,
            )
            seq_chunk_cumsum_tensor = from_dlpack(
                seq_chunk_cumsum,
                assumed_align=4,
            )

        num_logical_chunks = (
            len(chunk_indices) if chunk_indices is not None else nchunks
        )

        # -- Step 5: Compile on first call ------------------------------------
        if self._max_active_clusters is None:
            hardware_info = cutlass.utils.HardwareInfo()
            self._max_active_clusters = hardware_info.get_max_active_clusters(1)

        if self._compiled_kernel is None:
            self._compiled_kernel = cute.compile[GenerateLineInfo(True)](
                self._kernel_obj,
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
                self._max_active_clusters,
                stream,
            )

        # -- Step 6: Launch kernel --------------------------------------------
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

        # -- Step 7: Output ---------------------------------------------------
        out_view = out.permute(0, 3, 4, 1, 2).reshape(
            batch,
            seqlen,
            nheads,
            headdim,
        )
        fstate_out = fstate_torch.permute(3, 2, 1, 0) if return_final_states else None
        return out_view, fstate_out
